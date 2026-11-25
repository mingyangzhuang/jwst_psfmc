"""Test F200W forced flux at the real target against withheld pre-SN scenes."""

from argparse import ArgumentParser
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from scipy.ndimage import maximum_filter, shift
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from scripts.validate_empirical_psf import linear_fit, measure_kernel, noise_factor, write_csv
from scripts.validate_m10_epochs import EPOCHS, RA_DEC, check_grid, epoch_inputs, positions
from jwst_psfmc import RenderContext


def adopted_alignment(products, wcs, shape):
    pre = products/"preexplosion_coadd/88_preSN_f200w_data.fits"
    header = fits.getheader(pre)
    check_grid(wcs, WCS(header).celestial, shape)
    reference = header["REFEPOCH"]
    offsets = {}
    for i in range(1,header["NSTACK"]+1):
        label = f"{i:02d}"
        name = header[f"EPOCH{label}"]
        if name.startswith("deep_"):
            offsets[name.removeprefix("deep_")] = {
                "xy": [float(v) for v in header[f"SHIFT{label}"].split(",")],
                "source": header[f"SHSRC{label}"], "file": str(pre),
            }
    for epoch in EPOCHS[-2:]:
        aligned = products/f"aligned_images/88_deep_{epoch}_f200w_aligned_data.fits"
        h = fits.getheader(aligned)
        check_grid(wcs, WCS(h).celestial, shape)
        if not (h["ALIGNED"] and h["ALIGNVI"] and h["ALIGNREF"]==reference
                and h["ALIGNEP"]==f"deep_{epoch}" and h["ALIGNROL"]=="postSN"):
            raise ValueError(f"unexpected registration provenance for {aligned}")
        offsets[epoch] = {"xy": [h["ALIGNDX"],h["ALIGNDY"]], "source": h["ALIGNSRC"],
                          "file": str(aligned)}
    if set(offsets)!=set(EPOCHS):
        raise ValueError("registration metadata must cover all nine epochs")
    return reference, offsets


def registered_inputs(arrays, offsets):
    aligned = {}
    for epoch,(data,err,op) in arrays.items():
        dx,dy = offsets[epoch]["xy"]
        good = np.isfinite(data)&np.isfinite(err)&(err>0)
        xy = (dy,dx)
        image = shift(np.where(good,data,0.),xy,order=3)
        # The variance image is a spatial scale proxy. The actual
        # autocorrelation and aperture scatter are measured after matching.
        variance = shift(np.where(good,err**2,0.),xy,order=1)
        footprint = shift(good.astype(float),xy,order=0)>0
        safe = (variance>0)&~maximum_filter(~footprint,size=35,mode="constant",cval=1)
        psf = op.render_point(-dx,-dy,RenderContext((65,65)))
        psf = shift(psf,xy,order=3)[window(32,32,14)]
        aligned[epoch] = image,variance,safe,psf/psf.sum()
    return aligned


def paired_stamp(observed, pre):
    data,variance,valid,kernel = observed
    ref,ref_variance,ref_valid,ref_kernel = pre
    difference = fftconvolve(data,ref_kernel,mode="same")-fftconvolve(ref,kernel,mode="same")
    proxy = (fftconvolve(variance,ref_kernel**2,mode="same")
             + fftconvolve(ref_variance,kernel**2,mode="same"))
    # Ratios use the local, post-matching scatter calibration, not the raw proxy.
    return difference,np.sqrt(proxy.clip(min=0.)),valid&ref_valid


def window(x,y,half):
    return np.s_[y-half:y+half+1,x-half:x+half+1]


def audit_exposures(root):
    return {epoch: sorted(path.name.removeprefix("F200W_").removesuffix("_cutout.fits")
                          for path in (root/f"individual_exposure_cutouts/deep_{epoch}").glob("F200W_*_cutout.fits"))
            for epoch in EPOCHS}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cutouts",type=Path)
    parser.add_argument("early_psfs",type=Path)
    parser.add_argument("products",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    arrays,_,veto,wcs = epoch_inputs(args.cutouts,args.early_psfs,args.products/"PSFs")
    shape = arrays["ep01"][0].shape
    reference,offsets = adopted_alignment(args.products,wcs,shape)
    train,held = positions(arrays,veto,wcs)
    aligned = registered_inputs(arrays,offsets)
    x,y = wcs.world_to_pixel_values(*RA_DEC)
    ix,iy = np.rint([x,y]).astype(int)
    dx,dy = x-ix,y-iy
    rows = []
    for epoch in EPOCHS:
        shift_x,shift_y = offsets[epoch]["xy"]
        context = RenderContext((65,65))
        def projected_point(offset):
            ox,oy = offset
            raw = arrays[epoch][2].render_point(dx+ox-shift_x,dy+oy-shift_y,context)
            return shift(raw,(shift_y,shift_x),order=3)

        forward = projected_point((0,0))
        for template in EPOCHS[:7]:
            if epoch==template:
                continue
            diff,proxy,valid = paired_stamp(aligned[epoch],aligned[template])
            point = fftconvolve(forward,aligned[template][3],mode="same")
            training = [(u,v,diff[window(u,v,12)],proxy[window(u,v,12)],None) for u,v in train
                        if valid[window(u,v,12)].all()]
            if len(training)<20:
                raise ValueError(f"too few covariance training stamps for {epoch}/{template}")
            correlation,_,_ = measure_kernel(training,cutoff=8)
            for half in (5,7,9):
                stamp_size = 2*half+1
                local_point = point[window(32,32,half)]
                sky = np.ones_like(local_point)
                factor,_,shrinkage = noise_factor(correlation,local_point.shape)

                def fit(u,v):
                    stamp = window(u,v,half)
                    flux,err,_ = linear_fit(diff[stamp],[local_point,sky],proxy[stamp],factor)
                    return float(flux[0]),float(err[0])

                train_fits = [fit(u,v) for u,v,*_ in training]
                train_z = np.array([flux/sigma for flux,sigma in train_fits])
                scale = np.std(train_z,ddof=1)
                if not np.isfinite(scale) or scale<=0:
                    raise ValueError(f"invalid training scale for {epoch}/{template}")
                held_fits = [fit(u,v) for u,v in held if valid[window(u,v,half)].all()]
                held_z = [flux/sigma/scale for flux,sigma in held_fits]
                stamp = window(ix,iy,half)
                if len(held_z)<20 or not valid[stamp].all():
                    raise ValueError(f"insufficient controls or target coverage for {epoch}/{template}")
                target_flux,target_sigma = fit(ix,iy)
                sigma = target_sigma*scale
                floating = dict(floating_flux_uJy=None,floating_dx_px=None,
                                floating_dy_px=None,floating_at_bound=None)
                if epoch in EPOCHS[-2:]:
                    def profiled(offset):
                        model = fftconvolve(projected_point(offset),aligned[template][3],mode="same")
                        return linear_fit(diff[stamp],[model[window(32,32,half)],sky],proxy[stamp],factor)

                    match = least_squares(lambda offset: profiled(offset)[2],(0.,0.),
                                          bounds=(-1.,1.),max_nfev=50)
                    if not match.success:
                        raise RuntimeError(f"floating position failed for {epoch}/{template}")
                    refined_flux,_,_ = profiled(match.x)
                    floating.update(floating_flux_uJy=float(refined_flux[0]),
                                    floating_dx_px=float(match.x[0]),floating_dy_px=float(match.x[1]),
                                    floating_at_bound=bool(np.max(np.abs(match.x))>.99))
                rows.append(dict(epoch=epoch,template=template,stamp_size=stamp_size,
                                 flux_uJy=target_flux,sky_sigma_uJy=sigma,z=target_flux/sigma,
                                 n_train=len(training),n_held=len(held_z),training_scale=scale,
                                 held_sd=float(np.std(held_z,ddof=1)),
                                 held_within_95=int(np.count_nonzero(np.abs(held_z)<=1.96)),
                                 covariance_shrinkage=shrinkage,**floating))
    write_csv(args.output/"target_pairs.csv",rows)
    summary = {}
    for epoch in EPOCHS:
        for size in (11,15,19):
            group = [row for row in rows if row["epoch"]==epoch and row["stamp_size"]==size]
            fluxes = [row["flux_uJy"] for row in group]
            summary[f"{epoch}_{size}"] = dict(
                template_count=len(group),median_uJy=float(np.median(fluxes)),
                template_range_uJy=[min(fluxes),max(fluxes)],
                median_sky_sigma_uJy=float(np.median([row["sky_sigma_uJy"] for row in group])),
                median_held_sd=float(np.median([row["held_sd"] for row in group])))
            if epoch in EPOCHS[-2:]:
                floating = [row["floating_flux_uJy"] for row in group]
                summary[f"{epoch}_{size}"].update(
                    floating_median_uJy=float(np.median(floating)),
                    floating_template_range_uJy=[min(floating),max(floating)],
                    floating_at_bound=sum(row["floating_at_bound"] for row in group))
    report = dict(reference_grid=reference,alignment=offsets,forced_radec_deg=RA_DEC,
                  exposure_files_partial=audit_exposures(args.products),
                  exposure_independence_verified=False,
                  flux_convention="equivalent native phase-zero 0.36 arcsec aperture; no extinction correction",
                  fit_scope="empirical pre-epoch scenes; individual pair diagnostics; no combination of shared pairs",
                  results=summary)
    (args.output/"target_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Completed {len(rows)} paired real-target diagnostics.")


if __name__=="__main__":
    main()
