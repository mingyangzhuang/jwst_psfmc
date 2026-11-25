"""Measure all three late aligned cutouts against independent pre-SN templates."""

from argparse import ArgumentParser
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.ndimage import shift
from scipy.signal import fftconvolve

from jwst_psfmc import RenderContext, shared_reference_covariance
from scripts.validate_empirical_psf import linear_fit, measure_kernel, noise_factor
from scripts.validate_m10_epochs import RA_DEC
from scripts.validate_m10_shared_reference import BANDS, EPOCHS as DEEP_LATE, blank_centers, read_pair, response
from scripts.validate_m10_target import window


LATE = (*DEEP_LATE, "ddt_ep01")
PRE = ("deep_ep02", "deep_ep04", "deep_ep06")


def load_aligned(products, epoch, band, mas, wcs, shape, conversion):
    a = products/f"aligned_images/88_{epoch}_{band}_aligned_data.fits"
    d, err, h = read_pair(a, a.with_name(a.name.replace("_data","_error")), wcs, shape, conversion)
    if not (h["ALIGNED"] and h["ALIGNVI"] and h["ALIGNEP"]==epoch and h["ALIGNREF"]=="wide_ep01"):
        raise ValueError(f"unexpected alignment provenance in {a}")
    stem = f"nexus_central_{epoch}_{band}_PSF_4_c.fits"
    if epoch=="ddt_ep01":
        stem = f"prop_16602_{band.removesuffix('_60mas').upper()}_PSF_OBS_001_4_c"
        stem += ("_60mas" if mas==60 else "")+".fits"
    return d,err,response(products/"PSFs"/stem,mas),str(a)


def cross_match(late, early, phase, mas):
    data,err,op,_ = late
    ref,ref_err,ref_op,_ = early
    half = round(420/mas)
    context = RenderContext((2*half+1,)*2)
    templates = [renderer.render_point(0.,0.,context) for renderer in (op,ref_op)]
    science_psf, ref_psf = [item/item.sum() for item in templates]
    valid_science = np.isfinite(data)&np.isfinite(err)&(err>0)
    valid_ref = np.isfinite(ref)&np.isfinite(ref_err)&(ref_err>0)
    def match(image, sigma, kernel, good):
        weight = fftconvolve(good.astype(float),kernel,mode="same")
        abs_kernel = np.abs(kernel)
        support = fftconvolve(good.astype(float),abs_kernel,mode="same")/abs_kernel.sum()
        valid = (support >= .98)&(weight > 0)
        norm = np.where(valid,weight,1.)
        signal = fftconvolve(np.where(good,image,0.),kernel,mode="same")/norm
        variance = fftconvolve(np.where(good,sigma**2,0.),kernel**2,mode="same")/norm**2
        return signal,variance,valid
    a,avar,a_valid = match(data,err,ref_psf,valid_science)
    b,bvar,b_valid = match(ref,ref_err,science_psf,valid_ref)
    valid = a_valid&b_valid
    diff,proxy = a-b,avar+bvar
    point_context = RenderContext((4*half+1,)*2)
    point = fftconvolve(op.render_point(*phase,point_context),ref_psf,mode="same")
    return np.where(valid,diff,np.nan),np.where(valid,np.sqrt(proxy.clip(min=0)),np.nan),point


def fit_template(cutouts, psfs, band, mas, reference_epoch, ref_header, late, wcs, shape, conversion):
    h = [(ref_header[f"EPOCH{i:02}"],ref_header[f"SHIFT{i:02}"]) for i in range(1,ref_header["NSTACK"]+1)]
    dx,dy = map(float,dict(h)[reference_epoch].split(","))
    path = cutouts/f"88_{reference_epoch}_{band}_data.fits"
    ref,err,_ = read_pair(path,path.with_name(path.name.replace("_data","_error")),wcs,shape,conversion)
    # The pre-SN direct image is rendered on the saved late-epoch WCS.
    # Interpolate SCI and an ERR proxy, then train on the matched images.
    good = np.isfinite(ref)&np.isfinite(err)&(err>0)
    valid = shift(good.astype(float),(dy,dx),order=0)>0
    ref = np.where(valid,shift(np.where(good,ref,0.),(dy,dx),order=3),np.nan)
    err = np.where(valid,shift(np.where(good,err,0.),(dy,dx),order=1),np.nan)
    psf = psfs/f"nexus_central_{reference_epoch}_{band}_PSF_4_c.fits"
    early = ref,err,response(psf,mas),str(path)
    x,y = wcs.world_to_pixel_values(*RA_DEC)
    ix,iy = np.rint([x,y]).astype(int)
    matched = [cross_match(frame,early,(x-ix,y-iy),mas) for frame in late]
    patch_half = round(270/mas)
    veto_inputs = [(ref,err),*((np.where(np.isfinite(d),frame[0],np.nan),e)
                   for frame,(d,e,_) in zip(late,matched))]
    train,held = blank_centers(veto_inputs,(x,y),patch_half,mas,min_patches=2)
    record = dict(reference_science=str(path),reference_shift_xy=[dx,dy],train_count=len(train),
                  held_count=len(held))
    if min(len(train),len(held)) < 12:
        return dict(**record, skipped_reason="Too few common calibrated sky stamps")
    results = {}
    for radius in (150,210):
        half = round(radius/mas)
        sky = np.ones((2*half+1,)*2)
        points,factors = [],[]
        for d,e,p in matched:
            bank = [(u,v,d[window(u,v,patch_half)],e[window(u,v,patch_half)],None) for u,v in train]
            kernel,_,_ = measure_kernel(bank,cutoff=round(150/mas))
            factors.append(noise_factor(kernel,sky.shape)[0])
            c = p.shape[0]//2
            points.append(p[window(c,c,half)])

        def extract(u,v):
            region = window(u,v,half)
            estimates = []
            for (d,e,_),point,factor in zip(matched,points,factors):
                image,err = d[region],e[region]
                good = np.isfinite(image)&np.isfinite(err)&(err>0)
                if good.all():
                    local_fit = linear_fit(image,[point,sky],err,factor)
                else:
                    if good.sum() < sky.size/2:
                        raise ValueError(f"under half of target stamp supported for {band}/{reference_epoch}")
                    shared = factor[good.ravel()]
                    marginal = cholesky(shared@shared.T,lower=True)
                    local_fit = linear_fit(image[good],[point[good],sky[good]],err[good],marginal)
                estimates.append(local_fit[:2])
            return np.array([[coef[0] for coef,_ in estimates],[sigma[0] for _,sigma in estimates]])

        measured = np.array([extract(u,v) for u,v in train])
        standardized_cov = np.cov((measured[:,0]/measured[:,1]).T,ddof=1)
        estimate,sigma = extract(ix,iy)
        cov = standardized_cov*np.outer(sigma,sigma)
        if np.linalg.eigvalsh(cov)[0] <= 0:
            raise ValueError("target covariance is not positive definite")
        held_measurements = np.array([extract(u,v) for u,v in held])
        z = held_measurements[:,0]/held_measurements[:,1]/np.sqrt(standardized_cov.diagonal())
        contrasts = np.array([[-1,1,0],[-1,0,1],[0,-1,1]])
        contrast_z = []
        for beta,proxy in held_measurements:
            local = standardized_cov*np.outer(proxy,proxy)
            contrast_z.append((contrasts@beta)/np.sqrt((contrasts@local@contrasts.T).diagonal()))
        results[str(radius)] = dict(stamp_size=2*half+1, target_flux_nJy=(1000*estimate).tolist(),
            target_cov_nJy2=(1e6*cov).tolist(), held_scatter=np.std(z,axis=0,ddof=1).tolist(),
            held_contrast_scatter=np.std(contrast_z,axis=0,ddof=1).tolist(),
            target_supported_pixels=[int(np.isfinite(d[window(ix,iy,half)]).sum()) for d,_,_ in matched])
    return dict(**record,radii_mas=results)


def closure(products, band, late, wcs, shape, conversion):
    residuals, errors = [],[]
    for epoch,(image,_,_,_) in zip(LATE,late):
        path = products/f"difference_images_realign/88_{epoch}_{band}_diff.fits"
        d,e,_ = read_pair(path,path.with_stem(path.stem+"_error"),wcs,shape,conversion)
        residuals.append(image-d)
        errors.append(e)
    result = {}
    for j,epoch in enumerate(LATE[1:],1):
        residual = residuals[j]-residuals[0]
        sigma = np.hypot(errors[j],errors[0])
        good = np.isfinite(residual+sigma)&(sigma>0)
        median = np.median(residual[good])
        standardized = (residual[good]-median)/sigma[good]
        result[epoch] = dict(q01_q50_q99_in_nominal_err=np.percentile(standardized,[1,50,99]).tolist(),
                             pixels=int(good.sum()))
    return result


def recovered_f444w60_block(products, late, wcs, shape, conversion):
    """Check a pixel block using the numerically recoverable 60 mas reference."""
    diff, variances, used = [], [], []
    for epoch,(image,err,_,_) in zip(LATE,late):
        path = products/f"difference_images_realign/88_{epoch}_f444w_60mas_diff.fits"
        d,e,_ = read_pair(path,path.with_stem(path.stem+"_error"),wcs,shape,conversion)
        diff.append(d)
        used.append(image-d)
        variances.append(e**2-err**2)
    reference = used[0]
    ref_var = variances[0]
    ref_err = np.sqrt(np.where(ref_var>0,ref_var,np.nan))
    discrepancy = []
    for estimate in variances[1:]:
        good = np.isfinite(ref_err+estimate)
        discrepancy.append(np.percentile(abs(estimate[good]/ref_var[good]-1),[50,95,99]).tolist())
    inputs = [*((d,e) for d,e,_,_ in late),(reference,ref_err)]
    x,y = wcs.world_to_pixel_values(*RA_DEC)
    ix,iy = np.rint([x,y]).astype(int)
    half, patch_half = 4,5
    train,held = blank_centers(inputs,(x,y),patch_half,60)
    sky = np.ones((2*half+1,)*2)
    context = RenderContext(sky.shape)
    points = [op.render_point(x-ix,y-iy,context) for _,_,op,_ in late]
    factors = []
    for i,(d,e) in enumerate(inputs):
        bank = [(u,v,d[window(u,v,patch_half)],e[window(u,v,patch_half)],None) for u,v in train]
        factor = noise_factor(measure_kernel(bank,cutoff=3)[0],sky.shape)[0]
        z = []
        for u,v in train:
            s = window(u,v,half)
            b,err,_ = linear_fit(d[s],[points[min(i,2)],sky],e[s],factor)
            z.append(b[0]/err[0])
        factors.append(factor*np.std(z,ddof=1))
    n = sky.size
    design = np.zeros((3*n,6))
    for i,point in enumerate(points):
        design[i*n:(i+1)*n,2*i:2*i+2] = np.column_stack([point.ravel(),sky.ravel()])
    indices = [0,2,4]
    relative = np.array([[-1,1,0],[-1,0,1],[0,-1,1]])

    def fit(u,v):
        s = window(u,v,half)
        parts = []
        for (_,e),factor in zip(inputs,factors):
            weighted = e[s].reshape(-1,1)*factor
            parts.append(weighted@weighted.T)
        full = shared_reference_covariance(parts[:3],parts[3],[np.eye(n)]*3)
        inverse = cho_solve(cho_factor(full),design)
        covariance = np.linalg.inv(design.T@inverse)
        beta = covariance@inverse.T@np.concatenate([image[s].ravel() for image in diff])
        return beta[indices],covariance[np.ix_(indices,indices)]

    estimates,cov = fit(ix,iy)
    controls = []
    for u,v in held:
        beta,covariance = fit(u,v)
        controls.append((relative@beta)/np.sqrt((relative@covariance@relative.T).diagonal()))
    return dict(reference_reconstruction="ep08 aligned SCI minus round-2 difference, up to a constant",
                reference_variance="ep08 round-2 diff ERR squared minus aligned ERR squared",
                ep09_ddt_relative_reference_variance_discrepancy_p50_p95_p99=discrepancy,
                training_count=len(train),held_count=len(held),target_flux_nJy=(1e3*estimates).tolist(),
                target_cov_nJy2=(1e6*cov).tolist(),
                held_contrast_scatter=np.std(controls,axis=0,ddof=1).tolist())


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("products","cutouts","early_psfs","output"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    results = {}
    for band,mas in BANDS.items():
        reference = args.products/f"preexplosion_coadd/88_preSN_{band}_data.fits"
        data,header = fits.getdata(reference,header=True)
        wcs = WCS(header).celestial
        shape = data.shape
        conversion = proj_plane_pixel_area(wcs)*(np.pi/180)**2*1e12
        frames = [load_aligned(args.products,epoch,band,mas,wcs,shape,conversion) for epoch in LATE]
        results[band] = dict(
            late_images=[frame[3] for frame in frames],late_epochs=LATE,
            round2_reference_cancellation=closure(args.products,band,frames,wcs,shape,conversion),
            fits={epoch:fit_template(args.cutouts,args.early_psfs,band,mas,epoch,header,frames,
                                     wcs,shape,conversion) for epoch in PRE},
        )
        if band=="f444w_60mas":
            results[band]["round2_recovered_pixel_block"] = recovered_f444w60_block(
                args.products,frames,wcs,shape,conversion,
            )
        print(band,"done",flush=True)
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
