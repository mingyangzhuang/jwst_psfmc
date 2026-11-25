"""Audit object 88 inputs and test a conditional fit on real multi-epoch sky."""

from argparse import ArgumentParser
from itertools import combinations
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
import numpy as np
from scipy.linalg import solve_triangular
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates, maximum_filter, spline_filter
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from jwst_psfmc import PSFOperator, PSFSpec, RenderContext, SersicProfile
from scripts.validate_empirical_psf import linear_fit, measure_kernel, noise_factor, write_csv

EPOCHS = tuple(f"ep{index:02}" for index in range(1, 10))
RA_DEC = (268.457225, 65.233628)
SKY_HALF = 12
FIT_HALF = 7


def check_grid(first, second, shape):
    y, x = np.array([[0., shape[0]/2, shape[0]-1], [0., shape[1]/2, shape[1]-1]])
    transformed = second.world_to_pixel_values(*first.pixel_to_world_values(x, y))
    np.testing.assert_allclose(transformed, (x, y), atol=1e-5, rtol=0)


def epoch_inputs(cutouts, early_psfs, late_psfs):
    records, arrays, veto_arrays = {}, {}, []
    for i, epoch in enumerate(EPOCHS):
        prefix = f"88_deep_{epoch}"
        data_file = cutouts/f"{prefix}_f200w_data.fits"
        err_file = cutouts/f"{prefix}_f200w_error.fits"
        psf_file = (early_psfs if i<7 else late_psfs)/f"nexus_central_deep_{epoch}_f200w_PSF_4_c.fits"
        native_file = psf_file.with_name(psf_file.name.replace("_4_c", "_1_c"))
        data, header = fits.getdata(data_file, header=True)
        err, error_header = fits.getdata(err_file, header=True)
        veto, veto_header = fits.getdata(cutouts/f"{prefix}_f444w_data.fits", header=True)
        wcs = WCS(header).celestial
        if i==0:
            reference = wcs
        if data.shape != err.shape or data.shape != veto.shape or header["BUNIT"].strip()!="MJy/sr":
            raise ValueError(f"inconsistent direct cutout inputs for {epoch}")
        for other in (reference, WCS(error_header).celestial, WCS(veto_header).celestial):
            check_grid(wcs, other, data.shape)
        pixel_area_sr = proj_plane_pixel_area(wcs)*(np.pi/180)**2
        conversion = pixel_area_sr*1e12
        radius = 12
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        circular = np.hypot(xx,yy)<=radius
        raw = fits.getdata(psf_file).astype(float)
        native = fits.getdata(native_file).astype(float)
        cy, cx = (np.asarray(raw.shape)-1)//2
        ny, nx = (np.asarray(native.shape)-1)//2
        centered = raw[cy-4*radius:cy+4*radius+1:4,cx-4*radius:cx+4*radius+1:4]
        native = native[ny-radius:ny+radius+1,nx-radius:nx+radius+1]
        norm = centered[circular].sum()
        native_norm = native[circular].sum()
        response_agreement = np.linalg.norm((centered/norm-native/native_norm)[circular])
        # Restrict extended-source convolution to the same angular support
        # in every epoch. The phase-zero normalization is fixed separately.
        support = raw[cy-60:cy+61,cx-60:cx+61]
        operator = PSFOperator(PSFSpec(support,"effective",.25,(60,60),
                                       "centered 0.36 arcsec aperture",source=str(psf_file),scale=1/norm,
                                       native_pixel_scale=.03),oversamp=8)
        records[epoch] = dict(science=str(data_file),err=str(err_file),psf=str(psf_file),
                              pixel_area_sr=pixel_area_sr,observed_shape=list(data.shape),
                              native_psf_aperture_delta=response_agreement,
                              saved_exposure_metadata={k:header[k] for k in
                                  ("OBSERVTN","ASNTABLE","OBS_ID","FILENAME","MJD-BEG") if k in header})
        arrays[epoch] = (data.astype(float)*conversion,err.astype(float)*conversion,operator)
        if np.isfinite(veto).sum() > 0:
            veto_arrays.append(veto.astype(float)-np.nanmedian(veto))
    return arrays,records,np.nanmedian(veto_arrays,axis=0),reference


def positions(arrays, veto, wcs):
    filled = np.where(np.isfinite(veto),veto,0)
    smooth = gaussian_filter(filled,1.15)
    median = np.median(smooth)
    mad = 1.4826*np.median(np.abs(smooth-median))
    mask = binary_dilation((smooth>median+4*mad) | ~np.isfinite(veto),iterations=4)
    target_x,target_y = wcs.world_to_pixel_values(*RA_DEC)
    height,width = veto.shape
    valid = np.logical_and.reduce([np.isfinite(d)&np.isfinite(e)&(e>0) for d,e,_ in arrays.values()])
    accepted = []
    for y in range(SKY_HALF,height-SKY_HALF,2*SKY_HALF+1):
        for x in range(SKY_HALF,width-SKY_HALF,2*SKY_HALF+1):
            region = np.s_[y-SKY_HALF:y+SKY_HALF+1,x-SKY_HALF:x+SKY_HALF+1]
            if (np.hypot(x-target_x,y-target_y)>70 and
                    valid[region].all() and not np.any(mask[region])):
                accepted.append((x,y))
    np.random.default_rng(10010).shuffle(accepted)
    if len(accepted)<64:
        raise RuntimeError(f"only {len(accepted)} common blank patches")
    return accepted[:len(accepted)//2],accepted[len(accepted)//2:]


def analysis(arrays, train, held):
    s = 2*FIT_HALF+1
    shape = (s,s)
    context = RenderContext(shape)
    phases = ((0.,0.),(.22,-.27),(-.34,.18),(.44,.39))
    factors,templates,noise = {},{},{}
    for epoch,(data,err,operator) in arrays.items():
        patches = [(x,y,data[y-SKY_HALF:y+SKY_HALF+1,x-SKY_HALF:x+SKY_HALF+1],
                        err[y-SKY_HALF:y+SKY_HALF+1,x-SKY_HALF:x+SKY_HALF+1],None) for x,y in train]
        kernel,variance,pairs = measure_kernel(patches)
        factor,minimum,shrink = noise_factor(kernel,shape)
        factors[epoch] = factor
        noise[epoch] = dict(training_variance=variance,training_pairs=pairs,
                           min_eigenvalue=minimum,shrinkage=shrink,kernel=kernel.tolist())
        templates[epoch] = [
            (operator.render_point(dx,dy,context),
             operator.render_sersic(SersicProfile(1,2.3,1,.72,34),dx-.8,dy+.5,context))
            for dx,dy in phases
        ]
        training_z = []
        crop = np.s_[SKY_HALF-FIT_HALF:SKY_HALF+FIT_HALF+1,
                     SKY_HALF-FIT_HALF:SKY_HALF+FIT_HALF+1]
        for i, (_, _, image, sigma, _) in enumerate(patches):
            point, _ = templates[epoch][i%len(phases)]
            fit, error, _ = linear_fit(image[crop],[point,np.ones(shape)],sigma[crop],factor)
            training_z.append(fit[0]/error[0])
        sky_scale = np.std(training_z,ddof=1)
        if not np.isfinite(sky_scale) or sky_scale<=0:
            raise ValueError(f"invalid forced-photometry training scale in {epoch}")
        noise[epoch]["training_aperture_scale"] = float(sky_scale)
    rows = []
    count = len(EPOCHS)
    point_truth = np.array([0.]*7+[.040,.019])
    truth = np.r_[.160,point_truth,np.zeros(count)]
    sky = np.ones(shape)
    for i,(x,y) in enumerate(held):
        region = np.s_[y-FIT_HALF:y+FIT_HALF+1,x-FIT_HALF:x+FIT_HALF+1]
        phi = i%len(phases)
        for label in ("diagonal","finite","finite_skycal"):
            designs, vectors, blank = [],[],[]
            for j,epoch in enumerate(EPOCHS):
                data,err,_ = arrays[epoch]
                stamp = data[region]; sigma = err[region]
                point,host = templates[epoch][phi]
                factor = factors[epoch] if label!="diagonal" else None
                if label=="finite_skycal":
                    sigma = sigma*noise[epoch]["training_aperture_scale"]
                coef, uncertainty,_ = linear_fit(stamp,[point,sky],sigma,factor)
                blank.append((coef[0],uncertainty[0]))
                design = np.zeros((stamp.size,1+2*count))
                design[:,[0,1+j,1+count+j]] = np.column_stack([
                    host.ravel(),point.ravel(),sky.ravel()])/sigma.reshape(-1,1)
                vec = stamp.ravel()/sigma.ravel()
                if factor is not None:
                    transformed = solve_triangular(factor,np.column_stack([design,vec]),
                                                   lower=True,check_finite=False)
                    design,vec = transformed[:,:-1],transformed[:,-1]
                designs.append(design)
                vectors.append(vec)
            design = np.vstack(designs)
            vec = np.concatenate(vectors)
            covariance = np.linalg.inv(design.T@design)
            estimates = covariance@(design.T@(vec+design@truth))
            errors = np.sqrt(np.diag(covariance))
            np.testing.assert_allclose(estimates-truth,covariance@(design.T@vec),
                                       rtol=1e-9,atol=1e-11)
            rows.append(dict(x=x,y=y,phase=phi,backend=label,parameter="host",
                             truth=truth[0],flux=estimates[0],sigma=errors[0],blank_z=np.nan))
            for j,epoch in enumerate(EPOCHS):
                rows.append(dict(x=x,y=y,phase=phi,backend=label,parameter=epoch,
                                 truth=point_truth[j],flux=estimates[j+1],sigma=errors[j+1],
                                 blank_z=blank[j][0]/blank[j][1]))
    return rows,noise


def registration_check(arrays, wcs):
    """Estimate shifts of the surrounding scene after matching the PSFs."""
    ix, iy = np.rint(wcs.world_to_pixel_values(*RA_DEC)).astype(int)
    half = 105
    crop = np.s_[iy-half:iy+half+1,ix-half:ix+half+1]
    yy, xx = np.ogrid[-half:half+1,-half:half+1]
    distance = np.hypot(xx, yy)
    psf_context = RenderContext((25,25))
    scenes = {}
    for epoch,(image,_,operator) in arrays.items():
        data = image[crop]
        if data.shape != (2*half+1,)*2 or not np.isfinite(data).any():
            raise ValueError(f"registration check needs a covered stamp for {epoch}")
        point = operator.render_point(0,0,psf_context)
        valid = np.isfinite(data)
        safe = ~maximum_filter(~valid,size=31)
        scenes[epoch] = np.where(valid,data-np.nanmedian(data),0.), point/point.sum(), safe
    reference, ref_psf, reference_valid = scenes["ep01"]
    results = []
    for epoch,(data,psf,covered) in scenes.items():
        if epoch=="ep01":
            continue
        moving = fftconvolve(data,ref_psf,mode="same")
        fixed = fftconvolve(reference,psf,mode="same")
        coefficients = spline_filter(moving,order=3,mode="nearest")
        for inner,outer in ((6,22),(9,28),(28,60),(34,82)):
            common = covered & reference_valid
            valid = (distance>=inner)&(distance<=outer)&common
            outer_sky = (distance>82)&(distance<90)&common
            if valid.sum()<150 or outer_sky.sum()<100:
                raise ValueError(f"insufficient matched field for {epoch}")
            outer_residual = (moving-fixed)[outer_sky]
            noise = 1.4826*np.median(abs(outer_residual-np.median(outer_residual)))
            if noise<=0:
                raise ValueError(f"invalid registration noise scale for {epoch}")
            uy,ux = np.nonzero(valid)

            def residual(theta):
                dx,dy,amplitude,bkg = theta
                shifted = map_coordinates(coefficients,(uy-dy,ux-dx),order=3,
                                          mode="nearest",prefilter=False)
                return (amplitude*shifted+bkg-fixed[valid])/noise

            fit = least_squares(residual,(0.,0.,1.,0.),loss="soft_l1",
                                bounds=([-1.5,-1.5,.5,-np.inf],[1.5,1.5,1.5,np.inf]))
            if not fit.success:
                raise RuntimeError(f"registration check failed for {epoch}")
            dx,dy,amplitude,bkg = fit.x
            results.append(dict(epoch=epoch,inner_radius_px=inner,outer_radius_px=outer,
                                n_valid=int(np.count_nonzero(valid)),dx_px=dx,dy_px=dy,
                                near_bound=bool(max(abs(dx),abs(dy))>=1.49),
                                amplitude=amplitude,background_uJy=bkg))
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cutouts",type=Path)
    parser.add_argument("early_psfs",type=Path)
    parser.add_argument("late_psfs",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--registration-only",action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    arrays,records,veto,wcs = epoch_inputs(args.cutouts,args.early_psfs,args.late_psfs)
    if args.registration_only:
        registration = registration_check(arrays,wcs)
        write_csv(args.output/"registration.csv",registration)
        return
    train,held = positions(arrays,veto,wcs)
    print(f"{len(train)} training and {len(held)} held-out positions",flush=True)
    rows,noise = analysis(arrays,train,held)
    write_csv(args.output/"held_out.csv",rows)
    results = {}
    for backend in ("diagonal","finite","finite_skycal"):
        results[backend] = {"parameters":{}}
        vectors = []
        for parameter in ("host",*EPOCHS):
            group = [r for r in rows if r["backend"]==backend and r["parameter"]==parameter]
            errors = np.array([r["flux"]-r["truth"] for r in group])
            z = errors/np.array([r["sigma"] for r in group])
            stats = dict(n=len(group),median_bias_uJy=float(np.median(errors)),
                         sd_z=float(np.std(z,ddof=1)),coverage95=int(np.count_nonzero(abs(z)<=1.96)))
            if parameter!="host":
                blank_stats = np.array([r["blank_z"] for r in group])
                vectors.append(blank_stats)
                stats["blank_sd"] = float(np.std(blank_stats,ddof=1))
            results[backend]["parameters"][parameter] = stats
        correlation = np.corrcoef(vectors)
        results[backend]["cross_epoch_blank_correlation"] = {
            f"{EPOCHS[i]}_{EPOCHS[j]}": float(correlation[i,j])
            for i,j in combinations(range(len(EPOCHS)),2)
        }
    report = dict(records=records,training_xy=train,heldout_xy=held,noise=noise,results=results,
                  exposure_independence_verified=False,
                  inference_scope="conditional block-diagonal GLS on real direct-image sky; no transient fit")
    (args.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
