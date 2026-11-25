"""Validate shared pre-SN reference noise on the saved AT 2026abck differences."""

from argparse import ArgumentParser
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.optimize import least_squares

from jwst_psfmc import PSFOperator, PSFSpec, RenderContext, SersicProfile, shared_reference_covariance
from scripts.validate_empirical_psf import linear_fit, measure_kernel, noise_factor
from scripts.validate_m10_epochs import RA_DEC, check_grid
from scripts.validate_m10_target import window


BANDS = {"f200w": 30, "f444w": 30, "f444w_60mas": 60}
EPOCHS = ("deep_ep08", "deep_ep09")


def read_pair(path, error, wcs, shape, conversion):
    image, header = fits.getdata(path, header=True)
    sigma, err_header = fits.getdata(error, header=True)
    if image.shape != shape or sigma.shape != shape:
        raise ValueError(f"unexpected image shape at {path}")
    for grid in (header, err_header):
        check_grid(wcs, WCS(grid).celestial, shape)
    return image.astype(float)*conversion, sigma.astype(float)*conversion, header


def response(path, mas):
    psf = fits.getdata(path).astype(float)
    cy, cx = (np.asarray(psf.shape)-1)//2
    radius = round(360/mas)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    native = psf[cy-4*radius:cy+4*radius+1:4, cx-4*radius:cx+4*radius+1:4]
    aperture = native[np.hypot(x, y) <= 360/mas].sum()
    support = 4*min(15, cy//4, cx//4)
    data = psf[window(cx, cy, support)]
    return PSFOperator(PSFSpec(data, "effective", .25, (support, support),
                               "phase-zero 0.36 arcsec aperture", scale=1/aperture,
                               source=str(path), native_pixel_scale=mas/1000), oversamp=8)


def blank_centers(images, xy, half, mas, *, min_patches=40):
    valid = np.logical_and.reduce([np.isfinite(d)&np.isfinite(e)&(e > 0) for d, e in images])
    veto = np.median([gaussian_filter(np.where(valid, d, np.nanmedian(d)), 30/mas) for d, _ in images],
                     axis=0)
    center = np.median(veto[valid])
    mad = 1.4826*np.median(np.abs(veto[valid]-center))
    mask = binary_dilation(veto > center+4.5*mad, iterations=max(2, round(120/mas)))
    stride = 2*half+1
    coords = [(x,y) for y in range(half, valid.shape[0]-half, stride)
              for x in range(half, valid.shape[1]-half, stride)
              if np.hypot(x-xy[0], y-xy[1])*mas > 2100
              and np.all(valid[window(x,y,half)] & ~mask[window(x,y,half)])]
    np.random.default_rng(10110).shuffle(coords)
    if len(coords) < min_patches:
        raise ValueError(f"too few common blank stamps: {len(coords)}")
    return coords[:len(coords)//2], coords[len(coords)//2:]


def fit_band(products, band, mas):
    root = products/"preexplosion_coadd"
    ref_file = root/f"88_preSN_{band}_data.fits"
    raw, header = fits.getdata(ref_file, header=True)
    shape = raw.shape
    wcs = WCS(header).celestial
    conversion = proj_plane_pixel_area(wcs)*(np.pi/180)**2*1e12
    ref_data, ref_err, _ = read_pair(ref_file, root/f"88_preSN_{band}_error.fits", wcs, shape, conversion)
    x, y = wcs.world_to_pixel_values(*RA_DEC)
    ix, iy = np.rint([x,y]).astype(int)
    half = round(210/mas)
    patch_half = round(270/mas)
    context = RenderContext((2*half+1,)*2)
    science, differences, operators, points, provenance = [], [], [], [], []
    for epoch in EPOCHS:
        prefix = f"88_{epoch}_{band}"
        a = products/f"aligned_images/{prefix}_aligned_data.fits"
        d = products/f"diff_preSN/{prefix}_diff_preSN.fits"
        sci, err, h = read_pair(a, a.with_name(a.name.replace("_data", "_error")),
                                wcs, shape, conversion)
        diff, diff_err, dh = read_pair(d, d.with_stem(d.stem+"_error"), wcs, shape, conversion)
        if dh["TMPLN"] != header["NSTACK"] or dh["TMPLSCL"] != 1:
            raise ValueError(f"unexpected reference metadata in {d}")
        good = np.isfinite(sci+ref_data+diff+err+ref_err+diff_err)&(diff_err>0)
        offset = sci[good]-ref_data[good]-diff[good]
        sky = np.median(offset)
        image_delta = np.max(np.abs(offset-sky)/diff_err[good])
        error_delta = np.max(np.abs(np.hypot(err[good], ref_err[good])-diff_err[good])/diff_err[good])
        if max(image_delta, error_delta) > 1e-4:
            raise ValueError(f"failed image/error construction identity for {d}")
        op = response(products/f"PSFs/nexus_central_{epoch}_{band}_PSF_4_c.fits", mas)
        # The distributed array is already registered. Use its aligned-grid phase
        # for the mosaic point template. Interpolation changes are a systematic.
        point = op.render_point(x-ix, y-iy, context)
        science.append((sci, err))
        differences.append(diff)
        operators.append(op)
        points.append(point)
        provenance.append(dict(aligned=str(a), difference=str(d), sky_uJy_per_pixel=sky,
                               max_pixel_identity_sigma=image_delta, max_relative_error_identity=error_delta,
                               alignment_shift_px=[h["ALIGNDX"],h["ALIGNDY"]]))
    inputs = [*science, (ref_data, ref_err)]
    train, held = blank_centers(inputs, (x,y), patch_half, mas)
    factors, noise_scales = [], []
    sky_plane = np.ones(context.shape)
    for i, (image, err) in enumerate(inputs):
        bank = [(u,v,image[window(u,v,patch_half)],err[window(u,v,patch_half)],None) for u,v in train]
        kernel, _, _ = measure_kernel(bank, cutoff=round(150/mas))
        factor, _, _ = noise_factor(kernel, context.shape)
        point = points[min(i, 1)]
        zs = []
        for u,v in train:
            region = window(u,v,half)
            coef, sigma, _ = linear_fit(image[region], [point,sky_plane], err[region], factor)
            zs.append(coef[0]/sigma[0])
        factors.append(factor)
        noise_scales.append(np.std(zs, ddof=1))

    n = sky_plane.size
    design = np.zeros((2*n,4))
    for i, point in enumerate(points):
        design[i*n:(i+1)*n, 2*i:2*i+2] = np.column_stack([point.ravel(),sky_plane.ravel()])

    def local_fit(u,v):
        region = window(u,v,half)
        values = np.concatenate([d[region].ravel() for d in differences])
        parts = []
        for (_, err), scale, factor in zip(inputs, noise_scales, factors):
            local = err[region].reshape(-1,1)*scale*factor
            parts.append(local@local.T)
        full = shared_reference_covariance(parts[:2], parts[2], [np.eye(n)]*2)
        independent = full.copy()
        independent[:n,n:] = 0
        independent[n:,:n] = 0
        estimates = []
        for cov in (full, independent):
            inverse_design = cho_solve(cho_factor(cov), design)
            beta_cov = np.linalg.inv(design.T@inverse_design)
            beta = beta_cov@inverse_design.T@values
            select = np.ix_([0,2],[0,2])
            estimates.append((beta[[0,2]], beta_cov[select]))
        return estimates

    controls = []
    contrast = np.array([-1.,1.])
    for u,v in held:
        row = []
        for beta, cov in local_fit(u,v):
            row.append([*list(beta/np.sqrt(cov.diagonal())),
                        (contrast@beta)/np.sqrt(contrast@cov@contrast)])
        depth = [float(np.median(e[window(u,v,half)])) for _,e in science]
        controls.append((depth, row))
    depth_values = np.array([row[0] for row in controls])
    standardized = np.array([row[1] for row in controls])
    target_depth = [float(np.median(e[window(ix,iy,half)])) for _,e in science]
    order = np.argsort(np.linalg.norm(np.log(depth_values/target_depth),axis=1))
    similar = order[:max(20, len(order)//2)]
    result = {}
    for i,(label,(beta,cov)) in enumerate(zip(("shared","independent"),local_fit(ix,iy))):
        result[label] = dict(flux_nJy=(1000*beta).tolist(), sigma_nJy=(1000*np.sqrt(cov.diagonal())).tolist(),
                             corr=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]), contrast_nJy=1000*(contrast@beta),
                             contrast_sigma_nJy=1000*np.sqrt(contrast@cov@contrast),
                             held_sd=np.std(standardized[:,i],axis=0,ddof=1).tolist(),
                             similar_err_held_sd=np.std(standardized[similar,i],axis=0,ddof=1).tolist())
    output = dict(reference=str(ref_file), reference_epochs=[header[f"EPOCH{i:02}"]
                  for i in range(1,header["NSTACK"]+1)], audit=provenance, training_count=len(train),
                  held_count=len(held), similar_err_count=len(similar),
                  target_median_err_uJy_per_pixel=target_depth, held_median_err_quantiles_uJy_per_pixel=
                  np.percentile(depth_values,[5,50,95],axis=0).tolist(), calibration_scales=noise_scales, fit=result)
    if band.startswith("f444w"):
        # Signal-only injection on held-out difference stamps. Match the PSF
        # or substitute the other real epoch's ePSF to stress the host fit.
        host = operators[0].render_sersic(SersicProfile(1.,70/mas,1.,.67,32),x-ix,y-iy,context)
        sample = []
        for u,v in held:
            stamp = window(u,v,half)
            proxy = np.hypot(science[0][1][stamp],ref_err[stamp])
            # Estimate finite correlation directly on stored differences;
            # same fit and background are used in both injection arms.
            sample.append((differences[0][stamp],proxy))
        bank = [(u,v,differences[0][window(u,v,patch_half)],
                 np.hypot(science[0][1][window(u,v,patch_half)],ref_err[window(u,v,patch_half)]),None)
                for u,v in train]
        factor, _, _ = noise_factor(measure_kernel(bank,cutoff=round(150/mas))[0],context.shape)
        values, recentered = [], []
        for stamp, proxy in sample:
            def signed_host(data, point, galaxy):
                return linear_fit(data, [point,galaxy,sky_plane],proxy,factor)[0][1]

            baseline = signed_host(stamp,points[0],host)
            values.append([(signed_host(stamp+.25*p,points[0],host)-baseline)/.25 for p in points])

            def point_at(delta):
                return operators[0].render_point(x-ix+delta[0],y-iy+delta[1],context)

            def injection_residual(delta):
                return linear_fit(points[1], [point_at(delta),sky_plane],proxy,factor)[2]

            centroid = least_squares(injection_residual, (0.,0.), bounds=(-1.,1.))
            if not centroid.success:
                raise RuntimeError(f"alternate PSF centroid fit failed in {band}")
            dx, dy = centroid.x
            galaxy = operators[0].render_sersic(
                SersicProfile(1.,70/mas,1.,.67,32),x-ix+dx,y-iy+dy,context,
            )
            # Given the centroid, the injected flux delta is exactly linear.
            recentered.append(signed_host(points[1],point_at(centroid.x),galaxy))
        values = np.array(values)
        np.testing.assert_allclose(values[:,0],0,atol=1e-10)
        output["point_only_injection"] = dict(bright_point_nJy=250, host_re_mas=70,
            matched_max_absolute_host_fraction=np.max(np.abs(values[:,0])),
            other_epoch_psf_host_fraction_p05_p50_p95=np.percentile(values[:,1],[5,50,95]).tolist(),
            recentered_alternate_host_fraction_p05_p50_p95=np.percentile(recentered,[5,50,95]).tolist(),
            note="signal-only PSF stress on real sky; no additional source variance injected")
    return output


def fit_realign_flux_covariance(products, band, mas):
    """Use paired blank positions to calibrate flux-level covariance in round 2."""
    root = products/"difference_images_realign"
    epochs = (*EPOCHS, "ddt_ep01")
    first = root/f"88_{epochs[0]}_{band}_diff.fits"
    raw, header = fits.getdata(first,header=True)
    wcs = WCS(header).celestial
    shape = raw.shape
    conversion = proj_plane_pixel_area(wcs)*(np.pi/180)**2*1e12
    x,y = wcs.world_to_pixel_values(*RA_DEC)
    ix,iy = np.rint([x,y]).astype(int)
    half, patch_half = round(210/mas), round(270/mas)
    context = RenderContext((2*half+1,)*2)
    paths, inputs, points = [], [], []
    reference_epochs = [header[f"EPOCH{i:02}"] for i in range(1,header["NSTACK"]+1)]
    for epoch in epochs:
        file = root/f"88_{epoch}_{band}_diff.fits"
        d,e,h = read_pair(file,file.with_stem(file.stem+"_error"),wcs,shape,conversion)
        if [h[f"EPOCH{i:02}"] for i in range(1,h["NSTACK"]+1)] != reference_epochs:
            raise ValueError(f"reference membership differs in {file}")
        stem = f"nexus_central_{epoch}_{band}_PSF_4_c.fits"
        if epoch=="ddt_ep01":
            stem = f"prop_16602_{band.removesuffix('_60mas').upper()}_PSF_OBS_001_4_c"
            stem += ("_60mas" if mas==60 else "")+".fits"
        op = response(products/"PSFs"/stem,mas)
        paths.append(str(file))
        inputs.append((d,e))
        points.append(op.render_point(x-ix,y-iy,context))
    # Veto sources using the saved wide/deep pre-SN image on the same grid.
    # Round-2 uses a distinct coadd; here this image is used only for selection.
    pre = products/f"preexplosion_coadd/88_preSN_{band}_data.fits"
    ref,ref_err,_ = read_pair(pre,pre.with_name(pre.name.replace("_data","_error")),
                            wcs,shape,conversion)
    train,held = blank_centers([(ref,ref_err), *inputs],(x,y),patch_half,mas)
    factors = []
    for d,e in inputs:
        bank = [(u,v,d[window(u,v,patch_half)],e[window(u,v,patch_half)],None) for u,v in train]
        kernel,_,_ = measure_kernel(bank,cutoff=round(150/mas))
        factors.append(noise_factor(kernel,context.shape)[0])
    sky = np.ones(context.shape)

    def extract(u,v):
        region = window(u,v,half)
        fit = [linear_fit(d[region],[point,sky],e[region],factor)[:2]
               for (d,e),point,factor in zip(inputs,points,factors)]
        return np.array([[beta[0] for beta,_ in fit],[sigma[0] for _,sigma in fit]])

    measured = np.array([extract(u,v) for u,v in train])
    covariance = np.cov((measured[:,0]/measured[:,1]).T,ddof=1)
    beta,sigma = extract(ix,iy)
    target_cov = covariance*np.outer(sigma,sigma)
    relative = np.array([[-1.,1.,0.],[-1.,0.,1.],[0.,-1.,1.]])
    all_errors = []
    for use_cov in (covariance, np.diag(covariance.diagonal())):
        held_z = []
        for u,v in held:
            estimate,err = extract(u,v)
            projected = relative@(use_cov*np.outer(err,err))@relative.T
            held_z.append((relative@estimate)/np.sqrt(projected.diagonal()))
        all_errors.append(np.std(held_z,axis=0,ddof=1).tolist())
    return dict(image_paths=paths,epochs=epochs,reference_epochs=reference_epochs,
                training_count=len(train),held_count=len(held),target_flux_nJy=(1e3*beta).tolist(),
                target_empirical_cov_nJy2=(1e6*target_cov).tolist(),
                contrast_labels=["ep09-ep08","ddt-ep08","ddt-ep09"],
                held_contrast_scatter_full=all_errors[0],held_contrast_scatter_diagonal=all_errors[1])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("products",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    realign = {}
    results = {}
    for band, mas in BANDS.items():
        results[band] = fit_band(args.products,band,mas)
        print(band, results[band]["fit"], flush=True)
        realign[band] = fit_realign_flux_covariance(args.products,band,mas)
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    args.output.with_name("realign_flux_covariance.json").write_text(
        json.dumps(realign,indent=2,allow_nan=False)+"\n",
    )


if __name__ == "__main__":
    main()
