"""Compare legacy and effective-ePSF photometry on real difference stamps.

Run from the source checkout with python -m examples.demo_model_difference.
See docs/validation/M8_DELIVERY.md for the input layout and commands.
"""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr

from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet, PointSource,
    PSFOperator, PSFSpec, RenderContext, Uniform, evaluate_model_fit,
    load_emcee_results, load_model_results, load_model_snapshot,
    log_likelihood_cov_prepared, optimize_model_fit, plot_model_fit,
    plot_residual_diagnostics, prepare_for_fitting, prepare_model_fitting, psf_model,
    run_mcmc, run_model_mcmc, save_emcee_results, save_model_results, save_model_snapshot,
    summarize_model_results,
)
from jwst_psfmc.models import _LegacyPointSourceAdapter

SEED = 8808
NAMES = ("point.flux", "point.dx", "point.dy", "sky.level")
UNITS = ("uJy", "pixel", "pixel", "uJy/pixel")


def parameters(bounds, scales=(1., 1., 1., 1.), shifts=(0., 0., 0., 0.)):
    return ParameterSet([
        Parameter(name, initial=np.mean(scaled), prior=Uniform(*scaled), unit=unit)
        for name, unit, limits, scale, shift in zip(NAMES, UNITS, bounds.values(), scales, shifts)
        for scaled in (np.asarray(limits)*scale + shift,)
    ])


def periodic_covariance(prepared, err):
    if not np.all(prepared["valid"]):
        raise ValueError("the exact legacy-to-dense comparison requires a full stamp")
    # Inverse FFT yields the exact periodic covariance used by legacy.
    lag = np.fft.ifft2(prepared["power_spectrum"]).real
    y, x = np.indices(err.shape).reshape(2, -1)
    cov = lag[(y[:,None]-y)%err.shape[0], (x[:,None]-x)%err.shape[1]]
    return (cov+cov.T)/2 * np.outer(err.ravel(),err.ravel())


def make_figures(snapshot, stem):
    for name, plot in (("fit", plot_model_fit), ("residuals", plot_residual_diagnostics)):
        fig, _ = plot(snapshot)
        fig.savefig(stem.with_name(f"{stem.name}_{name}.png"), dpi=130)
        plt.close(fig)


def quadrature(fit, size):
    """Integrate centroid on a grid and use the conditional flux/sky Gaussian."""
    priors = [entry["prior"] for entry in fit.parameters.to_schema()["parameters"]]
    bounds = [(entry["lo"],entry["hi"]) for entry in priors]
    axis_edges = [np.linspace(*bounds[i],size+1) for i in (1,2)]
    axis_centers = [(edges[1:]+edges[:-1])/2 for edges in axis_edges]
    background = fit.noise.whiten(np.ones(fit.data.shape))
    data = fit.noise.whiten(fit.data)
    means = np.empty((size,size,2))
    sigmas = np.empty_like(means)
    correlations = np.empty((size,size))
    log_density = np.empty((size,size))
    for i,dx in enumerate(axis_centers[0]):
        for j,dy in enumerate(axis_centers[1]):
            pt = fit.model.render(dict(zip(NAMES,(1.,dx,dy,0.))),fit.context)
            design = np.column_stack([fit.noise.whiten(pt),background])
            inverse = np.linalg.inv(design.T@design)
            mean = inverse @ (design.T@data)
            residual = data-design@mean
            means[i,j] = mean
            sigmas[i,j] = np.sqrt(np.diag(inverse))
            correlations[i,j] = inverse[0,1] / np.prod(sigmas[i,j])
            log_density[i,j] = .5*(np.linalg.slogdet(inverse)[1] - residual@residual)
    linear_bounds = np.array([bounds[i] for i in (0,3)])
    lower = (linear_bounds[:,0]-means)/sigmas
    upper = (linear_bounds[:,1]-means)/sigmas
    nodes, integration_weights = np.polynomial.legendre.leggauss(48 if size==33 else 80)
    conditional_sigma = np.sqrt(1-correlations**2)[...,None]

    def rectangle(axis, top):
        other = 1-axis
        lo = np.maximum(lower[...,axis], -10)
        hi = np.maximum(lo,np.minimum(top,10))
        half = (hi-lo)/2
        z = ((hi+lo)/2)[...,None] + half[...,None]*nodes
        other_mean = correlations[...,None]*z
        a = (lower[...,other,None]-other_mean)/conditional_sigma
        b = (upper[...,other,None]-other_mean)/conditional_sigma
        mass = np.where(a>0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a))
        density = np.exp(-.5*z**2)/np.sqrt(2*np.pi)
        return half*((density*mass) @ integration_weights)

    total = rectangle(0,upper[...,0])
    np.testing.assert_allclose(total,rectangle(1,upper[...,1]),atol=1e-9,rtol=1e-9)
    raw_weights = np.exp(log_density-logsumexp(log_density))
    weights = raw_weights*total
    normalization = weights.sum()
    weights /= normalization
    probs = (.16,.5,.84)
    linear = []
    for axis, limits in enumerate(linear_bounds):
        def cdf(value):
            top = np.minimum(upper[...,axis],(value-means[...,axis])/sigmas[...,axis])
            return np.sum(raw_weights*rectangle(axis,top))/normalization

        linear.append([brentq(lambda value: cdf(value)-p,*limits) for p in probs])
    positional = [
        np.interp(probs,np.r_[0,weights.sum(axis=axis).cumsum()],edges)
        for axis, edges in zip((1,0),axis_edges)
    ]
    return np.asarray([linear[0],*positional,linear[1]]), float(normalization)


def experiment(cutout, psf_path, sky, out, steps, grid_only=False):
    full, header = fits.getdata(cutout, header=True)
    wcs = WCS(header).celestial
    x, y = wcs.world_to_pixel_values(*sky)
    ix, iy = np.rint([x, y]).astype(int)
    region = np.s_[iy-4:iy+5, ix-4:ix+5]
    if full[region].shape != (9, 9) or header.get("BUNIT") != "MJy/sr":
        raise ValueError("expected a 9×9 MJy/sr stamp at the catalog position")
    pixel_area_sr = proj_plane_pixel_area(wcs) * (np.pi/180)**2
    convert = pixel_area_sr*1e12
    data = full[region].astype(float)*convert
    err_path = cutout.with_name(cutout.name.replace("_diff.fits", "_diff_error.fits"))
    kernel_path = cutout.with_name(cutout.name.replace("_diff.fits", "_cov_kernel.fits"))
    err = fits.getdata(err_path)[region].astype(float)*convert
    kernel = fits.getdata(kernel_path)
    raw = fits.getdata(psf_path).astype(float)
    limit = 50*np.max(err)
    background_limit = 8*np.max(err)
    burn = steps//4
    args = prepare_for_fitting(data, err, raw, kernel, dx_init=x-ix, dy_init=y-iy,
                               nwalkers=24, flux_prior_bounds=(-limit,limit),
                               bkg_prior_half_width=background_limit, seed=SEED)
    packed = args["prepared"]
    noise = dict(backend="dense", covariance=periodic_covariance(packed,err),
                 normalization="relative", error_kind="difference_error_map")
    legacy_component = _LegacyPointSourceAdapter(
        "point", args["psf_os"], args["oversamp"], args["native_shape"], args["psf_prepare_fraction"]
    )
    bridge = prepare_model_fitting(data,CompositeModel([legacy_component,ConstantBackground("sky")]),
                                   parameters(args["prior_bounds"]),noise_options=noise,data_unit="uJy/pixel")
    # Put both amplitudes on the same phase-zero 9×9 aperture scale.
    crop_start = (np.asarray(raw.shape)-np.asarray(args["psf_os"].shape))//2
    offset_yx = ((np.asarray(raw.shape)-1)/2-crop_start-(np.asarray(args["psf_os"].shape)-1)/2)/4
    shift_x, shift_y = offset_yx[::-1]
    legacy_fraction = psf_model([1,-shift_x,-shift_y],args["psf_os"],4,
                                output_shape=data.shape,native_shape=args["native_shape"],
                                psf_prepare_fraction=args["psf_prepare_fraction"]).sum()
    origin = tuple((np.asarray(raw.shape)[::-1]-1)/2)
    yy, xx = np.indices(data.shape)-4
    norm = raw[(origin[1]+4*yy).astype(int),(origin[0]+4*xx).astype(int)].sum()
    spec = PSFSpec(raw,"effective",.25,origin,"centered 9×9 native pixel aperture",
                   source=str(psf_path),scale=1/norm,native_pixel_scale=np.sqrt(pixel_area_sr)*206264.806247)
    operator = PSFOperator(spec,max_offset=1.8)
    model = CompositeModel([PointSource("point"),ConstantBackground("sky")])
    fit = prepare_model_fitting(data, model, parameters(args["prior_bounds"],(legacy_fraction,1,1,1),
                                                        (0,shift_x,shift_y,0)),
                                context=RenderContext(data.shape,psf_operator=operator),
                                noise_options=noise,data_unit="uJy/pixel")
    scale = np.asarray([legacy_fraction,1,1,1])
    shift = np.asarray([0,shift_x,shift_y,0])
    if grid_only:
        output = {"case":out.name}
        for name, fit_case in (("legacy",bridge),("generic",fit)):
            values = []
            for size in (33,65):
                quantiles, mass = quadrature(fit_case,size)
                if name=="legacy":
                    quantiles = quantiles*scale[:,None]+shift[:,None]
                values.append(dict(size=size,quantiles=quantiles.tolist(),linear_prior_retained_mass=mass))
            output[name] = values
        return output
    started = perf_counter()
    legacy = run_mcmc(**args,nsteps=steps,ncores=1,progress=False,seed=SEED)
    legacy_time = perf_counter()-started
    save_emcee_results(out.with_name(f"{out.name}_legacy.npz"),legacy,burnin=burn,thin=1)
    old = load_emcee_results(out.with_name(f"{out.name}_legacy.npz"))
    draws = legacy.get_chain(discard=burn,flat=True)
    checks = []
    for vector in draws[::max(1,len(draws)//32)]:
        prior = log_likelihood_cov_prepared(vector,packed,args["psf_os"],4,
                                             args["native_shape"],args["psf_prepare_fraction"])
        checks.append(abs(prior-evaluate_model_fit(bridge,vector)["log_likelihood"]))
    if max(checks)>1e-8:
        raise RuntimeError("legacy likelihood bridge differs from the public likelihood")
    optim = optimize_model_fit(fit,objective="ml")
    result = run_model_mcmc(fit,initial=optim if optim.best.success else None,seed=SEED,
                            sampling_options=dict(nsteps=steps,burn_in=burn,nwalkers=24,
                                                  minimum_ess=300,progress=False))
    result = replace(result,metadata={**result.metadata,
        "input_files": [str(p) for p in (cutout,err_path,kernel_path,psf_path)],
        "stamp_center_parent_xy": [ix,iy], "catalog_radec_deg": list(sky),
        "pixel_area_sr": pixel_area_sr, "archive_scope": "signed difference photometry"})
    save_model_results(out,result)
    loaded = load_model_results(out)
    if not loaded.sampling_samples.size:
        raise RuntimeError(f"{out.name}: {loaded.status}: {loaded.stop_reason}")
    for name in ("sampling_samples","physical_samples","log_likelihood","log_prior"):
        np.testing.assert_array_equal(getattr(loaded,name),getattr(result,name))
    best = np.unravel_index(np.argmax(loaded.log_likelihood[burn:]),loaded.log_likelihood[burn:].shape)
    vector = loaded.sampling_samples[best[0]+burn,best[1]]
    snapshot_path = out.with_suffix(".fits")
    save_model_snapshot(snapshot_path,fit,vector,label="Highest-likelihood saved draw",
                        wcs=wcs[region],overwrite=True)
    snapshot = load_model_snapshot(snapshot_path)
    np.testing.assert_array_equal(snapshot["image"],evaluate_model_fit(fit,vector)["image"])
    np.testing.assert_allclose(WCS(snapshot["wcs_header"]).pixel_to_world_values(4,4),
                               wcs.pixel_to_world_values(ix,iy),atol=1e-10,rtol=0)
    make_figures(snapshot,out)
    converted_draws = draws*scale+shift
    np.testing.assert_array_equal(old["chain"],legacy.get_chain())
    new_draws = loaded.physical_samples[burn:].reshape(-1,4)
    return {
        "case": out.name, "status": loaded.status, "stop_reason": loaded.stop_reason,
        "legacy_seconds": legacy_time, "generic_seconds": loaded.diagnostics["runtime_s"],
        "legacy_tau": [float(v) if np.isfinite(v) else None for v in old["tau"]],
        "legacy_posterior": np.quantile(converted_draws,[.16,.5,.84],axis=0).T.tolist(),
        "generic_posterior": np.quantile(new_draws,[.16,.5,.84],axis=0).T.tolist(),
        "bridge_max_delta_log_likelihood": max(checks),
        "columns": ["aperture_flux_uJy","dx_pixel","dy_pixel","background_uJy_per_pixel"],
        "legacy_origin_shift_xy": [shift_x,shift_y], "legacy_aperture_factor": legacy_fraction,
        "generic_summary": summarize_model_results(loaded),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cutouts",type=Path)
    parser.add_argument("psfs",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--steps",type=int,default=5200)
    parser.add_argument("--grid-only",action="store_true",help="write the deterministic posterior reference")
    parser.add_argument("--ra",type=float,default=268.457225)
    parser.add_argument("--dec",type=float,default=65.233628)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    table = []
    for epoch in ("02","03","04"):
        for band in ("f200w","f444w_60mas"):
            stem = f"88_deep_ep{epoch}_{band}"
            table.append(experiment(args.cutouts/f"{stem}_diff.fits",
                                    args.psfs/f"nexus_central_deep_ep{epoch}_{band}_PSF_4_c.fits",
                                    (args.ra,args.dec),args.output/stem,args.steps,args.grid_only))
            print(stem,table[-1].get("status","reference complete"),flush=True)
    name = "quadrature.json" if args.grid_only else "comparison.json"
    (args.output/name).write_text(json.dumps(table,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
