"""Fit a mock direct image containing a point source and a resolved host."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet, PointSource,
    PSFOperator, PSFSpec, RenderContext, SersicSource, Uniform, evaluate_model_fit,
    load_model_results, load_model_snapshot, optimize_model_fit, plot_decomposition,
    plot_model_fit, plot_residual_diagnostics, prepare_model_fitting, run_model_mcmc,
    save_model_results, save_model_snapshot, summarize_decomposition,
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--psf", type=Path,default=Path(__file__).with_name("PSF")/"example1_f200w_PSF_4_c.fits")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    raw = fits.getdata(args.psf).astype(float)
    cy,cx = (np.array(raw.shape)-1)//2
    # Explicitly restrict renderer support to ±15 native pixels and
    # normalize to a centered 0.36 arcsec aperture at 30 mas.
    support = raw[cy-60:cy+61,cx-60:cx+61]
    y,x = np.ogrid[-12:13,-12:13]
    reference = support[12:109:4,12:109:4]
    psf = PSFSpec(support,"effective",.25,(60,60),"centered 0.36 arcsec aperture",
                  source=str(args.psf),scale=1/reference[np.hypot(x,y)<=12].sum(),native_pixel_scale=.03)
    context = RenderContext((25,25),psf_operator=PSFOperator(psf,oversamp=8))
    model = CompositeModel([PointSource("point"),SersicSource("host"),ConstantBackground("sky")])
    truth = {"point.flux": 1.1, "point.dx": .25, "point.dy": -.16, "host.flux": .8,
             "host.dx": .25, "host.dy": -.16, "host.re": 3.5, "host.n": 1.,
             "host.q": .72, "host.pa": 28., "sky.level": .003}
    parameters = ParameterSet([
        Parameter("point.flux",initial=1,prior=Uniform(0,3),unit="uJy"),
        Parameter("host.flux",initial=.6,prior=Uniform(0,3),unit="uJy"),
        Parameter("sky.level",initial=0,prior=Uniform(-.05,.05),unit="uJy/pixel"),
        *[Parameter(name,fixed=value) for name,value in truth.items()
          if name not in ("point.flux","host.flux","sky.level")],
    ])
    smooth = np.array([[0.,.13,0.],[.13,1.,.13],[0.,.13,0.]])
    smooth /= np.linalg.norm(smooth)
    from scipy.signal import correlate2d

    kernel = correlate2d(smooth,smooth,mode="full")
    err = np.full(context.shape,.008)
    white = np.random.default_rng(8109).normal(size=(27,27))
    data = model.render(truth,context)+err*convolve(white,smooth)[1:-1,1:-1]
    fit = prepare_model_fitting(data,model,parameters,context=context,
                                noise_options=dict(err=err,backend="dense",kernel=kernel,
                                                   boundary="finite",error_kind="mock_background"),
                                data_unit="uJy/pixel")
    optimum = optimize_model_fit(fit,objective="ml")
    result = run_model_mcmc(fit,initial=optimum,seed=8810,
                            sampling_options=dict(nwalkers=24,nsteps=2800,burn_in=600,
                                                  minimum_ess=400,progress=False))
    result = replace(result,metadata={**result.metadata,"mock_truth":truth,"pixel_area_arcsec2":.03**2,
                                      "inputs":{"psf":str(args.psf),"noise_seed":8109}})
    posterior_path = args.output/"direct.npz"
    save_model_results(posterior_path,result)
    saved = load_model_results(posterior_path)
    if not saved.sampling_samples.size:
        raise RuntimeError(f"{saved.status}: {saved.stop_reason}")
    np.testing.assert_array_equal(saved.sampling_samples,result.sampling_samples)
    summary = summarize_decomposition(fit,saved,aperture=np.hypot(x,y)<=12)
    snapshot_path = args.output/"direct.fits"
    save_model_snapshot(snapshot_path,fit,optimum.best.sample_vector,
                        label="Maximum likelihood, fixed mock geometry",overwrite=True)
    snapshot = load_model_snapshot(snapshot_path)
    np.testing.assert_array_equal(snapshot["image"],evaluate_model_fit(fit,optimum.best.sample_vector)["image"])
    np.testing.assert_array_equal(snapshot["image"],sum(snapshot["components"].values()))
    for name, plot in (("fit",plot_model_fit),("components",plot_decomposition),
                       ("residuals",plot_residual_diagnostics)):
        fig,_ = plot(snapshot)
        fig.savefig(args.output/f"direct_{name}.png",dpi=135)
        plt.close(fig)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(result.status,result.stop_reason)


if __name__ == "__main__":
    main()
