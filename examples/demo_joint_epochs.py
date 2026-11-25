"""Recover one host and a transient light curve from registered mock epochs."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d, correlate2d

from jwst_psfmc import (
    CompositeModel, ConstantBackground, JointObservation, Parameter, ParameterSet,
    PointSource, PSFOperator, PSFSpec, RenderContext, SersicSource, Uniform,
    evaluate_joint_model_fit, load_model_results, optimize_model_fit,
    prepare_joint_model_fitting, run_model_mcmc, save_model_results,
    shared_reference_covariance, summarize_model_results,
)


def mock_frames():
    epochs = ("pre", "rise", "late")
    model = CompositeModel([PointSource("point"), SersicSource("host"), ConstantBackground("sky")])
    definitions = [
        Parameter("host.flux", initial=1., prior=Uniform(0, 4), unit="uJy"),
        Parameter("host.n", fixed=1.), Parameter("host.re", fixed=2.3, unit="pixel"),
        Parameter("host.q", fixed=.72), Parameter("host.pa", fixed=34., unit="deg"),
        Parameter("pre.flux", fixed=0., unit="uJy"),
        Parameter("rise.flux", initial=.4, prior=Uniform(-1, 3), unit="uJy"),
        Parameter("late.flux", initial=.2, prior=Uniform(-1, 3), unit="uJy"),
    ]
    truth = {"host.flux": 1.5, "rise.flux": .84, "late.flux": .40}
    frames = []
    generator = np.random.default_rng(9012)
    grid = np.arange(-32,33)/4
    smoothing = np.array([[0.,.12,0.],[.12,1.,.12],[0.,.12,0.]])
    smoothing /= np.linalg.norm(smoothing)
    kernel = correlate2d(smoothing,smoothing)
    for i, name in enumerate(epochs):
        sigma_x, sigma_y = (.85+.13*i, .92+.10*i)
        psf = np.exp(-.5*((grid[None,:]/sigma_x)**2+(grid[:,None]/sigma_y)**2))
        psf /= psf[::4,::4].sum()
        spec = PSFSpec(psf,"effective",.25,(32.,32.),"mock common unit response",
                       source=f"independent mock epoch {name}",native_pixel_scale=.03)
        size = 13 + 2*i
        context = RenderContext((size,size),psf_operator=PSFOperator(spec,oversamp=4))
        offset = (.19*i,-.13*i)
        fixed = {"host.dx": -.23+offset[0], "host.dy": .35+offset[1],
                 "point.dx": .47+offset[0], "point.dy": -.50+offset[1]}
        aliases = {"point.flux": f"{name}.flux", "sky.level": f"{name}.sky"}
        for key,value in fixed.items():
            target = f"{name}.{key}"
            definitions.append(Parameter(target,fixed=value,unit="pixel"))
            aliases[key] = target
        definitions.append(Parameter(f"{name}.sky",initial=0,prior=Uniform(-.05,.05),
                                     unit="uJy/pixel"))
        truth[f"{name}.sky"] = (.003,-.002,.001)[i]
        local = {"host.flux":truth["host.flux"],"host.n":1.,"host.re":2.3,
                 "host.q":.72,"host.pa":34.,"point.flux":truth.get(f"{name}.flux",0.),
                 "sky.level":truth[f"{name}.sky"],**fixed}
        amplitude = .008+.002*i
        noise = amplitude*convolve2d(generator.normal(size=(size+2,size+2)),smoothing,mode="valid")
        data = model.render(local,context)+noise
        mask = np.zeros(data.shape,dtype=bool)
        mask[0,i] = True
        frames.append(JointObservation(
            name=name,band="F200W",data=data,model=model,context=context,aliases=aliases,
            exposures=(f"mock_sci_{name}",),registration="known frame-specific subpixel translation",
            calibration="common mock aperture scale at 30 mas",data_unit="uJy/pixel",
            noise_options=dict(backend="dense",err=np.full(data.shape,amplitude),mask=mask,
                               kernel=kernel,boundary="finite",error_kind="mock background"),
        ))
    return frames, ParameterSet(definitions), truth


def closed_form(fit, observations=None):
    values = fit.parameters.decode(fit.parameters.initial)
    names = fit.parameters.free_names
    all_columns = []
    all_vectors = []
    for observation in fit.observations if observations is None else observations:
        frame = observation.fit
        zero = {**values,**dict.fromkeys(names,0.)}
        baseline = frame.model.render(observation.local_values(zero),frame.context)
        design = []
        for name in names:
            image = frame.model.render(observation.local_values({**zero,name:1.}),frame.context)
            design.append(frame.noise.whiten(image-baseline))
        all_columns.append(np.column_stack(design))
        all_vectors.append(frame.noise.whiten(frame.data-baseline))
    matrix, data = np.vstack(all_columns), np.concatenate(all_vectors)
    used = np.any(matrix != 0.,axis=0)
    design = matrix[:,used]
    covariance = np.linalg.inv(design.T@design)
    values = covariance@(design.T@data)
    return {name:{"mean":value,"sigma":sigma}
            for name,value,sigma in zip(np.asarray(names)[used],values,np.sqrt(np.diag(covariance)))}


def reference_example():
    science = [np.eye(1)]*3
    covariance = shared_reference_covariance(science,np.array([[2.]]),[np.eye(1)]*3)
    # Difference frames measure the same toy relative signal, with a shared
    # noise term. This oracle isolates the error from removing cross blocks.
    correct = np.sqrt(1/np.linalg.inv(covariance).sum())
    naive = np.sqrt(1/np.linalg.inv(np.diag(covariance.diagonal())).sum())
    np.testing.assert_allclose(correct,np.sqrt(2+1/3))
    np.testing.assert_allclose(naive,1.)
    return dict(covariance=covariance.tolist(),shared_mean_sigma=correct,
                independent_marginals_sigma=naive)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    frames, parameters, truth = mock_frames()
    joint = prepare_joint_model_fitting(frames,parameters)
    reference = closed_form(joint)
    single = closed_form(joint, joint.observations[:1])
    optimum = optimize_model_fit(joint,objective="ml")
    if not optimum.best.success:
        raise RuntimeError(optimum.best.message)
    targets = np.array([reference[name]["mean"] for name in parameters.free_names])
    np.testing.assert_allclose(optimum.best.sample_vector,targets,rtol=0,atol=3e-6)
    posterior = run_model_mcmc(
        joint,initial=optimum,seed=9021,
        sampling_options=dict(nsteps=6400,burn_in=1100,nwalkers=28,minimum_ess=400,progress=False),
    )
    path = args.output/"joint.npz"
    save_model_results(path,posterior)
    restored = load_model_results(path)
    np.testing.assert_array_equal(restored.sampling_samples,posterior.sampling_samples)
    diagnostics = evaluate_joint_model_fit(joint,optimum.best.sample_vector,decompose=True)
    for frame in diagnostics["observations"].values():
        np.testing.assert_array_equal(frame["image"],sum(frame["components"].values()))
    recovered = summarize_model_results(restored)
    for name, target in reference.items():
        assert abs(recovered[name]["median"]-target["mean"]) < .15*target["sigma"]
    summary = {
        "epochs": list(diagnostics["observations"]), "truth":truth,
        "parameters": list(restored.free_names),"status":restored.status,
        "diagnostics":restored.diagnostics,"posterior":recovered,
        "gaussian_reference":reference,"pre_only_host":single["host.flux"],
        "shared_reference_oracle":reference_example(),
    }
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(restored.status,restored.stop_reason)


if __name__ == "__main__":
    main()
