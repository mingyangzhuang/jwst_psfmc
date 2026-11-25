"""Independent frames, shared parameters and shared-reference covariance."""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from jwst_psfmc import (
    CompositeModel, ConstantBackground, JointObservation, Parameter, ParameterSet,
    PointSource, PSFOperator, PSFSpec, RenderContext, SersicSource, Uniform,
    evaluate_joint_model_fit, load_model_results, optimize_model_fit,
    prepare_joint_model_fitting, run_model_mcmc, save_model_results,
    shared_reference_covariance,
)


def frame(name, data, model=None, **kwargs):
    return JointObservation(name=name, band="F200W", data=data,
                            model=model or CompositeModel([ConstantBackground("sky")]),
                            exposures=(name,), registration="known mock pixel grid",
                            calibration="same mock flux scale", **kwargs)


def test_independent_frames_share_prior_and_fit_host_and_transient():
    components = CompositeModel([PointSource("point"), SersicSource("host"), ConstantBackground("sky")])
    parameters = ParameterSet([
        Parameter("host.flux", initial=1., prior=Uniform(0, 4)),
        Parameter("host.dx", fixed=-.32), Parameter("host.dy", fixed=.24),
        Parameter("host.re", fixed=2.), Parameter("host.n", fixed=1.),
        Parameter("host.q", fixed=.7), Parameter("host.pa", fixed=25.),
        Parameter("point.dx", fixed=.42), Parameter("point.dy", fixed=-.25),
        Parameter("pre.flux", fixed=0.),
        Parameter("on.flux", initial=.3, prior=Uniform(-1, 2)),
        Parameter("pre.sky", initial=0, prior=Uniform(-.2, .2)),
        Parameter("on.sky", initial=0, prior=Uniform(-.2, .2)),
    ])
    truth = {"host.flux": 1.7, "on.flux": .9, "pre.sky": .01, "on.sky": -.02}
    vector = parameters.encode(truth)
    values = parameters.decode(vector)
    frames = []
    grid = np.arange(-24,25)/4
    for name, width, size in (("pre",.85,7), ("on",1.1,9)):
        response = np.exp(-.5*(grid[:,None]**2+grid[None,:]**2)/width**2)/(2*np.pi*width**2)
        spec = PSFSpec(response,"effective",.25,(24,24),"shared flux convention",native_pixel_scale=.03)
        context = RenderContext((size,size),psf_operator=PSFOperator(spec,oversamp=4))
        aliases = {"point.flux": f"{name}.flux", "sky.level": f"{name}.sky"}
        local = {**values, **{key:values[target] for key,target in aliases.items()}}
        data = components.render(local,context)
        mask = np.zeros(data.shape,dtype=bool)
        mask[-1,0] = True
        frames.append(frame(name,data,components,context=context,aliases=aliases,
                            noise_options={"err": np.full(data.shape,.02),"mask":mask}))
    fit = prepare_joint_model_fitting(frames,parameters)
    result = evaluate_joint_model_fit(fit,vector,decompose=True)
    assert parameters.ndim == 4
    assert result["log_prior"] == pytest.approx(-np.log(4*3*.4*.4))
    assert result["log_likelihood"] == pytest.approx(sum(
        np.count_nonzero(f["valid"])*(-.5*np.log(2*np.pi*.02**2))
        for f in result["observations"].values()
    ))
    assert not np.any(result["observations"]["pre"]["components"]["point"])
    for image in result["observations"].values():
        np.testing.assert_array_equal(sum(image["components"].values()),image["image"])
    optimum = optimize_model_fit(fit,objective="ml")
    assert optimum.best.success
    np.testing.assert_allclose(optimum.best.sample_vector,vector,atol=2e-6)
    second = frames[1]
    operator = second.context.psf_operator
    different = replace(second,context=replace(second.context,psf_operator=replace(
        operator,spec=replace(operator.spec,native_pixel_scale=.06))))
    with pytest.raises(ValueError,match="common native pixel scale"):
        prepare_joint_model_fitting((frames[0],different),parameters)


def difference_fixture():
    science = (np.diag([.3,.6]),np.diag([.7,.5,.9]))
    reference = np.array([[.3,.04],[.04,.5]])
    ops = (np.eye(2),np.array([[.6,.4],[.3,0],[0,1.]]))
    covariance = shared_reference_covariance(science,reference,ops)
    np.testing.assert_allclose(covariance[:2,2:],reference@ops[1].T)
    np.testing.assert_allclose(covariance[2:,2:],science[1]+ops[1]@reference@ops[1].T)
    obs = (
        frame("first",np.array([[.1,.4]]),kind="difference",reference_exposures=("ref",)),
        frame("second",np.array([[.4,-.2,.65]]),kind="difference",reference_exposures=("ref",),
              noise_options={"mask":np.array([[False,True,False]])}),
    )
    parameters = ParameterSet([Parameter("sky.level",initial=.35,prior=Uniform(-3,3))])
    return obs, parameters, covariance


def test_block_likelihood_masks_and_posterior_round_trip(tmp_path):
    obs, parameters, covariance = difference_fixture()
    fit = prepare_joint_model_fitting(obs,parameters,joint_covariance=covariance)
    valid = [0,1,2,4]
    data = np.array([.1,.4,.4,.65])
    noise = covariance[np.ix_(valid,valid)]
    terms = evaluate_joint_model_fit(fit,[.2],decompose=True)
    assert terms["log_likelihood"] == pytest.approx(multivariate_normal.logpdf(data-.2,cov=noise))
    assert sum(f["marginal_log_likelihood"] for f in terms["observations"].values()) != pytest.approx(terms["log_likelihood"])
    assert "log_likelihood" not in terms["observations"]["first"]
    assert terms["whitened_residual"].shape == (4,)
    precision = np.linalg.inv(noise)
    expected = (precision @ data).sum()/precision.sum()
    optimum = optimize_model_fit(fit,objective="ml")
    assert optimum.best.success
    assert optimum.best.sample_vector[0]==pytest.approx(expected,abs=1e-6)
    posterior = run_model_mcmc(fit,initial=optimum,seed=9,
                               sampling_options={"nsteps":100,"burn_in":15,"nwalkers":8})
    assert posterior.metadata["likelihood_path"]=="cached_linear"
    assert posterior.metadata["noise"]["backend"]=="block_dense"
    assert posterior.metadata["observations"][1]["reference_exposures"]==["ref"]
    sample = posterior.sampling_samples[-1,0]
    assert posterior.log_likelihood[-1,0]==pytest.approx(evaluate_joint_model_fit(fit,sample)["log_likelihood"])
    path = tmp_path/"joint.npz"
    save_model_results(path,posterior)
    loaded = load_model_results(path)
    assert loaded.metadata == json.loads(json.dumps(posterior.metadata))
    np.testing.assert_array_equal(loaded.sampling_samples,posterior.sampling_samples)


def test_shared_reference_creates_noise_floor():
    blocks = shared_reference_covariance([np.eye(1)]*3,np.array([[2.]]),[np.eye(1)]*3)
    variance = 1/np.linalg.inv(blocks).sum()
    naive = 1/np.linalg.inv(np.diag(np.diag(blocks))).sum()
    assert variance == pytest.approx(1/3+2)
    assert naive == pytest.approx(1)


def test_provenance_guards():
    obs, parameters, covariance = difference_fixture()
    with pytest.raises(ValueError,match="overlapping exposures"):
        prepare_joint_model_fitting(obs,parameters)
    with pytest.raises(ValueError,match="only per-observation masks"):
        prepare_joint_model_fitting((replace(obs[0],noise_options={"err":np.ones((1,2))}),obs[1]),
                                    parameters,joint_covariance=covariance)
    with pytest.raises(ValueError,match="known joint"):
        prepare_joint_model_fitting((replace(obs[0],aliases={"sky.level":"typo"}),),parameters)
    with pytest.raises(ValueError,match="one band, data unit and image kind"):
        prepare_joint_model_fitting((replace(obs[0],kind="direct",reference_exposures=()),obs[1]),parameters)
    with pytest.raises(ValueError,match="positive semidefinite"):
        shared_reference_covariance([np.eye(1)],np.array([[-.2]]),[np.eye(1)])


def test_nonlinear_block_density_agrees_at_saved_draw():
    model = CompositeModel([PointSource("p")])
    params = ParameterSet([Parameter("source_x",initial=0.,prior=Uniform(-.5,.5)),
                           Parameter("p.flux",fixed=1.),Parameter("p.dy",fixed=0.)])
    def point_response(dx,dy,context):
        y,x = np.indices(context.shape)
        return np.exp(-.5*((x-context.center[0]-dx)**2+(y-context.center[1]-dy)**2))
    context = RenderContext((3,3),point_response)
    data = point_response(.16,0,context)
    obs = [frame(str(i),data,model,context=context,aliases={"p.dx":"source_x"}) for i in range(2)]
    joint = prepare_joint_model_fitting(obs,params,joint_covariance=np.eye(18))
    result = run_model_mcmc(joint,seed=941,sampling_options={"nsteps":32,"burn_in":5})
    assert result.metadata["likelihood_path"]=="render_per_draw"
    assert result.log_likelihood[-1,0]==pytest.approx(
        evaluate_joint_model_fit(joint,result.sampling_samples[-1,0])["log_likelihood"])


def test_joint_alias_consumption_detects_typo_even_with_shared_default():
    model = CompositeModel([PointSource("point")])
    context = RenderContext((1, 1), lambda dx,dy,ctx: np.ones(ctx.shape))
    params = ParameterSet([
        Parameter("point.flux", initial=0., prior=Uniform(-1, 5)),
        Parameter("late.flux", initial=0., prior=Uniform(-1, 5)),
        Parameter("point.dx", fixed=0.), Parameter("point.dy", fixed=0.),
    ])
    options = {"err":np.ones((1, 1))}
    early = frame("early", np.array([[0.]]), model, context=context, noise_options=options)
    late = frame("late", np.array([[4.]]), model, context=context, noise_options=options,
                 aliases={"point.flux":"late.flux"})
    joint = prepare_joint_model_fitting([early, late], params)
    initial = evaluate_joint_model_fit(joint, [0., 0.])
    changed = evaluate_joint_model_fit(joint, [0., 4.])
    np.testing.assert_array_equal(initial["observations"]["early"]["image"],
                                  changed["observations"]["early"]["image"])
    np.testing.assert_array_equal(changed["observations"]["late"]["image"], [[4.]])
    assert changed["log_likelihood"]-initial["log_likelihood"] == pytest.approx(8.)
    with pytest.raises(ValueError, match=r"late.*point.*point.flxu"):
        prepare_joint_model_fitting([early,replace(late,aliases={"point.flxu":"late.flux"})],params)
    with pytest.raises(ValueError, match=r"do not use free parameters: late.flux"):
        prepare_joint_model_fitting([early,replace(late,aliases={})],params)


def test_exposure_ids_normalized_before_overlap_or_duplicates():
    obs, params, cov = difference_fixture()
    padded = (obs[0], replace(obs[1], reference_exposures=("ref ",)))
    with pytest.raises(ValueError,match="overlapping exposures"):
        prepare_joint_model_fitting(padded,params)
    joint = prepare_joint_model_fitting(padded,params,joint_covariance=cov)
    assert joint.metadata["observations"][1]["reference_exposures"] == ["ref"]
    with pytest.raises(ValueError,match="unique"):
        prepare_joint_model_fitting((replace(obs[0],exposures=("X"," X ")),obs[1]),params,
                                    joint_covariance=cov)


def test_dense_budget_is_checked_before_covariance_work(monkeypatch):
    import jwst_psfmc.joint as module
    obs = [frame(str(i),np.ones((9,9))) for i in range(3)]
    params = ParameterSet([Parameter("sky.level", fixed=0.)])
    def unexpected(*args, **kwargs):
        pytest.fail("covariance work began before the budget was checked")
    monkeypatch.setattr(module,"prepare_model_fitting",unexpected)
    monkeypatch.setattr(module,"block_diag",unexpected)
    monkeypatch.setattr(np.linalg,"eigvalsh",unexpected)
    with pytest.raises(ValueError,match="243 pixels.*workspace"):
        prepare_joint_model_fitting(obs,params,joint_covariance=np.eye(243),max_dense_bytes=315_000)
    with pytest.raises(ValueError,match="243 pixels.*workspace"):
        shared_reference_covariance([np.eye(81)]*3,np.eye(81),[np.eye(81)]*3,max_dense_bytes=315_000)


def test_dense_budget_uses_observed_pixels():
    obs = [frame(str(i),np.array([[0., np.nan]])) for i in range(2)]
    params = ParameterSet([Parameter("sky.level", fixed=0.)])
    fit = prepare_joint_model_fitting(obs,params,joint_covariance=np.eye(4),max_dense_bytes=200)
    assert fit.noise.n_valid == 2


@pytest.mark.parametrize("noise_options", [
    {"covariance": np.eye(4)},
    {"err": np.ones((2, 2)), "kernel": np.ones((1, 1)), "boundary": "finite"},
])
def test_independent_dense_frames_use_the_stricter_workspace_budget(noise_options):
    obs = frame("single", np.zeros((2, 2)), noise_options={"backend": "dense", **noise_options})
    params = ParameterSet([Parameter("sky.level", fixed=0.)])
    for joint_budget, frame_budget in [(700, 1000), (1000, 700)]:
        scoped = replace(obs, noise_options={**obs.noise_options, "max_dense_bytes": frame_budget})
        with pytest.raises(ValueError, match="4 pixels.*workspace budget.*700 allowed"):
            prepare_joint_model_fitting([scoped], params, max_dense_bytes=joint_budget)
    with pytest.raises(ValueError, match="4 pixels.*workspace budget.*700 allowed"):
        prepare_joint_model_fitting([obs], params, max_dense_bytes=700)

    masked = replace(obs, noise_options={**obs.noise_options,
                                         "mask": np.array([[False, True], [False, True]])})
    fit = prepare_joint_model_fitting([masked], params, max_dense_bytes=200)
    assert fit.observations[0].fit.noise.n_valid == 2
