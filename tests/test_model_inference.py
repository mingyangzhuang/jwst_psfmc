"""Analytic posterior and status checks for generic model fitting."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.special import logsumexp, ndtr

from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet,
    PSFOperator, RenderContext, SersicProfile, SersicSource, Uniform,
    evaluate_model_fit, load_model_results, optimize_model_fit, prepare_model_fitting,
    run_model_mcmc, save_model_results, summarize_model_results,
)
from tests.test_rendering import gaussian_spec


def linear_sersic_fit(correlated=False):
    shape = (5, 6)
    operator = PSFOperator(gaussian_spec("effective"), oversamp=8, max_offset=.5)
    context = RenderContext(shape, psf_operator=operator)
    model = CompositeModel([SersicSource("host"), ConstantBackground("sky")])
    flux, background = 1.1, .018
    profile = SersicProfile(1, .7, 1, .55, 26)
    unit_host = operator.render_sersic(profile, .25, -.2, context)
    err = np.linspace(.018, .023, unit_host.size).reshape(shape)
    sigma = err.ravel()
    covariance = np.diag(sigma**2)
    options = {"err": err}
    if correlated:
        correlation = np.eye(unit_host.size)
        for i, (y1, x1) in enumerate(np.ndindex(shape)):
            for j, (y2, x2) in enumerate(np.ndindex(shape)):
                if abs(y1-y2) + abs(x1-x2) == 1:
                    correlation[i, j] = .12
        covariance = (sigma[:, None] * correlation) * sigma
        options = {"backend": "dense", "covariance": covariance}
    noise = np.linalg.cholesky(covariance) @ np.random.default_rng(18).standard_normal(sigma.size)
    data = flux * unit_host + background + noise.reshape(shape)
    parameters = ParameterSet([
        Parameter("host.re", fixed=.7, unit="pixel"),
        Parameter("host.flux", .85, "Jy", prior=Uniform(.2, 2.2)),
        Parameter("host.n", fixed=1),
        Parameter("host.q", fixed=.55),
        Parameter("host.pa", fixed=26, unit="deg"),
        Parameter("host.dx", fixed=.25, unit="pixel"),
        Parameter("host.dy", fixed=-.2, unit="pixel"),
        Parameter("sky.level", .01, "Jy / pixel", prior=Uniform(-.09, .13)),
    ])
    fit = prepare_model_fitting(data, model, parameters, context=context,
                                noise_options=options, data_unit="Jy / pixel")
    design = np.column_stack([unit_host.ravel(), np.ones(unit_host.size)])
    precision = design.T @ np.linalg.solve(covariance, design)
    mean = np.linalg.solve(precision, design.T @ np.linalg.solve(covariance, data.ravel()))
    cov = np.linalg.solve(precision, np.eye(2))
    return fit, mean, cov


@pytest.fixture(scope="module", params=[False, True], ids=["white", "correlated"])
def linear_case(request):
    return linear_sersic_fit(request.param)


@pytest.mark.parametrize("backend", ["emcee", "nautilus"])
def test_single_sersic_posterior_against_independent_gls(linear_case, backend, tmp_path):
    fit, mean, covariance = linear_case
    opt = optimize_model_fit(fit, objective="ml", starts=[fit.parameters.initial, [.6, -.01]])
    assert opt.best.success
    assert len(opt.attempts) == 3
    np.testing.assert_allclose(opt.best.sample_vector, mean, atol=2e-7, rtol=0)
    if backend == "nautilus":
        pytest.importorskip("nautilus")
        opts = {"n_live": 110, "n_eff": 1800, "n_like_max": 15000}
        initial = None
    else:
        # Leave enough samples to pass the autocorrelation stability check across platforms.
        opts = {"nsteps": 12000, "burn_in": 500}
        initial = opt
    result = run_model_mcmc(fit, sampler=backend, initial=initial, seed=44, sampling_options=opts)
    assert result.status == "converged", result.diagnostics
    assert result.metadata["likelihood_path"] == "cached_linear"
    assert result.metadata["data_unit"] == "Jy / pixel"
    assert "data" in result.metadata["input_sha256"]
    assert result.metadata["psf"]["response_kind"] == "effective"
    assert result.free_names == ("host.flux", "sky.level")
    if backend == "nautilus":
        assert result.sampling_samples.ndim == 2
        assert abs(logsumexp(result.log_weights)) < 1e-12
        samples = result.physical_samples
        weights = np.exp(result.log_weights)
        index = (0,)
    else:
        assert result.sampling_samples.shape == (opts["nsteps"], 12, 2)
        samples = result.physical_samples[result.burn_in:].reshape(-1, 2)
        weights = None
        index = (0, 0)
    estimate = np.average(samples, axis=0, weights=weights)
    np.testing.assert_array_less(abs(estimate-mean), 5 * np.array(result.diagnostics["mcse"]))
    tolerance = np.sqrt(covariance.diagonal())
    for parameter, summary, at in zip(result.free_names, summarize_model_results(result).values(), range(2)):
        assert abs(summary["median"] - mean[at]) < .17 * tolerance[at], parameter
        assert abs((summary["q84"] - summary["q16"]) / 2 - tolerance[at]) < .18 * tolerance[at]
    eval_result = evaluate_model_fit(fit, result.sampling_samples[index])
    assert result.log_likelihood[index] == pytest.approx(eval_result["log_likelihood"], abs=1e-12)
    assert result.log_prior[index] == pytest.approx(eval_result["log_prior"], abs=1e-12)
    assert result.log_posterior[index] == pytest.approx(eval_result["log_posterior"], abs=1e-12)
    path = tmp_path / "posterior.model"
    save_model_results(path, result)
    with np.load(path.with_name("posterior.model.npz"), allow_pickle=False) as raw:
        assert all(raw[name].dtype.kind != "O" for name in raw.files)
    restored = load_model_results(path)
    assert restored.status == result.status
    assert restored.metadata["parameters"] == result.metadata["parameters"]
    assert restored.diagnostics == result.diagnostics
    np.testing.assert_array_equal(restored.sampling_samples, result.sampling_samples)
    np.testing.assert_array_equal(restored.log_likelihood, result.log_likelihood)
    assert summarize_model_results(restored) == summarize_model_results(result)


def background_fit(prior=None, transform="linear"):
    data = np.array([[1.2, 1.6], [1.35, 1.43]])
    parameters = ParameterSet([Parameter("sky.level", 1.25, prior=prior or Uniform(.1, 3.),
                                         transform=transform)])
    fit = prepare_model_fitting(data, CompositeModel([ConstantBackground("sky")]), parameters,
                                noise_options={"err": np.full(data.shape, .3)})
    return fit


def test_deterministic_seed_and_insufficient_sampling_and_spawn():
    fit = background_fit()
    opts = {"nsteps": 14, "burn_in": 4, "nwalkers": 8}
    first = run_model_mcmc(fit, seed=112, sampling_options=opts)
    second = run_model_mcmc(fit, seed=112, sampling_options={**opts, "ncores": 2})
    alternative = run_model_mcmc(fit, seed=113, sampling_options=opts)
    assert first.status == second.status == "insufficient_sampling"
    np.testing.assert_array_equal(first.sampling_samples, second.sampling_samples)
    assert first.metadata["sampler_state"]["rng_final"] == second.metadata["sampler_state"]["rng_final"]
    assert not np.array_equal(first.sampling_samples, alternative.sampling_samples)
    assert [first.metadata["seed_streams"][name]["spawn_key"]
            for name in ("initialization", "noise", "sampler")] == [(0,), (1,), (2,)]


def test_nested_one_dimensional_log_coordinate_and_early_stop(tmp_path):
    pytest.importorskip("nautilus")
    fit = background_fit(transform="log")
    opt = optimize_model_fit(fit)
    ml = optimize_model_fit(fit, objective="ml")
    estimate = fit.data.mean()
    variance = .3**2 / fit.data.size
    assert np.exp(ml.best.sample_vector[0]) == pytest.approx(estimate, abs=1e-7)
    # Physical uniform prior acquires a factor x in log coordinates.
    mode = (estimate + np.sqrt(estimate**2 + 4 * variance)) / 2
    assert np.exp(opt.best.sample_vector[0]) == pytest.approx(mode, abs=1e-7)
    result = run_model_mcmc(fit, sampler="nautilus", seed=16,
                            sampling_options={"n_live": 90, "n_eff": 1000, "n_like_max": 12000})
    assert result.status == "converged"
    assert result.metadata["sampler_state"]["auxiliary_unit_dimension"]
    np.testing.assert_allclose(result.physical_samples, np.exp(result.sampling_samples), atol=0, rtol=0)
    x = result.physical_samples[:, 0]
    reference_cdf = ndtr((x - estimate) / np.sqrt(variance))
    assert np.average(reference_cdf, weights=np.exp(result.log_weights)) == pytest.approx(.5, abs=.04)
    limited = run_model_mcmc(fit, sampler="nautilus", seed=16,
                             sampling_options={"n_live": 90, "n_eff": 10000, "n_like_max": 100})
    assert limited.status == "insufficient_sampling"
    save_model_results(tmp_path / "limited", limited)
    assert load_model_results(tmp_path / "limited").status == "insufficient_sampling"


@dataclass(frozen=True)
class BrokenModel:
    name: str = "broken"

    def render(self, values, context):
        raise RuntimeError("synthetic quadrature failure")


def test_numerical_failure_round_trip_and_future_schema(tmp_path):
    fit = prepare_model_fitting(np.zeros((2, 2)), CompositeModel([BrokenModel()]),
                                ParameterSet([Parameter("amplitude", 0, prior=Uniform(-1, 1))]),
                                noise_options={"err": np.ones((2, 2))})
    failed = run_model_mcmc(fit, seed=7, sampling_options={"nsteps": 4, "burn_in": 0})
    assert failed.status == "numerical_failure"
    assert "quadrature failure" in failed.stop_reason
    assert failed.physical_samples.shape == (0, 0, 1)
    save_model_results(tmp_path / "failure", failed)
    restored = load_model_results(tmp_path / "failure")
    assert restored.stop_reason == failed.stop_reason
    with pytest.raises(ValueError, match="no posterior"):
        summarize_model_results(restored)
    np.savez(tmp_path / "future.npz", metadata_json='{"schema_version":2}')
    with pytest.raises(ValueError, match="unsupported model archive schema"):
        load_model_results(tmp_path / "future")


@dataclass(frozen=True)
class AngleModel:
    name: str = "angle"

    def render(self, values, context):
        phi = 2 * np.deg2rad(values["angle.pa"])
        return np.array([[np.cos(phi), np.sin(phi)]])


def test_periodic_summary_tracks_the_seam_and_wraps_emcee():
    parameters = ParameterSet([Parameter("angle.pa", 89, "deg", prior=Uniform(-90, 90),
                                         period=180, origin=-90)])
    fit = prepare_model_fitting(np.array([[-1., 0.]]), CompositeModel([AngleModel()]), parameters,
                                noise_options={"err": [[.2, .2]]})
    result = run_model_mcmc(fit, seed=25, sampling_options={"nsteps": 900, "burn_in": 100})
    summary = summarize_model_results(result)["angle.pa"]
    assert abs(summary["median"]) > 85
    assert summary["q16"] > 75 and summary["q84"] < -75
    assert summary["resultant_length"] > .9
    assert "wrapped" in result.metadata["sampler_state"]["moves"]


def test_nested_periodic_spawn_preserves_weighted_circular_summary():
    pytest.importorskip("nautilus")
    parameters = ParameterSet([Parameter("angle.pa", 89, "deg", prior=Uniform(-90, 90),
                                         period=180, origin=-90)])
    fit = prepare_model_fitting(np.array([[-1., 0.]]), CompositeModel([AngleModel()]), parameters,
                                noise_options={"err": [[.2, .2]]})
    result = run_model_mcmc(fit, sampler="nautilus", seed=25,
                            sampling_options={"n_live": 90, "n_eff": 600, "ncores": 2})
    assert result.status == "converged"
    assert result.metadata["sampler_state"]["periodic_indices"] == [0]
    summary = summarize_model_results(result)["angle.pa"]
    assert abs(summary["median"]) > 85
    assert summary["q16"] > 75 and summary["q84"] < -75


def test_configuration_errors():
    white, _, _ = linear_sersic_fit()
    with pytest.raises(TypeError, match="unexpected"):
        run_model_mcmc(white, sampling_options={"n_eff": 100})
    with pytest.raises(ValueError, match="nwalkers"):
        run_model_mcmc(white, sampling_options={"nwalkers": 2, "nsteps": 10, "burn_in": 1})
    schema = white.parameters.to_schema()
    values = white.parameters.decode(white.parameters.initial)
    all_fixed = ParameterSet([Parameter(d["name"], fixed=values[d["name"]]) for d in schema["parameters"]])
    deterministic = prepare_model_fitting(white.data, white.model, all_fixed,
                                          context=white.context, noise_options={"err": np.ones(white.data.shape)})
    assert optimize_model_fit(deterministic).best.success
    with pytest.raises(ValueError, match="all parameters are fixed"):
        run_model_mcmc(deterministic)
    for name, lo, hi, message in (("host.n", .9, 2, "fixed n"), ("host.dx", -1, 1, "max_offset"),
                                  ("host.flux", -1, 2, "nonnegative")):
        definitions = [Parameter(key, fixed=value) if key != name else
                       Parameter(key, value, prior=Uniform(lo, hi)) for key, value in values.items()]
        fit = prepare_model_fitting(white.data, white.model, ParameterSet(definitions),
                                    context=white.context, noise_options={"err": np.ones(white.data.shape)})
        with pytest.raises(ValueError, match=message):
            run_model_mcmc(fit, sampling_options={"nsteps": 5, "burn_in": 0})


def free_shape_sersic_fit(*, add_noise=False):
    from tests.test_rendering import continuous_gaussian_convolution

    operator = PSFOperator(gaussian_spec("effective", extent=3), oversamp=6, max_offset=.5)
    context = RenderContext((7, 8), psf_operator=operator)
    model = CompositeModel([SersicSource("host"), ConstantBackground("sky")])
    truth = {"host.flux": 1.1, "host.re": 1.32, "host.q": .52, "host.pa": 38.,
             "host.dx": .19, "host.dy": -.22, "sky.level": .012, "host.n": .5}
    limits = {
        "host.flux": (.6, 1.7, 1.01), "host.re": (.9, 1.75, 1.22),
        "host.q": (.32, .83, .58), "host.pa": (10, 70, 35.),
        "host.dx": (-.4, .4, .12), "host.dy": (-.4, .4, -.13),
        "sky.level": (-.03, .055, .009),
    }
    parameters = ParameterSet([
        Parameter(name, start, prior=Uniform(lo, hi), transform="log" if name == "host.re" else "linear")
        for name, (lo, hi, start) in limits.items()
    ] + [Parameter("host.n", fixed=.5)])
    data = model.render(truth, context)
    sigma = .006
    if add_noise:
        analytic = continuous_gaussian_convolution(
            SersicProfile(truth["host.flux"], truth["host.re"], .5, truth["host.q"], truth["host.pa"]),
            context, truth["host.dx"], truth["host.dy"],
        )
        np.testing.assert_array_less(abs(data - analytic - truth["sky.level"]), 8e-4)
        data = analytic + truth["sky.level"] + sigma * np.random.default_rng(912).normal(size=context.shape)
    return prepare_model_fitting(data, model, parameters, context=context,
                                 noise_options={"err": np.full(context.shape, sigma)}), truth


def test_fixed_n_noiseless_recovers_all_seven_free_parameters():
    fit, truth = free_shape_sersic_fit()
    optimized = optimize_model_fit(fit, objective="ml", max_evaluations=2000)
    assert optimized.best.success, optimized.best
    physical = fit.parameters.decode(optimized.best.sample_vector)
    prediction = evaluate_model_fit(fit, optimized.best.sample_vector)["image"]
    assert np.linalg.norm(prediction - fit.data) < 2e-5
    for name, expected in truth.items():
        assert physical[name] == pytest.approx(expected, abs=.012 if name == "host.pa" else .0002)
