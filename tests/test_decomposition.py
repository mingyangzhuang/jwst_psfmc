"""Synthetic PSF+Sérsic models, reduction and posterior diagnostics."""

from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet, PointSource,
    PSFOperator, RenderContext, SersicProfile, SersicSource, Uniform,
    evaluate_model_fit, load_model_results, optimize_model_fit, prepare_model_fitting,
    run_model_mcmc, save_model_results, summarize_decomposition,
)
from tests.test_rendering import analytic_gaussian, continuous_gaussian_convolution, gaussian_spec


def components():
    return [PointSource("point"), SersicSource("host"), ConstantBackground("sky")]


VALUES = {"point.flux": 1.4, "point.dx": .18, "point.dy": -.21,
          "host.flux": .7, "host.dx": .18, "host.dy": -.21,
          "host.re": 1.4, "host.n": .5, "host.q": .67, "host.pa": 32.,
          "sky.level": .02}


def amplitude_fit(data, context, truth, *, sigma=.006, signed=False, ties=True, mask=None):
    limits = {"point.flux": (-2 if signed else 0, 12), "host.flux": (0, 4),
              "sky.level": (-.1, .1)}
    definitions = []
    for name, value in truth.items():
        if name in limits:
            lo, hi = limits[name]
            definitions.append(Parameter(name, np.clip(value, lo, hi), prior=Uniform(lo, hi)))
        elif ties and name in ("host.dx", "host.dy"):
            definitions.append(Parameter(name, tie=name.replace("host", "point")))
        else:
            definitions.append(Parameter(name, fixed=value))
    return prepare_model_fitting(
        data, CompositeModel(components()), ParameterSet(definitions), context=context,
        noise_options={"err": np.full(context.shape, sigma), "mask": mask},
    )


@pytest.mark.parametrize("zero", ["point", "host"])
def test_zero_component_matches_removal_with_masked_dense_covariance(zero):
    context = RenderContext((5, 6), psf_operator=PSFOperator(gaussian_spec("effective")))
    values = {**VALUES, f"{zero}.flux": 0}
    model = CompositeModel(components())
    data = model.render(VALUES, context)
    shape = context.shape
    std = np.linspace(.015, .03, np.prod(shape))
    covariance = np.diag(std**2) + np.outer(std, std) * .08
    mask = np.zeros(shape, dtype=bool)
    mask[1, 1] = True
    noise = {"backend": "dense", "covariance": covariance, "mask": mask}

    def prepare(model, values):
        params = ParameterSet([Parameter(name, fixed=value) for name, value in values.items()])
        return prepare_model_fitting(data, model, params, context=context, noise_options=noise)

    full = prepare(model, values)
    reduced = prepare(CompositeModel([c for c in components() if c.name != zero]),
                      {k: v for k, v in values.items() if not k.startswith(zero + ".")})
    evaluation = evaluate_model_fit(full, decompose=True)
    reference = evaluate_model_fit(reduced)
    assert_array_equal(reference["image"], evaluation["image"])
    assert evaluation["log_likelihood"] == reference["log_likelihood"]
    assert_array_equal(evaluation["components"][zero], np.zeros(shape))
    assert_array_equal(sum(evaluation["components"].values()), evaluation["image"])
    valid = ~mask
    remainder = (data - evaluation["image"])[valid]
    assert np.isnan(evaluation["data"][1, 1]) and np.isnan(evaluation["residual"][1, 1])
    assert_allclose(evaluation["standardized_residual"][valid],
                    remainder / np.sqrt(np.diag(covariance)[valid.ravel()]))
    expected = np.linalg.solve(np.linalg.cholesky(covariance[np.ix_(valid.ravel(), valid.ravel())]), remainder)
    assert_allclose(evaluation["whitened_residual"], expected, rtol=2e-13, atol=1e-15)


def test_tied_and_fixed_n_dimensions():
    definitions = []
    for name, value in VALUES.items():
        lo, hi = value - max(abs(value), .4) / 2, value + max(abs(value), .4) / 2
        definitions.append(Parameter(name, value, prior=Uniform(lo, hi)))
    assert ParameterSet(definitions).ndim == 11
    definitions = [Parameter(p.name, tie=p.name.replace("host", "point"))
                   if p.name in ("host.dx", "host.dy") else p for p in definitions]
    bound = ParameterSet(definitions)
    assert bound.ndim == 9
    values = bound.decode(bound.initial)
    assert values["point.dx"] == values["host.dx"] and values["point.dy"] == values["host.dy"]
    definitions = [Parameter("host.n", fixed=.5) if p.name == "host.n" else p for p in definitions]
    assert ParameterSet(definitions).ndim == 8


def test_weighted_draw_quantities_signed_point_and_archive(tmp_path):
    context = RenderContext((5, 6), psf_operator=PSFOperator(gaussian_spec("effective")))
    data = CompositeModel(components()).render(VALUES, context)
    initial_fit = amplitude_fit(data, context, VALUES, signed=True)
    definitions = [
        replace(p, fixed=None, initial=1.4, prior=Uniform(.8, 1.9), transform="log")
        if p.name == "host.re" else p for p in initial_fit.parameters._parameters.values()
    ]
    fit = prepare_model_fitting(data, initial_fit.model, ParameterSet(definitions), context=context,
                                noise_options={"err": np.full(context.shape, .006)})
    base = run_model_mcmc(fit, seed=6, sampling_options={"nsteps": 6, "burn_in": 1})
    physical = np.array([[-1., .7, .9, .03], [1., .6, 1.4, .01], [2., 1.6, 1.8, .04]])
    samples = physical.copy()
    samples[:, 2] = np.log(samples[:, 2])
    lp, ll = [], []
    for row in samples:
        evaluated = evaluate_model_fit(fit, row)
        lp.append(evaluated["log_prior"])
        ll.append(evaluated["log_likelihood"])
    result = replace(base, sampler="nautilus", burn_in=0, sampling_samples=samples,
                     physical_samples=physical, log_weights=np.log([.2, .5, .3]),
                     log_prior=np.array(lp), log_likelihood=np.array(ll))
    path = tmp_path / "decomposition"
    save_model_results(path, result)
    restored = load_model_results(path)
    aperture = np.zeros(context.shape, dtype=bool)
    aperture[1:4, 1:5] = True
    summary = summarize_decomposition(fit, restored, aperture=aperture, resolution_radius=1.5)
    assert summary["quantiles"]["host_total_flux"]["median"] == .6
    assert summary["host_fraction"]["total"]["defined_probability"] == pytest.approx(.8)
    assert summary["host_fraction"]["total"]["quantiles"]["median"] == pytest.approx(.6 / 1.6)
    assert summary["quantiles"]["host_re_circ"]["median"] == pytest.approx(1.4 * np.sqrt(.67))
    assert summary["resolution"]["unresolved"]
    assert not summary["resolution"]["radius_is_fixed"]
    assert summary["quantiles"]["host_re_circ"]["q95"] == pytest.approx(1.8 * np.sqrt(.67))
    assert summary["near_circular_probability"] == 0
    assert summary["correlations"]["host.n"]["host.flux"] is None
    assert summary["aperture_pixels"] == 12
    model = evaluate_model_fit(fit, samples[1], decompose=True)
    assert summary["quantiles"]["host_aperture_flux"]["median"] == pytest.approx(
        model["components"]["host"][aperture].sum()
    )
    shifted = prepare_model_fitting(data + .01, fit.model, fit.parameters, context=context,
                                    noise_options={"err": np.full(context.shape, .006)})
    with pytest.raises(ValueError, match="same model"):
        summarize_decomposition(shifted, restored)


def synthetic_case(ratio=2, re=1.4, q=.67, sigma=.006, *, mismatch=False, context_shape=(7, 8)):
    context = RenderContext(context_shape, psf_operator=PSFOperator(gaussian_spec("effective"),
                                                                   oversamp=8, max_offset=.5))
    truth = dict(VALUES, **{"host.re": re, "host.q": q, "point.flux": .7 * ratio})
    point_flux, host_flux = truth["point.flux"], truth["host.flux"]
    if mismatch:
        point_flux, host_flux = 1.4, 0.
    width = .22 if mismatch else 0.
    point = point_flux * analytic_gaussian(
        context, truth["point.dx"], truth["point.dy"], np.hypot(.68, width), np.hypot(.81, width)
    )
    host = continuous_gaussian_convolution(
        SersicProfile(host_flux, re, .5, q, truth["host.pa"]), context, truth["host.dx"], truth["host.dy"]
    )
    data = point + host + truth["sky.level"]
    return amplitude_fit(data, context, truth, sigma=sigma), truth


@pytest.mark.parametrize("ratio", [0, .5, 2, 10])
def test_resolved_flux_recovery_across_contrasts(ratio):
    fit, truth = synthetic_case(ratio=ratio)
    optimized = optimize_model_fit(fit, objective="ml")
    assert optimized.best.success
    best = fit.parameters.decode(optimized.best.sample_vector)
    assert best["host.flux"] == pytest.approx(truth["host.flux"], abs=.025)
    assert best["point.flux"] == pytest.approx(truth["point.flux"], abs=.025)


def test_compact_host_degeneracy_and_psf_mismatch_null():
    compact, _ = synthetic_case(re=.03, q=.99)
    result = run_model_mcmc(compact, seed=33, sampling_options={"nsteps": 1600, "burn_in": 250})
    summary = summarize_decomposition(compact, result, resolution_radius=.25)
    assert summary["resolution"]["unresolved"]
    assert summary["near_circular_probability"] == pytest.approx(1)
    assert summary["correlations"]["point.flux"]["host.flux"] < -.95
    true_psf, _ = synthetic_case(ratio=2, re=.5, q=1, mismatch=True)
    # Compare with a null at the fitted PSF width, holding the model and priors fixed.
    no_host = 1.4 * analytic_gaussian(true_psf.context, .18, -.21, .68, .81) + .02
    matched = amplitude_fit(no_host, true_psf.context, dict(VALUES, **{"host.re": .5, "host.q": 1}))
    recovered = optimize_model_fit(true_psf, objective="ml")
    control = optimize_model_fit(matched, objective="ml")
    faux = true_psf.parameters.decode(recovered.best.sample_vector)["host.flux"]
    baseline = matched.parameters.decode(control.best.sample_vector)["host.flux"]
    assert faux > .15
    assert baseline < .02
