"""Model composition and strict regression through the legacy bridge."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.special import erf

from jwst_psfmc import (
    CompositeModel,
    ConstantBackground,
    Parameter,
    ParameterSet,
    PointSource,
    RenderContext,
    Uniform,
    evaluate_model_fit,
    log_likelihood_cov_prepared,
    prepare_covariance_terms,
    prepare_model_fitting,
    prepare_psf_for_oversamp,
    psf_model,
)
from jwst_psfmc.models import _LegacyPointSourceAdapter


def gaussian_response(dx, dy, context):
    """Unit-flux Gaussian integrated over the native pixel edges."""
    x0, y0 = context.center
    ny, nx = context.shape
    norm = np.sqrt(2) * 0.6
    x_edges = (np.arange(nx + 1) - 0.5 - x0 - dx) / norm
    y_edges = (np.arange(ny + 1) - 0.5 - y0 - dy) / norm
    return np.outer(np.diff(erf(y_edges)), np.diff(erf(x_edges))) / 4


@pytest.mark.parametrize("shape,phase", [((5, 7), (0.2, -0.3)), ((4, 6), (0.25, 0.5))])
def test_composite_and_prepared_likelihood(shape, phase):
    model = CompositeModel([PointSource("agn"), ConstantBackground("sky")])
    context = RenderContext(shape, gaussian_response)
    dx, dy = phase
    parameters = ParameterSet([
        Parameter("agn.flux", initial=-3, prior=Uniform(-5, 5)),
        Parameter("agn.dx", fixed=dx, unit="pixel"),
        Parameter("agn.dy", fixed=dy, unit="pixel"),
        Parameter("sky.level", fixed=0.08),
    ])
    ny, nx = shape
    assert context.center == (nx / 2 - 0.5, ny / 2 - 0.5)
    x_center, y_center = nx / 2 - 0.5 + dx, ny / 2 - 0.5 + dy

    def integrated_axis(index, center):
        scale = np.sqrt(2) * 0.6
        return (erf((index + 0.5 - center) / scale) -
                erf((index - 0.5 - center) / scale)) / 2

    source = np.empty(shape)
    for y, x in np.ndindex(shape):
        source[y, x] = -3 * integrated_axis(x, x_center) * integrated_axis(y, y_center)
    expected = source + 0.08
    err = np.linspace(0.05, 0.12, source.size).reshape(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[0, 0] = True
    err[0, 0] = np.nan
    data = (expected + 0.02).copy()
    data[0, 0] = np.inf
    fit = prepare_model_fitting(data, model, parameters, context=context,
                                noise_options={"err": err, "mask": mask})
    data.fill(0)
    err.fill(1)
    mask.fill(False)

    result = evaluate_model_fit(fit)
    np.testing.assert_allclose(result["image"], expected, atol=1e-14, rtol=0)
    assert result["image"].dtype == np.float64
    parts = model.render_components(result["values"], context)
    assert list(parts) == ["agn", "sky"]
    np.testing.assert_allclose(parts["agn"], source, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(parts["sky"], np.full(shape, 0.08))
    sigma = np.linspace(0.05, 0.12, source.size)[1:]
    likelihood = -0.5 * np.sum((0.02 / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
    assert result["log_likelihood"] == pytest.approx(likelihood, abs=1e-13)
    assert result["log_prior"] == pytest.approx(-np.log(10))
    assert result["log_posterior"] == pytest.approx(likelihood - np.log(10), abs=1e-13)
    assert result["metadata"]["noise"]["invalid_pixels"] == [[0, 0]]
    assert result["metadata"]["noise"]["mask_mode"] == "exact_marginal"
    with pytest.raises(ValueError, match="prior support"):
        evaluate_model_fit(fit, [-6])
    with pytest.raises(ValueError, match="read-only"):
        fit.data[0, 1] = 0


def test_two_sources_share_one_background_and_tied_offsets():
    model = CompositeModel([PointSource("one"), PointSource("two"), ConstantBackground("sky")])
    context = RenderContext((4, 7), gaussian_response)
    parameters = ParameterSet([
        Parameter("one.flux", fixed=9),
        Parameter("two.flux", fixed=3),
        Parameter("one.dx", fixed=-0.25, unit="pixel"),
        Parameter("two.dx", tie="one.dx", unit="pixel"),
        Parameter("one.dy", fixed=0.5, unit="pixel"),
        Parameter("two.dy", tie="one.dy", unit="pixel"),
        Parameter("sky.level", fixed=0.3),
    ])
    expected = 12 * gaussian_response(-0.25, 0.5, context) + 0.3
    fit = prepare_model_fitting(expected, model, parameters, context=context,
                                noise_options={"err": np.ones(context.shape)})
    result = evaluate_model_fit(fit, [])
    assert result["log_prior"] == 0
    assert result["log_likelihood"] == pytest.approx(-expected.size * np.log(2 * np.pi) / 2)
    np.testing.assert_allclose(result["image"], expected, atol=1e-14, rtol=0)


@dataclass(frozen=True)
class BadComponent:
    name: str
    output: np.ndarray

    def render(self, values, context):
        return self.output


def test_model_validation():
    with pytest.raises(ValueError, match="at least one"):
        CompositeModel([])
    with pytest.raises(ValueError, match="duplicate"):
        CompositeModel([PointSource("src"), ConstantBackground("src")])
    with pytest.raises(ValueError, match="at most one"):
        CompositeModel([ConstantBackground("sky"), ConstantBackground("other")])
    with pytest.raises(ValueError, match="positive integers"):
        RenderContext((2, 0))
    context = RenderContext((4, 5))
    with pytest.raises(ValueError, match="requires a point_response"):
        PointSource("src").render({}, context)
    for output in (np.ones((5, 4)), np.ones(context.shape, dtype=complex),
                   np.full(context.shape, np.nan)):
        with pytest.raises(ValueError, match="component 'bad'"):
            CompositeModel([BadComponent("bad", output)]).render({}, context)
    with pytest.raises(ValueError, match="context shape"):
        prepare_model_fitting(np.zeros((4, 6)), CompositeModel([ConstantBackground("sky")]),
                              ParameterSet([]), context=context,
                              noise_options={"err": np.ones((4, 6))})


@pytest.mark.parametrize("shape", [(7, 9), (8, 6), (11, 13)])
@pytest.mark.parametrize("theta", [(2.0, 0.0, 0.0, 0.03), (-1, 0.23, -0.31, 0.08)])
def test_legacy_adapter_matches_psf_model_and_old_likelihood(shape, theta):
    y, x = np.mgrid[-20:21, -20:21]
    raw = np.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2) / 20)
    psf, native, fraction = prepare_psf_for_oversamp(raw, 4, (9, 9), return_fraction=True)
    context = RenderContext(shape)
    adapter = _LegacyPointSourceAdapter("obj", psf, 4, native, fraction)
    model = CompositeModel([adapter, ConstantBackground("sky")])
    values = dict(zip(("obj.flux", "obj.dx", "obj.dy", "sky.level"), theta))

    reference = psf_model(theta[:3], psf, 4, output_shape=shape, native_shape=native,
                          psf_prepare_fraction=fraction)
    assert np.array_equal(adapter.render(values, context), reference)
    np.testing.assert_array_equal(model.render(values, context), reference + theta[3])

    rng = np.random.default_rng(702)
    data = rng.normal(size=shape)
    err = rng.uniform(0.5, 2, size=shape)
    weight = rng.uniform(0.1, 1, size=shape)
    data[1, 2] = np.nan
    err[2, 1] = 0.0
    weight[3, 2] = 0.0
    weight[2, 3] = np.nan
    kernel = np.array([[0, 0.12, 0], [0.2, 1, 0.2], [0, 0.12, 0]])
    prepared = prepare_covariance_terms(data, err, kernel, fit_weight=weight)
    masked = np.where(prepared["valid"], reference + theta[3], 0.0)
    residual = (prepared["data"] - masked) / prepared["err"] * np.sqrt(prepared["fit_weight"])
    coeff = np.fft.fft2(residual)
    original = -0.5 * (np.sum(np.abs(coeff) ** 2 / prepared["power_spectrum"]) / residual.size)
    assert log_likelihood_cov_prepared(theta, prepared, psf, 4, native, fraction) == original
