"""Independent checks of the point and extended PSF renderers."""

import math

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import gammaincinv, ndtr

from jwst_psfmc import (
    CompositeModel, ConstantBackground, PSFOperator, PSFSpec, Parameter, ParameterSet,
    PointSource, RenderContext, SersicProfile, SersicSource, evaluate_model_fit,
    prepare_model_fitting, prepare_noise, render_intrinsic_sersic,
)


def gaussian_pixel_response(x, y, sx, sy):
    wx = ndtr((x + 0.5) / sx) - ndtr((x - 0.5) / sx)
    wy = ndtr((y + 0.5) / sy) - ndtr((y - 0.5) / sy)
    return np.outer(wy, wx)


def gaussian_spec(kind, spacing=1/16, extent=4, sx=0.68, sy=0.81):
    edge = round(extent / spacing)
    u = np.arange(-edge, edge + 1) * spacing
    if kind == "pre_pixel":
        samples = np.exp(-0.5 * ((u[:, None] / sy)**2 + (u[None, :] / sx)**2)) / (math.tau * sx * sy)
    else:
        samples = gaussian_pixel_response(u, u, sx, sy)
    return PSFSpec(samples, kind, spacing, (edge, edge), "analytic total flux",
                   source="synthetic separable Gaussian")


def analytic_gaussian(context, dx, dy, sx, sy):
    ny, nx = context.shape
    cx, cy = context.center
    return gaussian_pixel_response(np.arange(nx) - cx - dx, np.arange(ny) - cy - dy, sx, sy)


@pytest.mark.parametrize("kind", ["pre_pixel", "effective"])
@pytest.mark.parametrize("shape, dx, dy", [((7, 7), 0, 0), ((6, 8), .25, -.5), ((7, 8), .5, .25)])
def test_gaussian_point_and_galaxy_against_analytic_pixel_integrals(kind, shape, dx, dy):
    operator = PSFOperator(gaussian_spec(kind), oversamp=16, max_offset=.5)
    context = RenderContext(shape, psf_operator=operator)
    profile = SersicProfile(1, 1.1, .5, q=.61, pa=90)
    source_y = profile.re / math.sqrt(2 * math.log(2))
    source_x = profile.q * source_y
    point = operator.render_point(dx, dy, context)
    galaxy = operator.render_sersic(profile, dx, dy, context)
    expected_point = analytic_gaussian(context, dx, dy, .68, .81)
    expected_galaxy = analytic_gaussian(context, dx, dy, math.hypot(.68, source_x), math.hypot(.81, source_y))
    noise = prepare_noise(point, np.full(shape, .02))
    for image, expected in ((point, expected_point), (galaxy, expected_galaxy)):
        assert image.shape == shape
        assert image.dtype == np.float64
        assert np.max(abs(image - expected)) < 1.5e-3
        assert abs(image.sum() - expected.sum()) < 2e-3
        assert noise.quadratic(image - expected)**.5 < 0.1


def continuous_gaussian_convolution(profile, context, dx, dy, angles=256):
    """Integrate an analytic Gaussian pixel response directly in source polar coordinates."""
    # With u=b_n*(R/re)**(1/n), the normalized radial flux measure
    # is Gamma(2*n, 1). The angular measure is uniform in the ellipse.
    theta = np.arange(angles) * (math.tau / angles)
    phase = math.radians(profile.pa)
    x = np.cos(theta) * math.cos(phase) - profile.q * np.sin(theta) * math.sin(phase)
    y = np.cos(theta) * math.sin(phase) + profile.q * np.sin(theta) * math.cos(phase)
    ny, nx = context.shape
    cx, cy = context.center
    tx = (np.arange(nx) - cx - dx)[:, None]
    ty = (np.arange(ny) - cy - dy)[:, None]
    shape = 2 * profile.n
    b_n = gammaincinv(shape, 0.5)
    norm = profile.flux / (angles * math.gamma(shape))

    def angular_average(u):
        r = profile.re * (u / b_n)**profile.n
        px, py = tx - r * x, ty - r * y
        wx = ndtr((px + .5) / .68) - ndtr((px - .5) / .68)
        wy = ndtr((py + .5) / .81) - ndtr((py - .5) / .81)
        return (norm * u**(shape - 1) * math.exp(-u)) * (wy @ wx.T)

    return quad_vec(angular_average, 0, 80, epsabs=2e-10, epsrel=2e-10)[0]


@pytest.mark.parametrize("kind", ["pre_pixel", "effective"])
@pytest.mark.parametrize("n, re, q, pa, dx, dy", [
    (.5, 1.2, .45, 37, .17, -.23),
    (1, .7, .44, 26, .25, 0),
    (2, .7, .74, 108, .49, -.5),
    (4, .18, .39, 61, 0, 0),
    (4, .03, .30, 49, .49, -.5),
])
def test_continuous_gaussian_reference_for_tilted_hosts(kind, n, re, q, pa, dx, dy):
    profile = SersicProfile(1, re, n, q, pa)
    context = RenderContext((5, 6))
    expected = continuous_gaussian_convolution(profile, context, dx, dy)
    operator = PSFOperator(gaussian_spec(kind), oversamp=16, max_offset=.5)
    image = operator.render_sersic(profile, dx, dy, context)
    noise = prepare_noise(image, np.full(context.shape, .02))
    assert noise.quadratic(image - expected)**.5 < .1
    assert abs(image.sum() - expected.sum()) < 2e-3


def test_source_lattice_refinement_and_psf_support_refinement():
    profile = SersicProfile(1, .18, 4, .39, 61)
    context = RenderContext((5, 6))
    dx, dy = .17, -.23
    reference = continuous_gaussian_convolution(profile, context, dx, dy)
    np.testing.assert_allclose(
        continuous_gaussian_convolution(profile, context, dx, dy, angles=512), reference,
        rtol=0, atol=1e-8,
    )
    spec = gaussian_spec("effective")
    errors = [
        np.linalg.norm(PSFOperator(spec, oversamp=s, max_offset=.5).render_sersic(
            profile, dx, dy, context
        ) - reference)
        for s in (4, 8, 16)
    ]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .002
    support_images = [
        PSFOperator(gaussian_spec("effective", extent=extent), oversamp=16,
                    max_offset=.5).render_sersic(profile, dx, dy, context)
        for extent in (2, 3, 4)
    ]
    # Compare support changes directly. Remaining interpolation and source
    # quadrature errors can cancel some truncation error against the oracle.
    delta_23 = np.linalg.norm(support_images[1] - support_images[0])
    delta_34 = np.linalg.norm(support_images[2] - support_images[1])
    assert delta_34 < delta_23
    assert delta_34 < .002
    psf_errors = [
        np.linalg.norm(PSFOperator(gaussian_spec("pre_pixel", spacing=spacing)).render_point(
            dx, dy, context
        ) - analytic_gaussian(context, dx, dy, .68, .81))
        for spacing in (.25, .125, .0625)
    ]
    assert psf_errors[2] < psf_errors[1] < psf_errors[0]


@pytest.mark.parametrize("kind", ["pre_pixel", "effective"])
def test_compact_limit_at_off_lattice_phase(kind):
    operator = PSFOperator(gaussian_spec(kind), oversamp=8)
    context = RenderContext((6, 7), psf_operator=operator)
    dx, dy = -0.247, 0.501
    point = operator.render_point(dx, dy, context)
    extended = operator.render_sersic(SersicProfile(1, 1e-6, 4, .4, 33), dx, dy, context)
    np.testing.assert_allclose(extended, point, atol=2e-7, rtol=0)


def bilinear(samples, ox, oy, spacing, x, y):
    """Evaluate the compact interpolant directly, with explicit sample indices."""
    x, y = x / spacing + ox, y / spacing + oy
    h, w = samples.shape
    if not (0 <= x <= w - 1 and 0 <= y <= h - 1):
        return 0.0
    j, i = min(math.floor(x), w - 2), min(math.floor(y), h - 2)
    ax, ay = x - j, y - i
    return (1 - ay) * ((1 - ax) * samples[i, j] + ax * samples[i, j+1]) + ay * (
        (1 - ax) * samples[i+1, j] + ax * samples[i+1, j+1]
    )


def test_asymmetric_fractional_origin_matches_direct_discrete_convolution():
    samples = np.array([[0, 0, .08, 0], [0, .27, .38, .05], [.01, .1, .06, 0]])
    origin = (1.35, .7)
    scale = 1.3
    spec = PSFSpec(samples, "effective", .5, origin, "known reference", scale=scale)
    operator = PSFOperator(spec, oversamp=4, max_offset=.6)
    context = RenderContext((4, 5), psf_operator=operator)
    dx, dy = -.37, .55
    point = operator.render_point(dx, dy, context)
    profile = SersicProfile(1.25, .95, .5, .76, 90)
    galaxy = operator.render_sersic(profile, dx, dy, context)
    cx, cy = context.center
    source_y = profile.re / math.sqrt(2 * math.log(2))
    source_x = profile.q * source_y
    s = operator.oversamp
    coordinates = np.arange(-24, 25) / s
    weights = []
    for sigma in (source_x, source_y):
        weights.append(ndtr((coordinates + .5/s) / sigma) - ndtr((coordinates - .5/s) / sigma))
    wx, wy = weights
    expected = np.empty(context.shape)
    for i, j in np.ndindex(context.shape):
        x, y = j - cx - dx, i - cy - dy
        assert point[i, j] == pytest.approx(scale * bilinear(samples, *origin, .5, x, y), abs=1e-15)
        expected[i, j] = scale * profile.flux * sum(
            wx[a] * wy[b] * bilinear(samples, *origin, .5, x - u, y - v)
            for b, v in enumerate(coordinates) for a, u in enumerate(coordinates)
        )
    np.testing.assert_allclose(galaxy, expected, atol=5e-12, rtol=0)


@pytest.mark.parametrize("kind", ["pre_pixel", "effective"])
def test_shifted_crop_matches_large_stamp_and_retains_stamp_loss(kind):
    operator = PSFOperator(gaussian_spec(kind), oversamp=8, max_offset=2)
    small_context = RenderContext((4, 5))
    large_context = RenderContext((10, 11))
    profile = SersicProfile(2.5, 1.8, 2, .52, 31)
    # Small stamp covers rows 1:5, cols 4:9 of the large stamp. Its
    # geometric center is offset by (+1, -2) in native pixels.
    large = operator.render_sersic(profile, .13, -.45, large_context)
    small = operator.render_sersic(profile, -.87, 1.55, small_context)
    np.testing.assert_allclose(small, large[1:5, 4:9], atol=2e-15, rtol=0)
    assert 0 < small.sum() < large.sum() < profile.flux


def test_narrow_pre_pixel_response_recovers_unconvolved_pixel_integral():
    spacing = 1 / 128
    samples = np.zeros((3, 3))
    samples[1, 1] = spacing**-2
    spec = PSFSpec(samples, "pre_pixel", spacing, (1, 1), "unit integrated tent density")
    operator = PSFOperator(spec, oversamp=24, max_offset=0.5)
    context = RenderContext((5, 6))
    profile = SersicProfile(1, 1.25, .5, .7, 22)
    dx, dy = 0.25, 0
    image = operator.render_sersic(profile, dx, dy, context)
    expected = render_intrinsic_sersic(profile, context.shape, dx=dx, dy=dy)
    np.testing.assert_allclose(image, expected, atol=1.5e-4, rtol=0)


def test_phase_continuity_through_change_of_fft_crop_index():
    operator = PSFOperator(gaussian_spec("effective"), max_offset=.5)
    context = RenderContext((6, 7))
    profile = SersicProfile(1, .18, 4, .4, 37)
    images = [operator.render_sersic(profile, .25 + shift, -.25, context) for shift in (-1e-8, 0, 1e-8)]
    for image in (images[0], images[2]):
        np.testing.assert_allclose(image, images[1], atol=5e-9, rtol=0)


def test_finite_support_and_external_calibration_are_preserved():
    spec = PSFSpec(np.ones((3, 3)), "pre_pixel", 1, (.25, 1.75),
                   "external finite aperture", scale=.2, support_fraction=.8)
    operator = PSFOperator(spec)
    whole = operator.render_point(.25, -.5, RenderContext((9, 10)))
    expected = .2 * 2**2
    assert whole.sum() == pytest.approx(expected, abs=1e-15)
    assert spec.metadata["psf_support_fraction"] == .8
    crop = operator.render_point(.25, -.5, RenderContext((1, 2)))
    assert crop.sum() < whole.sum()
    effective = PSFOperator(PSFSpec(np.ones((5, 5)), "effective", .5, (2, 2),
                                    "arbitrary response calibration", scale=.4))
    assert effective.render_point(0, 0, RenderContext((3, 3))).sum() == pytest.approx(3.6)


def test_composite_model_and_psf_metadata_through_prepared_fit():
    operator = PSFOperator(gaussian_spec("effective"), max_offset=0.5)
    context = RenderContext((4, 5), psf_operator=operator)
    model = CompositeModel([PointSource("agn"), SersicSource("host"), ConstantBackground("sky")])
    values = {"agn.flux": -.3, "agn.dx": .25, "agn.dy": -.5, "host.flux": 1.4,
              "host.dx": .25, "host.dy": -.5, "host.re": .7, "host.n": 1,
              "host.q": .6, "host.pa": 48, "sky.level": .013}
    parameters = ParameterSet([Parameter(name, fixed=value) for name, value in values.items()])
    fit = prepare_model_fitting(np.zeros(context.shape), model, parameters, context=context,
                                noise_options={"err": np.ones(context.shape)})
    result = evaluate_model_fit(fit)
    expected = values["agn.flux"] * operator.render_point(.25, -.5, context)
    expected += operator.render_sersic(SersicProfile(1.4, .7, 1, .6, 48), .25, -.5, context) + .013
    np.testing.assert_allclose(result["image"], expected, atol=1e-15, rtol=0)
    assert result["metadata"]["psf"] == operator.metadata


def test_validation_and_immutable_psf_samples():
    original = np.eye(3)
    constructor = dict(samples=original, spacing=.25, origin=(1, 1), response_kind="effective",
                       normalization_reference="finite calibration", native_pixel_scale=.03)
    spec = PSFSpec(**constructor)
    assert spec.metadata["sample_scale_arcsec"] == pytest.approx(.0075)
    original[:] = 0
    np.testing.assert_array_equal(spec.samples, np.eye(3))
    with pytest.raises(ValueError, match="WRITEABLE"):
        spec.samples.setflags(write=True)
    constructor["samples"] = np.eye(3)
    for update in ({"response_kind": "unknown"}, {"response_kind": "mass"}, {"origin": (3, 0)},
                   {"spacing": 0}, {"scale": -1}, {"support_fraction": 1.1},
                   {"normalization_reference": ""}, {"samples": np.full((3, 3), np.nan)}):
        with pytest.raises(ValueError):
            PSFSpec(**(constructor | update))
    operator = PSFOperator(spec, max_source_cells=50)
    context = RenderContext((4, 5))
    with pytest.raises(ValueError, match="max_source_cells"):
        operator.render_sersic(SersicProfile(1, 1, 1), 0, 0, context)
    with pytest.raises(ValueError, match="max_offset"):
        operator.render_point(3, 0, context)
    with pytest.raises(ValueError, match="cannot be combined"):
        RenderContext((4, 5), point_response=lambda *args: None, psf_operator=operator)
    with pytest.raises(ValueError, match="requires a psf_operator"):
        SersicSource("host").render({}, context)
