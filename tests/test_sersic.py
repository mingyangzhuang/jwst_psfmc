"""Analytic and independent Cartesian checks of the intrinsic Sérsic model."""

import math

import numpy as np
import pytest
from astropy.modeling.models import Sersic2D
from scipy.integrate import quad
from scipy.special import erf

from jwst_psfmc import SersicProfile, integrate_sersic_grid, render_intrinsic_sersic


BN_EXPONENTIAL = 1.6783469900166605
BN_DE_VAUCOULEURS = 7.669249442500804


def gaussian_covariance(re, q, pa):
    sigma_sq = re**2 / (2 * math.log(2))
    angle = math.radians(pa)
    major = np.array([math.cos(angle), math.sin(angle)])
    return sigma_sq * (q**2 * np.eye(2) + (1 - q**2) * np.outer(major, major))


def gaussian_rectangle_image(profile, shape, dx, dy):
    """Closed-form axis-aligned Gaussian pixel integrals in native units."""
    ny, nx = shape
    cx, cy = (nx - 1) / 2 + dx, (ny - 1) / 2 + dy
    major = profile.re / math.sqrt(2 * math.log(2))
    sig_x, sig_y = major, major * profile.q
    if profile.pa == 90:
        sig_x, sig_y = sig_y, sig_x
    wx = np.diff(erf((np.arange(nx + 1) - 0.5 - cx) / (math.sqrt(2) * sig_x))) / 2
    wy = np.diff(erf((np.arange(ny + 1) - 0.5 - cy) / (math.sqrt(2) * sig_y))) / 2
    return profile.flux * np.outer(wy, wx)


def cartesian_integral(intensity, x0, x1, y0, y1):
    """Independent nested quadrature, splitting coordinates at the cusp."""
    x_cuts = [x0, 0, x1] if x0 < 0 < x1 else [x0, x1]
    y_cuts = [y0, 0, y1] if y0 < 0 < y1 else [y0, y1]
    total = 0.0
    for lo_x, hi_x in zip(x_cuts, x_cuts[1:]):
        for lo_y, hi_y in zip(y_cuts, y_cuts[1:]):
            def row(x):
                return quad(lambda y: intensity(x, y), lo_y, hi_y,
                            epsabs=1e-12, epsrel=2e-10)[0]
            total += quad(row, lo_x, hi_x, epsabs=1e-11, epsrel=2e-10)[0]
    return total


@pytest.mark.parametrize("pa", [37, 127, -143])
def test_rotated_gaussian_intensity_matches_cartesian_covariance(pa):
    profile = SersicProfile(3.7, 1.3, 0.5, q=0.41, pa=pa)
    covariance = gaussian_covariance(profile.re, profile.q, pa)
    inverse = np.linalg.inv(covariance)
    norm = 2 * math.pi * math.sqrt(np.linalg.det(covariance))
    x, y = np.array([-1.2, 0, 1.7]), np.array([0.4, 0, 0.8])
    coords = np.column_stack([x, y])
    exponent = -0.5 * np.einsum("ij,jk,ik->i", coords, inverse, coords)
    expected = profile.flux * np.exp(exponent) / norm
    np.testing.assert_allclose(profile.intensity(x, y), expected, rtol=4e-14)
    assert profile.b_n == pytest.approx(math.log(2), abs=1e-15)
    assert profile.re_circ == pytest.approx(1.3 * math.sqrt(0.41))


@pytest.mark.parametrize("n, bn", [(1, BN_EXPONENTIAL), (4, BN_DE_VAUCOULEURS)])
def test_profile_matches_astropy_with_independent_amplitude(n, bn):
    profile = SersicProfile(7.1, 1.65, n, q=0.62, pa=132)
    amplitude = (
        profile.flux * bn**(2 * n) * math.exp(-bn)
        / (math.tau * profile.q * n * profile.re**2 * math.factorial(2 * n - 1))
    )
    model = Sersic2D(amplitude=amplitude, r_eff=profile.re, n=n, ellip=1 - profile.q,
                     theta=math.radians(132))
    y, x = np.mgrid[-4:5, -5:6]
    np.testing.assert_allclose(profile.intensity(x / 3, y / 2), model(x / 3, y / 2),
                               rtol=6e-14, atol=1e-15)


@pytest.mark.parametrize("n", [0.5, 1, 2, 4])
def test_total_half_light_and_curve_of_growth(n):
    profile = SersicProfile(2.3, 1.6, n, q=0.31)
    known_bn = {0.5: math.log(2), 1: BN_EXPONENTIAL, 2: 3.672060748850897,
                4: BN_DE_VAUCOULEURS}[n]

    def ring_density(radius):
        return math.tau * profile.q * radius * float(profile.intensity(radius, 0))

    inner = quad(ring_density, 0, profile.re, epsabs=2e-12)[0]
    outer = quad(ring_density, profile.re, np.inf, epsabs=2e-12)[0]
    assert inner == pytest.approx(profile.flux / 2, abs=5e-10)
    assert inner + outer == pytest.approx(profile.flux, abs=5e-10)
    radii = np.array([0, 0.09, profile.re, 3.2, 15, np.inf])
    z = known_bn * (radii[:-1] / profile.re)**(1 / n)
    # For these n the gamma shape is integer and its CDF has a finite series.
    expected = 1 - np.exp(-z) * sum(z**k / math.factorial(k) for k in range(int(2*n)))
    np.testing.assert_allclose(profile.enclosed_fraction(radii), np.r_[expected, 1], atol=2e-14)


@pytest.mark.parametrize("shape, dx, dy, pa", [
    ((5, 7), 0, 0, 0),
    ((4, 6), 0, 0, 90),
    ((4, 7), 0.25, -0.5, 0),
    ((5, 6), 0.5, 0.25, 90),
])
def test_gaussian_pixel_integrals_and_native_origin(shape, dx, dy, pa):
    profile = SersicProfile(2.5, 0.8, 0.5, q=0.55, pa=pa)
    actual = render_intrinsic_sersic(profile, shape, dx=dx, dy=dy)
    expected = gaussian_rectangle_image(profile, shape, dx, dy)
    assert actual.shape == shape
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-11)


@pytest.mark.parametrize("offset", [1e-6, 1e-8, 1e-10])
def test_gaussian_arbitrarily_close_to_pixel_corner(offset):
    profile = SersicProfile(1, 0.1, 0.5, q=0.4)
    dx, dy = 0.5 + offset, 0.5 - offset
    actual = render_intrinsic_sersic(profile, (3, 3), dx=dx, dy=dy)
    expected = gaussian_rectangle_image(profile, (3, 3), dx, dy)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_exponential_narrow_rotated_pixels_match_independent_cartesian_integrals():
    flux, re, q, theta = 4.3, 0.35, 0.32, math.radians(26)
    profile = SersicProfile(flux, re, 1, q=q, pa=26)
    decay = BN_EXPONENTIAL / re
    central_intensity = flux * decay**2 / (math.tau * q)

    def intensity(x, y):
        radius = math.hypot(x * math.cos(theta) + y * math.sin(theta),
                            (-x * math.sin(theta) + y * math.cos(theta)) / q)
        return central_intensity * math.exp(-decay * radius)

    x_edges, y_edges = [-0.95, 0.07, 0.9], [-0.31, 0.16, 0.65]
    actual = integrate_sersic_grid(profile, x_edges, y_edges, atol=1e-12, rtol=1e-10)
    expected = np.array([
        [cartesian_integral(intensity, x0, x1, y0, y1) for x0, x1 in zip(x_edges, x_edges[1:])]
        for y0, y1 in zip(y_edges, y_edges[1:])
    ])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-11)


def test_compact_tilted_n4_matches_independent_cartesian_integrals():
    profile = SersicProfile(1.7, 0.18, 4, q=0.39, pa=61)
    center = profile.flux * BN_DE_VAUCOULEURS**8 / (
        math.tau * profile.q * 4 * profile.re**2 * math.factorial(7)
    )
    theta = math.radians(61)

    def intensity(x, y):
        radius = math.hypot(x * math.cos(theta) + y * math.sin(theta),
                            (-x * math.sin(theta) + y * math.cos(theta)) / profile.q)
        return center * math.exp(-BN_DE_VAUCOULEURS * (radius / profile.re)**0.25)

    x_edges, y_edges = [-1.02, -0.02, 0.98], [-1, 0, 1]
    actual = integrate_sersic_grid(profile, x_edges, y_edges, atol=1e-12, rtol=1e-10)
    expected = np.array([
        [cartesian_integral(intensity, x0, x1, y0, y1) for x0, x1 in zip(x_edges, x_edges[1:])]
        for y0, y1 in zip(y_edges, y_edges[1:])
    ])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-10)


@pytest.mark.parametrize("n, re, q, pa", [(0.5, 1.7, 0.5, 15), (2, 0.7, 0.74, 108), (4, 0.03, 0.3, 49)])
def test_subcell_refinement_and_tolerance_convergence(n, re, q, pa):
    profile = SersicProfile(1, re, n, q=q, pa=pa)
    shape = (3, 4)
    options = {"dx": 0.49, "dy": -0.5}
    reference = render_intrinsic_sersic(profile, shape, atol=1e-12, rtol=1e-11, **options)
    default = render_intrinsic_sersic(profile, shape, **options)
    np.testing.assert_allclose(default, reference, atol=2e-10, rtol=0)
    for oversamp in (2, 4):
        subcells = render_intrinsic_sersic(profile, shape, oversamp=oversamp, **options)
        assert subcells.shape == tuple(size * oversamp for size in shape)
        binned = subcells.reshape(shape[0], oversamp, shape[1], oversamp).sum(axis=(1, 3))
        np.testing.assert_allclose(binned, reference, rtol=0, atol=2e-10)


def test_flux_scales_and_crop_keeps_missing_wings():
    shape = (5, 6)
    geometry = dict(dx=-0.25, dy=0.5)
    profile = SersicProfile(4.7, 1.9, 4, q=0.56, pa=41)
    small = render_intrinsic_sersic(profile, shape, **geometry)
    large = render_intrinsic_sersic(profile, (9, 10), **geometry)
    unit = render_intrinsic_sersic(SersicProfile(1, 1.9, 4, q=0.56, pa=41), shape, **geometry)
    np.testing.assert_array_equal(small, large[2:-2, 2:-2])
    np.testing.assert_array_equal(small, 4.7 * unit)
    assert 0 < small.sum() < large.sum() < profile.flux


def test_zero_flux_and_circular_angle_degeneracy():
    zero = SersicProfile(0, 0.1, 4, pa=88)
    np.testing.assert_array_equal(zero.intensity([0, 1], 0), [0, 0])
    np.testing.assert_array_equal(render_intrinsic_sersic(zero, (3, 2), oversamp=2), np.zeros((6, 4)))
    assert zero.enclosed_fraction(zero.re) == pytest.approx(0.5)
    a, b = SersicProfile(1, 0.4, 2), SersicProfile(1, 0.4, 2, pa=105)
    np.testing.assert_array_equal(a.intensity([-1, 0, 1], 0.2), b.intensity([-1, 0, 1], 0.2))
    np.testing.assert_array_equal(render_intrinsic_sersic(a, (3, 3)), render_intrinsic_sersic(b, (3, 3)))


def test_far_wing_cell_mass_retains_small_positive_probability():
    # At these radii both lower Gamma CDF values round to one.
    profile = SersicProfile(1, 1, 0.5)
    value = integrate_sersic_grid(profile, [9, 10], [-0.25, 0.25])[0, 0]
    root_ln2 = math.sqrt(math.log(2))
    expected = 0.5 * (math.erfc(9 * root_ln2) - math.erfc(10 * root_ln2)) * erf(0.25 * root_ln2)
    assert expected > 0
    assert value / expected == pytest.approx(1, rel=1e-10)


@pytest.mark.parametrize("options", [
    {"flux": -1}, {"flux": float("nan")}, {"re": 0}, {"n": -1}, {"q": 1.2},
    {"q": 0}, {"pa": float("inf")}, {"n": 1e-12}, {"pa": True},
])
def test_invalid_physical_parameters(options):
    with pytest.raises(ValueError):
        SersicProfile(**({"flux": 1, "re": 1, "n": 1} | options))


def test_invalid_coordinates_grids_and_tolerances():
    profile = SersicProfile(1, 1, 1)
    for x in (float("inf"), [1j], [1, float("nan")]):
        with pytest.raises(ValueError):
            profile.intensity(x, 0)
    for radius in (-1, complex(1), float("nan")):
        with pytest.raises(ValueError):
            profile.enclosed_fraction(radius)
    for edges in ([1, 0], [0, 0], [[1, 2]], [0], [0, float("inf")]):
        with pytest.raises(ValueError, match="edges"):
            integrate_sersic_grid(profile, edges, [0, 1])
    for options in ({"oversamp": 0}, {"oversamp": 2.5}, {"oversamp": True},
                    {"dx": float("nan")}, {"rtol": 0}, {"atol": 2}):
        with pytest.raises(ValueError):
            render_intrinsic_sersic(profile, (3, 4), **options)
