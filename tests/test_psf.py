"""
Tests for jwst_psfmc.psf using the example FITS data in examples/.

These tests load the real oversampled PSF from examples/PSF/ and the real
difference-image stamps from examples/data/ to provide genuine end-to-end
integration coverage.  Paths are resolved relative to this file so the tests
work from any working directory.
"""

import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits

from jwst_psfmc.psf import (
    shift_psf_fourier,
    downsample_psf,
    match_shape_center,
    prepare_psf_for_oversamp,
    psf_model,
)

# Resolve paths relative to this test file (tests/ -> repo root -> examples/)
_EXAMPLES = Path(__file__).parent.parent / "examples"
_DATA = _EXAMPLES / "data"
_PSF  = _EXAMPLES / "PSF"

# The example FITS files live in examples/ and are intentionally not shipped
# inside the distribution (see README). Skip rather than error when running
# from an sdist, where tests/ is present but examples/ is not.
if not _DATA.is_dir():
    pytest.skip(
        "example data not available (examples/ is not shipped in the "
        "distribution) - clone the repository to run these tests",
        allow_module_level=True,
    )



# ---------------------------------------------------------------------------
# Fixtures – load bundled FITS once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def psf_ex2():
    """4× oversampled F200W PSF for example2 (detection)."""
    return fits.getdata(_PSF / "example2_f200w_PSF_4_c.fits").astype(np.float64)


@pytest.fixture(scope="session")
def psf_ex1():
    """4× oversampled F200W PSF for example1 (non-detection)."""
    return fits.getdata(_PSF / "example1_f200w_PSF_4_c.fits").astype(np.float64)


@pytest.fixture(scope="session")
def stamp_ex2():
    """9×9 detection stamp for example2."""
    data = fits.getdata(_DATA / "example2_f200w_diff.fits").astype(np.float64)
    cy, cx = data.shape[0] // 2, data.shape[1] // 2
    return data[cy - 4 : cy + 5, cx - 4 : cx + 5]


@pytest.fixture(scope="session")
def stamp_ex1():
    """9×9 non-detection stamp for example1."""
    data = fits.getdata(_DATA / "example1_f200w_diff.fits").astype(np.float64)
    cy, cx = data.shape[0] // 2, data.shape[1] // 2
    return data[cy - 4 : cy + 5, cx - 4 : cx + 5]


def _compact_gaussian(n, sigma=3.0):
    """Normalised Gaussian on an n x n grid, with negligible flux at the edges."""
    c = (n - 1) / 2
    y, x = np.mgrid[:n, :n]
    g = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _shift_without_nyquist(a, dx, dy):
    """Reference shift that zeroes the Nyquist bin first, so nothing complex is
    discarded. Differs from shift_psf_fourier only in that bin."""
    A = np.fft.fft2(a)
    ny, nx = a.shape
    if ny % 2 == 0:
        A[ny // 2, :] = 0
    if nx % 2 == 0:
        A[:, nx // 2] = 0
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    return np.real(np.fft.ifft2(A * np.exp(-2j * np.pi * (ky * dy + kx * dx))))


# ---------------------------------------------------------------------------
# shift_psf_fourier
# ---------------------------------------------------------------------------

class TestShiftPsfFourier:
    def test_zero_shift_is_identity(self, psf_ex2):
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        shifted = shift_psf_fourier(psf_os, 0.0, 0.0)
        np.testing.assert_allclose(shifted, psf_os, atol=1e-10)

    def test_flux_conservation(self, psf_ex2):
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        original_sum = psf_os.sum()
        for dx, dy in [(0.5, 0.0), (0.0, 0.5), (1.3, -0.7)]:
            shifted = shift_psf_fourier(psf_os, dx, dy)
            np.testing.assert_allclose(
                shifted.sum(), original_sum, rtol=1e-9,
                err_msg=f"Flux not conserved for dx={dx}, dy={dy}",
            )

    def test_shift_is_accurate(self):
        """The centroid moves by exactly the requested sub-pixel amount.

        Guards the package's central numerical claim: exact Fourier-space
        interpolation. Uses a compact Gaussian on an odd-sized grid so that
        neither wrap-around nor the even-grid Nyquist loss (below) contaminates
        the measurement.
        """
        a = _compact_gaussian(45, sigma=3.0)

        def centroid(z):
            w = np.clip(z, 0, None)
            yy, xx = np.mgrid[: z.shape[0], : z.shape[1]]
            return (yy * w).sum() / w.sum(), (xx * w).sum() / w.sum()

        cy0, cx0 = centroid(a)
        for dx, dy in [(0.25, 0.0), (0.0, -0.75), (1.5, 2.25)]:
            cy, cx = centroid(shift_psf_fourier(a, dx, dy))
            assert cx - cx0 == pytest.approx(dx, abs=1e-6)
            assert cy - cy0 == pytest.approx(dy, abs=1e-6)

    @pytest.mark.parametrize("n", [43, 45, 65])
    def test_shift_round_trip_is_exact_on_odd_grids(self, n):
        """Shifting by +d then -d returns the original to machine precision."""
        rng = np.random.default_rng(0)
        a = rng.random((n, n))
        back = shift_psf_fourier(shift_psf_fourier(a, 0.3, -0.4), -0.3, 0.4)
        np.testing.assert_allclose(back, a, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("dx,dy", [(0.3, -0.4), (0.5, 0.5),
                                       (1.25, -2.75), (0.1, 0.1)])
    def test_even_grid_nyquist_loss_vanishes_after_downsampling(self, psf_ex2, dx, dy):
        """The Nyquist artefact of even grids cannot reach the model output.

        The oversampled working grid is even (native_shape 11 x oversamp 4 = 44)
        even though both the PSF file and the data stamp are odd-sized. On an
        even axis a non-integer shift makes the Nyquist bin complex, breaking
        Hermitian symmetry, and taking the real part discards it — so the
        oversampled array does differ from an exact shift.

        That discarded component is precisely the one the block-sum removes: a
        boxcar of length 4 on a 44-grid has zeros at k = 11, 22, 33, and the
        Nyquist bin is k = 22. The native-resolution model is therefore exact.
        """
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        assert psf_os.shape == (44, 44)                     # even, the affected case

        shipped = shift_psf_fourier(psf_os, dx, dy)
        reference = _shift_without_nyquist(psf_os, dx, dy)

        # They differ on the oversampled grid ...
        assert np.abs(shipped - reference).max() / np.abs(shipped).max() > 1e-6

        # ... and agree exactly once block-summed to native resolution.
        a = downsample_psf(shipped, 4, recenter_peak=False)
        b = downsample_psf(reference, 4, recenter_peak=False)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-14 * np.abs(a).max())


# ---------------------------------------------------------------------------
# downsample_psf
# ---------------------------------------------------------------------------

class TestDownsamplePsf:
    def test_output_is_odd_shaped(self, psf_ex2):
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        native = downsample_psf(psf_os, oversamp=4)
        assert native.shape[0] % 2 == 1
        assert native.shape[1] % 2 == 1

    def test_non_negative_and_positive_sum(self, psf_ex2):
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        native = downsample_psf(psf_os, oversamp=4)
        assert native.sum() > 0
        assert native.min() >= 0.0

    def test_invalid_oversamp_raises(self, psf_ex2):
        with pytest.raises(ValueError, match="positive integer"):
            downsample_psf(psf_ex2, oversamp=0)

    def test_recenter_peak(self, psf_ex2):
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        native = downsample_psf(psf_os, oversamp=4, recenter_peak=True)
        cy, cx = np.array(native.shape) // 2
        py, px = np.unravel_index(np.argmax(native), native.shape)
        assert py == cy and px == cx


# ---------------------------------------------------------------------------
# prepare_psf_for_oversamp
# ---------------------------------------------------------------------------

class TestPreparePsfForOversamp:
    def test_native_shape_respected(self, psf_ex2):
        psf_crop, out_shape, frac = prepare_psf_for_oversamp(
            psf_ex2, oversamp=4, native_shape=(11, 11), return_fraction=True
        )
        assert out_shape == (11, 11)
        assert psf_crop.shape == (44, 44)
        assert 0.5 < frac <= 1.0

    def test_auto_native_shape_is_odd(self, psf_ex2):
        _, out_shape = prepare_psf_for_oversamp(psf_ex2, oversamp=4)
        assert out_shape[0] % 2 == 1
        assert out_shape[1] % 2 == 1

    def test_too_small_raises(self):
        tiny = np.ones((5, 5))
        with pytest.raises(ValueError, match="too small"):
            prepare_psf_for_oversamp(tiny, oversamp=4, native_shape=(11, 11))


# ---------------------------------------------------------------------------
# psf_model
# ---------------------------------------------------------------------------

class TestPsfModel:
    def test_output_shape(self, psf_ex2, stamp_ex2):
        psf_os, native_shape, frac = prepare_psf_for_oversamp(
            psf_ex2, oversamp=4, native_shape=(11, 11), return_fraction=True
        )
        model = psf_model(
            [1.0, 0.0, 0.0], psf_os, oversamp=4,
            output_shape=stamp_ex2.shape,
            native_shape=native_shape,
            psf_prepare_fraction=frac,
        )
        assert model.shape == stamp_ex2.shape

    def test_zero_flux_gives_zero_model(self, psf_ex2, stamp_ex2):
        psf_os, native_shape, frac = prepare_psf_for_oversamp(
            psf_ex2, oversamp=4, native_shape=(11, 11), return_fraction=True
        )
        model = psf_model(
            [0.0, 0.0, 0.0], psf_os, oversamp=4,
            output_shape=stamp_ex2.shape,
            native_shape=native_shape,
            psf_prepare_fraction=frac,
        )
        np.testing.assert_allclose(model, 0.0, atol=1e-15)

    def test_flux_scaling_linear(self, psf_ex2, stamp_ex2):
        psf_os, native_shape, frac = prepare_psf_for_oversamp(
            psf_ex2, oversamp=4, native_shape=(11, 11), return_fraction=True
        )
        m1 = psf_model([1.0, 0.0, 0.0], psf_os, 4, stamp_ex2.shape, native_shape, frac)
        m5 = psf_model([5.0, 0.0, 0.0], psf_os, 4, stamp_ex2.shape, native_shape, frac)
        np.testing.assert_allclose(m5, 5.0 * m1, rtol=1e-10)

    def test_return_meta(self, psf_ex2, stamp_ex2):
        psf_os, native_shape, frac = prepare_psf_for_oversamp(
            psf_ex2, oversamp=4, native_shape=(11, 11), return_fraction=True
        )
        _, meta = psf_model(
            [2.5, 0.1, -0.1], psf_os, 4, stamp_ex2.shape, native_shape, frac,
            return_meta=True,
        )
        assert "total_encircled_fraction" in meta
        assert 0.0 < meta["total_encircled_fraction"] <= 1.0

    def test_stamp_smaller_than_native_shape_loses_flux(self, psf_ex1):
        """native_shape=(11, 11) against a 9x9 stamp: the margin is cropped away.

        The model is evaluated on the 11x11 native grid and then cropped to the
        stamp, so the in-stamp sum must fall short of the requested total flux
        while the reported encircled fraction accounts for the difference.
        """
        psf_os, native_shape, frac = prepare_psf_for_oversamp(
            psf_ex1, oversamp=4, native_shape=(11, 11), return_fraction=True
        )
        flux = 0.5
        model, meta = psf_model([flux, 0.0, 0.0], psf_os, 4, (9, 9), native_shape,
                                frac, return_meta=True)
        assert model.shape == (9, 9)
        assert 0 < model.sum() < flux
        assert meta["total_encircled_fraction"] < 1.0
        assert model.sum() == pytest.approx(flux * meta["total_encircled_fraction"],
                                            rel=1e-6)
