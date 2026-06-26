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

    def test_shift_moves_peak_in_correct_direction(self, psf_ex2):
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        centre_col = np.unravel_index(np.argmax(psf_os), psf_os.shape)[1]
        shifted = shift_psf_fourier(psf_os, dx=4.0, dy=0.0)
        shifted_col = np.unravel_index(np.argmax(shifted), shifted.shape)[1]
        assert shifted_col > centre_col


# ---------------------------------------------------------------------------
# downsample_psf
# ---------------------------------------------------------------------------

class TestDownsamplePsf:
    def test_output_is_odd_shaped(self, psf_ex2):
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        native = downsample_psf(psf_os, oversamp=4)
        assert native.shape[0] % 2 == 1
        assert native.shape[1] % 2 == 1

    def test_positive_definite(self, psf_ex2):
        psf_os, _ = prepare_psf_for_oversamp(psf_ex2, oversamp=4, native_shape=(11, 11))
        native = downsample_psf(psf_os, oversamp=4)
        assert native.sum() > 0

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

    def test_non_detection_stamp(self, psf_ex1):
        psf_os, native_shape, frac = prepare_psf_for_oversamp(
            psf_ex1, oversamp=4, native_shape=(11, 11), return_fraction=True
        )
        model = psf_model([0.5, 0.0, 0.0], psf_os, 4, (9, 9), native_shape, frac)
        assert model.shape == (9, 9)
        assert model.sum() > 0
