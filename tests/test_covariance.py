"""
Tests for jwst_psfmc.covariance using the example FITS data in examples/.

Paths are resolved relative to this file so the tests work from any
working directory.
"""

import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits

from jwst_psfmc.covariance import (
    SplitCosineBellWindow,
    distance_grid,
    estimate_cov_kernel,
    find_zero_squares,
    get_source_mask,
    prepare_covariance_terms,
)

_EXAMPLES = Path(__file__).parent.parent / "examples"
_DATA = _EXAMPLES / "data"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def diff_ex2():
    return fits.getdata(_DATA / "example2_f200w_diff.fits").astype(np.float64)


@pytest.fixture(scope="session")
def err_ex2():
    return fits.getdata(_DATA / "example2_f200w_diff_error.fits").astype(np.float64)


@pytest.fixture(scope="session")
def kernel_ex2():
    return fits.getdata(_DATA / "example2_f200w_cov_kernel.fits").astype(np.float64)


@pytest.fixture(scope="session")
def diff_ex1():
    return fits.getdata(_DATA / "example1_f200w_diff.fits").astype(np.float64)


@pytest.fixture(scope="session")
def err_ex1():
    return fits.getdata(_DATA / "example1_f200w_diff_error.fits").astype(np.float64)


@pytest.fixture(scope="session")
def kernel_ex1():
    return fits.getdata(_DATA / "example1_f200w_cov_kernel.fits").astype(np.float64)


def _stamp(img, half=4):
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    return img[cy - half : cy + half + 1, cx - half : cx + half + 1]


# ---------------------------------------------------------------------------
# distance_grid
# ---------------------------------------------------------------------------

class TestDistanceGrid:
    def test_centre_is_zero(self):
        d = distance_grid((11, 11))
        assert d[5, 5] == pytest.approx(0.0)

    def test_shape(self):
        d = distance_grid((7, 9))
        assert d.shape == (7, 9)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            distance_grid((5,))


# ---------------------------------------------------------------------------
# SplitCosineBellWindow
# ---------------------------------------------------------------------------

class TestSplitCosineBellWindow:
    def test_centre_is_one(self):
        w = SplitCosineBellWindow((21, 21), alpha=0.3, beta=0.4)
        assert w[10, 10] == pytest.approx(1.0)

    def test_corner_is_zero(self):
        w = SplitCosineBellWindow((21, 21), alpha=0.5, beta=0.1)
        assert w[0, 0] == pytest.approx(0.0)

    def test_values_in_range(self):
        w = SplitCosineBellWindow((15, 15))
        assert w.min() >= 0.0
        assert w.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# estimate_cov_kernel
# ---------------------------------------------------------------------------

class TestEstimateCovKernel:
    def test_output_shape(self, diff_ex2):
        patch = diff_ex2[10:110, 10:110]
        k = estimate_cov_kernel(patch, size=15, r_in=2.0, r_out=6.0)
        assert k.shape == (15, 15)

    def test_peak_at_centre(self, diff_ex2):
        patch = diff_ex2[10:110, 10:110]
        k = estimate_cov_kernel(patch, size=15)
        assert np.argmax(k) == np.ravel_multi_index((7, 7), k.shape)

    def test_values_bounded(self, diff_ex2):
        patch = diff_ex2[10:110, 10:110]
        k = estimate_cov_kernel(patch, size=15)
        assert k.min() >= -1.0
        assert k.max() <= 1.0 + 1e-9

    def test_edge_is_zero(self, diff_ex2):
        patch = diff_ex2[10:110, 10:110]
        k = estimate_cov_kernel(patch, size=15, r_in=2.0, r_out=6.0)
        assert k[0, 0] == pytest.approx(0.0, abs=1e-12)

    def test_invalid_size_even(self, diff_ex2):
        with pytest.raises(ValueError, match="odd"):
            estimate_cov_kernel(diff_ex2[10:110, 10:110], size=14)

    def test_non_detection_epoch(self, diff_ex1):
        patch = diff_ex1[10:110, 10:110]
        k = estimate_cov_kernel(patch, size=15)
        assert k.shape == (15, 15)
        assert k[7, 7] > 0.5


# ---------------------------------------------------------------------------
# find_zero_squares
# ---------------------------------------------------------------------------

class TestFindZeroSquares:
    def test_all_free_returns_squares(self):
        arr = np.zeros((50, 50), dtype=int)
        squares = find_zero_squares(arr, a=10, all_sizes=False, max_nonzero=0)
        assert squares.shape[1] == 3
        assert len(squares) > 0

    def test_no_free_region(self):
        arr = np.ones((20, 20), dtype=int)
        squares = find_zero_squares(arr, a=5, all_sizes=False, max_nonzero=0)
        assert squares.shape == (0, 3)

    def test_result_columns(self):
        arr = np.zeros((30, 30), dtype=int)
        arr[5:15, 5:15] = 1
        squares = find_zero_squares(arr, a=5, all_sizes=False, max_nonzero=0)
        assert squares.ndim == 2
        assert squares.shape[1] == 3

    def test_tolerance(self):
        arr = np.zeros((20, 20), dtype=int)
        arr[3, 3] = 1
        s_strict = find_zero_squares(arr, a=5, all_sizes=False, max_nonzero=0)
        s_loose  = find_zero_squares(arr, a=5, all_sizes=False, max_nonzero=2)
        assert len(s_loose) >= len(s_strict)

    def test_on_bundled_mask(self):
        mask_int = fits.getdata(_DATA / "example2_f200w_mask.fits")
        squares = find_zero_squares(mask_int, a=20, all_sizes=False, max_nonzero=5)
        assert len(squares) > 0


# ---------------------------------------------------------------------------
# prepare_covariance_terms
# ---------------------------------------------------------------------------

class TestPrepareCovariance:
    def test_output_keys(self, diff_ex2, err_ex2, kernel_ex2):
        prepared = prepare_covariance_terms(_stamp(diff_ex2), _stamp(err_ex2), kernel_ex2)
        for key in ("data", "err", "valid", "fit_weight", "power_spectrum", "shape"):
            assert key in prepared

    def test_shape(self, diff_ex2, err_ex2, kernel_ex2):
        stamp = _stamp(diff_ex2)
        prepared = prepare_covariance_terms(stamp, _stamp(err_ex2), kernel_ex2)
        assert prepared["shape"] == stamp.shape

    def test_power_spectrum_positive(self, diff_ex2, err_ex2, kernel_ex2):
        prepared = prepare_covariance_terms(_stamp(diff_ex2), _stamp(err_ex2), kernel_ex2)
        assert np.all(prepared["power_spectrum"] > 0)

    def test_no_valid_pixels_raises(self):
        data = np.full((9, 9), np.nan)
        err  = np.ones((9, 9))
        kern = np.eye(9)
        with pytest.raises(ValueError, match="No valid pixels"):
            prepare_covariance_terms(data, err, kern)

    def test_non_detection_epoch(self, diff_ex1, err_ex1, kernel_ex1):
        prepared = prepare_covariance_terms(_stamp(diff_ex1), _stamp(err_ex1), kernel_ex1)
        assert prepared["shape"] == _stamp(diff_ex1).shape
        assert np.any(prepared["valid"])
