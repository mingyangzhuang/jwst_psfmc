"""
Tests for jwst_psfmc.covariance using the example FITS data in examples/.

Paths are resolved relative to this file so the tests work from any
working directory.
"""

import warnings

import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from jwst_psfmc.covariance import (
    SplitCosineBellWindow,
    distance_grid,
    estimate_cov_kernel,
    find_zero_squares,
    get_source_mask,
    kernel_power_spectrum,
    prepare_covariance_terms,
    whiten_image,
)

_EXAMPLES = Path(__file__).parent.parent / "examples"
_DATA = _EXAMPLES / "data"

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

    def test_oversized_kernel_is_cropped_to_stamp(self, diff_ex1, err_ex1, kernel_ex1):
        """The 15x15 kernel is larger than the 9x9 stamp and must be cropped.

        Centre-padding instead of cropping produces a negative offset and
        raises ValueError; this is the failure the demo notebook hit.
        """
        stamp = _stamp(diff_ex1)
        assert kernel_ex1.shape[0] > stamp.shape[0]
        prepared = prepare_covariance_terms(stamp, _stamp(err_ex1), kernel_ex1)
        assert prepared["shape"] == stamp.shape
        assert prepared["power_spectrum"].shape == stamp.shape
        assert np.any(prepared["valid"])


# ---------------------------------------------------------------------------
# kernel_power_spectrum / whiten_image
# ---------------------------------------------------------------------------


def _real_sky_patch():
    """Largest source-free square of the bundled F444W example."""
    diff = fits.getdata(_EXAMPLES / "data" / "example3_f444w_diff.fits").astype(float)
    mask = fits.getdata(_EXAMPLES / "data" / "example3_f444w_mask.fits").astype(bool)
    squares = find_zero_squares(mask.astype(np.int64), a=60, all_sizes=True,
                                max_nonzero=5)
    # 56 squares tie at the largest size here, and np.argsort defaults to an
    # unstable quicksort, so which tie ends up last depends on the numpy build.
    # That silently changed the patch between machines and moved the whitened
    # RMS ratio over a ~0.4 % range. Break the tie explicitly and deterministic-
    # ally: largest size, then smallest top, then smallest left.
    order = np.lexsort((squares[:, 1], squares[:, 0], -squares[:, 2]))
    top, left, size = squares[order[0]].astype(int)
    patch = diff[top:top + size, left:left + size].copy()
    bad = ~np.isfinite(patch)
    if bad.any():
        mean, _, std = sigma_clipped_stats(patch[np.isfinite(patch)], sigma=3)
        patch[bad] = np.random.default_rng(42).standard_normal(bad.sum()) * std + mean
        patch = patch - mean
    return patch


class TestKernelPowerSpectrum:
    def test_shape_and_positivity(self):
        kernel = estimate_cov_kernel(_real_sky_patch(), size=15)
        ps = kernel_power_spectrum(kernel, (64, 64))
        assert ps.shape == (64, 64)
        assert np.all(ps > 0)
        assert np.all(np.isfinite(ps))

    def test_mean_equals_kernel_centre(self):
        """mean(P) equals the kernel centre, which estimate_cov_kernel sets to 1."""
        patch = _real_sky_patch()
        kernel = estimate_cov_kernel(patch, size=15)
        ps = kernel_power_spectrum(kernel, patch.shape)
        assert np.all(ps > 1e-8)          # nothing clipped, so the identity is exact
        centre = kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]
        assert centre == pytest.approx(1.0, rel=1e-10)
        assert ps.mean() == pytest.approx(1.0, rel=1e-10)

    def test_kernel_larger_than_grid_is_cropped(self):
        """A kernel bigger than the image is centre-cropped, not centre-padded.

        This is the 15x15-kernel / 9x9-stamp case used by the PSF fitting demo.
        """
        kernel = estimate_cov_kernel(_real_sky_patch(), size=15)
        ps = kernel_power_spectrum(kernel, (9, 9))
        assert ps.shape == (9, 9)
        assert np.all(ps > 0)

    @pytest.mark.parametrize("shape", [(16, 16), (15, 15), (10, 9), (9, 10)])
    def test_delta_kernel_gives_flat_spectrum(self, shape):
        """A delta kernel means uncorrelated noise -> flat unit spectrum.

        Exercises even- and odd-sized grids: the kernel centre must land on
        index (0, 0), which plain ifftshift does not guarantee when even.
        """
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1.0
        assert np.allclose(kernel_power_spectrum(kernel, shape), 1.0)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            kernel_power_spectrum(np.ones(5), (8, 8))
        with pytest.raises(ValueError):
            kernel_power_spectrum(np.array([[np.nan]]), (8, 8))
        with pytest.raises(ValueError):
            kernel_power_spectrum(np.ones((3, 3)), (0, 8))


class TestWhitenImage:
    def test_preserves_rms_on_real_data(self):
        """Unit-mean spectrum => Parseval => the RMS is unchanged."""
        patch = _real_sky_patch()
        kernel = estimate_cov_kernel(patch, size=15)
        white = whiten_image(patch, kernel=kernel)
        # The kernel spectrum has unit mean, so Parseval preserves the variance.
        # The residual is set by how well the truncated, windowed kernel models
        # this particular patch; across the source-free squares of this image it
        # stays within a few parts in 1000.
        assert white.std() == pytest.approx(patch.std(), rel=5e-3)

    def test_reduces_pixel_correlation(self):
        patch = _real_sky_patch()
        kernel = estimate_cov_kernel(patch, size=15)
        white = whiten_image(patch, kernel=kernel)
        before = estimate_cov_kernel(patch, size=15)
        after = estimate_cov_kernel(white, size=15)
        c = before.shape[0] // 2
        assert before[c, c + 1] > 0.2                       # genuinely correlated
        assert abs(after[c, c + 1]) < 0.1                   # and whitened

    def test_kernel_and_power_spectrum_agree(self):
        patch = _real_sky_patch()
        kernel = estimate_cov_kernel(patch, size=15)
        ps = kernel_power_spectrum(kernel, patch.shape)
        assert np.allclose(whiten_image(patch, kernel=kernel),
                           whiten_image(patch, power_spectrum=ps))

    def test_delta_kernel_is_identity(self):
        patch = _real_sky_patch()[:32, :32]
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1.0
        assert np.allclose(whiten_image(patch, kernel=kernel), patch, atol=1e-10)

    def test_warns_when_spectrum_underflows(self):
        """Over-smoothed noise drives the modelled power below eps."""
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(7)
        patch = gaussian_filter(rng.standard_normal((96, 96)), 1.2)
        kernel = estimate_cov_kernel(patch, size=15)
        with pytest.warns(RuntimeWarning, match="clipped"):
            whiten_image(patch, kernel=kernel)

    def test_requires_exactly_one_divisor(self):
        patch = _real_sky_patch()
        kernel = estimate_cov_kernel(patch, size=15)
        with pytest.raises(ValueError):
            whiten_image(patch)
        with pytest.raises(ValueError):
            whiten_image(patch, kernel=kernel,
                         power_spectrum=kernel_power_spectrum(kernel, patch.shape))

    def test_rejects_bad_input(self):
        patch = _real_sky_patch()
        kernel = estimate_cov_kernel(patch, size=15)
        with pytest.raises(ValueError):
            whiten_image(np.ones(8), kernel=kernel)
        with pytest.raises(ValueError):
            whiten_image(np.full((8, 8), np.nan), kernel=kernel)
        with pytest.raises(ValueError):
            whiten_image(patch, power_spectrum=np.ones((4, 4)))


# ---------------------------------------------------------------------------
# get_source_mask
# ---------------------------------------------------------------------------


class TestGetSourceMask:
    def test_blank_sky_is_not_masked(self):
        """Detection must run on the same smoothed image the threshold came from.

        Thresholding the unsmoothed image with a threshold measured on the
        smoothed one is an effective cut far below 3 sigma and flagged ~22% of
        pure noise as sources.
        """
        rng = np.random.default_rng(0)
        mask = get_source_mask(rng.standard_normal((100, 100)))
        assert mask.mean() < 0.02, f"{100 * mask.mean():.1f}% of blank sky masked"

    def test_source_free_image_returns_all_false(self):
        """detect_sources returns None when nothing is found; do not crash."""
        mask = get_source_mask(np.zeros((50, 50)))
        assert mask.shape == (50, 50)
        assert mask.dtype == bool
        assert not mask.any()

    def test_real_source_is_detected_and_dilated(self):
        rng = np.random.default_rng(1)
        img = rng.standard_normal((100, 100))
        yy, xx = np.mgrid[:100, :100]
        img += 50.0 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 3.0 ** 2))
        mask = get_source_mask(img)
        assert mask[50, 50]
        assert 0.0 < mask.mean() < 0.2      # the source, not the whole frame


class TestEstimateCovKernelPatchSize:
    """A patch smaller than `size` used to return a silently truncated kernel.

    acf[cy - size//2 : cy + size//2 + 1] goes negative at the start, which
    NumPy reads as an offset from the end: a 12x12 patch with size=15 returned
    a 1x1 delta kernel. That produces a flat power spectrum, silently disabling
    the covariance correction — measured on the bundled detection example, the
    reported flux uncertainty drops to 0.72x the correct value, i.e. the ~30%
    underestimate the package exists to remove.
    """

    @staticmethod
    def _patch(n, seed=0):
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(seed)
        return gaussian_filter(rng.standard_normal((n, n)), 0.8)

    @pytest.mark.parametrize("n", [4, 8, 12, 13, 14])
    def test_patch_smaller_than_size_raises(self, n):
        with pytest.raises(ValueError, match="smaller than size"):
            estimate_cov_kernel(self._patch(n), size=15)

    @pytest.mark.parametrize("shape", [(60, 10), (10, 60)])
    def test_both_axes_are_checked(self, shape):
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(0)
        patch = gaussian_filter(rng.standard_normal(shape), 0.8)
        with pytest.raises(ValueError, match="smaller than size"):
            estimate_cov_kernel(patch, size=15)

    @pytest.mark.parametrize("n", [15, 16, 24, 40, 100])
    def test_large_enough_patch_gives_the_requested_shape(self, n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert estimate_cov_kernel(self._patch(n), size=15).shape == (15, 15)

    @pytest.mark.parametrize("n", [15, 20])
    def test_small_but_legal_patch_warns(self, n):
        """Legal, but the kernel is biased low and noisy."""
        with pytest.warns(RuntimeWarning, match="independent lags"):
            estimate_cov_kernel(self._patch(n), size=15)

    @pytest.mark.parametrize("n", [24, 60])
    def test_comfortable_patch_does_not_warn(self, n):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            estimate_cov_kernel(self._patch(n), size=15)

    def test_kernel_spanning_the_whole_patch_does_not_warn(self):
        """The demo notebook calls estimate_cov_kernel(patch_125, size=125).

        The window tapers to zero at r_out=6 px, far inside the patch, so the
        estimate is well sampled and must not trigger the warning even though
        `size` equals the patch side.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            k = estimate_cov_kernel(self._patch(125), size=125)
        assert k.shape == (125, 125)

    def test_real_data_path_is_unaffected(self):
        """The bundled sky patch must still produce a full-size kernel."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            k = estimate_cov_kernel(_real_sky_patch(), size=15)
        assert k.shape == (15, 15)
        assert k[7, 7] == pytest.approx(1.0, rel=1e-10)
