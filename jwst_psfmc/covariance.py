"""
Covariance kernel estimation and Fourier-space likelihood pre-computation.

Drizzled JWST images have strongly correlated pixel-to-pixel noise because
the co-addition algorithm spreads photon counts across multiple output pixels.
Treating such pixels as independent in a χ² fit leads to underestimated flux
uncertainties. This module provides tools to:

1. Measure the spatial covariance structure from source-free sky regions.
2. Encode that structure as a power-spectrum filter for an efficient
   Fourier-space likelihood (see :func:`prepare_covariance_terms`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import grey_dilation
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from photutils.utils import circular_footprint
from photutils.segmentation import detect_sources, make_2dgaussian_kernel


# ---------------------------------------------------------------------------
# Window function helpers
# ---------------------------------------------------------------------------

def distance_grid(shape: tuple[int, int]) -> np.ndarray:
    """Return a 2-D array of Euclidean distances from the array centre.

    Parameters
    ----------
    shape : (ny, nx)
        Output array shape. Should have odd elements for a well-defined centre.

    Returns
    -------
    dist : ndarray of shape *shape*, float64
        Each element equals the Euclidean distance from the geometric centre
        ``((ny-1)/2, (nx-1)/2)``.

    Raises
    ------
    ValueError
        If *shape* does not have exactly two elements.
    """
    if len(shape) != 2:
        raise ValueError("shape must have exactly 2 elements")
    y_cen = (shape[0] - 1) / 2.0
    x_cen = (shape[1] - 1) / 2.0
    y_vals, x_vals = np.ogrid[: shape[0], : shape[1]]
    return np.hypot(x_vals - x_cen, y_vals - y_cen)


def SplitCosineBellWindow(
    shape: tuple[int, int],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """Generate a 2-D split-cosine-bell (Tukey) radial window function.

    The window equals 1.0 inside radius ``r_inner = beta * max_r``, rolls off
    with a half-cosine taper over the annulus from ``r_inner`` to
    ``r_outer = r_inner + alpha * max_r``, and equals 0.0 beyond that.

    Parameters
    ----------
    shape : (ny, nx)
        Output array shape.
    alpha : float, optional
        Taper width as a fraction of ``max_r = (min(shape)-1)/2``.
        Must be in [0, 1]. Default ``0.5``.
    beta : float, optional
        Flat-top inner radius as a fraction of ``max_r``.
        Must be in [0, 1]. Default ``0.5``.

    Returns
    -------
    window : ndarray of shape *shape*, float64
        Window values in [0, 1].
    """
    dist = distance_grid(shape)
    max_r = (min(shape) - 1.0) / 2.0
    r_inner = beta * max_r
    taper_width = alpha * max_r
    r_outer = r_inner + taper_width

    if taper_width > 0:
        r = dist - r_inner
        result = 0.5 * (1.0 + np.cos(np.pi * r / taper_width))
    else:
        result = np.ones(shape, dtype=float)

    result[dist < r_inner] = 1.0
    result[dist > r_outer] = 0.0
    return result


# ---------------------------------------------------------------------------
# Source masking
# ---------------------------------------------------------------------------

def get_source_mask(image: np.ndarray, npixels: int = 15) -> np.ndarray:
    """Detect sources and return a boolean mask with dilated footprints.

    Uses ``photutils`` segmentation on a Gaussian-smoothed image to detect
    compact sources, then expands each mask footprint by 2 pixels using a
    circular structuring element.

    Parameters
    ----------
    image : ndarray, 2-D
        Science image (e.g. a difference image or direct image).
    npixels : int, optional
        Minimum number of connected pixels above threshold to classify a region
        as a source. Default ``15``.

    Returns
    -------
    source_mask : ndarray of shape ``image.shape``, bool
        *True* where a source (or its dilated footprint) is present.

    Notes
    -----
    For difference-image photometry, run this function on **both** the
    reference and science images before subtraction to generate masks that
    exclude real astrophysical sources. Masking before subtraction prevents
    real source emission from leaking into the difference image and
    contaminating the covariance kernel estimate.
    """
    image_conv = convolve(image, make_2dgaussian_kernel(3, 5), normalize_kernel=True)
    _, median, std = sigma_clipped_stats(image_conv, sigma=3)
    threshold = median + 3.0 * std
    segm = detect_sources(image, threshold, npixels=npixels)
    mask = segm.data > 0
    footprint = circular_footprint(2)
    source_mask = grey_dilation(mask, footprint=footprint)
    return source_mask.astype(bool)


# ---------------------------------------------------------------------------
# Source-free region finder
# ---------------------------------------------------------------------------

def find_zero_squares(
    arr: np.ndarray,
    a: int,
    all_sizes: bool = True,
    max_nonzero: int = 0,
) -> np.ndarray:
    """Find all axis-aligned squares in a 2-D mask that contain few non-zero pixels.

    Uses a summed-area table (integral image) for O(n²·s) complexity rather
    than O(n²·s³), making it efficient even for large arrays.

    Parameters
    ----------
    arr : array_like, 2-D
        Input binary or integer mask.  Zero = free; non-zero = occupied.
    a : int
        Minimum side length (≥ 1) of the squares to search for.
    all_sizes : bool, optional
        If *True*, enumerate every valid square of side ≥ *a*.
        If *False*, report for each top-left corner only the single *largest*
        valid square (greedy scan from large to small). Default *True*.
    max_nonzero : int, optional
        Maximum number of non-zero (occupied) pixels tolerated inside a
        candidate square. Default ``0`` (strictly source-free).

    Returns
    -------
    squares : ndarray of shape (N, 3), int
        Each row is ``(top, left, size)`` giving the top-row index, left-column
        index (both 0-based), and side length. The square spans rows
        ``[top, top+size-1]`` and columns ``[left, left+size-1]``. Returns a
        shape-(0, 3) array if no square is found.

    Raises
    ------
    ValueError
        If *arr* is not 2-D, *a* < 1, or *max_nonzero* < 0.
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if a <= 0:
        raise ValueError("a must be >= 1")
    if max_nonzero < 0:
        raise ValueError("max_nonzero must be >= 0")

    ny, nx = arr.shape
    max_s = min(ny, nx)
    if a > max_s:
        return np.zeros((0, 3), dtype=int)

    occ = (arr != 0).astype(np.int64)
    S = np.zeros((ny + 1, nx + 1), dtype=np.int64)
    S[1:, 1:] = occ.cumsum(axis=0).cumsum(axis=1)

    squares: list[tuple[int, int, int]] = []

    if all_sizes:
        for s in range(a, max_s + 1):
            totals = S[s:, s:] - S[:-s, s:] - S[s:, :-s] + S[:-s, :-s]
            ys, xs = np.nonzero(totals <= max_nonzero)
            for y, x in zip(ys, xs):
                squares.append((int(y), int(x), int(s)))
    else:
        assigned = np.zeros((ny, nx), dtype=bool)
        size_map = -np.ones((ny, nx), dtype=int)
        for s in range(max_s, a - 1, -1):
            totals = S[s:, s:] - S[:-s, s:] - S[s:, :-s] + S[:-s, :-s]
            ys, xs = np.nonzero(totals <= max_nonzero)
            for y, x in zip(ys, xs):
                if not assigned[y, x]:
                    assigned[y, x] = True
                    size_map[y, x] = int(s)
        ys, xs = np.nonzero(size_map >= a)
        for y, x in zip(ys, xs):
            squares.append((int(y), int(x), int(size_map[y, x])))

    if not squares:
        return np.zeros((0, 3), dtype=int)
    return np.array(squares, dtype=int)


# ---------------------------------------------------------------------------
# Kernel estimation
# ---------------------------------------------------------------------------

def estimate_cov_kernel(
    sky_patch: np.ndarray,
    size: int = 15,
    r_in: float = 2.0,
    r_out: float = 6.0,
) -> np.ndarray:
    """Estimate the pixel-to-pixel covariance kernel from a source-free sky patch.

    The algorithm:

    1. Replaces NaN pixels with Gaussian random noise at the local RMS to
       avoid Fourier ringing artefacts.
    2. Subtracts the sigma-clipped mean.
    3. Computes the 2-D autocorrelation function via ``fftconvolve``.
    4. Extracts a central sub-image of shape ``(size, size)``.
    5. Tapers to zero beyond radius *r_out* using a split-cosine-bell window.

    Parameters
    ----------
    sky_patch : ndarray, 2-D
        Source-free region of the difference (or science) image. Should be
        ≳ 3–5 × *size* in each dimension for a reliable estimate.
    size : int, optional
        Side length of the output kernel in pixels. Must be odd ≥ 3.
        Default ``15``.
    r_in : float, optional
        Inner flat radius (pixels) of the cosine-bell taper. Default ``2.0``.
    r_out : float, optional
        Outer zero radius (pixels) of the taper. Default ``6.0``.

    Returns
    -------
    cov_kernel : ndarray of shape (size, size), float64
        Normalised, windowed covariance kernel. Peak value is 1.0 before
        windowing; windowing tapers it smoothly to 0.0 at the edges.

    Raises
    ------
    ValueError
        If *size* is even or less than 3, or if *sky_patch* is not 2-D.
    """
    sky_patch = np.array(sky_patch, dtype=np.float64)
    if sky_patch.ndim != 2:
        raise ValueError("sky_patch must be a 2-D array")
    if size < 3 or size % 2 == 0:
        raise ValueError("size must be an odd integer >= 3")
    if r_out <= r_in:
        raise ValueError("r_out must be greater than r_in")

    mean, median, std = sigma_clipped_stats(sky_patch, sigma=3, maxiters=10)
    nan_mask = np.isnan(sky_patch)
    if nan_mask.any():
        rng = np.random.default_rng()
        sky_patch[nan_mask] = rng.standard_normal(nan_mask.sum()) * std + mean

    img = sky_patch - mean

    acf = fftconvolve(img, img[::-1, ::-1], mode="same")
    if acf.max() == 0:
        raise ValueError("Autocorrelation is all-zero; sky_patch may be constant")
    acf /= acf.max()

    cy, cx = np.array(acf.shape) // 2
    half = size // 2
    kernel = acf[cy - half : cy + half + 1, cx - half : cx + half + 1]

    max_r = (size - 1) / 2.0
    alpha = (r_out - r_in) / max_r
    beta = r_in / max_r
    alpha = float(np.clip(alpha, 0.0, 1.0))
    beta = float(np.clip(beta, 0.0, 1.0))

    window = SplitCosineBellWindow(kernel.shape, alpha=alpha, beta=beta)
    return kernel * window


# ---------------------------------------------------------------------------
# Covariance-aware likelihood pre-computation
# ---------------------------------------------------------------------------

def prepare_covariance_terms(
    data: np.ndarray,
    err: np.ndarray,
    kernel: np.ndarray,
    fit_weight: np.ndarray | None = None,
) -> dict:
    """Pre-compute static terms for the Fourier-space covariance likelihood.

    This function should be called *once* before the MCMC loop. It encodes the
    covariance structure as the FFT power spectrum of the kernel, allowing the
    likelihood to be evaluated in O(N log N) at every MCMC step.

    The Fourier-space log-likelihood is:

    .. math::

        \\ln \\mathcal{L} = -\\frac{1}{2} \\sum_{\\mathbf{k}}
            \\frac{|\\hat{r}(\\mathbf{k})|^2}{P(\\mathbf{k}) \\cdot N}

    where :math:`\\hat{r}` is the FFT of the weighted, normalised residual
    and :math:`P(\\mathbf{k})` is the power spectrum of the covariance kernel.

    Parameters
    ----------
    data : ndarray, 2-D
        Difference image stamp (science flux units).
    err : ndarray, 2-D, same shape as *data*
        Per-pixel 1-σ uncertainty (standard deviation, same units as *data*).
    kernel : ndarray, 2-D
        Covariance kernel from :func:`estimate_cov_kernel`. May be smaller
        than *data*; it will be centre-padded to match.
    fit_weight : ndarray, 2-D or None, optional
        Optional pixel weight map (e.g. a soft central aperture). Pixels with
        weight ≤ 0 or non-finite are excluded from the fit. If *None*, all
        valid pixels are weighted equally.

    Returns
    -------
    prepared : dict
        Keys:

        * ``'data'`` – masked data array (invalid pixels zeroed).
        * ``'err'`` – masked error array.
        * ``'valid'`` – boolean mask of pixels used in the fit.
        * ``'fit_weight'`` – weight map (ones if not supplied).
        * ``'power_spectrum'`` – FFT power spectrum of the kernel,
          clipped to ≥ 1 × 10⁻⁸.
        * ``'shape'`` – ``(ny, nx)`` of the data stamp.

    Raises
    ------
    ValueError
        If array shapes are incompatible, the kernel contains non-finite values,
        or no valid pixels remain after masking.
    """
    data = np.asarray(data, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if data.ndim != 2 or err.ndim != 2 or kernel.ndim != 2:
        raise ValueError("data, err, and kernel must all be 2-D arrays")
    if data.shape != err.shape:
        raise ValueError("data and err must have the same shape")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel must not contain NaN or Inf")

    valid = np.isfinite(data) & np.isfinite(err) & (err > 0)

    if fit_weight is None:
        fit_weight = np.ones_like(data, dtype=np.float64)
    else:
        fit_weight = np.asarray(fit_weight, dtype=np.float64)
        if fit_weight.shape != data.shape:
            raise ValueError("fit_weight must have the same shape as data")
        fit_weight = np.where(np.isfinite(fit_weight) & (fit_weight > 0), fit_weight, 0.0)

    effective_valid = valid & (fit_weight > 0)
    if not np.any(effective_valid):
        raise ValueError("No valid pixels remain after applying masks and fit_weight")

    data_use = np.where(effective_valid, data, 0.0)
    err_use = np.where(effective_valid, err, np.nanmedian(err[valid]))

    ny, nx = data.shape
    ky, kx = kernel.shape

    if ky > ny:
        sy = (ky - ny) // 2
        kernel = kernel[sy : sy + ny, :]
        ky = kernel.shape[0]
    if kx > nx:
        sx = (kx - nx) // 2
        kernel = kernel[:, sx : sx + nx]
        kx = kernel.shape[1]

    kernel_padded = np.zeros((ny, nx), dtype=np.float64)
    y0 = (ny - ky) // 2
    x0 = (nx - kx) // 2
    kernel_padded[y0 : y0 + ky, x0 : x0 + kx] = kernel

    kernel_shifted = np.fft.ifftshift(kernel_padded)
    power_spectrum = np.real(np.fft.fft2(kernel_shifted))
    power_spectrum = np.clip(power_spectrum, 1e-8, None)

    return {
        "data": data_use,
        "err": err_use,
        "valid": effective_valid,
        "fit_weight": np.where(effective_valid, fit_weight, 0.0),
        "power_spectrum": power_spectrum,
        "shape": data.shape,
    }
