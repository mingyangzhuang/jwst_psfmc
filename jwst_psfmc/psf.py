"""
PSF manipulation utilities for sub-pixel shifting, downsampling, and model evaluation.

All Fourier operations use float64 to avoid sub-pixel interpolation noise.
The oversampled PSF is shifted in Fourier space (exact interpolation), then
block-summed to native resolution. Flux bookkeeping uses an *encircled-energy
fraction* so that flux lost outside the fitting stamp is correctly accounted for.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import fourier_shift
import matplotlib.pyplot as plt


def shift_psf_fourier(psf: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift a 2-D PSF array in Fourier space (exact, band-limited interpolation).

    Parameters
    ----------
    psf : ndarray of shape (ny, nx)
        Input PSF array (typically oversampled). Will be cast to float64.
    dx : float
        Shift along the x-axis (column direction) in pixels of the *input* grid.
        Positive values move the PSF peak to the right.
    dy : float
        Shift along the y-axis (row direction) in pixels of the *input* grid.
        Positive values move the PSF peak upward.

    Returns
    -------
    shifted : ndarray of shape (ny, nx), float64
        The shifted PSF, normalised so its sum equals that of the input.
    """
    psf = np.asarray(psf, dtype=np.float64)
    psf_ft = np.fft.fftn(psf)
    shifted = np.fft.ifftn(fourier_shift(psf_ft, shift=[dy, dx])).real
    return shifted


def downsample_psf(psf_os: np.ndarray, oversamp: int, recenter_peak: bool = False) -> np.ndarray:
    """Downsample an oversampled PSF to native resolution by block-summing.

    The function symmetrically centre-crops the oversampled PSF to the largest
    even multiple of *oversamp* in each axis, then reshapes and sums each
    (oversamp × oversamp) super-pixel. A final crop enforces an odd native size
    so the discrete peak has a unique central pixel.

    Parameters
    ----------
    psf_os : ndarray of shape (ny_os, nx_os)
        Oversampled PSF.
    oversamp : int
        Integer oversampling factor (e.g. 4 for a 4× oversampled PSF).
    recenter_peak : bool, optional
        If *True*, roll the downsampled PSF so its maximum pixel lands exactly
        on the geometric centre. Leave *False* during fitting — re-centering
        would erase any intentional sub-pixel shift.

    Returns
    -------
    psf : ndarray of shape (ny_nat, nx_nat), float64
        Native-resolution PSF with odd-shaped dimensions.

    Raises
    ------
    ValueError
        If *oversamp* is not a positive integer, or if *psf_os* is smaller
        than *oversamp* in either dimension.
    """
    if oversamp <= 0 or int(oversamp) != oversamp:
        raise ValueError("oversamp must be a positive integer")

    oversamp = int(oversamp)
    psf_os = np.asarray(psf_os, dtype=np.float64)
    ny, nx = psf_os.shape

    if ny < oversamp or nx < oversamp:
        raise ValueError("psf_os must be at least as large as oversamp in both dimensions")

    def _center_crop(arr: np.ndarray, out_ny: int, out_nx: int) -> np.ndarray:
        sy = (arr.shape[0] - out_ny) // 2
        sx = (arr.shape[1] - out_nx) // 2
        return arr[sy : sy + out_ny, sx : sx + out_nx]

    crop_ny = (ny // oversamp) * oversamp
    crop_nx = (nx // oversamp) * oversamp
    psf_os = _center_crop(psf_os, crop_ny, crop_nx)

    psf = psf_os.reshape(
        crop_ny // oversamp, oversamp,
        crop_nx // oversamp, oversamp,
    ).sum(axis=(1, 3))

    out_ny, out_nx = psf.shape
    if out_ny % 2 == 0:
        psf = _center_crop(psf, out_ny - 1, out_nx)
    if out_nx % 2 == 0:
        psf = _center_crop(psf, psf.shape[0], psf.shape[1] - 1)

    if recenter_peak:
        cy, cx = np.array(psf.shape) // 2
        py, px = np.unravel_index(np.nanargmax(psf), psf.shape)
        if py != cy or px != cx:
            psf = np.roll(psf, shift=(cy - py, cx - px), axis=(0, 1))

    return psf


def match_shape_center(
    arr: np.ndarray,
    target_shape: tuple[int, int],
    pad_value: float = 0.0,
    return_fraction: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Centre-crop or zero-pad a 2-D array to *target_shape*.

    Parameters
    ----------
    arr : ndarray, 2-D
        Input array.
    target_shape : (ny, nx)
        Desired output shape.
    pad_value : float, optional
        Fill value used when padding. Default ``0.0``.
    return_fraction : bool, optional
        If *True*, also return the fraction of the input flux retained after
        the crop (always 1.0 when padding, ≤ 1.0 when cropping).

    Returns
    -------
    out : ndarray of shape *target_shape*
        Re-shaped array.
    fraction : float
        Returned only when *return_fraction* is *True*. Ratio of
        ``nansum(cropped) / nansum(input)``.

    Raises
    ------
    ValueError
        If *arr* is not 2-D or *target_shape* contains non-positive integers.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")

    target_ny, target_nx = map(int, target_shape)
    if target_ny <= 0 or target_nx <= 0:
        raise ValueError("target_shape must contain positive integers")

    input_sum = float(np.nansum(arr))
    ny, nx = arr.shape

    if ny > target_ny:
        sy = (ny - target_ny) // 2
        arr = arr[sy : sy + target_ny, :]
    if nx > target_nx:
        sx = (nx - target_nx) // 2
        arr = arr[:, sx : sx + target_nx]

    cropped_sum = float(np.nansum(arr))

    if arr.shape != (target_ny, target_nx):
        out = np.full((target_ny, target_nx), pad_value, dtype=arr.dtype)
        sy = (target_ny - arr.shape[0]) // 2
        sx = (target_nx - arr.shape[1]) // 2
        out[sy : sy + arr.shape[0], sx : sx + arr.shape[1]] = arr
        arr = out

    if return_fraction:
        if input_sum > 0 and np.isfinite(input_sum):
            fraction = cropped_sum / input_sum
        else:
            fraction = np.nan
        return arr, fraction

    return arr


def prepare_psf_for_oversamp(
    psf_os: np.ndarray,
    oversamp: int,
    native_shape: tuple[int, int] | None = None,
    return_fraction: bool = False,
) -> tuple:
    """Crop the oversampled PSF to exactly the region needed for fitting.

    Call this *once* before the MCMC loop. The cropped PSF is then passed to
    :func:`psf_model` at every likelihood evaluation, avoiding repeated
    (and potentially inconsistent) cropping.

    Parameters
    ----------
    psf_os : ndarray of shape (ny_os, nx_os)
        Raw oversampled PSF (e.g. from a star-based PSF model FITS file).
    oversamp : int
        Integer oversampling factor.
    native_shape : (ny, nx) or None, optional
        Desired shape at native resolution. If given, the oversampled PSF is
        cropped to ``(ny * oversamp, nx * oversamp)``. If *None*, the
        maximum odd native shape that fits inside *psf_os* is used.
    return_fraction : bool, optional
        If *True*, also return the encircled-energy fraction retained in the
        crop (i.e. ``nansum(cropped) / nansum(psf_os)``).

    Returns
    -------
    psf_crop : ndarray of shape (ny_os_crop, nx_os_crop), float64
        Cropped oversampled PSF, ready to pass to :func:`psf_model`.
    out_native_shape : (ny, nx)
        Corresponding native-resolution shape (always odd in both axes).
    fraction : float
        Encircled-energy fraction. Only returned when *return_fraction* is
        *True*.

    Raises
    ------
    ValueError
        If *oversamp* is not a positive integer, or if *psf_os* is too small
        for the requested *native_shape*.
    """
    if oversamp <= 0 or int(oversamp) != oversamp:
        raise ValueError("oversamp must be a positive integer")

    oversamp = int(oversamp)
    psf_os = np.asarray(psf_os, dtype=np.float64)
    total_sum = float(np.nansum(psf_os))
    ny, nx = psf_os.shape

    if native_shape is not None:
        if len(native_shape) != 2:
            raise ValueError("native_shape must be a 2-element (ny, nx) tuple")

        crop_ny = int(native_shape[0]) * oversamp
        crop_nx = int(native_shape[1]) * oversamp

        if ny < crop_ny or nx < crop_nx:
            raise ValueError(
                f"psf_os shape ({ny}, {nx}) is too small for native_shape "
                f"{native_shape} with oversamp={oversamp}. "
                f"Need at least ({crop_ny}, {crop_nx})."
            )
        out_native_shape = (int(native_shape[0]), int(native_shape[1]))
    else:
        if ny < oversamp or nx < oversamp:
            raise ValueError("psf_os must be at least as large as oversamp in both dimensions")

        native_ny = (ny // oversamp)
        native_nx = (nx // oversamp)
        if native_ny % 2 == 0:
            native_ny -= 1
        if native_nx % 2 == 0:
            native_nx -= 1

        out_native_shape = (native_ny, native_nx)
        crop_ny = native_ny * oversamp
        crop_nx = native_nx * oversamp

    sy = (ny - crop_ny) // 2
    sx = (nx - crop_nx) // 2
    psf_crop = psf_os[sy : sy + crop_ny, sx : sx + crop_nx]

    if total_sum > 0 and np.isfinite(total_sum):
        fraction = float(np.nansum(psf_crop)) / total_sum
    else:
        fraction = np.nan

    if return_fraction:
        return psf_crop, out_native_shape, fraction
    return psf_crop, out_native_shape


def psf_model(
    param: np.ndarray,
    psf_os: np.ndarray,
    oversamp: int = 4,
    output_shape: tuple[int, int] | None = None,
    native_shape: tuple[int, int] | None = None,
    psf_prepare_fraction: float = 1.0,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Generate a PSF model image from a (flux, dx, dy) parameter vector.

    The model is constructed by:

    1. Shifting the pre-cropped oversampled PSF in Fourier space by
       ``(dx * oversamp, dy * oversamp)`` pixels on the oversampled grid.
    2. Block-summing to native resolution.
    3. Matching the result to *native_shape* and/or *output_shape* by
       centre-cropping or zero-padding (flux losses are tracked via
       *psf_prepare_fraction* and the crop fractions).
    4. Normalising the in-stamp PSF shape to unit sum, then scaling by
       ``flux * total_encircled_fraction``.

    Parameters
    ----------
    param : array-like of length 3
        ``[flux, dx_native, dy_native]``.  *flux* is the total source flux
        in the image units. *dx_native* and *dy_native* are sub-pixel shifts
        in native-pixel units relative to the PSF reference centre.
    psf_os : ndarray
        Pre-cropped oversampled PSF, as returned by
        :func:`prepare_psf_for_oversamp`.
    oversamp : int, optional
        Integer oversampling factor. Default ``4``.
    output_shape : (ny, nx) or None, optional
        If given, the native PSF is further matched (crop/pad) to this shape
        before scaling.  Usually set to ``prepared['shape']`` from
        :func:`~jwst_psfmc.covariance.prepare_covariance_terms`.
    native_shape : (ny, nx) or None, optional
        Expected native PSF shape after downsampling. Used to enforce a
        consistent shape even if the downsampled PSF has rounding artefacts.
    psf_prepare_fraction : float, optional
        Encircled-energy fraction from :func:`prepare_psf_for_oversamp`.
        Multiplied with any additional crop fractions to track total flux.
        Default ``1.0``.
    return_meta : bool, optional
        If *True*, also return a metadata dictionary with flux bookkeeping.

    Returns
    -------
    model : ndarray of shape *output_shape* (or *native_shape* if no
        *output_shape*), float64
        The scaled PSF model image (background not included).
    meta : dict
        Returned only when *return_meta* is *True*. Keys:

        * ``'psf_prepare_fraction'`` – fraction from the preparation step.
        * ``'model_match_fraction'`` – fraction from shape-matching inside
          this call.
        * ``'total_encircled_fraction'`` – product of the two above.
        * ``'psf_native_shape'`` – native shape after downsampling.
        * ``'output_shape'`` – shape of the returned model array.
        * ``'model_sum'`` – ``sum(model)``.
        * ``'total_flux_parameter'`` – the input *flux* parameter value.

    Raises
    ------
    ValueError
        If PSF normalisation fails (non-positive or non-finite sum after
        downsampling and shape-matching).
    """
    flux, dx_native, dy_native = float(param[0]), float(param[1]), float(param[2])

    dx = dx_native * oversamp
    dy = dy_native * oversamp

    psf_shifted = shift_psf_fourier(psf_os, dx, dy)
    psf = downsample_psf(psf_shifted, oversamp, recenter_peak=False)

    model_fraction = 1.0
    if native_shape is not None and psf.shape != tuple(native_shape):
        psf, frac = match_shape_center(psf, native_shape, pad_value=0.0, return_fraction=True)
        model_fraction *= frac

    if output_shape is not None:
        psf, frac = match_shape_center(psf, output_shape, pad_value=0.0, return_fraction=True)
        model_fraction *= frac

    psf_sum = float(psf.sum())
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        raise ValueError("PSF normalisation failed: non-positive or non-finite sum")
    psf = psf / psf_sum

    psf_native_shape = psf.shape
    total_encircled_fraction = psf_prepare_fraction * model_fraction
    model = flux * total_encircled_fraction * psf

    if return_meta:
        return model, {
            "psf_prepare_fraction": psf_prepare_fraction,
            "model_match_fraction": model_fraction,
            "total_encircled_fraction": total_encircled_fraction,
            "psf_native_shape": psf_native_shape,
            "output_shape": model.shape,
            "model_sum": float(np.sum(model)),
            "total_flux_parameter": flux,
        }
    return model


def inspect_psf_shift(
    psf_os: np.ndarray,
    oversamp: int = 4,
    delta: float = 0.5,
    output_shape: tuple[int, int] | None = None,
    native_shape: tuple[int, int] | None = None,
    psf_prepare_fraction: float = 1.0,
) -> tuple:
    """Visual sanity-check for the PSF shift sign convention.

    Renders five panels: centred, ±dx, ±dy. The red ``+`` marks the geometric
    centre; the white ``×`` marks the peak pixel.

    Parameters
    ----------
    psf_os : ndarray
        Pre-cropped oversampled PSF.
    oversamp : int, optional
        Integer oversampling factor. Default ``4``.
    delta : float, optional
        Shift magnitude to test in native pixels. Default ``0.5``.
    output_shape : (ny, nx) or None, optional
        Forwarded to :func:`psf_model`.
    native_shape : (ny, nx) or None, optional
        Forwarded to :func:`psf_model`.
    psf_prepare_fraction : float, optional
        Forwarded to :func:`psf_model`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
    """
    configs = [
        ("centre", [1.0, 0.0, 0.0]),
        (f"+dx={delta}", [1.0, delta, 0.0]),
        (f"-dx={delta}", [1.0, -delta, 0.0]),
        (f"+dy={delta}", [1.0, 0.0, delta]),
        (f"-dy={delta}", [1.0, 0.0, -delta]),
    ]

    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 4))
    axes = np.atleast_1d(axes)

    for ax, (title, p) in zip(axes, configs):
        m = psf_model(
            p, psf_os,
            oversamp=oversamp,
            output_shape=output_shape,
            native_shape=native_shape,
            psf_prepare_fraction=psf_prepare_fraction,
        )
        ax.imshow(m, origin="lower", cmap="viridis")
        cy, cx = np.array(m.shape) // 2
        py, px = np.unravel_index(np.nanargmax(m), m.shape)
        ax.plot(cx, cy, marker="+", color="r", ms=12, mew=2)
        ax.plot(px, py, marker="x", color="w", ms=10, mew=2)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    return fig, axes
