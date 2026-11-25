"""Prepared fixed-covariance Gaussian likelihoods on image stamps."""

from dataclasses import dataclass, field
import math
from numbers import Integral

import numpy as np
from scipy.linalg import cholesky, solve_triangular


_LOG_2PI = math.log(2 * math.pi)
_DEFAULT_DENSE_BYTES = 128 * 1024**2
_KERNEL_SYMMETRY_ATOL = 1e-7


def _image(array, name, shape=None):
    array = np.asarray(array)
    if array.ndim != 2 or array.size == 0 or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} must have shape {shape or '(ny, nx) with positive dimensions'}")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    return array


def _fft_quadratic(standardized, power):
    coefficients = np.fft.fft2(standardized)
    return np.sum(np.abs(coefficients) ** 2 / power) / standardized.size


def _legacy_log_likelihood_from_residual(residual, prepared):
    """Evaluate the original weighted, relative objective on its prepared grid."""
    standardized = (residual / prepared["err"]) * np.sqrt(prepared["fit_weight"])
    return -0.5 * _fft_quadratic(standardized, prepared["power_spectrum"])


def _correlation_kernel(kernel):
    kernel = _image(kernel, "kernel").astype(float)
    if any(size % 2 == 0 for size in kernel.shape):
        raise ValueError("kernel must have odd dimensions with zero lag at its center")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel must be finite")
    center = kernel[tuple(size // 2 for size in kernel.shape)]
    if not np.isclose(center, 1.0, rtol=0, atol=1e-12):
        raise ValueError("correlation kernel must have unit zero lag")
    if np.any(np.abs(kernel) > 1 + 1e-12):
        raise ValueError("correlation kernel coefficients must have magnitude at most one")
    mirrored = kernel[::-1, ::-1]
    if np.max(np.abs(kernel - mirrored)) > _KERNEL_SYMMETRY_ATOL:
        raise ValueError("correlation kernel must be centrosymmetric")
    corrected = (kernel + mirrored) / (2 * center)
    return corrected, float(np.max(np.abs(corrected - kernel)))


def _periodic_grid(kernel, shape):
    ky, kx = kernel.shape
    ny, nx = shape
    if ky > ny or kx > nx:
        raise ValueError("periodic kernel cannot exceed the image grid; specify a smaller kernel")
    lag_y = (np.arange(ky) - ky // 2) % ny
    lag_x = (np.arange(kx) - kx // 2) % nx
    grid = np.zeros(shape)
    grid[np.ix_(lag_y, lag_x)] = kernel
    return grid


def _dense_kernel_covariance(correlation, shape, indices, sigma, boundary):
    y, x = np.unravel_index(indices, shape)
    ny, nx = shape
    cy, cx = (size // 2 for size in correlation.shape)
    matrix = np.zeros((indices.size, indices.size), order="F")
    for j, (yj, xj) in enumerate(zip(y, x)):
        if boundary == "periodic":
            coeff = correlation[(y - yj) % ny, (x - xj) % nx]
            matrix[:, j] = (coeff * sigma) * sigma[j]
        else:
            dy, dx = y - yj + cy, x - xj + cx
            inside = (dy >= 0) & (dy < correlation.shape[0]) & (dx >= 0) & (dx < correlation.shape[1])
            coeff = correlation[dy[inside], dx[inside]]
            matrix[inside, j] = (coeff * sigma[inside]) * sigma[j]
    return matrix


@dataclass(frozen=True, eq=False)
class PreparedNoise:
    """Fixed Gaussian noise operator returned by :func:`prepare_noise`.

    Attributes
    ----------
    valid : ndarray of bool
        Read-only image mask with True for observed pixels.
    logdet : float
        Log determinant of the observed covariance, including the error map.
    n_valid : int
        Number of observed pixels.
    metadata : dict
        JSON-compatible description of the noise, boundary and mask.

    Notes
    -----
    Residuals passed to this object are in data units. Error scaling is done
    here. Covariance is fixed and inference is conditional on the input
    uncertainties and any estimated correlation kernel.
    """

    valid: np.ndarray
    logdet: float
    _indices: np.ndarray = field(repr=False)
    _sigma: np.ndarray | None = field(repr=False)
    _factor: np.ndarray | None = field(repr=False)
    _power: np.ndarray | None = field(repr=False)
    _sqrt_power: np.ndarray | None = field(repr=False)
    _spectrum_range: tuple[float, float] | None
    _kernel_adjustment: float | None
    _backend: str
    _boundary: str
    _error_kind: str
    _normalization: str

    @property
    def n_valid(self):
        """Number of observed pixels."""
        return self._indices.size

    @property
    def metadata(self):
        """Return independent Gaussian likelihood metadata and mask positions."""
        metadata = {
            "backend": self._backend,
            "boundary": self._boundary,
            "error_kind": self._error_kind,
            "normalization": self._normalization,
            "mask_mode": "exact_marginal" if self.n_valid < self.valid.size else "full_grid",
            "weight_mode": "none",
            "approximate": False,
            "shape": list(self.valid.shape),
            "n_valid": self.n_valid,
            "n_invalid": self.valid.size - self.n_valid,
            "invalid_pixels": np.argwhere(~self.valid).tolist(),
        }
        if self._spectrum_range is not None:
            metadata["spectrum"] = {
                "raw_min": self._spectrum_range[0],
                "raw_max": self._spectrum_range[1],
                "floor": None,
                "clipped_fraction": 0.0,
            }
        if self._kernel_adjustment is not None:
            metadata["kernel_max_abs_adjustment"] = self._kernel_adjustment
        return metadata

    def _observed(self, residual):
        residual = _image(residual, "residual", self.valid.shape)
        selected = residual.ravel()[self._indices].astype(float, copy=False)
        if not np.all(np.isfinite(selected)):
            raise ValueError("residual must be finite on observed pixels")
        return selected

    def _whiten_observed(self, observed):
        if self._factor is not None:
            return solve_triangular(self._factor, observed, lower=True, check_finite=False)
        standardized = observed / self._sigma
        if self._sqrt_power is None:
            return standardized
        coefficients = np.fft.fft2(standardized.reshape(self.valid.shape))
        return np.fft.ifft2(coefficients / self._sqrt_power).real.ravel()

    def whiten(self, residual):
        """Apply a whitening operator in observed C-order pixel order.

        Parameters
        ----------
        residual : array-like
            Residual image in data units, with the preparation shape.

        Returns
        -------
        ndarray
            One value per observed pixel. Dense uses a lower triangular
            solve. FFT first divides by err and then whitens periodically.

        Raises
        ------
        ValueError
            If the shape is wrong or observed residuals are nonfinite.
        """
        return self._whiten_observed(self._observed(residual))

    def standardize(self, residual):
        """Return residual / marginal sigma in observed C-order pixel order."""
        return self._observed(residual) / self._sigma

    def quadratic(self, residual):
        """Evaluate the covariance-weighted squared residual.

        Parameters
        ----------
        residual : array-like
            Residual image in data units, with the preparation shape.

        Returns
        -------
        float
            The observed quadratic form.

        Raises
        ------
        ValueError
            If the shape is wrong or observed residuals are nonfinite.
        """
        observed = self._observed(residual)
        if self._power is not None:
            standardized = (observed / self._sigma).reshape(self.valid.shape)
            return float(_fft_quadratic(standardized, self._power))
        whitened = self._whiten_observed(observed)
        return float(whitened @ whitened)

    def log_likelihood(self, residual):
        """Evaluate the prepared Gaussian log likelihood.

        Parameters
        ----------
        residual : array-like
            Residual image in data units, with the preparation shape.

        Returns
        -------
        float
            Absolute Gaussian log likelihood by default. Relative
            normalization retains only the quadratic term.

        Raises
        ------
        ValueError
            If the shape is wrong or observed residuals are nonfinite.
        """
        chi2 = self.quadratic(residual)
        if self._normalization == "absolute":
            chi2 += self.logdet + self.n_valid * _LOG_2PI
        return -0.5 * chi2


def _valid_pixels(data, mask):
    valid = np.isfinite(data)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype.kind != "b" or mask.shape != data.shape:
            raise ValueError("mask must be a boolean array with the data shape")
        valid &= ~mask
    return valid


def _check_dense_budget(n_pixels, budget, *, extra_bytes=0):
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Integral) or budget <= 0:
        raise ValueError("max_dense_bytes must be a positive integer")
    estimated_bytes = 6 * np.dtype(float).itemsize * n_pixels**2 + extra_bytes
    if estimated_bytes > budget:
        raise ValueError(
            f"dense covariance for {n_pixels} pixels exceeds the dense workspace "
            f"budget ({estimated_bytes} estimated bytes, {budget} allowed)"
        )


def prepare_noise(
    data, err=None, *, backend="diagonal", mask=None, covariance=None,
    kernel=None, boundary=None, error_kind="unspecified", normalization="absolute",
    max_dense_bytes=_DEFAULT_DENSE_BYTES,
):
    """Validate inputs and cache a fixed Gaussian covariance operator.

    Parameters
    ----------
    data : array-like, shape (ny, nx)
        Image in flux per pixel. Nonfinite data are treated as unobserved.
    err : array-like, shape (ny, nx), optional
        Standard deviations in image units. Must be positive and finite on
        observed pixels. Omit when supplying a full covariance.
    backend : {'diagonal', 'dense', 'fft_periodic'}, optional
        Diagonal independent errors, dense Cholesky, or a full-grid FFT.
    mask : array-like of bool, optional
        True excludes a pixel. The FFT backend requires every pixel.
    covariance : array-like, shape (ny*nx, ny*nx), optional
        Full covariance in squared image units, with C-order pixel indices.
        Dense only. Only the observed principal submatrix is used.
    kernel : array-like, optional
        Odd-shaped correlation kernel, with unit zero lag at its center.
        Dense or FFT only. Periodic kernels must fit on the image grid.
        Opposite lags may differ by at most 1e-7 on the unit-normalized
        scale and are averaged. The largest adjustment is recorded.
    boundary : {'finite', 'periodic'} or None, optional
        Required for a dense kernel. Finite uses zero correlation outside
        the kernel support. FFT uses periodic boundaries by construction.
    error_kind : str, optional
        Uncertainty provenance label, e.g. 'background_only' or
        'pipeline_total'. This does not change the covariance.
    normalization : {'absolute', 'relative'}, optional
        Include or omit the fixed Gaussian normalization, respectively.
    max_dense_bytes : int, optional
        Approximate workspace budget for dense preparation. The estimate
        includes multiple float64 observed covariance-sized arrays and
        excludes the caller's input arrays.

    Returns
    -------
    PreparedNoise
        Fixed covariance operator. Nonpositive periodic spectra are rejected
        before fitting, without flooring or clipping.

    Raises
    ------
    ValueError
        If inputs are incompatible, invalid, have no observed pixels, exceed
        the dense budget, or specify an invalid covariance.
    """
    data = _image(data, "data")
    shape = data.shape
    if backend not in ("diagonal", "dense", "fft_periodic"):
        raise ValueError(f"unknown noise backend {backend!r}")
    if normalization not in ("absolute", "relative"):
        raise ValueError("normalization must be 'absolute' or 'relative'")
    if not isinstance(error_kind, str) or not error_kind.strip():
        raise ValueError("error_kind must be a nonempty provenance label")
    valid = _valid_pixels(data, mask)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        raise ValueError("no valid pixels remain")
    if backend == "fft_periodic" and indices.size != data.size:
        raise ValueError("fft_periodic requires a fully observed grid; use dense for a mask")

    if covariance is not None:
        if backend != "dense" or err is not None or kernel is not None or boundary is not None:
            raise ValueError("explicit covariance requires dense without err, kernel, or boundary")
        sigma = None
        boundary = "explicit"
    else:
        sigma = _image(err, "err", shape).ravel()[indices].astype(float, copy=False)
        if not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("err must be positive and finite on observed pixels")
        if backend == "diagonal":
            if kernel is not None or boundary is not None:
                raise ValueError("diagonal noise does not use kernel or boundary")
            boundary = "independent"
        elif kernel is None:
            raise ValueError(f"{backend} requires a correlation kernel or an explicit dense covariance")
        elif backend == "fft_periodic":
            if boundary not in (None, "periodic"):
                raise ValueError("fft_periodic requires a periodic boundary")
            boundary = "periodic"
        elif boundary not in ("finite", "periodic"):
            raise ValueError("dense kernel requires an explicit 'finite' or 'periodic' boundary")

    if backend == "dense":
        _check_dense_budget(indices.size, max_dense_bytes)

    power = sqrt_power = factor = spectrum_range = kernel_adjustment = None
    if kernel is not None:
        correlation, kernel_adjustment = _correlation_kernel(kernel)
        if boundary == "periodic":
            correlation = _periodic_grid(correlation, shape)
            power = np.fft.fft2(correlation).real.copy()
            spectrum_range = float(power.min()), float(power.max())
            if not np.all(np.isfinite(power) & (power > 0)):
                raise ValueError(
                    f"periodic covariance must be positive definite (raw power min={spectrum_range[0]:.6g})"
                )

    if backend == "dense":
        if covariance is None:
            matrix = _dense_kernel_covariance(correlation, shape, indices, sigma, boundary)
        else:
            covariance = _image(covariance, "covariance", (data.size, data.size))
            matrix = np.array(covariance[np.ix_(indices, indices)], dtype=float, order="F")
        if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0):
            raise ValueError("observed covariance must be finite and symmetric")
        variance = matrix.diagonal().copy() if sigma is None else None
        try:
            factor = cholesky(matrix, lower=True, overwrite_a=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError("observed covariance must be positive definite") from exc
        logdet = 2 * np.log(factor.diagonal()).sum()
        power = None
        if variance is not None:
            sigma = np.sqrt(variance)
    else:
        logdet = 2 * np.log(sigma).sum()
        if power is not None:
            logdet += np.log(power).sum()
            sqrt_power = np.sqrt(power)

    for array in (valid, indices, sigma, factor, power, sqrt_power):
        if array is not None:
            array.setflags(write=False)
    return PreparedNoise(
        valid, float(logdet), indices, sigma, factor, power, sqrt_power,
        spectrum_range, kernel_adjustment, backend, boundary, error_kind, normalization,
    )
