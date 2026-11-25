"""Image geometry shared by model components."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from .sersic import SersicProfile


@dataclass(frozen=True)
class RenderContext:
    """Native output geometry and optional point response.

    Parameters
    ----------
    shape : (int, int)
        Image shape in (ny, nx) order.
    point_response : callable, optional
        Called as `point_response(dx, dy, context)` for a point of unit flux.
        It returns the point response in flux per output pixel. It supplies
        its own PSF response and normalization semantics.

    psf_operator : PSFOperator, optional
        Shared response for point and Sérsic sources. Mutually exclusive
        with the point-only callback.

    Raises
    ------
    ValueError
        If the shape or the response is invalid.
    """

    shape: tuple[int, int]
    point_response: Callable[[float, float, "RenderContext"], np.ndarray] | None = None
    psf_operator: "PSFOperator | None" = None

    def __post_init__(self):
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise ValueError("render shape must be two positive integers") from exc
        if len(shape) != 2 or any(
            not isinstance(size, Integral) or isinstance(size, (bool, np.bool_)) or size <= 0
            for size in shape
        ):
            raise ValueError("render shape must be two positive integers")
        if self.point_response is not None and not callable(self.point_response):
            raise ValueError("point_response must be callable")
        if self.psf_operator is not None and (
            not isinstance(self.psf_operator, PSFOperator) or self.point_response is not None
        ):
            raise ValueError("psf_operator must be a PSFOperator and cannot be combined with point_response")
        object.__setattr__(self, "shape", tuple(map(int, shape)))

    @property
    def center(self) -> tuple[float, float]:
        """The geometric stamp center as (x, y) in native pixel indices."""
        ny, nx = self.shape
        return (nx - 1) / 2, (ny - 1) / 2


@dataclass(frozen=True, eq=False)
class PSFSpec:
    """Calibrated samples of a locally shift-invariant PSF response.

    Parameters
    ----------
    samples : array-like, shape (ny, nx)
        Finite pre-pixel density in inverse square native pixels for unit
        source flux, or effective pixel response per unit source flux.
    response_kind : {'pre_pixel', 'effective'}
        Samples of a continuous density, or samples of an already integrated
        pixel response. Both use bilinear interpolation, truncated to the
        sampled rectangle. Negative samples from empirical models are allowed.
    spacing : float
        Isotropic sample spacing in native pixels, not an oversampling factor.
    origin : (float, float)
        (x, y) sample indices of zero offset, which may be fractional.
    normalization_reference : str
        The independent flux calibration, e.g. an encircled-energy aperture.
        No stamp-dependent normalization is applied.
    source : str, optional
        PSF identifier or description of its provenance.
    scale : float, optional
        Multiplicative calibration applied once to the supplied samples.
    support_fraction : float or None, optional
        Independently known retained fraction of the calibration reference.
        None represents an unknown fraction. It is metadata, not a rescaling.
    native_pixel_scale : float or None, optional
        Native angular pixel scale in arcsec, for provenance.
    """

    samples: np.ndarray = field(repr=False)
    response_kind: str
    spacing: float
    origin: tuple[float, float]
    normalization_reference: str
    source: str = ""
    scale: float = 1.0
    support_fraction: float | None = None
    native_pixel_scale: float | None = None
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        if self.response_kind not in ("pre_pixel", "effective"):
            raise ValueError("response_kind must be 'pre_pixel' or 'effective'")
        if not isinstance(self.normalization_reference, str) or not self.normalization_reference.strip():
            raise ValueError("normalization_reference must describe the calibration")
        if not isinstance(self.source, str):
            raise ValueError("source must be a string")
        values = np.asarray(self.samples)
        if values.ndim != 2 or min(values.shape) < 2 or values.dtype.kind not in "iuf":
            raise ValueError("PSF samples must be a real 2D array with at least two samples per axis")
        if not np.all(np.isfinite(values)):
            raise ValueError("PSF samples must be finite")
        values = np.asarray(values, dtype="<f8")
        spacing = _real_scalar(self.spacing, "spacing")
        scale = _real_scalar(self.scale, "scale")
        if spacing <= 0 or scale <= 0:
            raise ValueError("spacing and calibration scale must be positive")
        try:
            ox, oy = self.origin
        except (TypeError, ValueError) as exc:
            raise ValueError("origin must contain x and y sample indices") from exc
        ox, oy = _real_scalar(ox, "origin x"), _real_scalar(oy, "origin y")
        if not (0 <= ox <= values.shape[1] - 1 and 0 <= oy <= values.shape[0] - 1):
            raise ValueError("origin must lie within the PSF sample grid")
        if self.support_fraction is not None:
            fraction = _real_scalar(self.support_fraction, "support_fraction")
            if not 0 < fraction <= 1:
                raise ValueError("support_fraction must lie in (0, 1]")
            object.__setattr__(self, "support_fraction", fraction)
        if self.native_pixel_scale is not None:
            pixel_scale = _real_scalar(self.native_pixel_scale, "native_pixel_scale")
            if pixel_scale <= 0:
                raise ValueError("native_pixel_scale must be positive")
            object.__setattr__(self, "native_pixel_scale", pixel_scale)
        payload = values.tobytes()
        with np.errstate(over="ignore", invalid="ignore"):
            calibrated = values * scale
        if not np.all(np.isfinite(calibrated)) or not np.any(calibrated > 0):
            raise ValueError("calibrated PSF must be finite with positive response somewhere")
        object.__setattr__(self, "samples", np.frombuffer(payload, dtype="<f8").reshape(values.shape))
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "origin", (ox, oy))
        object.__setattr__(self, "_sha256", hashlib.sha256(payload).hexdigest())

    @property
    def support(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return x and y bounds of the tabulated response in native pixels."""
        ox, oy = self.origin
        ny, nx = self.samples.shape
        return ((-ox * self.spacing, (nx - 1 - ox) * self.spacing),
                (-oy * self.spacing, (ny - 1 - oy) * self.spacing))

    @property
    def metadata(self) -> dict:
        """Return the calibration, sampling and input provenance."""
        return {
            "response_kind": self.response_kind,
            "sample_shape": list(self.samples.shape),
            "spacing_native_pixels": self.spacing,
            "origin_xy": list(self.origin),
            "tabulated_support_xy": [list(axis) for axis in self.support],
            "normalization_reference": self.normalization_reference,
            "calibration_scale": self.scale,
            "psf_support_fraction": self.support_fraction,
            "source": self.source,
            "samples_sha256": self._sha256,
            "native_pixel_scale_arcsec": self.native_pixel_scale,
            "sample_scale_arcsec": None if self.native_pixel_scale is None else self.spacing * self.native_pixel_scale,
        }


def _real_scalar(value, label):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    return value


def _tent_integral(t):
    left = np.clip(t + 1, 0, 1)
    right = np.clip(t, 0, 1)
    return 0.5 * left**2 + right * (1 - 0.5 * right)


def _pixel_weights(coordinates, ni, origin, spacing):
    # Integrate the nodal linear basis over a native pixel, clipped to the
    # tabulated support. Clipping also restricts the two end-node basis tents.
    low = np.clip((coordinates - 0.5) / spacing + origin, 0, ni - 1)
    high = np.clip((coordinates + 0.5) / spacing + origin, 0, ni - 1)
    nodes = np.arange(ni)
    weights = _tent_integral(high[:, None] - nodes) - _tent_integral(low[:, None] - nodes)
    return csr_matrix(weights * spacing)


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)


@lru_cache(maxsize=2)
def _source_masses(profile, half_x, half_y, s):
    # This lattice follows the source center. A fixed 3x3 group receives
    # adaptive cell integrals so a compact cusp can collapse onto the same
    # exact centroid used by the point renderer.
    from .sersic import integrate_sersic_grid

    x = np.arange(-half_x, half_x + 1, dtype=float) / s
    y = np.arange(-half_y, half_y + 1, dtype=float) / s
    masses = np.zeros((y.size, x.size))
    for dy, wy in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
        for dx, wx in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
            masses += (wx * wy / (4 * s**2)) * profile.intensity(
                x[None, :] + dx / (2 * s), y[:, None] + dy / (2 * s)
            )
    edges = (np.arange(4) - 1.5) / s
    masses[half_y - 1:half_y + 2, half_x - 1:half_x + 2] = integrate_sersic_grid(
        profile, edges, edges
    )
    # Frames with the same padded grid and intrinsic profile share this
    # lattice. PSF convolution and pixel phase still vary by frame.
    masses.setflags(write=False)
    return masses


@dataclass(frozen=True, eq=False)
class PSFOperator:
    """Shared continuous pixel response and linear Sérsic convolution.

    Parameters
    ----------
    spec : PSFSpec
        Explicit response samples and flux calibration.
    oversamp : int, optional
        Source cells per native pixel per axis. The source is integrated over
        these cells and its mass is assigned to their centers for convolution.
    max_offset : float, optional
        Maximum absolute supported source offset on either native axis.
    max_source_cells : int, optional
        Guard on the size of the padded source grid, before allocation.

    Notes
    -----
    Pre-pixel density is integrated over the uniform native pixel exactly for
    the bilinear interpolant. Effective samples are evaluated at pixel centers.
    Both paths use the resulting same pixel response for points and galaxies.
    The cell-center approximation for the intrinsic source converges on grid
    refinement. Consult the measured domain and error budget in the M4 report.
    """

    spec: PSFSpec
    oversamp: int = 8
    max_offset: float = 2.0
    max_source_cells: int = 500_000

    def __post_init__(self):
        if not isinstance(self.spec, PSFSpec):
            raise ValueError("spec must be a PSFSpec")
        for name in ("oversamp", "max_source_cells"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        max_offset = _real_scalar(self.max_offset, "max_offset")
        if max_offset < 0:
            raise ValueError("max_offset must be nonnegative")
        object.__setattr__(self, "max_offset", max_offset)

    @property
    def metadata(self) -> dict:
        """Return response calibration and convolution sampling settings."""
        return {
            **self.spec.metadata,
            "source_oversamp": self.oversamp,
            "max_offset_native_pixels": self.max_offset,
            "max_source_cells": self.max_source_cells,
            "interpolation": "bilinear",
            "boundary": "zero_outside_tabulated_support",
        }

    def _offset(self, dx, dy):
        dx, dy = _real_scalar(dx, "dx"), _real_scalar(dy, "dy")
        if max(abs(dx), abs(dy)) > self.max_offset:
            raise ValueError("source offset exceeds PSFOperator.max_offset")
        return dx, dy

    def _response(self, x, y):
        spec = self.spec
        ox, oy = spec.origin
        ny, nx = spec.samples.shape
        if spec.response_kind == "effective":
            iy, ix = np.meshgrid(y / spec.spacing + oy, x / spec.spacing + ox, indexing="ij")
            return spec.scale * map_coordinates(spec.samples, (iy, ix), order=1, mode="constant", cval=0.0)
        wx = _pixel_weights(x, nx, ox, spec.spacing)
        wy = _pixel_weights(y, ny, oy, spec.spacing)
        return spec.scale * (wx @ (wy @ spec.samples).T).T

    def render_point(self, dx: float, dy: float, context: RenderContext) -> np.ndarray:
        """Render a unit-flux point with offsets in native pixels.

        Parameters
        ----------
        dx, dy : float
            Source offset relative to the geometric center.
        context : RenderContext
            Native stamp geometry.

        Returns
        -------
        ndarray
            Calibrated pixel response with the exact stamp shape.

        Raises
        ------
        ValueError
            If an offset is nonfinite or outside the supported range.
        """
        dx, dy = self._offset(dx, dy)
        ny, nx = context.shape
        cx, cy = context.center
        return self._response(np.arange(nx) - cx - dx, np.arange(ny) - cy - dy)

    def render_sersic(self, profile: "SersicProfile", dx: float, dy: float, context: RenderContext) -> np.ndarray:
        """Render an extended source using padded linear convolution.

        Parameters
        ----------
        profile : SersicProfile
            Intrinsic Sérsic profile on the PSF flux calibration scale.
        dx, dy : float
            Source offset relative to the stamp center in native pixels.
        context : RenderContext
            Native stamp geometry.

        Returns
        -------
        ndarray
            Convolved flux per native pixel. Finite support and stamp losses
            are retained.

        Raises
        ------
        ValueError
            If offsets or padded source grid size are unsupported.
        RuntimeError
            If intrinsic quadrature fails to converge.
        """
        dx, dy = self._offset(dx, dy)
        ny, nx = context.shape
        if profile.flux == 0:
            return np.zeros(context.shape)
        cx, cy = context.center
        s = self.oversamp
        (xl, xh), (yl, yh) = self.spec.support
        pad = 0.5 if self.spec.response_kind == "pre_pixel" else 0
        kx = math.ceil((max(abs(xl), abs(xh)) + pad) * s) + 1
        ky = math.ceil((max(abs(yl), abs(yh)) + pad) * s) + 1
        mx = math.ceil((cx + self.max_offset) * s) + kx + 1
        my = math.ceil((cy + self.max_offset) * s) + ky + 1
        n_cells = (2 * my + 1) * (2 * mx + 1)
        if n_cells > self.max_source_cells:
            raise ValueError(f"padded source grid needs {n_cells} cells, above max_source_cells")
        # Express the first output pixel as an integer lattice index and a
        # fractional phase. The kernel carries that phase, and the full
        # convolution is indexed explicitly, including for even stamps.
        first_x, first_y = (-cx - dx) * s, (-cy - dy) * s
        ix, iy = math.floor(first_x), math.floor(first_y)
        lattice_x = (np.arange(-kx, kx + 1) + (first_x - ix)) / s
        lattice_y = (np.arange(-ky, ky + 1) + (first_y - iy)) / s
        kernel = self._response(lattice_x, lattice_y)
        masses = _source_masses(profile, mx, my, s)
        convolved = fftconvolve(masses, kernel, mode="full")
        out_x = ix + mx + kx + s * np.arange(nx)
        out_y = iy + my + ky + s * np.arange(ny)
        return convolved[np.ix_(out_y, out_x)]
