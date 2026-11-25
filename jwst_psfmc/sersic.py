"""Intrinsic Sérsic profiles and reference integration on rectangular cells."""

from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc, gammaincc, gammaincinv, gammaln

from .rendering import RenderContext


_LOG_FLOAT_MAX = math.log(np.finfo(float).max)
_RADIAL_QUANTILES = (1e-8, 1e-4, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-4, 1 - 1e-8)


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _real_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    return array.astype(float, copy=False)


@dataclass(frozen=True)
class SersicProfile:
    """Intrinsic elliptical Sérsic profile centered at (0, 0).

    Parameters
    ----------
    flux : float
        Total flux integrated over the infinite intrinsic profile, nonnegative.
    re : float
        Semimajor-axis half-light radius in native pixels.
    n : float
        Positive Sérsic index.
    q : float, optional
        Minor to major axis ratio in (0, 1].
    pa : float, optional
        Position angle in degrees from +x toward +y in array coordinates.
        The angle is represented modulo 180 degrees.

    Raises
    ------
    ValueError
        If a parameter is invalid or the float64 normalization is nonfinite.
    """

    flux: float
    re: float
    n: float
    q: float = 1.0
    pa: float = 0.0
    _b_n: float = field(init=False, repr=False)
    _log_b_n: float = field(init=False, repr=False)
    _log_re: float = field(init=False, repr=False)
    _log_i0: float = field(init=False, repr=False)
    _cos_pa: float = field(init=False, repr=False)
    _sin_pa: float = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("flux", "re", "n", "q", "pa"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.flux < 0:
            raise ValueError("flux must be nonnegative")
        if self.re <= 0 or self.n <= 0 or not 0 < self.q <= 1:
            raise ValueError("re and n must be positive, and q must lie in (0, 1]")
        b_n = float(gammaincinv(2 * self.n, 0.5))
        if not math.isfinite(b_n) or b_n <= 0:
            raise ValueError("Sérsic index is beyond float64 normalization")
        log_b_n, log_re = math.log(b_n), math.log(self.re)
        log_i0 = -math.inf
        if self.flux != 0:
            log_i0 = (
                math.log(self.flux) + 2 * self.n * log_b_n - math.log(math.tau)
                - math.log(self.q) - math.log(self.n) - 2 * log_re - gammaln(2 * self.n)
            )
            if not math.isfinite(log_i0):
                raise ValueError("Sérsic normalization is nonfinite")
        pa = self.pa % 180.0
        if self.q == 1 or pa == 0:
            cos_pa, sin_pa = 1.0, 0.0
        elif pa == 90:
            cos_pa, sin_pa = 0.0, 1.0
        else:
            radians = math.radians(pa)
            cos_pa, sin_pa = math.cos(radians), math.sin(radians)
        object.__setattr__(self, "pa", pa)
        object.__setattr__(self, "_b_n", b_n)
        object.__setattr__(self, "_log_b_n", log_b_n)
        object.__setattr__(self, "_log_re", log_re)
        object.__setattr__(self, "_log_i0", log_i0)
        object.__setattr__(self, "_cos_pa", cos_pa)
        object.__setattr__(self, "_sin_pa", sin_pa)

    @property
    def b_n(self) -> float:
        """Dimensionless constant that places half the light within `re`."""
        return self._b_n

    @property
    def re_circ(self) -> float:
        """Circularized half-light radius in native pixels."""
        return self.re * math.sqrt(self.q)

    def _radial_argument(self, radius):
        with np.errstate(divide="ignore", over="ignore", under="ignore"):
            return np.exp(self._log_b_n + (np.log(radius) - self._log_re) / self.n)

    def intensity(self, x, y) -> np.ndarray:
        """Evaluate the intrinsic surface brightness.

        Parameters
        ----------
        x, y : array-like
            Broadcast-compatible native pixel coordinates relative to the
            source center. +x increases column index and +y increases row index.

        Returns
        -------
        ndarray
            Brightness in flux per square native pixel, as float64. Scalar
            coordinates produce a zero-dimensional array.

        Raises
        ------
        ValueError
            If the coordinates are invalid or the evaluated intensity
            exceeds float64.
        """
        x, y = np.broadcast_arrays(_real_array(x, "x"), _real_array(y, "y"))
        if not np.all(np.isfinite(x) & np.isfinite(y)):
            raise ValueError("x and y must be finite")
        if self.flux == 0:
            return np.zeros(x.shape, dtype=float)
        with np.errstate(over="ignore", under="ignore"):
            major = x * self._cos_pa + y * self._sin_pa
            minor = -x * self._sin_pa + y * self._cos_pa
            radius = np.hypot(major, minor / self.q)
            intensity = np.exp(self._log_i0 - self._radial_argument(radius))
        if not np.all(np.isfinite(intensity)):
            raise ValueError("intrinsic intensity is outside the float64 range")
        return np.asarray(intensity)

    def enclosed_fraction(self, radius) -> np.ndarray:
        """Return the normalized curve of growth in an aligned ellipse.

        Parameters
        ----------
        radius : array-like
            Nonnegative semimajor-axis radius in native pixels. Infinity
            returns one.

        Returns
        -------
        ndarray
            Dimensionless fraction of the unit-flux profile inside the
            ellipse, including for a profile with flux zero.

        Raises
        ------
        ValueError
            If a radius is negative, NaN, or nonreal.
        """
        radius = _real_array(radius, "radius")
        if np.any(np.isnan(radius) | (radius < 0)):
            raise ValueError("radius must be nonnegative and must not be NaN")
        return np.asarray(gammainc(2 * self.n, self._radial_argument(radius)))


def _ray_interval(vx, vy, x0, x1, y0, y1):
    """Intersect a ray with the cell and return the two nonnegative radii."""
    near, far = 0.0, math.inf
    for direction, low, high in ((vx, x0, x1), (vy, y0, y1)):
        if direction == 0:
            if low > 0 or high < 0:
                return 0.0, 0.0
        else:
            first, last = low / direction, high / direction
            near = max(near, min(first, last))
            far = min(far, max(first, last))
    return near, far


def _scalar_radial_argument(profile, radius):
    if radius == 0:
        return 0.0
    log_arg = profile._log_b_n + (math.log(radius) - profile._log_re) / profile.n
    return math.exp(log_arg) if log_arg < _LOG_FLOAT_MAX else math.inf


def _cell_fraction(profile, radii, x0, x1, y0, y1, atol, rtol):
    """Integrate analytic radial mass over angle in elliptical coordinates."""
    cos_pa, sin_pa, q = profile._cos_pa, profile._sin_pa, profile.q

    def angle(x, y):
        major = x * cos_pa + y * sin_pa
        minor = -x * sin_pa + y * cos_pa
        return math.atan2(minor / q, major) % math.tau

    corners = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    axes = ((1, 0), (0, 1), (-1, 0), (0, -1))
    # The active rectangle edge changes only at a corner ray. The axis rays
    # also split off directions where the Cartesian ray denominators vanish.
    cuts = {0.0, math.tau} | {angle(x, y) for x, y in corners + axes if x or y}
    ax, bx, ay, by = cos_pa, -q * sin_pa, sin_pa, q * cos_pa

    # Close to an edge through the center, most of the radial mass can turn
    # on across a tiny angle. Bracket that transition with rays to where
    # selected enclosed-flux ellipses cross each cell edge.
    for edge_pair, along, across, other_edges in (
        ((x0, x1), (ax, bx), (ay, by), (y0, y1)),
        ((y0, y1), (ay, by), (ax, bx), (x0, x1)),
    ):
        projection = math.hypot(*along)
        offset = math.atan2(along[1], along[0])
        for edge in edge_pair:
            if edge == 0:
                continue
            for radius in radii:
                extent = radius * projection
                if abs(edge) >= extent:
                    continue
                angle_span = math.acos(edge / extent)
                for theta in (offset - angle_span, offset + angle_span):
                    other = radius * (across[0] * math.cos(theta) + across[1] * math.sin(theta))
                    if other_edges[0] <= other <= other_edges[1]:
                        cuts.add(theta % math.tau)
    cuts = sorted(cuts)

    def ray(theta):
        c, s = math.cos(theta), math.sin(theta)
        return _ray_interval(ax * c + bx * s, ay * c + by * s, x0, x1, y0, y1)

    shape = 2 * profile.n

    def mass_per_angle(theta):
        near, far = ray(theta)
        if near >= far:
            return 0.0
        a = _scalar_radial_argument(profile, near)
        b = _scalar_radial_argument(profile, far)
        if a >= profile.b_n:
            # The survival function keeps the difference accurate in the wings.
            return max(0.0, float(gammaincc(shape, a) - gammaincc(shape, b)))
        return max(0.0, float(gammainc(shape, b) - gammainc(shape, a)))

    fraction = error = 0.0
    epsabs = 0.5 * math.tau * atol / (len(cuts) - 1)
    for left, right in zip(cuts, cuts[1:]):
        if right - left <= epsabs:  # the normalized radial mass is at most one
            error += (right - left) / math.tau
            continue
        near, far = ray((left + right) / 2)
        if near >= far:
            continue
        value, uncertainty, _, *status = quad(
            mass_per_angle, left, right, epsabs=epsabs, epsrel=0.5 * rtol,
            limit=100, full_output=True,
        )
        if status:
            raise RuntimeError(status[0])
        fraction += value / math.tau
        error += uncertainty / math.tau
    if not math.isfinite(fraction) or error > max(atol, rtol * fraction):
        raise RuntimeError("intrinsic cell quadrature did not converge")
    return min(fraction, 1.0)


def integrate_sersic_grid(
    profile: SersicProfile, x_edges, y_edges, *, atol: float = 1e-10, rtol: float = 1e-9,
) -> np.ndarray:
    """Integrate the intrinsic flux in each rectangular cell.

    Parameters
    ----------
    profile : SersicProfile
        Intrinsic profile. Cell edges are relative to its center.
    x_edges, y_edges : array-like
        One-dimensional, finite, strictly increasing edges in native
        pixels. Edge spacing may be nonuniform.
    atol : float, optional
        Absolute integration tolerance per cell, as a fraction of total flux.
    rtol : float, optional
        Relative integration tolerance per cell.

    Returns
    -------
    ndarray
        Float64 flux per cell in (y, x) order. The values retain the flux
        loss outside the grid.

    Raises
    ------
    ValueError
        If edges or tolerances are invalid.
    RuntimeError
        If quadrature reports a convergence failure.

    Notes
    -----
    This is a deterministic reference integrator. It analytically integrates
    radial mass on each ray, then adaptively integrates over angle using the
    cell geometry. Each tolerance applies to an individual cell.
    """
    x_edges, y_edges = _real_array(x_edges, "x_edges"), _real_array(y_edges, "y_edges")
    for name, edges in (("x_edges", x_edges), ("y_edges", y_edges)):
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
            raise ValueError(f"{name} must be a finite one-dimensional array of at least two edges")
        if not np.all(edges[1:] > edges[:-1]):
            raise ValueError(f"{name} must be strictly increasing")
    atol, rtol = _finite_scalar(atol, "atol"), _finite_scalar(rtol, "rtol")
    if not 0 < atol < 1 or not 0 < rtol < 1:
        raise ValueError("atol and rtol must lie in (0, 1)")
    image = np.zeros((y_edges.size - 1, x_edges.size - 1), dtype=float)
    if profile.flux == 0:
        return image
    quantiles = gammaincinv(2 * profile.n, _RADIAL_QUANTILES)
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        radii = np.exp(profile._log_re + profile.n * (np.log(quantiles) - profile._log_b_n))
    radii = radii[np.isfinite(radii) & (radii > 0)].tolist()
    x_edges, y_edges = x_edges.tolist(), y_edges.tolist()
    for y, (y0, y1) in enumerate(zip(y_edges, y_edges[1:])):
        for x, (x0, x1) in enumerate(zip(x_edges, x_edges[1:])):
            try:
                fraction = _cell_fraction(profile, radii, x0, x1, y0, y1, atol, rtol)
            except RuntimeError as exc:
                raise RuntimeError(f"intrinsic quadrature failed in cell (y={y}, x={x}): {exc}") from exc
            image[y, x] = profile.flux * fraction
    return image


def render_intrinsic_sersic(
    profile: SersicProfile, shape: tuple[int, int], *, dx: float = 0.0, dy: float = 0.0,
    oversamp: int = 1, atol: float = 1e-10, rtol: float = 1e-9,
) -> np.ndarray:
    """Integrate an unconvolved Sérsic profile on the native or subpixel grid.

    Parameters
    ----------
    profile : SersicProfile
        Intrinsic profile to render.
    shape : (int, int)
        Native stamp shape in (ny, nx) order.
    dx, dy : float, optional
        Offsets relative to the geometric stamp center, in native pixels.
    oversamp : int, optional
        Number of equal subcells per native pixel along each axis.
    atol, rtol : float, optional
        Absolute fractional-flux and relative tolerances per output cell.

    Returns
    -------
    ndarray
        Flux per cell of shape (ny * oversamp, nx * oversamp).

    Raises
    ------
    ValueError
        If the stamp geometry, offsets or tolerances are invalid.
    RuntimeError
        If quadrature reports a convergence failure.
    """
    context = RenderContext(shape)
    if isinstance(oversamp, (bool, np.bool_)) or not isinstance(oversamp, Integral) or oversamp < 1:
        raise ValueError("oversamp must be a positive integer")
    dx, dy = _finite_scalar(dx, "dx"), _finite_scalar(dy, "dy")
    ny, nx = context.shape
    cx, cy = context.center
    x_edges = np.arange(nx * oversamp + 1, dtype=float) / oversamp - 0.5 - cx - dx
    y_edges = np.arange(ny * oversamp + 1, dtype=float) / oversamp - 0.5 - cy - dy
    return integrate_sersic_grid(profile, x_edges, y_edges, atol=atol, rtol=rtol)
