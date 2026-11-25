"""Physical parameters, normalized priors and coordinates for generic fitting."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri
from scipy.stats import truncnorm

__all__ = [
    "Uniform", "LogUniform", "Normal", "TruncatedNormal",
    "Parameter", "ParameterSet", "spawn_seed_streams",
]


def _finite(value: Real, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _prior_bounds(lo: float, hi: float) -> tuple[float, float]:
    lo, hi = _finite(lo, "lo"), _finite(hi, "hi")
    if not lo < hi:
        raise ValueError("prior bounds must be strictly increasing")
    return lo, hi


def _log_ratio(hi: float, lo: float) -> float:
    if hi - lo <= lo:
        return math.log1p((hi - lo) / lo)
    return math.log(hi) - math.log(lo)


@dataclass(frozen=True)
class Uniform:
    """Normalized uniform prior on the inclusive physical interval [lo, hi].

    Parameters
    ----------
    lo, hi : float
        Finite, strictly increasing physical bounds with finite width.

    Raises
    ------
    ValueError
        If the interval is invalid.
    """

    lo: float
    hi: float

    def __post_init__(self):
        lo, hi = _prior_bounds(self.lo, self.hi)
        if not math.isfinite(hi - lo):
            raise ValueError("uniform prior width must be finite")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def _support(self):
        return self.lo, self.hi

    def _logpdf(self, x):
        return -math.log(self.hi - self.lo)

    def _ppf(self, u):
        return self.lo + u * (self.hi - self.lo)


@dataclass(frozen=True)
class LogUniform:
    """Normalized log-uniform prior on a finite positive physical interval.

    Parameters
    ----------
    lo, hi : float
        Positive, finite, strictly increasing physical bounds.

    Raises
    ------
    ValueError
        If the interval is invalid.
    """

    lo: float
    hi: float

    def __post_init__(self):
        lo, hi = _prior_bounds(self.lo, self.hi)
        if lo <= 0:
            raise ValueError("log-uniform prior requires positive bounds")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def _support(self):
        return self.lo, self.hi

    def _logpdf(self, x):
        return -math.log(x) - math.log(_log_ratio(self.hi, self.lo))

    def _ppf(self, u):
        offset = u * _log_ratio(self.hi, self.lo)
        if offset <= 1:
            return self.lo + self.lo * math.expm1(offset)
        return math.exp(math.log(self.lo) + offset)


@dataclass(frozen=True)
class Normal:
    """Normalized Gaussian prior in physical units.

    Parameters
    ----------
    mean : float
        Finite center.
    sigma : float
        Finite positive standard deviation.

    Raises
    ------
    ValueError
        If the center or scale is invalid.
    """

    mean: float
    sigma: float

    def __post_init__(self):
        object.__setattr__(self, "mean", _finite(self.mean, "mean"))
        sigma = _finite(self.sigma, "sigma")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        object.__setattr__(self, "sigma", sigma)

    @property
    def _support(self):
        return -math.inf, math.inf

    def _logpdf(self, x):
        offset = (x - self.mean) / self.sigma
        return -0.5 * offset * offset - math.log(self.sigma) - 0.5 * math.log(2 * math.pi)

    def _ppf(self, u):
        return self.mean + self.sigma * float(ndtri(u))


@dataclass(frozen=True)
class TruncatedNormal(Normal):
    """Gaussian prior renormalized on a finite physical interval.

    Parameters
    ----------
    mean, sigma : float
        Finite center and finite positive scale of the parent Gaussian.
    lo, hi : float
        Finite, strictly increasing physical truncation bounds.

    Raises
    ------
    ValueError
        If the parent Gaussian or truncation interval is invalid.
    """

    lo: float
    hi: float

    def __post_init__(self):
        super().__post_init__()
        lo, hi = _prior_bounds(self.lo, self.hi)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        a, b = self._standard_bounds
        if not math.isfinite(a) or not math.isfinite(b) or not a < b:
            raise ValueError("standardized truncation bounds must be finite and increasing")

    @property
    def _support(self):
        return self.lo, self.hi

    @property
    def _standard_bounds(self):
        return (self.lo - self.mean) / self.sigma, (self.hi - self.mean) / self.sigma

    def _logpdf(self, x):
        offset = (x - self.mean) / self.sigma
        return float(truncnorm.logpdf(offset, *self._standard_bounds)) - math.log(self.sigma)

    def _ppf(self, u):
        return self.mean + self.sigma * float(truncnorm.ppf(u, *self._standard_bounds))


Prior = Uniform | LogUniform | Normal | TruncatedNormal


@dataclass(frozen=True)
class Parameter:
    """Define one named physical quantity.

    Parameters
    ----------
    name : str
        Unique name within a set. Dotted names such as ``host.dx`` are allowed.
    initial : float, optional
        Physical starting value for a free parameter.
    unit : str, optional
        Physical unit label. Ties require exactly matching labels.
    prior : Uniform, LogUniform, Normal or TruncatedNormal, optional
        Normalized physical prior, required for a free parameter.
    fixed : float, optional
        Fixed physical value.
    tie : str, optional
        Name of another parameter with the same physical value.
    transform : {'linear', 'log'}, optional
        Sampling coordinate of a free parameter. Log uses the natural log.
    bounds : tuple of (float or None, float or None), optional
        Inclusive physical validity limits. ``None`` omits that limit.
        A free prior or a tie's root support must lie entirely within them.
    period : float, optional
        Positive physical period. A tie inherits its root's period and origin.
    origin : float, optional
        Lower end of the canonical interval for periodic roots. Defaults to 0.

    Notes
    -----
    A free parameter supplies ``prior`` and ``initial``. A fixed parameter
    supplies ``fixed``. An equality alias supplies ``tie``. Initial and fixed
    periodic values may be given anywhere on the circle.
    """

    name: str
    initial: float | None = None
    unit: str = ""
    prior: Prior | None = None
    fixed: float | None = None
    tie: str | None = None
    transform: str = "linear"
    bounds: tuple[float | None, float | None] | None = None
    period: float | None = None
    origin: float | None = None


def _wrap(parameter: Parameter, value: float) -> float:
    if parameter.period is None:
        return value
    origin = parameter.origin
    period = parameter.period
    upper = origin + period
    if origin <= value < upper:
        return value
    if value == upper:
        return origin
    offset = value - origin
    if not math.isfinite(offset):
        offset = (value % period) - (origin % period)
    wrapped = origin + offset % period
    return wrapped if wrapped < upper else origin


def _within(bounds, support):
    return (bounds is None or
            ((bounds[0] is None or bounds[0] <= support[0]) and
             (bounds[1] is None or support[1] <= bounds[1])))


def _validate(parameter: Parameter) -> Parameter:
    name = parameter.name
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("parameter name must be a nonempty, trimmed string")
    if not isinstance(parameter.unit, str):
        raise ValueError(f"{name}: unit must be a string")
    if sum(value is not None for value in (parameter.prior, parameter.fixed, parameter.tie)) != 1:
        raise ValueError(f"{name}: specify exactly one of prior, fixed or tie")
    if parameter.transform not in ("linear", "log"):
        raise ValueError(f"{name}: unknown transform {parameter.transform!r}")
    if parameter.bounds is not None:
        try:
            lo, hi = parameter.bounds
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: bounds must contain two entries") from exc
        lo = None if lo is None else _finite(lo, f"{name} lower bound")
        hi = None if hi is None else _finite(hi, f"{name} upper bound")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"{name}: bounds must be increasing")
        parameter = replace(parameter, bounds=(lo, hi))
    if parameter.tie is not None:
        if not isinstance(parameter.tie, str) or not parameter.tie:
            raise ValueError(f"{name}: tie must name another parameter")
        if parameter.period is not None or parameter.origin is not None:
            raise ValueError(f"{name}: a tie inherits periodicity from its root")
    if parameter.period is not None:
        period = _finite(parameter.period, f"{name} period")
        origin = 0.0 if parameter.origin is None else _finite(parameter.origin, f"{name} origin")
        if period <= 0 or not origin < origin + period < math.inf:
            raise ValueError(f"{name}: invalid periodic interval")
        parameter = replace(parameter, period=period, origin=origin)
    elif parameter.origin is not None:
        raise ValueError(f"{name}: origin requires a period")
    if parameter.prior is None:
        if parameter.initial is not None or parameter.transform != "linear":
            raise ValueError(f"{name}: initial and transform are only for free parameters")
        if parameter.fixed is not None:
            value = _wrap(parameter, _finite(parameter.fixed, f"{name} fixed value"))
            if not _within(parameter.bounds, (value, value)):
                raise ValueError(f"{name}: fixed value is outside physical bounds")
            return replace(parameter, fixed=value)
        return parameter

    prior = parameter.prior
    if not isinstance(prior, (Uniform, LogUniform, Normal)):
        raise ValueError(f"{name}: unsupported physical prior")
    lo, hi = prior._support
    if not _within(parameter.bounds, (lo, hi)):
        raise ValueError(f"{name}: prior support exceeds physical bounds")
    if parameter.period is not None:
        origin, period = parameter.origin, parameter.period
        if parameter.transform != "linear" or (lo, hi) != (origin, origin + period):
            raise ValueError(f"{name}: periodic prior must span its canonical interval in linear coordinates")
    if parameter.transform == "log" and not (lo > 0 and math.log(lo) < math.log(hi)):
        raise ValueError(f"{name}: log transform requires distinct, strictly positive prior bounds")
    initial = _wrap(parameter, _finite(parameter.initial, f"{name} initial"))
    if not lo <= initial <= hi or not math.isfinite(prior._logpdf(initial)):
        raise ValueError(f"{name}: initial value is outside the usable prior support")
    return replace(parameter, initial=initial)


class ParameterSet:
    """Prepare parameters and resolve equality ties before fitting.

    Parameters
    ----------
    parameters : iterable of Parameter
        Declaration order determines output mapping order and ``free_names``.

    Raises
    ------
    ValueError
        If a definition, name, prior, physical bound or tie is inconsistent.

    Notes
    -----
    ``free_names`` is the ordered tuple of independent sampled names. ``ndim``
    is its length. Fixed and tied parameters add neither a sampling dimension
    nor a prior factor. An all-fixed set represents one deterministic model.
    """

    def __init__(self, parameters: Iterable[Parameter]):
        self._parameters = {}
        for definition in parameters:
            if not isinstance(definition, Parameter):
                raise ValueError("parameter definitions must be Parameter instances")
            parameter = _validate(definition)
            if parameter.name in self._parameters:
                raise ValueError(f"duplicate parameter name: {parameter.name}")
            self._parameters[parameter.name] = parameter
        self._free = tuple(p for p in self._parameters.values() if p.prior is not None)
        self._roots = {}
        for name in self._parameters:
            self._resolve_root(name)

    def _resolve_root(self, name):
        path = {}
        current = name
        while current not in self._roots:
            if current in path:
                raise ValueError(f"cyclic parameter tie involving {current}")
            parameter = self._parameters[current]
            if parameter.tie is None:
                self._roots[current] = parameter
                break
            if parameter.tie not in self._parameters:
                raise ValueError(f"{current}: missing tie target {parameter.tie!r}")
            path[current] = parameter
            current = parameter.tie
        root = self._roots[current]
        support = root.prior._support if root.prior is not None else (root.fixed, root.fixed)
        for alias in path.values():
            if alias.unit != root.unit:
                raise ValueError(f"{alias.name}: tie requires matching units with {root.name}")
            if not _within(alias.bounds, support):
                raise ValueError(f"{alias.name}: tie bounds exclude values allowed by {root.name}")
            self._roots[alias.name] = root

    @property
    def free_names(self) -> tuple[str, ...]:
        """Return the free parameter names in sampling-vector order."""
        return tuple(p.name for p in self._free)

    @property
    def ndim(self) -> int:
        """Return the number of free parameters."""
        return len(self._free)

    @property
    def initial(self) -> NDArray[np.float64]:
        """Return a new vector of initial values in sampling coordinates."""
        return self.encode({p.name: p.initial for p in self._free})

    def _vector(self, vector):
        values = np.asarray(vector)
        if values.shape != (self.ndim,) or values.dtype.kind not in "iuf":
            raise ValueError(f"expected a real vector with shape ({self.ndim},)")
        return values.astype(float, copy=False)

    @staticmethod
    def _encode_value(parameter, value):
        return math.log(value) if parameter.transform == "log" else value

    def _decode_free(self, vector):
        result = {}
        for parameter, sample in zip(self._free, vector):
            sample = float(sample)
            if not math.isfinite(sample):
                raise ValueError(f"{parameter.name}: sampling value must be finite")
            lo, hi = parameter.prior._support
            if parameter.transform == "log":
                low_log, high_log = math.log(lo), math.log(hi)
                if not low_log <= sample <= high_log:
                    raise ValueError(f"{parameter.name}: sampling value is outside prior support")
                # Map exact log endpoints back to their physical endpoints.
                if sample == low_log:
                    value = lo
                elif sample == high_log:
                    value = hi
                else:
                    value = min(hi, max(lo, math.exp(sample)))
            else:
                value = _wrap(parameter, sample)
                if not lo <= value <= hi:
                    raise ValueError(f"{parameter.name}: sampling value is outside prior support")
            result[parameter.name] = value
        return result

    def decode(self, vector: ArrayLike) -> dict[str, float]:
        """Decode a sampling vector into physical values in declaration order.

        Parameters
        ----------
        vector : array-like of float
            One finite sampling vector of shape ``(ndim,)``.

        Returns
        -------
        dict of str to float
            Canonical physical values for free, fixed and tied parameters.

        Raises
        ------
        ValueError
            If the vector shape, numeric type or prior support is invalid.
        """
        free = self._decode_free(self._vector(vector))
        values = {}
        for name in self._parameters:
            root = self._roots[name]
            values[name] = free[root.name] if root.prior is not None else root.fixed
        return values

    def encode(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        """Encode physical values in ``free_names`` order.

        Parameters
        ----------
        values : mapping of str to float
            All free physical values. Optional fixed and tied entries must
            agree with the corresponding values after periodic wrapping.

        Returns
        -------
        numpy.ndarray
            Sampling vector with shape ``(ndim,)``.

        Raises
        ------
        ValueError
            If names are missing or unknown, or values violate the definitions.
        """
        if values.keys() - self._parameters.keys() or set(self.free_names) - values.keys():
            raise ValueError("physical values must contain all free names and only known names")
        free = {}
        for parameter in self._free:
            value = _wrap(parameter, _finite(values[parameter.name], parameter.name))
            lo, hi = parameter.prior._support
            if not lo <= value <= hi:
                raise ValueError(f"{parameter.name}: physical value is outside prior support")
            free[parameter.name] = value
        for name in values.keys() - free.keys():
            root = self._roots[name]
            expected = root.fixed if root.prior is None else free[root.name]
            actual = _wrap(root, _finite(values[name], name))
            if root.period is None:
                agrees = actual == expected
            else:
                difference = math.remainder(actual - expected, root.period)
                agrees = abs(difference) <= 4 * math.ulp(root.period)
            if not agrees:
                raise ValueError(f"{name}: physical value conflicts with its fixed value or tie")
        return np.array([self._encode_value(p, free[p.name]) for p in self._free], dtype=float)

    def log_prior_in_sampling_space(self, vector: ArrayLike) -> float:
        """Evaluate the log prior density with the change-of-variable Jacobian.

        Parameters
        ----------
        vector : array-like of float
            One vector in sampling coordinates. Periodic coordinates must be
            in their half-open canonical intervals for density evaluation.

        Returns
        -------
        float
            Log prior density, zero for an all-fixed set, or minus infinity
            for a nonfinite value or a value outside support.

        Raises
        ------
        ValueError
            If the vector shape or numeric type is invalid.
        """
        coordinates = self._vector(vector)
        try:
            free = self._decode_free(coordinates)
        except ValueError:
            return -math.inf
        total = 0.0
        for parameter, sample in zip(self._free, coordinates):
            if parameter.period is not None and not (
                parameter.origin <= sample < parameter.origin + parameter.period
            ):
                return -math.inf
            total += parameter.prior._logpdf(free[parameter.name])
            if parameter.transform == "log":
                total += float(sample)
        return total

    def prior_transform(self, cube: ArrayLike) -> NDArray[np.float64]:
        """Map independent unit-cube coordinates to sampling coordinates.

        Parameters
        ----------
        cube : array-like of float
            One vector with shape ``(ndim,)`` and entries in [0, 1]. An
            unbounded prior requires interior quantiles. A periodic upper
            endpoint maps to the equivalent lower endpoint.

        Returns
        -------
        numpy.ndarray
            Sampling vector from the declared physical priors. A nested
            sampler using this transform should evaluate the likelihood alone.

        Raises
        ------
        ValueError
            If the cube is invalid or a quantile is nonfinite.
        """
        unit = self._vector(cube)
        if not np.all(np.isfinite(unit) & (unit >= 0) & (unit <= 1)):
            raise ValueError("unit-cube coordinates must be finite and in [0, 1]")
        values = {}
        for parameter, fraction in zip(self._free, unit):
            lo, hi = parameter.prior._support
            if fraction == 0:
                value = lo
            elif fraction == 1:
                value = hi
            else:
                value = parameter.prior._ppf(float(fraction))
                if math.isfinite(value):
                    value = min(hi, max(lo, value))
            values[parameter.name] = value
        return self.encode(values)

    def to_schema(self) -> dict:
        """Return a JSON-compatible snapshot of the parameter definitions.

        Returns
        -------
        dict
            Versioned definitions in declaration order, including prior kind,
            canonical initial and fixed values, ties and units.
        """
        definitions = []
        for parameter in self._parameters.values():
            entry = asdict(parameter)
            if parameter.prior is not None:
                entry["prior"]["kind"] = type(parameter.prior).__name__
            if parameter.bounds is not None:
                entry["bounds"] = list(parameter.bounds)
            definitions.append(entry)
        return {"version": 1, "parameters": definitions, "free_names": list(self.free_names)}


def spawn_seed_streams(seed: int | None = None) -> dict[str, np.random.SeedSequence]:
    """Spawn independent streams for initialization, noise, and sampling.

    Parameters
    ----------
    seed : int or None, optional
        Nonnegative root entropy. ``None`` obtains entropy from the OS.

    Returns
    -------
    dict of str to numpy.random.SeedSequence
        Streams named ``initialization``, ``noise``, and ``sampler`` in that
        order. Record each child's ``state`` for provenance. Spawn descendants
        from the appropriate child for separate workers.

    Raises
    ------
    ValueError
        If the seed is invalid.
    """
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or
                             not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    children = np.random.SeedSequence(seed).spawn(3)
    return dict(zip(("initialization", "noise", "sampler"), children))
