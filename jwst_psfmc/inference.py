"""Optimization and posterior inference for prepared image models."""

from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import version
from inspect import signature
import math
import multiprocessing as mp
from numbers import Integral
from time import perf_counter

import emcee
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from ._provenance import runtime_provenance
from .fitting import PreparedModelFit
from .joint import PreparedJointFit, _pack
from .models import ConstantBackground, PointSource, SersicSource, _COMPONENT_FIELDS
from .parameters import spawn_seed_streams


class _NumericalFailure(RuntimeError):
    """A supported sampling coordinate failed to render or evaluate."""


@dataclass(frozen=True, eq=False)
class FitResult:
    """Posterior draws and diagnostics from :func:`run_model_mcmc`.

    The two sample arrays have shape (step, walker, parameter) for emcee
    and (draw, parameter) for Nautilus. All log arrays omit the parameter
    axis. Log priors and posteriors are densities in sampling coordinates.
    The schema stores fixed values and equality ties in `metadata`.
    """

    sampler: str
    status: str
    stop_reason: str
    free_names: tuple[str, ...]
    sampling_samples: np.ndarray
    physical_samples: np.ndarray
    log_likelihood: np.ndarray
    log_prior: np.ndarray
    log_weights: np.ndarray | None
    burn_in: int
    diagnostics: dict
    metadata: dict

    @property
    def log_posterior(self) -> np.ndarray:
        """Return the sampling-space log posterior, up to a constant."""
        return self.log_likelihood + self.log_prior


@dataclass(frozen=True, eq=False)
class OptimizationAttempt:
    """Outcome for one optimizer start in sampling coordinates."""

    start: np.ndarray
    sample_vector: np.ndarray
    objective_value: float
    success: bool
    message: str
    n_evaluations: int


@dataclass(frozen=True, eq=False)
class OptimizationResult:
    """Multistart ML or MAP optimization result in sampling coordinates."""

    objective: str
    attempts: tuple[OptimizationAttempt, ...]
    best_index: int
    free_names: tuple[str, ...]

    @property
    def best(self) -> OptimizationAttempt:
        """Return the highest-objective successful attempt, if available."""
        return self.attempts[self.best_index]


def _support(root):
    return root.prior._support if root.prior is not None else (root.fixed, root.fixed)


def _periodic_axes(parameters):
    return tuple((i, p.origin, p.period) for i, p in enumerate(parameters._free)
                 if p.period is not None)


def _validate_renderer_support(fit, roots):
    for component in fit.model.components:
        name = component.name
        kind = type(component)
        fields = _COMPONENT_FIELDS.get(kind, ())
        for field in fields:
            if f"{name}.{field}" not in roots:
                raise ValueError(f"missing model parameter {name}.{field}")
        operator = fit.context.psf_operator
        if kind is SersicSource:
            if operator is None:
                raise ValueError("SersicSource requires a psf_operator in RenderContext")
            if roots[f"{name}.n"].prior is not None:
                raise ValueError("Sérsic inference requires fixed n until a continuous n domain is validated")
            for field in ("re", "n", "q"):
                lo, hi = _support(roots[f"{name}.{field}"])
                if not 0 < lo <= hi < np.inf or (field == "q" and hi > 1):
                    raise ValueError(f"{name}.{field} prior support exceeds the renderer's physical domain")
            if _support(roots[f"{name}.flux"])[0] < 0:
                raise ValueError(f"{name}.flux prior support must be nonnegative")
            period = roots[f"{name}.pa"].period
            if period is not None and period != 180:
                raise ValueError(f"{name}.pa period must be 180 degrees")
        if kind is PointSource and operator is None and fit.context.point_response is None:
            raise ValueError("PointSource requires a point response")
        if operator is not None and kind in (SersicSource, PointSource):
            for field in ("dx", "dy"):
                lo, hi = _support(roots[f"{name}.{field}"])
                if lo < -operator.max_offset or hi > operator.max_offset:
                    raise ValueError(f"{name}.{field} prior support exceeds PSFOperator.max_offset")


@dataclass(eq=False)
class _ModelDensity:
    """Picklable density with optional exact caching of linear responses."""

    fit: PreparedModelFit | PreparedJointFit
    linear_names: tuple[str, ...] = ()
    design: np.ndarray | None = None
    whitened_data: np.ndarray | None = None
    constant: float = 0.0
    aliases: dict | None = None

    def _local_values(self, values):
        return values if not self.aliases else {
            **values, **{local: values[target] for local, target in self.aliases.items()},
        }

    def _templates(self):
        aliases = self.aliases or {}
        roots = self.fit.parameters._roots
        roots = {**roots, **{local: roots[target] for local, target in aliases.items()}}
        _validate_renderer_support(self.fit, roots)
        names = []
        for component in self.fit.model.components:
            kind = type(component)
            if kind not in (ConstantBackground, PointSource, SersicSource):
                return
            shape_keys = {
                ConstantBackground: (),
                PointSource: ("dx", "dy"),
                SersicSource: ("dx", "dy", "re", "n", "q", "pa"),
            }[kind]
            if any(roots[f"{component.name}.{key}"].prior is not None
                   for key in shape_keys):
                return
            amplitude = "level" if kind is ConstantBackground else "flux"
            names.append(f"{component.name}.{amplitude}")
        values = self._local_values(self.fit.parameters.decode(self.fit.parameters.initial))
        try:
            templates = [component.render({**values, name: 1.0}, self.fit.context)
                         for name, component in zip(names, self.fit.model.components)]
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise _NumericalFailure(f"linear template evaluation failed: {exc}") from exc
        return tuple(aliases.get(name, name) for name in names), templates

    def prepare(self):
        linear = self._templates()
        if linear is None:
            return
        self.linear_names, templates = linear
        self.design = np.column_stack([self.fit.noise.whiten(template) for template in templates])
        self.whitened_data = self.fit.noise.whiten(self.fit.data)
        self.constant = self.fit.noise.log_likelihood(np.zeros(self.fit.context.shape))

    def _render(self, values):
        return self.fit.model.render(self._local_values(values), self.fit.context)

    def _linear_likelihood(self, values):
        amplitudes = np.array([values[key] for key in self.linear_names])
        resid = self.whitened_data - self.design @ amplitudes
        return self.constant - 0.5 * (resid @ resid)

    def _likelihood_values(self, values):
        if self.design is not None:
            return self._linear_likelihood(values)
        return self.fit.noise.log_likelihood(self.fit.data - self._render(values))

    def _likelihood(self, vector):
        values = self.fit.parameters.decode(vector)
        try:
            logl = self._likelihood_values(values)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise _NumericalFailure(f"likelihood evaluation failed: {exc}") from exc
        if not np.isfinite(logl):
            raise _NumericalFailure("likelihood is nonfinite inside the prior")
        return float(logl)

    def __call__(self, vector):
        prior = self.fit.parameters.log_prior_in_sampling_space(vector)
        if not np.isfinite(prior):
            return -np.inf, -np.inf, -np.inf
        logl = self._likelihood(vector)
        return logl + prior, logl, prior

    def nested_likelihood(self, vector):
        return self._likelihood(vector[:self.fit.parameters.ndim])

    def nested_transform(self, cube):
        ndim = self.fit.parameters.ndim
        # Keep unbounded prior transforms finite at an exact cube endpoint.
        unit = np.clip(cube[:ndim], np.nextafter(0., 1.), np.nextafter(1., 0.))
        values = self.fit.parameters.prior_transform(unit)
        return np.r_[values, cube[ndim:]] if ndim < len(cube) else values


@dataclass(eq=False)
class _JointDensity(_ModelDensity):
    frames: tuple[_ModelDensity, ...] = ()

    def prepare(self):
        self.frames = tuple(_ModelDensity(obs.fit, aliases=obs.aliases)
                            for obs in self.fit.observations)
        if self.fit.noise is None:
            for frame in self.frames:
                frame.prepare()
            if any(frame.design is None for frame in self.frames):
                return
            names = [frame.linear_names for frame in self.frames]
            self.linear_names = tuple(dict.fromkeys(name for keys in names for name in keys))
            designs = [dict(zip(self.linear_names, np.zeros((len(self.linear_names), frame.design.shape[0]))))
                       for frame in self.frames]
            for frame, columns in zip(self.frames, designs):
                for name, column in zip(frame.linear_names, frame.design.T):
                    columns[name] += column
            self.design = np.vstack([np.column_stack(list(columns.values())) for columns in designs])
            self.whitened_data = np.concatenate([frame.whitened_data for frame in self.frames])
            self.constant = sum(frame.constant for frame in self.frames)
            return

        templates = [frame._templates() for frame in self.frames]
        if any(template is None for template in templates):
            return
        self.linear_names = tuple(dict.fromkeys(name for names, _ in templates for name in names))
        data = _pack(frame.fit.data for frame in self.frames)
        designs = [dict(zip(self.linear_names, np.zeros((len(self.linear_names), *frame.fit.data.shape))))
                   for frame in self.frames]
        for (names, columns), design in zip(templates, designs):
            for name, column in zip(names, columns):
                design[name] += column
        self.design = np.column_stack([
            self.fit.noise.whiten(_pack(design[name] for design in designs))
            for name in self.linear_names
        ])
        self.whitened_data = self.fit.noise.whiten(data)
        self.constant = self.fit.noise.log_likelihood(np.zeros_like(data))

    def _likelihood_values(self, values):
        if self.design is not None:
            return self._linear_likelihood(values)
        if self.fit.noise is None:
            return sum(frame._likelihood_values(values) for frame in self.frames)
        residual = _pack(frame.fit.data-frame._render(values) for frame in self.frames)
        return self.fit.noise.log_likelihood(residual)


def _density_for(fit):
    return _JointDensity(fit) if isinstance(fit, PreparedJointFit) else _ModelDensity(fit)


def _bounds(parameters):
    bounds = []
    for parameter in parameters._free:
        lo, hi = _support(parameter)
        if parameter.transform == "log":
            lo, hi = math.log(lo), math.log(hi)
        if parameter.period is not None:
            hi = np.nextafter(hi, lo)
        bounds.append((lo, hi))
    return bounds


def _count(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _start(parameters, initial):
    vector = parameters.initial if initial is None else np.asarray(initial)
    if not np.isfinite(parameters.log_prior_in_sampling_space(vector)):
        raise ValueError("initial vector is outside prior support")
    return np.array(vector, dtype=float, copy=True)


def _linear_start(density, start, bounds):
    """Use weighted least squares only if every free root is an amplitude."""
    parameters = density.fit.parameters
    if density.design is None or not parameters.ndim:
        return None
    columns = np.zeros((density.design.shape[0], parameters.ndim))
    fixed = np.zeros(density.design.shape[0])
    index = {name: i for i, name in enumerate(parameters.free_names)}
    used = set()
    for name, template in zip(density.linear_names, density.design.T):
        root = parameters._roots[name]
        if root.prior is None:
            fixed += root.fixed * template
        else:
            columns[:, index[root.name]] += template
            used.add(root.name)
    if used != set(index):
        return None
    values, _, rank, _ = np.linalg.lstsq(columns, density.whitened_data - fixed, rcond=None)
    if rank != parameters.ndim:
        return None
    try:
        vector = parameters.encode(dict(zip(parameters.free_names, values)))
    except ValueError:
        return None
    if np.array_equal(start, vector) or not all(lo <= x <= hi for (lo, hi), x in zip(bounds, vector)):
        return None
    return vector


def optimize_model_fit(fit, *, objective="map", starts=None, max_evaluations=2000):
    """Fit a prepared model from one or more sampling-space starting vectors.

    Parameters
    ----------
    fit : PreparedModelFit or PreparedJointFit
        Prepared single-frame or joint image model.
    objective : {'map', 'ml'}, optional
        Maximize posterior density in sampling coordinates or likelihood.
        Prior support constrains both objectives.
    starts : iterable of array-like, optional
        Initial sampling-space vectors. The prepared initial vector is the
        default. A feasible fixed-morphology least-squares start is added.
    max_evaluations : int, optional
        Powell's evaluation limit per start.

    Returns
    -------
    OptimizationResult
        Every attempt and the best successful attempt, or the best finite
        attempt if none converged. All-fixed models evaluate deterministically.

    Raises
    ------
    ValueError
        If the configuration, prior support, or a start is invalid.
    """
    if objective not in ("map", "ml"):
        raise ValueError("objective must be 'map' or 'ml'")
    max_evaluations = _count(max_evaluations, "max_evaluations", 1)
    parameters = fit.parameters
    vectors = [parameters.initial] if starts is None else [_start(parameters, v) for v in starts]
    if not vectors:
        raise ValueError("starts must contain at least one vector")
    bounds = _bounds(parameters)
    density = _density_for(fit)

    def loss(vector):
        posterior, logl, _ = density(vector)
        return -(posterior if objective == "map" else logl)

    attempts = []
    try:
        density.prepare()
        linear = _linear_start(density, vectors[0], bounds)
        if linear is not None:
            vectors.append(linear)
    except (_NumericalFailure, np.linalg.LinAlgError) as exc:
        return OptimizationResult(objective, tuple(
            OptimizationAttempt(v, v.copy(), -np.inf, False, str(exc), 0) for v in vectors
        ), 0, parameters.free_names)
    for start in vectors:
        evaluated = 0

        def counted_loss(vector):
            nonlocal evaluated
            evaluated += 1
            return loss(vector)

        try:
            start_value = -counted_loss(start)
            if not parameters.ndim:
                attempt = OptimizationAttempt(start, start.copy(), start_value, True, "all parameters fixed", 1)
            elif max_evaluations == 1:
                attempt = OptimizationAttempt(start, start.copy(), start_value, False,
                                              "evaluation limit reached", 1)
            else:
                result = minimize(counted_loss, start, method="Powell", bounds=bounds,
                                  options={"maxfev": max_evaluations - 1, "xtol": 1e-6, "ftol": 1e-10})
                value = -float(result.fun)
                finite_result = np.isfinite(value)
                vector = np.asarray(result.x)
                if value < start_value or not finite_result:
                    vector, value = start.copy(), start_value
                attempt = OptimizationAttempt(start, vector, value,
                                              bool(result.success and finite_result),
                                              str(result.message), evaluated)
        except _NumericalFailure as exc:
            attempt = OptimizationAttempt(start, start.copy(), -np.inf, False, str(exc), evaluated)
        attempts.append(attempt)
    best = max(range(len(attempts)), key=lambda i: (attempts[i].success, attempts[i].objective_value))
    return OptimizationResult(objective, tuple(attempts), best, parameters.free_names)


def _wrap_axes(coords, axes):
    for index, origin, period in axes:
        wrapped = origin + (coords[..., index] - origin) % period
        coords[..., index] = np.where(wrapped < origin + period, wrapped, origin)
    return coords


@dataclass(frozen=True)
class _PeriodicMove:
    axes: tuple

    def __call__(self, coords, random):
        proposals = coords.copy()
        for index, _, period in self.axes:
            proposals[:, index] += random.normal(scale=.07 * period, size=len(coords))
        return _wrap_axes(proposals, self.axes), np.zeros(len(coords))


def _walkers(parameters, center, rng, count, init_scale):
    scales = (parameters.prior_transform(np.full(parameters.ndim, .84))
              - parameters.prior_transform(np.full(parameters.ndim, .16))) * init_scale
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("walker initialization scales must be finite and positive")
    axes = _periodic_axes(parameters)
    accepted = []
    for _ in range(128):
        proposals = center + rng.normal(size=(count, parameters.ndim)) * scales
        _wrap_axes(proposals, axes)
        accepted.extend(vector for vector in proposals
                        if np.isfinite(parameters.log_prior_in_sampling_space(vector)))
        if len(accepted) >= count:
            result = np.asarray(accepted[:count])
            if emcee.ensemble.walkers_independent(result):
                return result
            break
    raise ValueError("could not initialize a distinct full-rank ensemble inside the prior")


def _physical_samples(parameters, samples):
    physical = samples.copy()
    for index, parameter in enumerate(parameters._free):
        if parameter.transform == "log":
            lo, hi = _support(parameter)
            source = samples[..., index]
            physical[..., index] = np.where(
                source == math.log(lo), lo,
                np.where(source == math.log(hi), hi, np.exp(source)),
            )
    return physical


def _features(physical, periodic):
    angles = {i: (origin, period) for i, origin, period in periodic}
    for i in range(physical.shape[-1]):
        values = physical[..., i]
        if i in angles:
            origin, period = angles[i]
            phase = math.tau * (values - origin) / period
            yield np.stack((np.cos(phase), np.sin(phase)), axis=-1)
        else:
            yield values[..., None]


def _emcee_diagnostics(physical, axes, minimum_ess, tau_multiple):
    t, walkers, ndim = physical.shape
    tau, change, ess, mcse = [], [], [], []
    for feature in _features(physical, axes):
        if t < 16:
            tau.append(None)
            change.append(None)
            ess.append(None)
            mcse.append(None)
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            full = emcee.autocorr.integrated_time(feature, tol=0)
            first_half = emcee.autocorr.integrated_time(feature[:t//2], tol=0)
        if not np.all(np.isfinite(full) & np.isfinite(first_half) & (full > 0) & (first_half > 0)):
            tau.append(None)
            change.append(None)
            ess.append(None)
            mcse.append(None)
            continue
        slowest = max(1., float(np.max(full)))
        score = min(t * walkers, t * walkers / slowest)
        tau.append(slowest)
        change.append(float(np.max(np.abs(full - first_half) / full)))
        ess.append(float(score))
        mcse.append(float(np.std(feature, axis=(0, 1), ddof=1).max() / math.sqrt(score)))
    converged = ndim > 0 and all(
        a is not None and t >= tau_multiple * a and b <= .2 and c >= minimum_ess
        for a, b, c in zip(tau, change, ess)
    )
    return converged, {"tau": tau, "tau_relative_change": change, "ess": ess, "mcse": mcse,
                       "post_burn_in_steps": t, "minimum_ess": minimum_ess,
                       "tau_multiple": tau_multiple,
                       "mcse_definition": "approximate mean MCSE; periodic axes use max of sine/cosine MCSE"}


def _pool(ncores):
    ncores = _count(ncores, "ncores", 1)
    return nullcontext(None) if ncores == 1 else mp.get_context("spawn").Pool(ncores)


def _random_state(state):
    key, values, position, has_gauss, cached = state
    return [key, values.tolist(), position, has_gauss, cached]


def _run_emcee(density, streams, initial, *, nsteps=2000, burn_in=500, nwalkers=None,
               init_scale=.05, minimum_ess=400, tau_multiple=50, ncores=1, progress=False):
    parameters = density.fit.parameters
    ndim = parameters.ndim
    nsteps = _count(nsteps, "nsteps", 1)
    burn_in = _count(burn_in, "burn_in", 0)
    if burn_in >= nsteps:
        raise ValueError("burn_in must be smaller than nsteps")
    nwalkers = max(12, 2 * ndim + 2) if nwalkers is None else _count(nwalkers, "nwalkers", max(2, 2 * ndim))
    minimum_ess = _count(minimum_ess, "minimum_ess", 1)
    tau_multiple = _count(tau_multiple, "tau_multiple", 1)
    if not np.isfinite(init_scale) or init_scale <= 0:
        raise ValueError("init_scale must be finite and positive")
    rng = np.random.default_rng(streams["initialization"])
    center = _start(parameters, initial)
    positions = _walkers(parameters, center, rng, nwalkers, init_scale)
    axes = _periodic_axes(parameters)
    move = emcee.moves.StretchMove()
    moves = [(move, .75), (emcee.moves.MHMove(_PeriodicMove(axes)), .25)] if axes else move
    random = np.random.RandomState(streams["sampler"].generate_state(4))
    with _pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, density, pool=pool, moves=moves,
            blobs_dtype=[("log_likelihood", float), ("log_prior", float)],
        )
        sampler.random_state = random.get_state()
        sampler.run_mcmc(positions, nsteps, progress=progress)
    coordinates = sampler.get_chain()
    physical = _physical_samples(parameters, coordinates)
    converged, diagnostics = _emcee_diagnostics(physical[burn_in:], axes, minimum_ess, tau_multiple)
    diagnostics["acceptance_fraction"] = sampler.acceptance_fraction.tolist()
    blobs = sampler.get_blobs()
    return dict(
        status="converged" if converged else "insufficient_sampling",
        stop_reason=("within-ensemble tau and ESS targets met" if converged
                     else "post-burn-in tau length, stability or ESS target not met"),
        sampling_samples=coordinates, physical_samples=physical,
        log_likelihood=blobs["log_likelihood"], log_prior=blobs["log_prior"], log_weights=None,
        burn_in=burn_in, diagnostics=diagnostics,
    ), {"initial_center": center.tolist(), "initial_walkers": positions.tolist(),
        "rng_initial": _random_state(random.get_state()), "rng_final": _random_state(sampler.random_state),
        "moves": "stretch plus symmetric wrapped MH" if axes else "stretch",
        "nsteps": nsteps, "nwalkers": nwalkers, "init_scale": float(init_scale), "ncores": int(ncores)}


def _load_nautilus():
    try:
        from nautilus import Sampler
    except ModuleNotFoundError as exc:
        if exc.name != "nautilus":
            raise
        raise ImportError('Install the Nautilus backend with pip install "jwst-psfmc[nautilus]"') from exc
    return Sampler


def _empty(sampler, ndim):
    shape = (0, 0) if sampler == "emcee" else (0,)
    return dict(sampling_samples=np.empty(shape + (ndim,)), physical_samples=np.empty(shape + (ndim,)),
                log_likelihood=np.empty(shape), log_prior=np.empty(shape),
                log_weights=None if sampler == "emcee" else np.empty(shape), burn_in=0)


def _run_nested(density, streams, *, n_live=200, n_eff=1000, n_like_max=None,
                ncores=1, progress=False):
    Sampler = _load_nautilus()
    parameters = density.fit.parameters
    ndim = parameters.ndim
    nested_dim = max(2, ndim)
    n_live = _count(n_live, "n_live", nested_dim + 1)
    n_eff = _count(n_eff, "n_eff", 1)
    cap = np.inf if n_like_max is None else _count(n_like_max, "n_like_max", 1)
    seed = int(streams["sampler"].generate_state(1, dtype=np.uint64)[0])
    periodic = [i for i, _, _ in _periodic_axes(parameters)]
    with _pool(ncores) as pool:
        sampler = Sampler(
            density.nested_transform, density.nested_likelihood, n_dim=nested_dim,
            periodic=periodic or None, n_live=n_live, seed=seed, pool=(pool, None),
        )
        complete = sampler.run(n_eff=n_eff, n_like_max=cap, discard_exploration=True, verbose=progress)
        posterior = sampler.posterior() if sampler.explored and np.sum(sampler.shell_n) > 0 else None
    result = _empty("nautilus", ndim)
    ess, mcse = 0., None
    if posterior is not None:
        samples, logw, logl = posterior
        samples = samples[:, :ndim]
        logw = logw - logsumexp(logw)
        if not np.all(np.isfinite(logl) & np.isfinite(logw)):
            raise _NumericalFailure("Nautilus returned a nonfinite posterior")
        priors = np.array([parameters.log_prior_in_sampling_space(v) for v in samples])
        physical = _physical_samples(parameters, samples)
        weights = np.exp(logw)
        ess = float(np.exp(-logsumexp(2 * logw)))
        mcse = []
        for feature in _features(physical, _periodic_axes(parameters)):
            mean = weights @ feature
            var = weights @ (feature - mean)**2
            mcse.append(float(np.sqrt(var.max() / ess)))
        result.update(sampling_samples=samples, physical_samples=physical,
                      log_likelihood=logl, log_prior=priors, log_weights=logw)
    converged = bool(complete and posterior is not None and ess >= n_eff * (1 - 1e-12))
    result.update(
        status="converged" if converged else "insufficient_sampling",
        stop_reason="nested exploration and weighted ESS target met" if converged
        else "nested exploration or weighted ESS target not reached before stopping",
        diagnostics={"weighted_ess": ess, "mcse": mcse, "n_likelihood_calls": int(sampler.n_like),
                     "minimum_ess": n_eff, "exploration_discarded": True,
                     "mcse_definition": "Kish weighted mean approximation; periodic axes use sine/cosine"},
    )
    return result, {"seed": seed, "rng_final": sampler.rng.bit_generator.state,
                    "periodic_indices": periodic, "auxiliary_unit_dimension": ndim == 1,
                    "n_live": n_live, "n_like_max": n_like_max, "ncores": int(ncores)}


def run_model_mcmc(fit: PreparedModelFit | PreparedJointFit, *, sampler="emcee", initial=None, seed=None,
                   sampling_options=None) -> FitResult:
    """Sample a prepared model using emcee or optional Nautilus.

    Parameters
    ----------
    fit : PreparedModelFit or PreparedJointFit
        Deterministic single-frame or joint image fit. Sérsic inference uses fixed n.
    sampler : {'emcee', 'nautilus'}, optional
        emcee uses the sampling-space prior. Nautilus draws from the same
        unit-cube prior transform and evaluates the likelihood alone.
    initial : array-like or OptimizationResult, optional
        emcee ensemble center. Defaults to the prepared initial vector.
    seed : int, optional
        Root seed for independent initialization and sampler streams.
    sampling_options : mapping, optional
        emcee accepts nsteps, burn_in, nwalkers, init_scale, minimum_ess,
        tau_multiple, ncores and progress. Nautilus accepts n_live, n_eff,
        n_like_max, ncores and progress. Multiple cores use spawn.

    Returns
    -------
    FitResult
        Draws, diagnostics, and a converged, insufficient_sampling or
        numerical_failure status. Configuration errors raise before sampling.

    Raises
    ------
    ValueError
        If the sampler, prior, initialization or configuration is invalid.
    ImportError
        If the requested Nautilus dependency is unavailable.
    """
    if sampler not in ("emcee", "nautilus"):
        raise ValueError("sampler must be 'emcee' or 'nautilus'")
    if not fit.parameters.ndim:
        raise ValueError("all parameters are fixed; use evaluate_model_fit")
    if sampler == "nautilus" and initial is not None:
        raise ValueError("Nautilus initializes from the prior; omit initial")
    optimization = None
    if isinstance(initial, OptimizationResult):
        if initial.free_names != fit.parameters.free_names or not initial.best.success:
            raise ValueError("initial optimization must be successful with the same free_names")
        optimization = {
            "objective": initial.objective,
            "best_index": initial.best_index,
            "attempts": [
                dict(start=attempt.start.tolist(), solution=attempt.sample_vector.tolist(),
                     objective_value=attempt.objective_value if np.isfinite(attempt.objective_value) else None,
                     success=attempt.success, message=attempt.message, n_evaluations=attempt.n_evaluations)
                for attempt in initial.attempts
            ],
        }
        initial = initial.best.sample_vector
    options = {} if sampling_options is None else dict(sampling_options)
    runner = _run_emcee if sampler == "emcee" else _run_nested
    # Check option names even if the rendering setup fails.
    args = (initial,) if sampler == "emcee" else ()
    streams = spawn_seed_streams(seed)
    density = _density_for(fit)
    signature(runner).bind(density, streams, *args, **options)
    if sampler == "nautilus":
        _load_nautilus()
    start = perf_counter()
    try:
        density.prepare()
        result, state = runner(density, streams, *args, **options)
    except (_NumericalFailure, np.linalg.LinAlgError, FloatingPointError) as exc:
        result = _empty(sampler, fit.parameters.ndim)
        result.update(status="numerical_failure", stop_reason=str(exc), diagnostics={})
        state = {}
    result["diagnostics"]["runtime_s"] = perf_counter() - start
    from . import __version__

    fit_metadata = fit.metadata if isinstance(fit, PreparedJointFit) else {
        **fit.provenance,
        "components": [{"name": c.name, "type": type(c).__name__} for c in fit.model.components],
        "geometry": {"shape": list(fit.context.shape), "center_xy": list(fit.context.center),
                     "coordinate_convention": "array[y, x]; offsets from geometric stamp center in native pixels"},
        "psf": None if fit.context.psf_operator is None else fit.context.psf_operator.metadata,
        "noise": fit.noise.metadata,
    }
    metadata = {
        **fit_metadata,
        **runtime_provenance(),
        "parameters": fit.parameters.to_schema(),
        "evidence": {"reported": False, "likelihood_normalization": fit_metadata["noise"]["normalization"]},
        "package_version": __version__,
        "sampler_version": version("emcee" if sampler == "emcee" else "nautilus-sampler"),
        "seed_streams": {name: child.state for name, child in streams.items()},
        "sampling_options": options,
        "sampler_state": state,
        "optimization": optimization,
        "likelihood_path": "cached_linear" if density.design is not None else "render_per_draw",
    }
    return FitResult(sampler=sampler, free_names=fit.parameters.free_names, metadata=metadata, **result)


def summarize_model_results(result: FitResult) -> dict:
    """Return 16th, 50th and 84th percentiles of physical free parameters.

    Parameters
    ----------
    result : FitResult
        Uses post-burn-in emcee draws or all weighted Nautilus draws.

    Returns
    -------
    dict
        Per-name percentiles. Periodic quantiles are unwrapped around the
        mean direction and mapped back to the canonical interval. The
        resultant length describes angular concentration.

    Raises
    ------
    ValueError
        If the result has no posterior draws.
    """
    samples = result.physical_samples[result.burn_in:].reshape(-1, len(result.free_names))
    if not len(samples):
        raise ValueError("result has no posterior draws")
    weights = None if result.log_weights is None else np.exp(result.log_weights)
    definitions = {d["name"]: d for d in result.metadata["parameters"]["parameters"]}
    summaries = {}
    for index, name in enumerate(result.free_names):
        values = samples[:, index]
        period, origin = definitions[name]["period"], definitions[name]["origin"]
        stats = {}
        if period is not None:
            phase = math.tau * (values - origin) / period
            direction = np.average(np.exp(1j * phase), weights=weights)
            center = origin + np.angle(direction) * period / math.tau
            values = center + (values - center + period / 2) % period - period / 2
            stats.update(period=period, resultant_length=float(abs(direction)))
        if weights is None:
            quantiles = np.percentile(values, [16, 50, 84])
        else:
            order = np.argsort(values)
            cdf = np.cumsum(weights[order])
            quantiles = values[order[np.searchsorted(cdf, cdf[-1] * np.array([.16, .5, .84]))]]
        if period is not None:
            quantiles = origin + (quantiles - origin) % period
        summaries[name] = {**stats, **dict(zip(("q16", "median", "q84"), map(float, quantiles)))}
    return summaries
