"""Draw-level flux and degeneracy summaries for PSF+Sérsic models."""

import numpy as np
from scipy.special import logsumexp

from .fitting import PreparedModelFit
from .inference import FitResult
from .models import ConstantBackground, PointSource, SersicSource


def _check_match(fit, result):
    metadata = result.metadata
    components = [{"name": c.name, "type": type(c).__name__} for c in fit.model.components]
    operator = fit.context.psf_operator
    if (result.free_names != fit.parameters.free_names
            or metadata["parameters"] != fit.parameters.to_schema()
            or metadata["components"] != components
            or metadata["noise"] != fit.noise.metadata
            or metadata["psf"] != (None if operator is None else operator.metadata)
            or metadata.get("data_unit") != fit.provenance.get("data_unit")
            or metadata.get("input_sha256") != fit.provenance.get("input_sha256")):
        raise ValueError("posterior and prepared fit must describe the same model and inputs")


def _flux_summary(values, log_weights):
    valid = np.isfinite(values) & np.isfinite(log_weights)
    if not np.any(valid):
        return None
    values = values[valid]
    weights = log_weights[valid]
    order = np.argsort(values)
    cdf = np.cumsum(np.exp(weights[order] - logsumexp(weights)))
    quantiles = values[order[np.searchsorted(cdf, cdf[-1] * np.array([.16, .5, .84, .95]))]]
    return dict(zip(("q16", "median", "q84", "q95"), map(float, quantiles)))


def _fraction(host, point):
    total = host + point
    valid = (host >= 0) & (point >= 0) & (total > 0)
    return np.divide(host, total, where=valid, out=np.full_like(total, np.nan))


def _amplitude_fluxes(component, amplitudes, samples, fit, aperture):
    name = component.name
    flux_name = f"{name}.flux"
    shape_keys = ("dx", "dy")
    if isinstance(component, SersicSource):
        shape_keys += ("re", "q", "pa", "n")
    fixed = all(fit.parameters._roots[f"{name}.{key}"].prior is None for key in shape_keys)

    def aperture_flux(values):
        return float(component.render(values, fit.context)[aperture].sum())

    if fixed:
        values = fit.parameters.decode(samples[0])
        return aperture_flux({**values, flux_name: 1.0}) * amplitudes
    return np.array([aperture_flux(fit.parameters.decode(row)) for row in samples])


def _correlations(columns, weights):
    names = tuple(columns)
    matrix = np.column_stack(tuple(columns.values()))
    centered = matrix - weights @ matrix
    cross = (centered * weights[:, None]).T @ centered
    sd = np.sqrt(cross.diagonal())
    fixed = {i for i, name in enumerate(names) if np.all(columns[name] == columns[name][0])}
    return {
        left: {
            right: (float(np.clip(cross[i, j] / (sd[i] * sd[j]), -1, 1))
                    if i not in fixed and j not in fixed and sd[i] > 0 and sd[j] > 0 else None)
            for j, right in enumerate(names)
        }
        for i, left in enumerate(names)
    }


def summarize_decomposition(
    fit: PreparedModelFit, result: FitResult, *, point="point", host="host",
    aperture=None, resolution_radius=None, boundary_fraction=.05, boundary_mass=.25,
) -> dict:
    """Summarize PSF+Sérsic quantities from corresponding posterior draws.

    Parameters
    ----------
    fit : PreparedModelFit
        Original model, PSF, data and covariance.
    result : FitResult
        Posterior of this fit. Uses post-burn-in emcee draws or Nautilus
        weights and records the associated sampler status.
    point, host : str, optional
        Names of the point and Sérsic components.
    aperture : ndarray of bool, optional
        Selected output pixels. Defaults to the whole stamp, independently
        of the likelihood mask. True includes a pixel.
    resolution_radius : float, optional
        Caller-calibrated intrinsic semimajor resolution threshold in native
        pixels. A probability of at least one half below this threshold is
        flagged as unresolved.
    boundary_fraction : float, optional
        Fraction of each finite physical support interval defining its
        lower and upper edges.
    boundary_mass : float, optional
        Posterior probability above which either edge is flagged.

    Returns
    -------
    dict
        Quantiles of total and aperture fluxes, fractions and circularized
        radius, posterior correlations, boundary probabilities and optional
        resolution flag. Host fractions are conditional on nonnegative
        component fluxes and positive total, with their defined probability.
        Fluxes follow the PSF normalization reference. A 95th percentile
        is a posterior bound conditional on the fitted model and its priors.

    Raises
    ------
    ValueError
        If the posterior, aperture or diagnostic settings are invalid.
    """
    _check_match(fit, result)
    components = {component.name: component for component in fit.model.components}
    if not isinstance(components.get(point), PointSource) or not isinstance(components.get(host), SersicSource):
        raise ValueError("point and host must name a PointSource and a SersicSource")
    aperture = np.ones(fit.context.shape, dtype=bool) if aperture is None else np.asarray(aperture)
    if aperture.shape != fit.context.shape or aperture.dtype.kind != "b" or not np.any(aperture):
        raise ValueError("aperture must be a nonempty boolean mask with the stamp shape")
    if not 0 < boundary_fraction < .5 or not 0 < boundary_mass <= 1:
        raise ValueError("invalid boundary fraction or mass threshold")
    if resolution_radius is not None and (not np.isfinite(resolution_radius) or resolution_radius <= 0):
        raise ValueError("resolution_radius must be finite and positive")

    samples = result.sampling_samples[result.burn_in:].reshape(-1, len(result.free_names))
    if not len(samples):
        raise ValueError("result has no posterior draws")
    log_weights = (np.full(len(samples), -np.log(len(samples))) if result.log_weights is None
                   else result.log_weights - logsumexp(result.log_weights))
    weights = np.exp(log_weights)
    physical = result.physical_samples[result.burn_in:].reshape(samples.shape)
    indices = {name: i for i, name in enumerate(result.free_names)}

    def column(name):
        root = fit.parameters._roots[name]
        if root.prior is None:
            return np.full(len(samples), root.fixed)
        return physical[:, indices[root.name]]

    def probability(selected):
        return min(1.0, float(weights[selected].sum()))

    p, h = column(f"{point}.flux"), column(f"{host}.flux")
    radius, q = column(f"{host}.re"), column(f"{host}.q")
    point_aperture = _amplitude_fluxes(components[point], p, samples, fit, aperture)
    host_aperture = _amplitude_fluxes(components[host], h, samples, fit, aperture)
    fractions = {"total": _fraction(h, p), "aperture": _fraction(host_aperture, point_aperture)}
    columns = {name: column(name) for name in (
        f"{point}.flux", f"{host}.flux", f"{host}.re", f"{host}.n",
    )}
    for component in fit.model.components:
        if isinstance(component, ConstantBackground):
            name = f"{component.name}.level"
            columns[name] = column(name)
    boundaries = {}
    for name in result.free_names:
        root = fit.parameters._roots[name]
        lo, hi = root.prior._support
        if root.period is None and np.isfinite(lo) and np.isfinite(hi):
            width = (hi - lo) * boundary_fraction
            if np.isfinite(width):
                values = column(name)
                lower = probability(values <= lo + width)
                upper = probability(values >= hi - width)
                boundaries[name] = {"lower_probability": lower, "upper_probability": upper,
                                    "flagged": max(lower, upper) >= boundary_mass}
    resolution = None
    if resolution_radius is not None:
        below = probability(radius < resolution_radius)
        resolution = {"threshold_native_pixels": resolution_radius, "probability_below": below,
                      "radius_is_fixed": fit.parameters._roots[f"{host}.re"].prior is None,
                      "unresolved": below >= .5}
    draws = {"point_total_flux": p, "host_total_flux": h, "source_total_flux": p + h,
             "point_aperture_flux": point_aperture, "host_aperture_flux": host_aperture,
             "source_aperture_flux": point_aperture + host_aperture,
             "host_re": radius, "host_re_circ": radius * np.sqrt(q)}
    return {
        "sampler_status": result.status,
        "quantiles": {name: _flux_summary(values, log_weights) for name, values in draws.items()},
        "host_fraction": {
            name: {"quantiles": _flux_summary(values, log_weights),
                   "defined_probability": probability(np.isfinite(values))}
            for name, values in fractions.items()
        },
        "correlations": _correlations(columns, weights),
        "prior_boundary": {"interval_fraction": boundary_fraction,
                           "mass_threshold": boundary_mass, "parameters": boundaries},
        "resolution": resolution,
        "near_circular_q_threshold": .95,
        "near_circular_probability": probability(q >= .95),
        "axis_ratio_is_fixed": fit.parameters._roots[f"{host}.q"].prior is None,
        "aperture_pixels": int(aperture.sum()),
        "data_unit": fit.provenance.get("data_unit"),
        "flux_normalization_reference": fit.context.psf_operator.spec.normalization_reference,
    }
