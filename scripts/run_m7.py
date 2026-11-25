"""Run the predeclared synthetic M7 validation from a checkout or source distribution.

Usage: python -m scripts.run_m7
"""

from argparse import ArgumentParser
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import ndtr
from scipy.stats import binomtest, chisquare, truncnorm

from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet, PointSource,
    PSFOperator, RenderContext, SersicProfile, SersicSource, Uniform,
    estimate_cov_kernel, evaluate_model_fit, prepare_model_fitting, run_model_mcmc,
)
from jwst_psfmc.inference import _ModelDensity
from jwst_psfmc._provenance import runtime_provenance
from tests.test_rendering import analytic_gaussian, continuous_gaussian_convolution, gaussian_spec


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/validation/m7"
AMPLITUDES = ("point.flux", "host.flux", "sky.level")
BOUNDS = {}
RUN = {}
ROWS = []
RANKS = []
FALLBACKS = []


def _rss_mib():
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def _streams(family, case, trial, root):
    parent = np.random.SeedSequence(root, spawn_key=(family, case, trial))
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in parent.spawn(3))


def _record(case, trial, family, seed_noise, seed_sampler, elapsed, **details):
    ROWS.append(dict(
        case_id=case, realization_id=trial, test_family=family,
        baseline_commit=RUN["baseline_commit"], code_commit=RUN["code_commit"],
        prior_id="uniform-physical-v1", psf_id="gaussian-v1", renderer_id="m4-16x",
        seed_noise=seed_noise, seed_sampler=seed_sampler, runtime_s=elapsed,
        peak_memory_mb=_rss_mib(), **details,
    ))


def _kernel(config):
    r = config["finite_neighbor_corr"]
    return np.array([[0., r, 0.], [r, 1., r], [0., r, 0.]])


def _correlation(shape, config):
    coordinates = np.array(list(np.ndindex(shape)))
    adjacent = abs(coordinates[:, None] - coordinates[None, :]).sum(axis=2) == 1
    return np.eye(len(coordinates)) + config["finite_neighbor_corr"] * adjacent


def _context(shape, config, kind="effective"):
    p = config["psf"]
    spec = gaussian_spec(kind, spacing=p["spacing"], extent=p["half_width"],
                         sx=p["sigma_x"], sy=p["sigma_y"])
    return RenderContext(tuple(shape), psf_operator=PSFOperator(
        spec, oversamp=p["source_oversamp"], max_offset=p["max_offset"]
    ))


def _fixed_profile(config, n, re, flux=1):
    return SersicProfile(flux, re, n, config["host_q"], config["host_pa"])


def _numeric(config):
    results = []
    for case in (dict(zip(config["numeric_fields"], row)) for row in config["numeric"]):
        context = _context(config["stamp_shape"], config, case["kind"])
        op = context.psf_operator
        profile = SersicProfile(1, case["re"], case["n"], case["q"], case["pa"])
        dx, dy = case["dx"], case["dy"]
        t0 = perf_counter()
        image = op.render_sersic(profile, dx, dy, context) + case["contrast"] * op.render_point(dx, dy, context)
        elapsed = perf_counter() - t0
        expected_host = continuous_gaussian_convolution(profile, context, dx, dy, angles=512)
        refined = continuous_gaussian_convolution(profile, context, dx, dy, angles=1024)
        p = config["psf"]
        point = analytic_gaussian(context, dx, dy, p["sigma_x"], p["sigma_y"])
        expected = expected_host + case["contrast"] * point
        sigma = np.linalg.norm(expected_host) / case["host_sn"]
        epsilon = float(np.linalg.norm((image - expected) / sigma))
        reference_error = float(np.linalg.norm((refined - expected_host) / sigma))
        passed = (epsilon <= config["criteria"]["numeric_epsilon_max"]
                  and reference_error <= config["criteria"]["reference_epsilon_max"])
        results.append({**case, "epsilon": epsilon, "reference_epsilon": reference_error,
                        "stamp_flux_error": float(image.sum()-expected.sum()),
                        "runtime_s": elapsed, "psf_sha256": op.spec.metadata["samples_sha256"],
                        "passes_budget": passed})
        _record(case["id"], 0, "numeric", None, None, elapsed, parameter_name="image",
                status="pass" if passed else "outside_budget", render_error=epsilon,
                noise_backend="diagonal", boundary_mode="independent", mask_id="none", kernel_id="none")
    return results


def _production_fit(config, context, data, model, re, n, noise_options):
    dx, dy = config["centroid_xy"]
    components, definitions = [], []
    for name, component in (("point", PointSource), ("host", SersicSource)):
        if model not in (name, "composite"):
            continue
        components.append(component(name))
        key = f"{name}.flux"
        lo, hi = BOUNDS[key]
        definitions += [Parameter(key, (lo + hi)/2, prior=Uniform(lo, hi)),
                        Parameter(f"{name}.dx", fixed=dx), Parameter(f"{name}.dy", fixed=dy)]
    if model in ("host", "composite"):
        definitions += [Parameter("host.re", fixed=re), Parameter("host.n", fixed=n),
                        Parameter("host.q", fixed=config["host_q"]),
                        Parameter("host.pa", fixed=config["host_pa"])]
    components.append(ConstantBackground("sky"))
    limits = BOUNDS["sky.level"]
    definitions.append(Parameter("sky.level", 0., prior=Uniform(*limits)))
    return prepare_model_fitting(data, CompositeModel(components), ParameterSet(definitions),
                                 context=context, noise_options=noise_options)


def _linear_design(fit):
    density = _ModelDensity(fit)
    density.prepare()
    design = density.design
    precision = design.T @ design
    covariance = np.linalg.solve(precision, np.eye(len(precision)))
    factor = np.linalg.cholesky(covariance)
    lo, hi = np.array([BOUNDS[name] for name in fit.parameters.free_names]).T
    initial = fit.parameters.decode(fit.parameters.initial)
    aperture = {
        name: float(component.render({**initial, name: 1.}, fit.context).sum())
        for name, component in zip(fit.parameters.free_names, fit.model.components)
        if name.endswith(".flux")
    }
    return density, design, covariance, factor, lo, hi, aperture


def _posterior(prepared, white, rng, count, realization):
    _, design, covariance, factor, lo, hi, _ = prepared
    mean = covariance @ (design.T @ white)
    accepted = []
    remaining = count
    for _ in range(100):
        proposal = mean + rng.normal(size=(max(1024, 2*remaining), len(mean))) @ factor.T
        kept = proposal[np.all((lo <= proposal) & (proposal <= hi), axis=1)][:remaining]
        accepted.append(kept)
        remaining -= len(kept)
        if not remaining:
            return np.concatenate(accepted)
    # Condition one Gaussian marginal on its support before drawing the rest
    # from the Gaussian conditional. The final interval rejection is still
    # exact, and avoids wasting almost all proposals on one prohibited axis.
    FALLBACKS.append(realization)
    std = np.sqrt(covariance.diagonal())
    a, b = (lo - mean)/std, (hi - mean)/std
    marginal_mass = np.where(a > 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))
    chosen = int(marginal_mass.argmin())
    other = np.arange(len(mean)) != chosen
    beta = covariance[other, chosen] / covariance[chosen, chosen]
    conditional = covariance[np.ix_(other, other)] - np.outer(beta, covariance[other, chosen])
    conditional_factor = np.linalg.cholesky(conditional)
    for _ in range(100):
        n = max(1024, 2*remaining)
        proposal = np.empty((n, len(mean)))
        proposal[:, chosen] = truncnorm.rvs(a[chosen], b[chosen], loc=mean[chosen],
                                          scale=std[chosen], size=n, random_state=rng)
        proposal[:, other] = (
            mean[other] + (proposal[:, chosen] - mean[chosen])[:, None] * beta
            + rng.normal(size=(n, other.sum())) @ conditional_factor.T
        )
        kept = proposal[np.all((lo <= proposal) & (proposal <= hi), axis=1)][:remaining]
        accepted.append(kept)
        remaining -= len(kept)
        if not remaining:
            return np.concatenate(accepted)
    raise RuntimeError("both exact posterior rejection budgets exhausted")


def _trial_summaries(case, trial, family, fit, posterior, truth, white, setup, seeds, elapsed, mask_id="none"):
    _, design, _, _, _, _, aperture = setup
    names = fit.parameters.free_names
    columns = {name: posterior[:, i] for i, name in enumerate(names)}
    values = dict(zip(names, truth))
    for name, amplitude in aperture.items():
        columns[name.replace(".flux", ".aperture_flux")] = columns[name] * amplitude
        values[name.replace(".flux", ".aperture_flux")] = values[name] * amplitude
    columns["log_likelihood"] = -.5 * np.sum((white - posterior @ design.T)**2, axis=1)
    values["log_likelihood"] = -.5 * np.sum((white - design @ truth)**2)
    noise = fit.noise.metadata
    for name, column in columns.items():
        lo, q16, median, q84, hi, q95 = np.quantile(column, [.025, .16, .5, .84, .975, .95])
        sd = float(column.std(ddof=1))
        _record(case, trial, family, seeds[1], seeds[2], elapsed, parameter_name=name,
                truth=float(values[name]), median=float(median), q16=float(q16), q84=float(q84),
                q025=float(lo), q975=float(hi), q95=float(q95), posterior_sd=sd,
                ess=len(posterior), mcse=sd/np.sqrt(len(posterior)), status="exact_independent",
                noise_backend=noise["backend"], boundary_mode=noise["boundary"],
                mask_id=mask_id, kernel_id="stationary-neighbors" if noise["backend"] != "diagonal" else "none")
        if family == "sbc":
            RANKS.append(dict(case_id=case, realization_id=trial, parameter_name=name,
                              rank=int(np.sum(column < values[name])), draws=len(column), seed_truth=seeds[0]))


def _inference(config, context):
    results = []
    cache = {}
    dx, dy = config["centroid_xy"]
    shape = context.shape
    for case_index, row in enumerate(config["inference"]):
        case = dict(zip(config["inference_fields"], row))
        model, re, n = case["model"], case["re"], case["n"]
        host = context.psf_operator.render_sersic(_fixed_profile(config, n, re), dx, dy, context)
        point = context.psf_operator.render_point(dx, dy, context)
        signal = point if model == "point" else host
        ratio = _correlation(shape, config) if case["noise"] == "finite" else np.eye(signal.size)
        sigma = np.sqrt(signal.ravel() @ np.linalg.solve(ratio, signal.ravel())) / case["host_sn"]
        covariance = ratio * sigma**2
        chol = np.linalg.cholesky(covariance)
        noise_options = ({"backend": "dense", "covariance": covariance} if case["noise"] == "finite"
                         else {"err": np.full(shape, sigma)})
        zeros = np.zeros(shape)
        fit = _production_fit(config, context, zeros, model, re, n, noise_options)
        setup = _linear_design(fit)
        design = setup[1]
        template = dict(zip(AMPLITUDES, (point, host, np.ones(shape))))
        x = np.column_stack([template[name].ravel() for name in fit.parameters.free_names])
        reference_precision = x.T @ np.linalg.solve(covariance, x)
        np.testing.assert_allclose(design.T @ design, reference_precision, rtol=2e-12)
        names = fit.parameters.free_names
        truth_fixed = np.array([dict(zip(AMPLITUDES, case["truth"]))[name] for name in names])
        for family_id, family in ((1, "sbc"), (2, "fixed_truth")):
            if family == "sbc" and case["group"] != "core":
                continue
            for trial in range(config[f"{family}_trials"]):
                started = perf_counter()
                seeds = _streams(family_id, case_index, trial, config["root_seed"])
                truth = (np.random.default_rng(seeds[0]).uniform(setup[4], setup[5])
                         if family == "sbc" else truth_fixed)
                data = (x @ truth + chol @ np.random.default_rng(seeds[1]).standard_normal(signal.size)).reshape(shape)
                white = fit.noise.whiten(data)
                try:
                    posterior = _posterior(setup, white, np.random.default_rng(seeds[2]),
                                           config["independent_posterior_draws"], (case["id"], family, trial))
                except RuntimeError as exc:
                    _record(case["id"], trial, family, seeds[1], seeds[2], perf_counter()-started,
                            parameter_name="*", status="numerical_failure", failure_reason=str(exc))
                    continue
                _trial_summaries(case["id"], trial, family, fit, posterior, truth, white, setup,
                                 seeds, perf_counter()-started)
                if trial == 0 and family == "fixed_truth" and case["group"] == "core":
                    observed = _production_fit(config, context, data, model, re, n, noise_options)
                    score = evaluate_model_fit(observed, truth)["log_likelihood"]
                    residual = data.ravel() - x @ truth
                    sign, determinant = np.linalg.slogdet(covariance)
                    assert sign == 1
                    oracle = -.5*(residual @ np.linalg.solve(covariance, residual)
                                  + determinant + signal.size * np.log(2*np.pi))
                    np.testing.assert_allclose(score, oracle, rtol=0, atol=2e-10)
                    cache[case["id"]] = (observed, posterior, seeds, setup, white)
        results.append({**case, "sigma": float(sigma), "noise": fit.noise.metadata,
                        "parameters": fit.parameters.to_schema(),
                        "render_config": context.psf_operator.metadata,
                        "psf_fwhm_geomean_native": float(np.sqrt(config["psf"]["sigma_x"] * config["psf"]["sigma_y"])
                                                         * np.sqrt(8*np.log(2)))})
    return results, cache


def _robustness(config, context):
    shape = context.shape
    dx, dy = config["centroid_xy"]
    n, re = .5, 1.4
    psf = config["psf"]
    stress = config["stress_settings"]
    sx, sy = psf["sigma_x"], psf["sigma_y"]
    pflux = stress["pure_point_flux"]
    op = context.psf_operator
    host = op.render_sersic(_fixed_profile(config, n, re), dx, dy, context)
    true_unit = analytic_gaussian(context, dx, dy, sx, sy)
    y, x = np.indices(shape)
    ring = np.hypot(x-context.center[0]-dx, y-context.center[1]-dy)
    ctrue = _correlation(shape, config) * .012**2
    chol = np.linalg.cholesky(ctrue)
    results = []
    gradient = stress["sky_gradient_per_x_pixel"]*(x-context.center[0])
    wing = stress["wing_width_factor"]
    designs = {
        "matched": pflux * true_unit,
        "wider_psf": pflux*analytic_gaussian(context, dx, dy, sx+stress["width_increment_per_axis"],
                                             sy+stress["width_increment_per_axis"]),
        "psf_wings": pflux*((1-stress["wing_fraction"])*true_unit + stress["wing_fraction"]
                           * analytic_gaussian(context, dx, dy, sx*wing, sy*wing)),
        "psf_ellipticity": pflux*analytic_gaussian(context, dx, dy, sx+stress["ellipticity_width_shift"],
                                                  sy-stress["ellipticity_width_shift"]),
        "psf_centroid": pflux*analytic_gaussian(context, dx+stress["centroid_x_shift"], dy, sx, sy),
        "background_gradient": pflux*true_unit + gradient,
        "neighbor": pflux*true_unit + stress["neighbor_flux"]
                    * analytic_gaussian(context, *stress["neighbor_offset_xy"], sx, sy),
        "non_sersic_host": pflux*true_unit + .6*host + .4 * analytic_gaussian(context, -.8, .5, sx, sy),
    }
    masks = {
        "mask_core": ring < .8, "mask_wings": (ring > 1.8) & (ring < 2.3),
        "mask_edge": x == 0, "mask_cluster": (x >= 1) & (x <= 2) & (y >= 1) & (y <= 2),
    }
    for case, kind in config["robustness"]:
        image = designs.get(kind, pflux * true_unit + host) + .02
        injected_host = 1. if kind not in designs or kind == "non_sersic_host" else 0.
        mask = masks.get(kind)
        options = {"backend": "dense", "covariance": ctrue, "mask": mask}
        if kind == "finite_as_periodic":
            options = {"backend": "fft_periodic", "err": np.full(shape, .012), "kernel": _kernel(config)}
        elif kind == "err_scale":
            options["covariance"] = ctrue * stress["err_scale"]**2
        fit = _production_fit(config, context, np.zeros(shape), "composite", re, n, options)
        setup = _linear_design(fit)
        reference = _production_fit(config, context, np.zeros(shape), "composite", re, n,
                                    {"backend": "dense", "covariance": ctrue, "mask": mask})
        c0 = _linear_design(reference)[2]
        results.append({"id": case, "kind": kind, "noise": fit.noise.metadata,
                        "effective_sd_ratio_point_host_sky": np.sqrt(setup[2].diagonal()/c0.diagonal()).tolist(),
                        "observed_pixels": fit.noise.n_valid})
        for trial in range(config["stress_trials"]):
            started = perf_counter()
            # Paired noise across strategies makes the sensitivity difference interpretable.
            seeds = _streams(3, 0, trial, config["root_seed"])
            data = image + (chol @ np.random.default_rng(seeds[1]).normal(size=image.size)).reshape(shape)
            white = fit.noise.whiten(data)
            try:
                posterior = _posterior(setup, white, np.random.default_rng(seeds[2]),
                                       config["independent_posterior_draws"], (case, "stress", trial))
            except RuntimeError as exc:
                _record(case, trial, "stress", seeds[1], seeds[2], perf_counter()-started,
                        parameter_name="*", status="numerical_failure", failure_reason=str(exc))
                continue
            _trial_summaries(case, trial, "stress", fit, posterior, np.array([pflux, injected_host, .02]),
                             white, setup, seeds, perf_counter()-started, kind if mask is not None else "none")
    return results


def _interval(k, n):
    result = binomtest(int(k), int(n)).proportion_ci(confidence_level=.95, method="wilson")
    return [float(result.low), float(result.high)]


def _radius_sbc(config, context):
    settings = config["radius_sbc"]
    grid = np.linspace(*settings["re_prior"], settings["grid_points"])
    covariance = _correlation(context.shape, config)*settings["sigma"]**2
    options = {"backend": "dense", "covariance": covariance}
    template = _production_fit(config, context, np.zeros(context.shape), "host", 1.4, settings["n"], options)
    values = template.parameters.decode(template.parameters.initial)
    values.update({"host.flux": 1., "sky.level": .02})

    def pixels(radius):
        return template.model.render({**values, "host.re": radius}, context)

    images = np.asarray([pixels(radius) for radius in grid])
    white_grid = np.asarray([template.noise.whiten(image) for image in images])
    factor = np.linalg.cholesky(covariance)
    y, x = np.indices(context.shape)
    cx, cy = context.center
    aperture = np.hypot(x - cx, y - cy) < 2
    aperture_curve = (images[:, aperture] - .02).sum(axis=1)
    refinement = []
    draws = config["independent_posterior_draws"]
    for trial in range(config["sbc_trials"]):
        started = perf_counter()
        seeds = _streams(6, 0, trial, config["root_seed"])
        truth = np.random.default_rng(seeds[0]).uniform(*settings["re_prior"])
        signal = pixels(truth)
        data = signal + (factor @ np.random.default_rng(seeds[1]).normal(size=factor.shape[0])).reshape(context.shape)
        obs = template.noise.whiten(data)
        logp = -.5 * np.sum((white_grid-obs)**2, axis=1)
        p = np.exp(logp-logp.max())
        cdf = cumulative_trapezoid(p, grid, initial=0)
        cdf /= cdf[-1]
        if trial < 8:
            coarse = cumulative_trapezoid(p[::2], grid[::2], initial=0)
            refinement.append(float(np.max(abs(cdf[::2]-coarse/coarse[-1]))))
        posterior = np.interp(np.random.default_rng(seeds[2]).random(draws), cdf, grid)
        entries = {"host.re": (posterior, truth),
                   "host.re_circ": (posterior*np.sqrt(config["host_q"]), truth*np.sqrt(config["host_q"])),
                   "host.aperture_flux": (np.interp(posterior, grid, aperture_curve),
                                          float((signal-.02)[aperture].sum()))}
        for name, (samples, expected) in entries.items():
            lo, q16, median, q84, hi, q95 = np.quantile(samples, [.025, .16, .5, .84, .975, .95])
            sd = float(samples.std(ddof=1))
            _record(settings["id"], trial, "sbc", seeds[1], seeds[2], perf_counter()-started,
                    parameter_name=name, truth=float(expected), median=float(median),
                    q16=float(q16), q84=float(q84), q025=float(lo), q975=float(hi), q95=float(q95),
                    posterior_sd=sd, ess=draws, mcse=sd/np.sqrt(draws), status="grid_quadrature",
                    noise_backend="dense", boundary_mode="explicit", mask_id="none",
                    kernel_id="stationary-neighbors")
            RANKS.append(dict(case_id=settings["id"], realization_id=trial, parameter_name=name,
                              rank=int(np.sum(samples < expected)), draws=draws, seed_truth=seeds[0]))
    return dict(**settings, spot_checks=len(refinement), max_cdf_difference=max(refinement),
                passes_grid_refinement=max(refinement) <= settings["cdf_refinement_max"])


def _aggregate(config):
    sbc = []
    groups = sorted({(row["case_id"], row["parameter_name"]) for row in RANKS})
    alpha = config["criteria"]["sbc_familywise_alpha"] / max(1, len(groups))
    nbins = config["criteria"]["sbc_histogram_bins"]
    counts_expected = np.bincount(np.arange(config["independent_posterior_draws"] + 1) * nbins
                                  // (config["independent_posterior_draws"] + 1))
    for case, param in groups:
        ranks = [r["rank"] for r in RANKS if (r["case_id"], r["parameter_name"]) == (case, param)]
        hist = np.bincount(np.asarray(ranks) * nbins // (config["independent_posterior_draws"] + 1), minlength=nbins)
        pvalue = float(chisquare(hist, len(ranks)*counts_expected/counts_expected.sum()).pvalue)
        sbc.append(dict(case=case, parameter=param, histogram=hist.tolist(), n=len(ranks),
                        pvalue=pvalue, threshold=alpha, passes=pvalue >= alpha))
    metrics = []
    cases = [(family, case) for family in ("fixed_truth", "stress")
             for case in sorted({r["case_id"] for r in ROWS if r["test_family"] == family})]
    primary = {"b01": "point.flux", "b02": "host.flux", "b03": "host.flux"}
    alpha = config["criteria"]["coverage_familywise_alpha"] / (len(primary)*2)
    for family, case in cases:
        chosen = primary.get(case, "host.flux")
        data = [r for r in ROWS if r["test_family"] == family and r["case_id"] == case
                and r["parameter_name"] == chosen]
        if not data:
            metrics.append(dict(case=case, family=family, successful=0))
            continue
        n = len(data)
        v = lambda name: np.array([r[name] for r in data])
        k68 = int(((v("q16") <= v("truth")) & (v("truth") <= v("q84"))).sum())
        k95 = int(((v("q025") <= v("truth")) & (v("truth") <= v("q975"))).sum())
        bias = float(np.mean(v("median") - v("truth")))
        standardized = bias / float(np.median(v("posterior_sd")))
        attempts = config["fixed_truth_trials"] if family == "fixed_truth" else config["stress_trials"]
        values = dict(case=case, family=family, successful=n, attempts=attempts,
                      mean_bias=bias, bias_over_posterior_sd=standardized,
                      coverage68=k68/n, ci68=_interval(k68, n), coverage95=k95/n, ci95=_interval(k95, n),
                      median_width68=float(np.median(v("q84")-v("q16"))),
                      median_q95=float(np.median(v("q95"))))
        if case in primary:
            values["passes"] = bool(n == attempts
                and abs(standardized) <= config["criteria"]["population_abs_mean_bias_over_median_sd"]
                and binomtest(k68, n, .68).pvalue >= alpha
                and binomtest(k95, n, .95).pvalue >= alpha)
        metrics.append(values)
    return sbc, metrics


def _sampler_checks(config, cache):
    results = []
    for case, (fit, reference, seeds, _, _) in cache.items():
        for backend in (("emcee", "nautilus") if case == "b03" else ("emcee",)):
            limits = ({"nsteps": 8000, "burn_in": 1000} if backend == "emcee"
                      else {"n_live": 110, "n_eff": 1800, "n_like_max": 22000})
            try:
                result = run_model_mcmc(fit, sampler=backend, seed=seeds[2], sampling_options=limits)
            except ImportError as exc:
                results.append(dict(case=case, sampler=backend, status="optional_unavailable", reason=str(exc)))
                continue
            diagnostics = result.diagnostics
            if not result.physical_samples.size:
                results.append(dict(case=case, sampler=backend, status=result.status, diagnostics=diagnostics))
                continue
            fit_samples = result.physical_samples[result.burn_in:].reshape(-1, fit.parameters.ndim)
            weights = None if result.log_weights is None else np.exp(result.log_weights)
            estimate = np.average(fit_samples, axis=0, weights=weights)
            mcse = np.asarray([np.nan if x is None else x for x in diagnostics["mcse"]])
            score = abs(estimate-reference.mean(axis=0))/np.hypot(mcse, reference.std(axis=0)/np.sqrt(len(reference)))
            results.append(dict(case=case, sampler=backend, status=result.status,
                                max_combined_mcse=float(score.max()) if np.all(np.isfinite(score)) else None,
                                diagnostics=diagnostics))
    return results


def _estimated_kernels(config, context):
    # Sky fields are generated on a much larger periodic grid. The imported
    # cutouts and evaluation stamp have nonperiodic finite boundaries.
    side = 65
    grid = np.zeros((side, side))
    taps = _kernel(config)
    grid[np.ix_([side-1, 0, 1], [side-1, 0, 1])] = taps
    root_power = np.sqrt(np.fft.fft2(grid).real)
    shape = context.shape
    true_covariance = _correlation(shape, config)*.012**2
    fit = _production_fit(config, context, np.zeros(shape), "composite", 1.4, .5,
                          {"backend": "dense", "covariance": true_covariance})
    truth_sd = np.sqrt(_linear_design(fit)[2].diagonal())
    results = []
    for trial in range(config["estimated_kernel_trials"]):
        seeds = _streams(4, 0, trial, config["root_seed"])
        large = np.random.default_rng(seeds[1]).normal(size=grid.shape)
        field = np.fft.ifft2(np.fft.fft2(large)*root_power).real
        kernel = estimate_cov_kernel(field[16:49, 16:49], size=7, r_in=1, r_out=3)
        for backend in ("dense", "fft_periodic"):
            started = perf_counter()
            options = {"backend": backend, "err": np.full(shape, .012), "kernel": kernel}
            if backend == "dense":
                options["boundary"] = "finite"
            identity = f"estimated-{trial}"
            try:
                fit = _production_fit(config, context, np.zeros(shape), "composite", 1.4, .5, options)
                values = np.sqrt(_linear_design(fit)[2].diagonal()) / truth_sd
            except ValueError as exc:
                failure = dict(realization=trial, backend=backend, status="rejected", reason=str(exc))
                results.append(failure)
                _record("c15", trial, "estimated_kernel", seeds[1], seeds[2], perf_counter()-started,
                        parameter_name="*", status="rejected", failure_reason=str(exc),
                        noise_backend=backend, boundary_mode=options.get("boundary", "periodic"),
                        kernel_id=identity, mask_id="none")
                continue
            results.append(dict(realization=trial, backend=backend, status="accepted",
                                ratios=dict(zip(fit.parameters.free_names, values.tolist())),
                                spectrum=fit.noise.metadata.get("spectrum")))
            for name, value in zip(fit.parameters.free_names, values):
                _record("c15", trial, "estimated_kernel", seeds[1], seeds[2], perf_counter()-started,
                        parameter_name=name, status="accepted", uncertainty_ratio=float(value),
                        noise_backend=backend, boundary_mode=fit.noise.metadata["boundary"],
                        kernel_id=identity, mask_id="none")
    return results


def _predictive(config, cache):
    fit, posterior, _, setup, white = cache["b03"]
    _, design, _, _, _, _, _ = setup
    shape = fit.context.shape
    factor = fit.noise._factor
    sigma = np.linalg.norm(factor, axis=1)
    y, x = np.indices(shape)
    dx, dy = config["centroid_xy"]
    radii = np.hypot(x - fit.context.center[0] - dx, y - fit.context.center[1] - dy)
    regions = ((0, 1), (1, 2), (2, 3), (3, 6))
    names = ["whitened_q", "standardized_horizontal_lag1", "standardized_vertical_lag1"]
    names += [f"radial_mean_{lo}_{hi}" for lo, hi in regions]

    def discrepancy(whitened):
        z = ((factor @ whitened) / sigma).reshape(shape)
        centered = z - z.mean()
        variance = np.mean(centered**2)
        return [whitened @ whitened,
                np.mean(centered[:, 1:]*centered[:, :-1])/variance,
                np.mean(centered[1:]*centered[:-1])/variance] + [
                    z[(radii >= lo) & (radii < hi)].mean() for lo, hi in regions
                ]

    seeds = _streams(5, 0, 0, config["root_seed"])
    rng = np.random.default_rng(seeds[1])
    count = config["posterior_predictive_draws"]
    draws = posterior[rng.choice(len(posterior), size=count, replace=False)]
    observed = np.asarray([discrepancy(white - design @ draw) for draw in draws])
    replicated = np.asarray([discrepancy(rng.standard_normal(white.size)) for _ in draws])
    return dict(seed=seeds[1], draws=count, case="b03",
                discrepancies={name: {
                    "observed_median": float(np.median(observed[:, i])),
                    "replicated_q05_q95": np.quantile(replicated[:, i], [.05, .95]).tolist(),
                    "paired_tail_fraction": float(np.mean(replicated[:, i] >= observed[:, i])),
                } for i, name in enumerate(names)})


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(config, numeric, metrics, sbc, cache, path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
    ax = axes[0,0]
    ax.scatter([r["runtime_s"]*1000 for r in numeric], [r["epsilon"] for r in numeric],
               c=["#237c86" if r["group"] == "core" else "#b55340" for r in numeric])
    ax.axhline(config["criteria"]["numeric_epsilon_max"], ls="--", c="0.3")
    ax.set(xlabel="Render time (ms)", ylabel="Noise-weighted error", yscale="log", title="Independent rendering")
    ax = axes[0,1]
    for row in sbc:
        ax.plot(np.arange(len(row["histogram"])), row["histogram"], alpha=.48)
    ax.axhline(config["sbc_trials"]/config["criteria"]["sbc_histogram_bins"], ls="--", c="0.3")
    ax.set(xlabel="Rank bin", ylabel="Trials", title="SBC ranks by observable")
    ax = axes[1,0]
    core = [r for r in metrics if r["family"] == "fixed_truth" and "coverage68" in r]
    for dx, target in ((-.08, "68"), (.08, "95")):
        for index, row in enumerate(core):
            p = row[f"coverage{target}"]
            lo, hi = row[f"ci{target}"]
            ax.errorbar(index+dx, p, yerr=[[max(0, p-lo)], [max(0, hi-p)]], fmt="o")
    ax.axhline(.68, ls="--", c=".6")
    ax.axhline(.95, ls="--", c=".6")
    ax.set(xticks=range(len(core)), xticklabels=[r["case"] for r in core],
           ylabel="Coverage (95% Wilson CI)", ylim=(0,1.05), title="Fixed truth")
    ax2 = ax.twinx()
    ax2.plot(range(len(core)), [r["bias_over_posterior_sd"] for r in core], "k+", label="signed bias")
    ax2.set_ylabel("Mean bias / median posterior SD")
    ax = axes[1,1]
    stress = [r for r in metrics if r["family"] == "stress" and "mean_bias" in r]
    ax.bar([r["case"] for r in stress], [r["mean_bias"] for r in stress], color="#687da1")
    ax.axhline(0, c=".3")
    ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="Host flux bias", title="Paired synthetic mismatches")
    if "b03" in cache:
        fit, reference, _, _, _ = cache["b03"]
        ax = axes[2,0]
        ax.hist(reference[:, 1], bins=27, color="#237c86")
        ax.set(xlabel="Host flux", ylabel="Independent draws", title="Resolved host posterior (b03)")
        residual = evaluate_model_fit(fit, np.median(reference, axis=0), decompose=True)["standardized_residual"]
        ax = axes[2,1]
        shown = ax.imshow(residual, origin="lower", cmap="coolwarm", vmin=-3, vmax=3)
        ax.set(title="Standardized residual (b03)")
        fig.colorbar(shown, ax=ax, shrink=.68)
    else:
        for ax in axes[2]:
            ax.text(.5, .5, "b03 realization unavailable", ha="center", transform=ax.transAxes)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _acceptance(config, numeric, sbc, metrics, samplers, radius_sbc, rows, ranks):
    checks = []

    def record(family, case, passed, **details):
        checks.append(dict(family=family, case=case, passes=bool(passed), **details))

    criteria = config["criteria"]
    numeric_rows = {row["id"]: row for row in numeric}
    expected_numeric = [dict(zip(config["numeric_fields"], row)) for row in config["numeric"]]
    for case in expected_numeric:
        if case["group"] != "core":
            continue
        value = numeric_rows.get(case["id"], {})
        epsilon, ref = value.get("epsilon"), value.get("reference_epsilon")
        record("numeric", case["id"], epsilon is not None and ref is not None
               and epsilon <= criteria["numeric_epsilon_max"] and ref <= criteria["reference_epsilon_max"],
               completed=int(bool(value)), expected=1, epsilon=epsilon, reference_epsilon=ref)

    cases = [dict(zip(config["inference_fields"], row)) for row in config["inference"]]
    core = {case["id"]: case["model"] for case in cases if case["group"] == "core"}
    expected_sbc = {}
    for case, model in core.items():
        components = ("point", "host") if model == "composite" else (model,)
        expected_sbc[case] = {"log_likelihood", "sky.level"} | {
            f"{name}.{field}" for name in components for field in ("flux", "aperture_flux")
        }
    radius_id = config["radius_sbc"]["id"]
    expected_sbc[radius_id] = {"host.re", "host.re_circ", "host.aperture_flux"}
    sbc_lookup = {(row["case"], row["parameter"]): row for row in sbc}
    for case, names in expected_sbc.items():
        for name in sorted(names):
            value = sbc_lookup.get((case, name), {})
            ids = [r["realization_id"] for r in ranks
                   if r["case_id"] == case and r["parameter_name"] == name
                   and r["draws"] == config["independent_posterior_draws"]]
            expected = config["sbc_trials"]
            record("sbc", case, value.get("passes", False)
                   and sorted(ids) == list(range(expected)), parameter=name,
                   completed=len(set(ids)), expected=expected,
                   pvalue=value.get("pvalue"), threshold=value.get("threshold"))
    record("radius_refinement", radius_id, radius_sbc["passes_grid_refinement"],
           max_cdf_difference=radius_sbc["max_cdf_difference"])

    fixed = {row["case"]: row for row in metrics if row["family"] == "fixed_truth"}
    primary = {"point": "point.flux", "host": "host.flux", "composite": "host.flux"}
    expected_samplers = {(case, "emcee") for case in core}
    if "b03" in core:
        expected_samplers.add(("b03", "nautilus"))
    for case, model in core.items():
        value = fixed.get(case, {})
        ids = [r["realization_id"] for r in rows if r["case_id"] == case
               and r["test_family"] == "fixed_truth" and r["parameter_name"] == primary[model]
               and r["status"] == "exact_independent"]
        expected = config["fixed_truth_trials"]
        record("fixed_truth", case, value.get("passes", False)
               and sorted(ids) == list(range(expected)), completed=len(set(ids)), expected=expected,
               coverage68=value.get("coverage68"), coverage95=value.get("coverage95"),
               standardized_bias=value.get("bias_over_posterior_sd"))
    reported = {(row["case"], row["sampler"]): row for row in samplers}
    threshold = criteria["sampler_parity_max_combined_mcse"]
    for case, sampler in sorted(expected_samplers):
        value = reported.get((case, sampler), {})
        score = value.get("max_combined_mcse")
        record("sampler_parity", case, value.get("status") == "converged" and score is not None
               and score <= threshold, sampler=sampler, status=value.get("status", "missing"),
               max_combined_mcse=score, threshold=threshold)
    return {"passes": all(check["passes"] for check in checks), "checks": checks}


def _write_manifest(path, manifest):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _run(design, out):
    for state in (BOUNDS, RUN, ROWS, RANKS, FALLBACKS):
        state.clear()
    design_bytes = design.read_bytes()
    config = json.loads(design_bytes)
    BOUNDS.update(config["priors"])
    started = perf_counter()
    build = runtime_provenance()["build"]
    RUN.update(
        baseline_commit="38f0834d5fdd977c9c761907d0ab59a82bc5ae90",
        code_commit=build["git_commit"],
        package_snapshot_sha256=build["source_sha256"],
        design_sha256=hashlib.sha256(design_bytes).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256((ROOT / "tests/test_rendering.py").read_bytes()).hexdigest(),
        dirty=build["git_dirty"],
        python=sys.version.split()[0], numpy=np.__version__,
        synthetic_only=True, scipy=version("scipy"), astropy=version("astropy"),
        emcee=version("emcee"),
    )
    context = _context(config["stamp_shape"], config)
    numeric = _numeric(config)
    inference, cache = _inference(config, context)
    radius_sbc = _radius_sbc(config, context)
    robustness = _robustness(config, context)
    estimated_kernels = _estimated_kernels(config, context)
    predictive = _predictive(config, cache) if "b03" in cache else {"status": "unavailable"}
    sbc, metrics = _aggregate(config)
    print("Numerical, exact-SBC and robustness batches complete; running sampler spot checks.", flush=True)
    samplers = _sampler_checks(config, cache)
    _write_csv(out / "results.csv", ROWS)
    _write_csv(out / "sbc_ranks.csv", RANKS)
    _plot(config, numeric, metrics, sbc, cache, out / "diagnostics.png")
    acceptance = _acceptance(config, numeric, sbc, metrics, samplers, radius_sbc, ROWS, RANKS)
    manifest = dict(status="completed", run=RUN, acceptance=acceptance,
                    numeric=numeric, inference=inference, robustness=robustness,
                    sbc=sbc, metrics=metrics, samplers=samplers,
                    radius_sbc=radius_sbc, conditional_rejection_fallbacks=FALLBACKS,
                    estimated_kernels=estimated_kernels, posterior_predictive=predictive,
                    failed_realizations=[row for row in ROWS if row["status"] == "numerical_failure"],
                    wall_s=perf_counter()-started, peak_memory_mb=_rss_mib())
    _write_manifest(out / "manifest.json", manifest)
    print(f"Wrote {len(ROWS)} result rows and {len(RANKS)} ranks in {manifest['wall_s']:.1f} s.")
    if not acceptance["passes"]:
        failed = [f"{check['family']}/{check['case']}" for check in acceptance["checks"] if not check["passes"]]
        print("M7 acceptance failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"M7 acceptance passed ({len(acceptance['checks'])} checks).")
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=OUT / "design.json")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    directory = os.open(out, os.O_RDONLY)
    try:
        try:
            fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"output directory is already in use: {out}")
        path = out / "manifest.json"
        pending = dict(status="in_progress", acceptance={"passes": False, "checks": []},
                       design=str(args.design))
        _write_manifest(path, pending)
        try:
            return _run(args.design, out)
        except Exception as exc:
            _write_manifest(path, dict(pending, status="error",
                                       error={"type": type(exc).__name__, "message": str(exc)}))
            raise
    finally:
        os.close(directory)


if __name__ == "__main__":
    sys.exit(main())
