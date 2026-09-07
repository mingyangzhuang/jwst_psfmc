"""
I/O utilities: save/load MCMC results.

Results are stored as compressed NumPy ``.npz`` archives that contain the
full chain, the thinned flat chain, posterior summaries, autocorrelation
times, and sampler diagnostics. A dedicated :func:`load_emcee_results`
helper unpacks the archive into a clean dictionary without requiring the
caller to know the internal key names.

Example data files (difference images, error maps, covariance kernels, and
PSF models) are distributed separately in the ``examples/`` directory of the
repository — see the README for download instructions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import emcee


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_emcee_results(
    path: str | Path,
    sampler: emcee.EnsembleSampler,
    burnin: int = 500,
    thin: int = 4,
    labels: tuple[str, ...] = ("flux", "dx", "dy", "bkg"),
    include_log_prob: bool = True,
) -> dict:
    """Save an ``emcee`` sampler to a compressed ``.npz`` archive.

    Parameters
    ----------
    path : str or Path
        Output file path (the ``.npz`` extension is appended automatically
        by ``numpy.savez_compressed`` if absent).
    sampler : emcee.EnsembleSampler
        Completed sampler.
    burnin : int, optional
        Number of burn-in steps to discard when computing the flat chain
        and summaries. Default ``500``.
    thin : int, optional
        Thinning factor applied to the flat chain. Default ``4``.
    labels : tuple of str, optional
        Parameter names in chain order.
    include_log_prob : bool, optional
        If *True*, attempt to store the full log-probability chain (and flat
        version) in the archive. A sampler that cannot supply it is tolerated:
        the keys are then absent from both the archive and the returned
        payload. Default *True*.

    Returns
    -------
    payload : dict
        The data saved to the archive (useful for immediate inspection without
        re-loading from disk). The stored ``'tau'`` is the integrated
        autocorrelation time estimated on the **post-burn-in** chain, in units
        of steps, and ``'tau_ok'`` records whether that estimate succeeded.

    Raises
    ------
    ValueError
        If the number of *labels* does not match the sampler ``ndim``, or
        if the chain is empty after applying *burnin* and *thin*.
    AttributeError
        If *sampler* has no chain because it was never run.

    Notes
    -----
    ``np.savez_compressed`` appends ``.npz`` unless the name already ends in
    it, so ``save_emcee_results("run_v1.2", ...)`` writes ``run_v1.2.npz``.
    :func:`load_emcee_results` applies the same rule, so either spelling of
    the path loads it back.
    """
    from .mcmc import summarize_emcee  # local import to avoid circularity

    chain = sampler.get_chain()
    ndim_check = chain.shape[-1]

    if len(labels) != ndim_check:
        raise ValueError(
            f"Number of labels ({len(labels)}) must match sampler ndim ({ndim_check})"
        )

    flat_chain = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    if flat_chain.size == 0:
        raise ValueError("Empty chain after burn-in/thinning; reduce burnin or thin")

    summary = summarize_emcee(sampler, burnin=burnin, thin=thin, labels=labels)

    try:
        # Measure tau on the post-burn-in chain: including burn-in leaves the
        # transient in the series and inflates tau, so tau_ok reads False for
        # chains that have in fact converged. The chain is deliberately not
        # thinned here - tau is what justifies a thinning factor, and a chain
        # thinned first cannot resolve a tau below that factor. emcee already
        # returns the result in units of steps.
        tau = np.asarray(sampler.get_autocorr_time(discard=burnin), dtype=float)
        tau_ok = True
        tau_error = ""
    except Exception as exc:
        tau = np.full(ndim_check, np.nan, dtype=float)
        tau_ok = False
        tau_error = f"{type(exc).__name__}: {exc}"

    q50 = np.array([summary[lb]["median"] for lb in labels], dtype=float)
    qminus = np.array([summary[lb]["minus_1sigma"] for lb in labels], dtype=float)
    qplus = np.array([summary[lb]["plus_1sigma"] for lb in labels], dtype=float)

    payload: dict = {
        "chain": chain,
        "flat_chain": flat_chain,
        "acceptance_fraction": np.asarray(sampler.acceptance_fraction, dtype=float),
        "burnin": np.array(burnin, dtype=int),
        "thin": np.array(thin, dtype=int),
        "labels": np.asarray(labels),
        "ndim": np.array(ndim_check, dtype=int),
        "nwalkers": np.array(chain.shape[1], dtype=int),
        "nsteps": np.array(chain.shape[0], dtype=int),
        "tau": tau,
        "tau_ok": np.array(tau_ok),
        "tau_error": np.array(tau_error),
        "summary_median": q50,
        "summary_minus_1sigma": qminus,
        "summary_plus_1sigma": qplus,
    }

    if include_log_prob:
        try:
            payload["log_prob"] = sampler.get_log_prob()
            payload["flat_log_prob"] = sampler.get_log_prob(
                discard=burnin, thin=thin, flat=True
            )
        except Exception:
            pass

    np.savez_compressed(path, **payload)
    return payload


def load_emcee_results(path: str | Path) -> dict:
    """Load an MCMC result archive saved by :func:`save_emcee_results`.

    Parameters
    ----------
    path : str or Path
        Path to the archive, with or without the ``.npz`` extension. The
        suffix is appended when the name does not already end in ``.npz``,
        matching what :func:`save_emcee_results` writes — including for names
        that contain a dot, such as ``"sn2023abc_v1.2"``. A file stored
        without the suffix is also loaded if it exists.

    Returns
    -------
    result : dict
        Keys and their meanings:

        * ``'chain'`` – full chain, shape ``(nsteps, nwalkers, ndim)``.
        * ``'flat_chain'`` – thinned post-burn-in flat chain, shape
          ``(nsamples, ndim)``.
        * ``'labels'`` – list of parameter name strings.
        * ``'burnin'``, ``'thin'`` – ints used when saving.
        * ``'summary'`` – dict of ``{label: {'median', 'minus_1sigma',
          'plus_1sigma'}}`` for each parameter.
        * ``'tau_error'`` – empty string, or the reason tau could not be
          estimated.
        * ``'tau'`` – integrated autocorrelation time array (NaN if not
          converged when saved).
        * ``'tau_ok'`` – bool.
        * ``'acceptance_fraction'`` – per-walker acceptance fraction array.
        * Additional optional keys: ``'log_prob'``, ``'flat_log_prob'`` -
          present only if they were stored and readable at save time.

        ``ndim``, ``nwalkers`` and ``nsteps`` are written to the archive by
        :func:`save_emcee_results` but are not returned here; read them from
        ``chain.shape`` if needed.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist at *path*.
    """
    path = Path(path)

    # Mirror numpy.savez_compressed, which appends ".npz" unless the name
    # already ends with it. Path.with_suffix cannot be used: for a name like
    # "run_v1.2" it would *replace* the trailing ".2" and look for
    # "run_v1.npz", and testing `path.suffix` alone would skip appending
    # entirely. Both give a FileNotFoundError for a file that was written.
    candidate = path if path.name.endswith(".npz") else path.with_name(path.name + ".npz")
    if not candidate.exists() and path.exists():
        candidate = path        # archive stored under a name without the suffix
    if not candidate.exists():
        raise FileNotFoundError(
            f"MCMC result archive not found: {candidate}"
            + ("" if candidate == path else f" (nor {path})")
        )
    path = candidate

    res = np.load(path, allow_pickle=False)

    labels = [str(lb) for lb in res["labels"]]
    burnin = int(res["burnin"])
    thin = int(res["thin"])
    q50 = res["summary_median"]
    qminus = res["summary_minus_1sigma"]
    qplus = res["summary_plus_1sigma"]

    summary = {
        lb: {
            "median": float(q50[i]),
            "minus_1sigma": float(qminus[i]),
            "plus_1sigma": float(qplus[i]),
        }
        for i, lb in enumerate(labels)
    }

    out: dict = {
        "chain": res["chain"],
        "flat_chain": res["flat_chain"],
        "labels": labels,
        "burnin": burnin,
        "thin": thin,
        "summary": summary,
        "tau": res["tau"],
        "tau_ok": bool(res["tau_ok"]),
        "tau_error": str(res["tau_error"]),
        "acceptance_fraction": res["acceptance_fraction"],
    }

    for key in ("log_prob", "flat_log_prob"):
        if key in res:
            out[key] = res[key]

    return out
