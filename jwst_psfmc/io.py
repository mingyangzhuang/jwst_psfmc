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
        If *True*, store the full log-probability chain (and flat version)
        in the archive. Default *True*.

    Returns
    -------
    payload : dict
        The exact data saved to the archive (useful for immediate inspection
        without re-loading from disk).

    Raises
    ------
    ValueError
        If the number of *labels* does not match the sampler ``ndim``, or
        if the chain is empty after applying *burnin* and *thin*.
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
        tau = np.asarray(sampler.get_autocorr_time(), dtype=float)
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
        Path to the ``.npz`` file (with or without the extension).

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
        * ``'tau'`` – integrated autocorrelation time array (NaN if not
          converged when saved).
        * ``'tau_ok'`` – bool.
        * ``'acceptance_fraction'`` – per-walker acceptance fraction array.
        * Additional optional keys: ``'log_prob'``, ``'flat_log_prob'``.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist at *path*.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"MCMC result archive not found: {path}")

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
