"""
MCMC fitting machinery for PSF photometry on JWST difference images.

Fitted parameters (in order):
    0  flux         – total source flux (image units, e.g. MJy sr⁻¹)
    1  dx           – sub-pixel x-shift of the PSF centroid (native pixels)
    2  dy           – sub-pixel y-shift of the PSF centroid (native pixels)
    3  bkg          – flat residual background level (same units as flux)

The log-likelihood is evaluated in Fourier space to account for correlated
drizzle noise (see :func:`log_likelihood_cov_prepared`).

Typical workflow
----------------
>>> prepared = prepare_covariance_terms(data, err, cov_kernel)
>>> psf_os, native_shape, frac = prepare_psf_for_oversamp(
...     psf_raw, oversamp=4, native_shape=(11, 11), return_fraction=True)
>>> fit = prepare_for_fitting(data, err, psf_os, cov_kernel,
...                           dx_init=0.05, dy_init=-0.10)
>>> sampler = run_mcmc(**fit)
>>> summary = summarize_emcee(sampler, burnin=500, thin=4)
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import emcee
from astropy.stats import sigma_clipped_stats

from .psf import psf_model, prepare_psf_for_oversamp
from .covariance import prepare_covariance_terms


# ---------------------------------------------------------------------------
# Default prior configuration
# ---------------------------------------------------------------------------

#: Flat prior bounds applied when none are provided by the caller.
DEFAULT_PRIOR_BOUNDS: dict[str, tuple[float, float]] = {
    "flux": (-10.0, 100.0),
    "dx": (-1.0, 1.0),
    "dy": (-1.0, 1.0),
    "bkg": (-0.1, 0.1),
}


def _resolve_prior_bounds(prior_bounds: dict | None) -> dict:
    bounds = DEFAULT_PRIOR_BOUNDS.copy()
    if prior_bounds is not None:
        bounds.update(prior_bounds)
    return bounds


# ---------------------------------------------------------------------------
# Probability functions
# ---------------------------------------------------------------------------

def log_prior(
    param: np.ndarray,
    prior_bounds: dict | None = None,
) -> float:
    """Evaluate the flat (uniform) log-prior for the four model parameters.

    Parameters
    ----------
    param : array-like of length 4
        ``[flux, dx, dy, bkg]``.
    prior_bounds : dict or None, optional
        Per-parameter ``(lo, hi)`` bounds. Missing keys fall back to
        :data:`DEFAULT_PRIOR_BOUNDS`.  Bounds are *open* intervals.

    Returns
    -------
    lp : float
        0.0 if all parameters are within their bounds, −∞ otherwise.
    """
    flux, dx, dy, bkg = param
    bounds = _resolve_prior_bounds(prior_bounds)
    if not (bounds["flux"][0] < flux < bounds["flux"][1]):
        return -np.inf
    if not (bounds["dx"][0] < dx < bounds["dx"][1]):
        return -np.inf
    if not (bounds["dy"][0] < dy < bounds["dy"][1]):
        return -np.inf
    if not (bounds["bkg"][0] < bkg < bounds["bkg"][1]):
        return -np.inf
    return 0.0


def log_likelihood_cov_prepared(
    param: np.ndarray,
    prepared: dict,
    psf_os: np.ndarray,
    oversamp: int,
    native_shape: tuple[int, int] | None = None,
    psf_prepare_fraction: float = 1.0,
) -> float:
    """Fourier-space covariance log-likelihood.

    Computes

    .. math::

        \\ln \\mathcal{L} = -\\frac{1}{2N}
            \\sum_{\\mathbf{k}} \\frac{|\\hat{r}(\\mathbf{k})|^2}{P(\\mathbf{k})}

    where :math:`\\hat{r}` is the FFT of the pixel-weighted normalised
    residual ``(data − model) / err × sqrt(weight)`` and :math:`P` is the
    covariance power spectrum stored in *prepared*.

    Parameters
    ----------
    param : array-like of length 4
        ``[flux, dx, dy, bkg]``.
    prepared : dict
        Output of :func:`~jwst_psfmc.covariance.prepare_covariance_terms`.
    psf_os : ndarray
        Pre-cropped oversampled PSF from
        :func:`~jwst_psfmc.psf.prepare_psf_for_oversamp`.
    oversamp : int
        Integer oversampling factor.
    native_shape : (ny, nx) or None, optional
        Native PSF shape; forwarded to :func:`~jwst_psfmc.psf.psf_model`.
    psf_prepare_fraction : float, optional
        Encircled-energy fraction; forwarded to :func:`~jwst_psfmc.psf.psf_model`.

    Returns
    -------
    log_like : float
        Log-likelihood value.
    """
    bkg = float(param[3])
    model = (
        psf_model(
            param[:3],
            psf_os,
            oversamp,
            output_shape=prepared["shape"],
            native_shape=native_shape,
            psf_prepare_fraction=psf_prepare_fraction,
        )
        + bkg
    )

    model_use = np.where(prepared["valid"], model, 0.0)
    resid = prepared["data"] - model_use
    r_norm = (resid / prepared["err"]) * np.sqrt(prepared["fit_weight"])

    r_norm_fft = np.fft.fft2(r_norm)
    chi2 = (
        np.sum(np.abs(r_norm_fft) ** 2 / prepared["power_spectrum"])
        / r_norm.size
    )
    return -0.5 * chi2


def log_prob_prepared(
    param: np.ndarray,
    prepared: dict,
    psf_os: np.ndarray,
    oversamp: int,
    prior_bounds: dict | None = None,
    native_shape: tuple[int, int] | None = None,
    psf_prepare_fraction: float = 1.0,
) -> float:
    """Log-posterior = log-prior + log-likelihood (covariance-aware).

    This is the function passed directly to :class:`emcee.EnsembleSampler`.

    Parameters
    ----------
    param : array-like of length 4
        ``[flux, dx, dy, bkg]``.
    prepared : dict
        Output of :func:`~jwst_psfmc.covariance.prepare_covariance_terms`.
    psf_os : ndarray
        Pre-cropped oversampled PSF.
    oversamp : int
        Oversampling factor.
    prior_bounds : dict or None, optional
        Per-parameter prior bounds.
    native_shape : (ny, nx) or None, optional
        Native PSF shape.
    psf_prepare_fraction : float, optional
        Encircled-energy fraction.

    Returns
    -------
    log_post : float
        Log-posterior value (−∞ if outside priors).
    """
    lp = log_prior(param, prior_bounds=prior_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_cov_prepared(
        param,
        prepared,
        psf_os,
        oversamp,
        native_shape=native_shape,
        psf_prepare_fraction=psf_prepare_fraction,
    )


# ---------------------------------------------------------------------------
# High-level preparation and run functions
# ---------------------------------------------------------------------------

def prepare_for_fitting(
    data: np.ndarray,
    err: np.ndarray,
    psf_os: np.ndarray,
    cov_kernel: np.ndarray,
    dx_init: float = 0.0,
    dy_init: float = 0.0,
    oversamp: int = 4,
    native_shape: tuple[int, int] = (11, 11),
    nwalkers: int = 48,
    flux_prior_scale: float = 30.0,
    pos_prior_half_width: float = 1.0,
    bkg_prior_half_width: float = 0.1,
    fit_weight: np.ndarray | None = None,
) -> dict:
    """Prepare all inputs needed to call :func:`run_mcmc`.

    Accepts NumPy arrays directly (no file paths). This is the array-oriented
    counterpart of the old server-specific ``prepare_for_emcee``.

    Parameters
    ----------
    data : ndarray, 2-D
        Difference-image stamp centred on the source (science flux units).
    err : ndarray, 2-D
        Corresponding per-pixel 1-σ uncertainty map, same shape as *data*.
    psf_os : ndarray, 2-D
        Raw oversampled PSF (e.g. from a star-based PSF model FITS file).
    cov_kernel : ndarray, 2-D
        Covariance kernel from
        :func:`~jwst_psfmc.covariance.estimate_cov_kernel`.
    dx_init : float, optional
        Initial guess for the sub-pixel x-shift in native pixels. Default ``0``.
    dy_init : float, optional
        Initial guess for the sub-pixel y-shift in native pixels. Default ``0``.
    oversamp : int, optional
        Integer oversampling factor of *psf_os*. Default ``4``.
    native_shape : (ny, nx), optional
        Desired native PSF shape. Default ``(11, 11)``.
    nwalkers : int, optional
        Number of ``emcee`` walkers. Must be even and ≥ 2 × ndim (= 8).
        Default ``48``.
    flux_prior_scale : float, optional
        Upper flux prior bound = ``flux_prior_scale × max(MAX, 0.1)``, where
        *MAX* is the peak pixel value of the stamp. Default ``30``.
    pos_prior_half_width : float, optional
        Half-width of the uniform prior on dx and dy in native pixels.
        Default ``1.0``.
    bkg_prior_half_width : float, optional
        Half-width of the uniform prior on the background offset. Default ``0.1``.
    fit_weight : ndarray or None, optional
        Optional pixel weight map; forwarded to
        :func:`~jwst_psfmc.covariance.prepare_covariance_terms`.

    Returns
    -------
    fit_args : dict
        All keyword arguments accepted by :func:`run_mcmc`, plus an extra
        ``'pos'`` key with the initial walker positions of shape
        ``(nwalkers, 4)``.
    """
    psf_os_crop, out_native_shape, psf_prepare_fraction = prepare_psf_for_oversamp(
        psf_os, oversamp, native_shape=native_shape, return_fraction=True
    )

    prepared = prepare_covariance_terms(data, err, cov_kernel, fit_weight=fit_weight)

    MAX = float(np.nanmax(data))
    _, _, std = sigma_clipped_stats(data, sigma=3.0, maxiters=10)
    RMS = float(max(2.0 * max(MAX, 0.0), std, 0.2))
    if MAX <= 0:
        MAX = 0.1
        RMS = 0.2

    prior_bounds = {
        "flux": (-3.0, flux_prior_scale * max(MAX, 0.1)),
        "dx": (dx_init - pos_prior_half_width, dx_init + pos_prior_half_width),
        "dy": (dy_init - pos_prior_half_width, dy_init + pos_prior_half_width),
        "bkg": (-bkg_prior_half_width, bkg_prior_half_width),
    }

    initial = np.array([max(10.0 * MAX, 0.1), dx_init, dy_init, 0.0])
    perturbation = np.array([RMS, 0.1, 0.1, 0.01])

    rng = np.random.default_rng()
    pos = initial + perturbation * rng.standard_normal((nwalkers, 4))
    pos[:, 0] = np.clip(pos[:, 0], prior_bounds["flux"][0] + 0.1, prior_bounds["flux"][1] - 0.1)
    pos[:, 1] = np.clip(pos[:, 1], prior_bounds["dx"][0] + 0.01, prior_bounds["dx"][1] - 0.01)
    pos[:, 2] = np.clip(pos[:, 2], prior_bounds["dy"][0] + 0.01, prior_bounds["dy"][1] - 0.01)
    pos[:, 3] = np.clip(pos[:, 3], prior_bounds["bkg"][0] + 0.01, prior_bounds["bkg"][1] - 0.01)

    return {
        "prepared": prepared,
        "psf_os": psf_os_crop,
        "oversamp": oversamp,
        "prior_bounds": prior_bounds,
        "native_shape": out_native_shape,
        "psf_prepare_fraction": psf_prepare_fraction,
        "nwalkers": nwalkers,
        "pos": pos,
    }


def run_mcmc(
    prepared: dict,
    psf_os: np.ndarray,
    oversamp: int,
    prior_bounds: dict | None,
    native_shape: tuple[int, int] | None,
    psf_prepare_fraction: float,
    nwalkers: int,
    pos: np.ndarray,
    nsteps: int = 2000,
    ncores: int | None = None,
    progress: bool = True,
    **_extra,
) -> emcee.EnsembleSampler:
    """Run the ``emcee`` ensemble sampler with an optional multiprocessing pool.

    Parameters
    ----------
    prepared : dict
        Output of :func:`~jwst_psfmc.covariance.prepare_covariance_terms`.
    psf_os : ndarray
        Pre-cropped oversampled PSF.
    oversamp : int
        Integer oversampling factor.
    prior_bounds : dict or None
        Per-parameter uniform prior bounds.
    native_shape : (ny, nx) or None
        Native PSF shape.
    psf_prepare_fraction : float
        Encircled-energy fraction of the prepared PSF.
    nwalkers : int
        Number of MCMC walkers.
    pos : ndarray of shape (nwalkers, 4)
        Initial walker positions.
    nsteps : int, optional
        Total number of MCMC steps. Default ``2000``.
    ncores : int or None, optional
        Number of parallel worker processes. ``None`` uses
        ``min(nwalkers, os.cpu_count())``. Set to ``1`` to disable
        multiprocessing (useful in notebooks).
    progress : bool, optional
        Display a ``tqdm`` progress bar. Default *True*.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        Completed sampler with the full chain stored in memory.

    Notes
    -----
    The acceptance fraction should be between ~0.2 and ~0.5.  If it is
    consistently below 0.1, widen the prior bounds or increase *nsteps*.
    After the run, check convergence with
    ``sampler.get_autocorr_time(quiet=True)``; aim for
    ``nsteps / tau > 50``.
    """
    ndim = 4

    if ncores is None:
        import os
        ncores = min(nwalkers, os.cpu_count() or 1)

    sampler_kwargs = dict(
        args=(
            prepared,
            psf_os,
            oversamp,
            prior_bounds,
            native_shape,
            psf_prepare_fraction,
        )
    )

    if ncores > 1:
        with mp.Pool(processes=ncores) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_prob_prepared, pool=pool, **sampler_kwargs
            )
            sampler.run_mcmc(pos, nsteps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_prepared, **sampler_kwargs
        )
        sampler.run_mcmc(pos, nsteps, progress=progress)

    return sampler


# ---------------------------------------------------------------------------
# Posterior summarisation
# ---------------------------------------------------------------------------

def summarize_emcee(
    sampler: emcee.EnsembleSampler,
    burnin: int = 0,
    thin: int = 1,
    labels: tuple[str, ...] = ("flux", "dx", "dy", "bkg"),
) -> dict:
    """Summarise posterior chains as median and 1-σ (16/84 percentile) errors.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Completed sampler.
    burnin : int, optional
        Number of steps to discard as burn-in. Default ``0``.
    thin : int, optional
        Thinning factor. Default ``1`` (no thinning).
    labels : tuple of str, optional
        Parameter names in chain order. Default ``('flux', 'dx', 'dy', 'bkg')``.

    Returns
    -------
    summary : dict
        For each label, a dict with keys ``'median'``, ``'minus_1sigma'``,
        and ``'plus_1sigma'``.

    Raises
    ------
    ValueError
        If the chain is empty after applying *burnin* and *thin*.
    """
    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    if flat.size == 0:
        raise ValueError("Empty chain after burn-in/thinning; reduce burnin or thin")

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    return {
        label: {
            "median": float(q50[i]),
            "minus_1sigma": float(q50[i] - q16[i]),
            "plus_1sigma": float(q84[i] - q50[i]),
        }
        for i, label in enumerate(labels)
    }


def summarize_flux_from_chain(
    sampler: emcee.EnsembleSampler,
    burnin: int = 0,
    thin: int = 1,
    psf_os: np.ndarray | None = None,
    oversamp: int = 4,
    native_shape: tuple[int, int] | None = None,
    output_shape: tuple[int, int] | None = None,
    psf_prepare_fraction: float = 1.0,
) -> dict:
    """Summarise total and in-stamp flux posteriors accounting for PSF encircled energy.

    For each posterior sample the PSF model is evaluated (with the sampled
    position shift) to obtain the exact fraction of flux landing in the stamp.
    This is more accurate than using the median position alone when the source
    is near the stamp edge.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Completed sampler.
    burnin : int, optional
        Burn-in steps to discard.
    thin : int, optional
        Thinning factor.
    psf_os : ndarray
        Pre-cropped oversampled PSF.
    oversamp : int, optional
        Oversampling factor. Default ``4``.
    native_shape : (ny, nx) or None, optional
        Native PSF shape.
    output_shape : (ny, nx) or None, optional
        Stamp shape (used to determine in-stamp fraction).
    psf_prepare_fraction : float, optional
        Encircled-energy fraction from :func:`~jwst_psfmc.psf.prepare_psf_for_oversamp`.

    Returns
    -------
    result : dict
        Keys ``'total_flux'``, ``'stamp_flux_sum'``, and
        ``'total_encircled_fraction'``, each containing
        ``{'median', 'minus_1sigma', 'plus_1sigma'}``.

    Raises
    ------
    ValueError
        If *psf_os* is not provided or the chain is empty.
    """
    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    if flat.size == 0:
        raise ValueError("Empty chain after burn-in/thinning")
    if psf_os is None:
        raise ValueError("psf_os is required")

    total_flux = flat[:, 0]
    stamp_flux = np.empty_like(total_flux)
    total_fraction = np.empty_like(total_flux)

    for i, param in enumerate(flat):
        m, meta = psf_model(
            param[:3],
            psf_os,
            oversamp=oversamp,
            native_shape=native_shape,
            output_shape=output_shape,
            psf_prepare_fraction=psf_prepare_fraction,
            return_meta=True,
        )
        stamp_flux[i] = float(np.sum(m))
        total_fraction[i] = meta["total_encircled_fraction"]

    def _q(arr: np.ndarray) -> dict:
        q16, q50, q84 = np.percentile(arr, [16, 50, 84])
        return {
            "median": float(q50),
            "minus_1sigma": float(q50 - q16),
            "plus_1sigma": float(q84 - q50),
        }

    return {
        "total_flux": _q(total_flux),
        "stamp_flux_sum": _q(stamp_flux),
        "total_encircled_fraction": _q(total_fraction),
    }
