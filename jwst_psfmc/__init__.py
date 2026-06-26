"""
jwst_psfmc
==========

PSF photometry with MCMC for JWST (and HST) drizzled difference images.

Accounts for correlated pixel noise introduced by the drizzle algorithm via
a Fourier-space covariance likelihood. The four fitted parameters are:

    flux   – total source flux in the same units as the image
    dx     – sub-pixel x-shift of the PSF centre (native pixels)
    dy     – sub-pixel y-shift of the PSF centre (native pixels)
    bkg    – flat residual background

Public API
----------
PSF manipulation
    shift_psf_fourier, downsample_psf, prepare_psf_for_oversamp,
    psf_model, match_shape_center, inspect_psf_shift

Covariance estimation
    estimate_cov_kernel, prepare_covariance_terms,
    SplitCosineBellWindow, get_source_mask, find_zero_squares

MCMC fitting
    prepare_for_fitting, run_mcmc,
    log_prior, log_likelihood_cov_prepared, log_prob_prepared,
    summarize_emcee, summarize_flux_from_chain

I/O
    save_emcee_results, load_emcee_results

Visualisation
    plot_psf_fit_triptych, plot_chains, plot_corner
"""

__version__ = "0.1.0"

from .psf import (
    shift_psf_fourier,
    downsample_psf,
    match_shape_center,
    prepare_psf_for_oversamp,
    psf_model,
    inspect_psf_shift,
)
from .covariance import (
    SplitCosineBellWindow,
    distance_grid,
    estimate_cov_kernel,
    prepare_covariance_terms,
    get_source_mask,
    find_zero_squares,
)
from .mcmc import (
    log_prior,
    log_likelihood_cov_prepared,
    log_prob_prepared,
    prepare_for_fitting,
    run_mcmc,
    summarize_emcee,
    summarize_flux_from_chain,
)
from .io import (
    save_emcee_results,
    load_emcee_results,
)
from .plot import (
    plot_psf_fit_triptych,
    plot_chains,
    plot_corner,
)

__all__ = [
    # psf
    "shift_psf_fourier",
    "downsample_psf",
    "match_shape_center",
    "prepare_psf_for_oversamp",
    "psf_model",
    "inspect_psf_shift",
    # covariance
    "SplitCosineBellWindow",
    "distance_grid",
    "estimate_cov_kernel",
    "prepare_covariance_terms",
    "get_source_mask",
    "find_zero_squares",
    # mcmc
    "log_prior",
    "log_likelihood_cov_prepared",
    "log_prob_prepared",
    "prepare_for_fitting",
    "run_mcmc",
    "summarize_emcee",
    "summarize_flux_from_chain",
    # io
    "save_emcee_results",
    "load_emcee_results",
    # plot
    "plot_psf_fit_triptych",
    "plot_chains",
    "plot_corner",
]
