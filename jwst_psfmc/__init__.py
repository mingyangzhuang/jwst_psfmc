"""
jwst_psfmc
==========

PSF photometry with MCMC for JWST (and HST) drizzled images.

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
    estimate_cov_kernel, kernel_power_spectrum, whiten_image,
    prepare_covariance_terms,
    SplitCosineBellWindow, get_source_mask, find_zero_squares

MCMC fitting
    prepare_for_fitting, run_mcmc,
    log_prior, log_likelihood_cov_prepared, log_prob_prepared,
    summarize_emcee, summarize_posterior, summarize_flux_from_chain

Parameter definitions
    Parameter, ParameterSet, Uniform, LogUniform, Normal, TruncatedNormal,
    spawn_seed_streams

Model evaluation
    RenderContext, PSFSpec, PSFOperator, PointSource, SersicSource,
    ConstantBackground, CompositeModel,
    prepare_noise, prepare_model_fitting, evaluate_model_fit
    JointObservation, PreparedJointFit, prepare_joint_model_fitting,
    evaluate_joint_model_fit, shared_reference_covariance
    optimize_model_fit, run_model_mcmc, FitResult, summarize_model_results
    summarize_decomposition

Intrinsic Sérsic profiles
    SersicProfile, integrate_sersic_grid, render_intrinsic_sersic

I/O
    save_emcee_results, load_emcee_results
    save_nautilus_results, load_nautilus_results
    save_model_results, load_model_results
    save_model_snapshot, load_model_snapshot

Visualisation
    plot_psf_fit_triptych, plot_chains, plot_corner
    plot_model_fit, plot_decomposition, plot_residual_diagnostics
"""

__version__ = "0.4.0"

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
    kernel_power_spectrum,
    whiten_image,
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
    summarize_posterior,
    summarize_flux_from_chain,
)
from .parameters import (
    Parameter,
    ParameterSet,
    Uniform,
    LogUniform,
    Normal,
    TruncatedNormal,
    spawn_seed_streams,
)
from .rendering import RenderContext, PSFSpec, PSFOperator
from .models import PointSource, SersicSource, ConstantBackground, CompositeModel
from .noise import prepare_noise
from .fitting import prepare_model_fitting, evaluate_model_fit
from .joint import (
    JointObservation, PreparedJointFit, prepare_joint_model_fitting,
    evaluate_joint_model_fit, shared_reference_covariance,
)
from .inference import (
    FitResult, optimize_model_fit, run_model_mcmc, summarize_model_results,
)
from .decomposition import summarize_decomposition
from .sersic import SersicProfile, integrate_sersic_grid, render_intrinsic_sersic
from .io import (
    save_emcee_results,
    load_emcee_results,
    save_nautilus_results,
    load_nautilus_results,
    save_model_results,
    load_model_results,
)
from .plot import (
    plot_psf_fit_triptych,
    plot_chains,
    plot_corner,
    plot_model_fit,
    plot_decomposition,
    plot_residual_diagnostics,
)
from .snapshot import save_model_snapshot, load_model_snapshot

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
    "kernel_power_spectrum",
    "whiten_image",
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
    "summarize_posterior",
    "summarize_flux_from_chain",
    # parameters
    "Parameter",
    "ParameterSet",
    "Uniform",
    "LogUniform",
    "Normal",
    "TruncatedNormal",
    "spawn_seed_streams",
    # model evaluation
    "RenderContext",
    "PSFSpec",
    "PSFOperator",
    "PointSource",
    "SersicSource",
    "ConstantBackground",
    "CompositeModel",
    "prepare_noise",
    "prepare_model_fitting",
    "evaluate_model_fit",
    "JointObservation",
    "PreparedJointFit",
    "prepare_joint_model_fitting",
    "evaluate_joint_model_fit",
    "shared_reference_covariance",
    "FitResult",
    "optimize_model_fit",
    "run_model_mcmc",
    "summarize_model_results",
    "summarize_decomposition",
    # intrinsic Sérsic profiles
    "SersicProfile",
    "integrate_sersic_grid",
    "render_intrinsic_sersic",
    # io
    "save_emcee_results",
    "load_emcee_results",
    "save_nautilus_results",
    "load_nautilus_results",
    "save_model_results",
    "load_model_results",
    "save_model_snapshot",
    "load_model_snapshot",
    # plot
    "plot_psf_fit_triptych",
    "plot_chains",
    "plot_corner",
    "plot_model_fit",
    "plot_decomposition",
    "plot_residual_diagnostics",
]
