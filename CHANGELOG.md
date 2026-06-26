# Changelog

All notable changes to `jwst_psfmc` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] – 2025-06-25

### Added
- `jwst_psfmc.psf`: Fourier-space PSF shifting (`shift_psf_fourier`),
  block-sum downsampling (`downsample_psf`), oversampled PSF preparation
  (`prepare_psf_for_oversamp`), model evaluation with encircled-energy
  fraction tracking (`psf_model`), and visual shift sanity check
  (`inspect_psf_shift`).
- `jwst_psfmc.covariance`: 2-D covariance kernel estimation from source-free
  sky patches (`estimate_cov_kernel`), Fourier-space power-spectrum
  pre-computation (`prepare_covariance_terms`), split-cosine-bell window
  function (`SplitCosineBellWindow`), source masking via `photutils`
  segmentation (`get_source_mask`), and integral-image based source-free
  square finder (`find_zero_squares`).
- `jwst_psfmc.mcmc`: flat log-prior (`log_prior`), Fourier-space covariance
  log-likelihood (`log_likelihood_cov_prepared`), log-posterior
  (`log_prob_prepared`), array-oriented fitting preparation
  (`prepare_for_fitting`), `emcee` ensemble sampler runner with optional
  multiprocessing pool (`run_mcmc`), posterior summarisation (`summarize_emcee`,
  `summarize_flux_from_chain`).
- `jwst_psfmc.io`: compressed `.npz` save/load (`save_emcee_results`,
  `load_emcee_results`), `importlib.resources`-based bundled data path helpers
  .
- `jwst_psfmc.plot`: data / model / residual triptych (`plot_psf_fit_triptych`),
  walker chain traces (`plot_chains`), `corner.py` wrapper (`plot_corner`).
- Bundled example data: JWST NEXUS F200W cutouts for source 43, epochs
  deep_ep02 (non-detection) and deep_ep03 (detection), plus 4× oversampled
  PSF models for each epoch.
- `examples/demo_psf_photometry.ipynb`: end-to-end demonstration notebook
  covering covariance estimation, MCMC for a non-detection (3-σ upper limit)
  and a detection, chain diagnostics, corner plots, and result I/O.
- GitHub Actions CI workflow (Python 3.10 and 3.11).
- Full NumPy-style docstrings on all public functions.

[Unreleased]: https://github.com/mingyangzhuang/jwst_psfmc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mingyangzhuang/jwst_psfmc/releases/tag/v0.1.0
