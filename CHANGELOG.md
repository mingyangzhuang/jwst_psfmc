# Changelog

All notable changes to `jwst_psfmc` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- The recommended Gaussian-equivalent 3σ upper limit is now
  `max(median, 0) + 3σ` rather than `median + 3σ`. Unclamped, it is
  anti-conservative when the posterior median comes out negative — routine for
  a non-detection on a downward noise fluctuation — and would claim a limit
  deeper than the noise allows. The notebook reports the unclamped value when
  the clamp engages.

## [0.2.1] – 2026-09-08

### Fixed
- Added `tqdm` to the runtime dependencies. `run_mcmc` defaults to
  `progress=True`, but `emcee` does not declare `tqdm` itself, so a clean
  `pip install jwst-psfmc` produced no progress bar at all: emcee falls back to
  a `_NoOpPBar` and only logs "You must install the tqdm library to use
  progress indicators with emcee" to stderr. On runs that take hours the
  absence of a bar is easy to misread as a hung process.
- `tests/test_covariance.py` selected its sky patch with `argsort` on square
  size, but 56 squares tie at the largest size and numpy's default quicksort is
  unstable, so the chosen corner varied by build — (242, 112) under quicksort,
  (215, 111) under heapsort. The candidates give whitened/original RMS ratios
  spanning 0.9958 to 1.0016, which failed CI on Python 3.10 against a `rel=1e-3`
  assertion. The tie is now broken explicitly with `lexsort`, and the tolerance
  reflects kernel-versus-patch modelling rather than machine noise.

### Changed
- Three figures from the demo notebooks are now shown in the README (whitening,
  a detection triptych, and the two-run upper-limit posteriors). They are
  referenced by absolute URLs, since the README is also the PyPI long
  description and repository-relative paths do not resolve there.
- README badge cache window shortened from 3600 s to 300 s: GitHub's Camo proxy
  had cached the v0.1.0 badge and kept serving it after 0.2.0 was published.

## [0.2.0] – 2026-09-07

### Added
- `jwst_psfmc.covariance.kernel_power_spectrum` — builds the Fourier power
  spectrum of a covariance kernel on an arbitrary image grid, centre-cropping
  or centre-padding the kernel as needed. Exposed as
  `jwst_psfmc.kernel_power_spectrum`.
- `jwst_psfmc.covariance.whiten_image` — divides an image's Fourier amplitudes
  by `sqrt(P(k))` to remove pixel-to-pixel noise correlations. Warns when the
  modelled noise power underflows the `eps` floor, a case in which the result
  is amplified and unreliable. Exposed as `jwst_psfmc.whiten_image`.
- `examples/demo_psf_photometry.ipynb`: new **Detection** section (example2) —
  fit preparation, MCMC, convergence diagnostics, posterior summary with
  detection significance, encircled-energy-corrected total flux via
  `summarize_flux_from_chain`, corner plot and residual triptych.
- Both 3σ upper-limit conventions are now documented and computed side by side
  in the README and in `examples/demo_psf_photometry.ipynb`: the 99.7th
  posterior percentile (distribution-free) and `median + 3σ` (the
  Gaussian-equivalent form used in most transient and SN-rate literature),
  with guidance on when they diverge and why the choice must be stated.
- `prepare_for_fitting` gained a `flux_prior_min` keyword (default `-3.0`,
  reproducing the previous behaviour) for the lower bound of the uniform flux
  prior, which was previously hard-coded. `flux_prior_scale` governs only the
  upper bound, so a source that has faded relative to the reference epoch —
  genuinely negative flux in the difference image — could be truncated by the
  prior and report a spuriously tight error bar. `tests/test_mcmc.py` covers
  the new keyword.
- Unit tests for `kernel_power_spectrum` and `whiten_image`, exercising
  odd- and even-sized grids, oversized kernels, the delta-kernel identity,
  RMS preservation on the bundled data, and the underflow warning.
- `tests/test_io.py` — round-trip coverage for `save_emcee_results` /
  `load_emcee_results` against a real short chain: chain values, persisted
  summary versus a fresh `summarize_emcee`, and metadata.
- Tests that `shift_psf_fourier` moves the centroid by exactly the requested
  sub-pixel amount, and that a +d / -d round trip is exact on odd-sized grids.
- A test pinning an invariant of the oversampled pipeline: the working grid is
  even (`native_shape` 11 × `oversamp` 4 = 44) even though the PSF file and the
  data stamp are both odd, so a non-integer shift loses the Nyquist bin on the
  oversampled array — but that is exactly the component the block-sum of 4
  annihilates, leaving the native-resolution model exact.

### Fixed
- The covariance kernel is now placed at the Fourier origin by an explicit
  roll rather than `np.fft.ifftshift`. The two agree for odd-sized grids (all
  shipped examples), but `ifftshift` misplaced the kernel centre by one pixel
  on even-sized grids, corrupting the power spectrum and hence the likelihood.
- `examples/demo_covariance_kernel.ipynb` whitened by the patch's own
  periodogram rather than the kernel power spectrum. That is a phase-only
  transform: it discards the data and pins the output RMS near 1 regardless
  of the input, so the whitened RMS did not match the original.
- `examples/demo_psf_photometry.ipynb` built the power-spectrum panel by
  centre-padding the kernel into the stamp. The kernel (15×15) is larger than
  the stamp (9×9), so the pad offset went negative and the cell raised
  `ValueError: could not broadcast input array`. It now calls
  `prepare_covariance_terms`, which crops before padding.
- `examples/demo_psf_photometry.ipynb` was missing its detection section
  entirely, so the save/reload cell raised `NameError` on `sampler_det`.

### Fixed
- `run_mcmc` accepted `**_extra`, which existed only so `run_mcmc(**prepared)`
  would tolerate stray keys — but `prepare_for_fitting` returns exactly the
  keys `run_mcmc` names, so it bought nothing while silently absorbing
  misspellings (`nstep=`, `n_cores=`, `nwalker=`) that then fell back to
  defaults. Removed; typos now raise `TypeError`.
- `summarize_flux_from_chain` and `plot_psf_fit_triptych` defaulted to model
  geometry that disagrees with what was fitted, producing a plausible but
  wrongly normalised result. With `output_shape=None`,
  `summarize_flux_from_chain` applies no crop, so `'stamp_flux_sum'` is not an
  in-stamp flux and `'total_encircled_fraction'` collapses to exactly 1.0 with
  zero spread; with `native_shape=None`, `plot_psf_fit_triptych` skips the
  native-shape matching step and plots a model ~22 % brighter than the fitted
  one, putting structure in the residual panel of a good fit. Both now emit a
  `RuntimeWarning` in that state.

### Added (API)
- `summarize_flux_from_chain` and `plot_psf_fit_triptych` accept
  `fit=<prepare_for_fitting dict>`, which fills in `psf_os`, `oversamp`,
  `native_shape`, `psf_prepare_fraction` and (for the former) `output_shape`
  from `fit['prepared']['shape']`. This is the reliable way to make the model
  geometry match the fit; explicit arguments still take precedence.
  `plot_psf_fit_triptych`'s `psf_os` and `oversamp` are now optional, and
  raise `ValueError` if neither they nor *fit* are given.

- `estimate_cov_kernel` returned a silently truncated kernel when the sky
  patch was smaller than `size`. The central crop indexes
  `acf[cy - size//2 : cy + size//2 + 1]`, whose start goes negative for a small
  patch; NumPy reads that as an offset from the end, so a 12x12 patch with
  `size=15` returned a **1x1 delta kernel**. That yields a flat power spectrum
  and silently disables the covariance correction — measured on the bundled
  detection example, the reported flux uncertainty falls to 0.72x the correct
  value, reintroducing exactly the ~30 % underestimate the package exists to
  remove. Patches smaller than `size` on either axis now raise `ValueError`.
  A `RuntimeWarning` is also issued when the patch is smaller than
  `4 × min(r_out, (size - 1) / 2)` — 24 px with the defaults — where the
  autocorrelation is estimated from too few independent lags (measured ~13 %
  low at that ratio). The threshold is deliberately tied to the taper radius
  rather than to `size`, since the window zeroes the kernel well inside
  `size`; it therefore does not fire for a kernel that spans its own patch, as
  in `examples/demo_covariance_kernel.ipynb`. The error message, the warning
  and the docstring all quote the same figure.
- `get_source_mask` measured its detection threshold on a Gaussian-smoothed
  image but ran `detect_sources` on the *unsmoothed* one. Smoothing lowers the
  noise by ~4x, so the nominal 3σ cut was really ~0.7σ and 21.9 % of pure noise
  was flagged as sources, eating the source-free area the covariance kernel is
  estimated from. Detection now runs on the same smoothed image, and a
  genuinely source-free image returns an all-False mask instead of raising
  `AttributeError`.
- `load_emcee_results` could not read a file `save_emcee_results` had just
  written whenever the path contained a dot. `numpy.savez_compressed` appends
  `.npz` unless the name already ends in it, so `"sn2023abc_v1.2"` is stored as
  `sn2023abc_v1.2.npz`, but the loader appended the suffix only when
  `Path.suffix` was empty and so raised `FileNotFoundError`. It now applies the
  same rule numpy does, and still accepts the on-disk name.
- `save_emcee_results` estimated the autocorrelation time over the chain
  *including* burn-in, while every other quantity it stores discards it. On a
  chain with a 300-step transient this inflated `tau` from ~1 to ~122, so the
  stored `tau_ok` convergence flag read False for chains that had converged.
  `tau` is now measured on the post-burn-in chain.

### Fixed (packaging)
- The sdist shipped `tests/` but not `examples/`, so running the tests from a
  released source archive produced 24 errors and 10 failures. The test modules
  now skip with an explanatory message when the example data is absent; a
  repository checkout still runs the full suite.

### Changed
- Documentation corrections throughout, from a full audit of every public
  docstring against the implementation. Most consequential: the `mcmc` module's
  "Typical workflow" example pre-cropped the PSF with
  `prepare_psf_for_oversamp` and passed the result to `prepare_for_fitting`,
  which prepares it again — the second pass measures an encircled-energy
  fraction of 1.0 instead of the true value, biasing the fitted flux low by
  ~20 % for anyone who followed the example. Also corrected: `shift_psf_fourier`
  claimed a normalisation step it does not perform (and did not mention that
  the shift is circular); `prepare_psf_for_oversamp` promised an output shape
  "always odd in both axes" that an even `native_shape` violates;
  `downsample_psf` described its crop as symmetric and to an "even multiple";
  `estimate_cov_kernel` described a "local RMS" that is global and omitted two
  `ValueError`s it raises; `whiten_image` overstated how rarely clipping fires;
  `prepare_covariance_terms` described its `err` output as masked; the flux
  prior's hard `-3.0` lower bound and the odd-`native_shape` requirement were
  undocumented; `load_emcee_results` omitted `tau_error` and did not say the
  round trip drops `ndim`/`nwalkers`/`nsteps`, ; `plot_psf_fit_triptych` claimed to
  require exactly one of three arguments without enforcing it; and
  `plot_corner` did not state that it returns a flat axes array or that it
  injects three `corner` defaults.
- `prepare_covariance_terms` now delegates to `kernel_power_spectrum` instead
  of repeating the crop/pad/FFT logic. Behaviour is unchanged for odd-sized
  stamps.
- README: expanded the worked-examples section to describe what each notebook
  demonstrates and which functions it exercises; updated the citation to the
  submitted paper reference.
- Documented that `prepare_for_fitting`'s `native_shape` may be larger than the
  data stamp: the default `(11, 11)` against a 9×9 stamp leaves a one-pixel
  margin so an off-centre source is not truncated at the stamp edge.


## [0.1.0] – 2026-09-07

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
  `load_emcee_results`).
- `jwst_psfmc.plot`: data / model / residual triptych (`plot_psf_fit_triptych`),
  walker chain traces (`plot_chains`), `corner.py` wrapper (`plot_corner`).
- Example data in the repository's `examples/` directory (not shipped in the
  installed package): JWST NEXUS F200W cutouts for AT 2025amoq, epochs
  deep_ep02 (non-detection) and deep_ep03 (detection), a F444W example for the
  covariance demo, plus 4× oversampled PSF models. Clone the repository to run
  the notebooks.
- `examples/demo_psf_photometry.ipynb`: end-to-end demonstration notebook
  covering covariance estimation, MCMC for a non-detection (3-σ upper limit)
  and a detection, chain diagnostics, corner plots, and result I/O.
- `examples/demo_covariance_kernel.ipynb`: covariance kernel pipeline from
  source masking through windowing, power spectrum, and noise whitening.
- GitHub Actions CI workflow (Python 3.10 and 3.11).
- Release workflow publishing to PyPI via Trusted Publishing (OIDC), with a
  TestPyPI dry-run target; see `RELEASING.md`.
- Full NumPy-style docstrings on all public functions.

[Unreleased]: https://github.com/mingyangzhuang/jwst_psfmc/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/mingyangzhuang/jwst_psfmc/releases/tag/v0.2.1
[0.2.0]: https://github.com/mingyangzhuang/jwst_psfmc/releases/tag/v0.2.0
[0.1.0]: https://github.com/mingyangzhuang/jwst_psfmc/releases/tag/v0.1.0
