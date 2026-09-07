# jwst_psfmc

**PSF photometry with MCMC for JWST (and HST) drizzled images.**

[![CI](https://github.com/mingyangzhuang/jwst_psfmc/actions/workflows/ci.yml/badge.svg)](https://github.com/mingyangzhuang/jwst_psfmc/actions)
[![PyPI](https://img.shields.io/pypi/v/jwst-psfmc?cacheSeconds=300)](https://pypi.org/project/jwst-psfmc/)
[![Python](https://img.shields.io/pypi/pyversions/jwst-psfmc?cacheSeconds=300)](https://pypi.org/project/jwst-psfmc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Scientific Background

JWST NIRCam images processed with the [drizzle algorithm](https://drizzlepac.readthedocs.io)
exhibit **spatially correlated pixel-to-pixel noise**: photons detected in a single detector
pixel are spread across several output pixels during co-addition.  If these correlations are
ignored in a standard χ² PSF fit, the flux uncertainties are systematically *underestimated*.
  Even in full MCMC fitting with correlated-noise likelihoods, neglecting the covariance kernel
  underestimates flux uncertainties by ~30% in F200W (Zhuang et al., *NEXUS: Transient Searches
  and First Results from Year One Observations*, submitted).

`jwst_psfmc` solves this by:

1. **Measuring the covariance structure** directly from source-free sky regions of the
   drizzled image using an autocorrelation estimator.
2. **Encoding the covariance as a Fourier-space power spectrum**, enabling an exact
   correlated-noise log-likelihood that is *O(N log N)* per MCMC step.
3. **Running an ensemble MCMC sampler** ([`emcee`](https://emcee.readthedocs.io)) to obtain
   full posterior distributions for the four model parameters:

   | Parameter | Description |
   |-----------|-------------|
   | `flux`    | Total source flux (same units as the image) |
   | `dx`      | Sub-pixel x-shift of the PSF centroid (native pixels) |
   | `dy`      | Sub-pixel y-shift of the PSF centroid (native pixels) |
   | `bkg`     | Flat residual background level |

The PSF is shifted with **exact Fourier-space interpolation** (no interpolation kernel
artefacts) and downsampled from the 4× oversampled PSF model by block-summing.

---

## Installation

```bash
pip install jwst-psfmc
```

Or from source:

```bash
git clone https://github.com/mingyangzhuang/jwst_psfmc.git
cd jwst_psfmc
pip install -e ".[dev]"
```

**Dependencies:** `numpy`, `scipy`, `astropy`, `photutils`, `emcee`, `corner`, `matplotlib`

---

## Example Data

The example FITS files (difference images, error maps, covariance kernels, and PSF models)
live in `examples/data/` and `examples/PSF/` inside the repository. They are **not**
bundled with the PyPI package — clone the repo to use them:

```bash
git clone https://github.com/mingyangzhuang/jwst_psfmc.git
cd jwst_psfmc/examples
jupyter notebook demo_psf_photometry.ipynb
```

| File | Description |
|------|-------------|
| `examples/data/example1_f200w_diff.fits` | Difference image — non-detection |
| `examples/data/example1_f200w_diff_error.fits` | Uncertainty map — non-detection |
| `examples/data/example1_f200w_cov_kernel.fits` | Covariance kernel — non-detection |
| `examples/data/example2_f200w_diff.fits` | Difference image — detection |
| `examples/data/example2_f200w_diff_error.fits` | Uncertainty map — detection |
| `examples/data/example2_f200w_cov_kernel.fits` | Covariance kernel — detection |
| `examples/PSF/example1_f200w_PSF_4_c.fits` | 4× oversampled PSF model — non-detection |
| `examples/PSF/example2_f200w_PSF_4_c.fits` | 4× oversampled PSF model — detection |

---

## Quick Start

The core API accepts NumPy arrays directly — load your own FITS files however you prefer:

```python
from astropy.io import fits
import numpy as np
import jwst_psfmc as jpm

# ── 1. Load your data ──────────────────────────────────────────────────────
data    = fits.getdata("examples/data/example2_f200w_diff.fits")
err     = fits.getdata("examples/data/example2_f200w_diff_error.fits")
kernel  = fits.getdata("examples/data/example2_f200w_cov_kernel.fits")
psf_raw = fits.getdata("examples/PSF/example2_f200w_PSF_4_c.fits")

# ── 2. Extract a 9×9 stamp centred on the source ───────────────────────────
cy, cx = data.shape[0] // 2, data.shape[1] // 2
stamp     = data[cy-4:cy+5, cx-4:cx+5]
stamp_err = err[cy-4:cy+5, cx-4:cx+5]

# ── 3. Prepare MCMC inputs ─────────────────────────────────────────────────
fit_args = jpm.prepare_for_fitting(
    data=stamp, err=stamp_err,
    psf_os=psf_raw, cov_kernel=kernel,
    dx_init=0.0, dy_init=0.0,
    oversamp=4, native_shape=(11, 11), nwalkers=48,
)

# ── 4. Run MCMC ────────────────────────────────────────────────────────────
sampler = jpm.run_mcmc(**fit_args, nsteps=2000, ncores=4, progress=True)

# ── 5. Summarise posterior ─────────────────────────────────────────────────
summary = jpm.summarize_emcee(sampler, burnin=500, thin=4)
print(f"flux = {summary['flux']['median']:.4f} "
      f"+{summary['flux']['plus_1sigma']:.4f} / "
      f"-{summary['flux']['minus_1sigma']:.4f}")

# ── 6. Visualise ───────────────────────────────────────────────────────────
fig_trip, _ = jpm.plot_psf_fit_triptych(
    stamp, stamp_err, fit_args["psf_os"], oversamp=4,
    native_shape=fit_args["native_shape"],
    psf_prepare_fraction=fit_args["psf_prepare_fraction"],
    summary=summary, fig_title="Detection example — F200W",
)
fig_trip.savefig("triptych.pdf", bbox_inches="tight")

fig_chains, _ = jpm.plot_chains(sampler, burnin=500)
fig_corner, _ = jpm.plot_corner(sampler, burnin=500, thin=4)

# ── 7. Save / reload results ───────────────────────────────────────────────
jpm.save_emcee_results("example2_f200w_mcmc.npz", sampler)
res = jpm.load_emcee_results("example2_f200w_mcmc.npz")
print(res["summary"])
```

### Deriving a flux upper limit from a non-detection

Residual small-scale background fluctuations can mimic low-level source
emission at 1–2σ. We use a **two-run workflow**:

1. **Detection check** — broad priors (±1 pix on dx/dy) to confirm the
   source is genuinely absent.
2. **Upper limit** — tight priors (±0.1 pix) to derive the 3σ bound from
   the posterior flux distribution.

Two conventions for the 3σ bound are supported, and they answer slightly
different questions:

| Convention | Definition | When to prefer it |
|---|---|---|
| **99.7th percentile** | The flux below which 99.7 % of the posterior samples fall | Makes no assumption about the shape of the posterior. Correct when the flux posterior is skewed or truncated (e.g. by a non-negative flux prior). |
| **3 × σ** | `median + 3 × std` of the posterior | Matches the convention most transient and SN-rate papers quote, so it is the comparable number when placing limits alongside published work. |

For a clean non-detection with a flux prior spanning negative values the
posterior is near-Gaussian and centred on zero, so the two agree closely;
their divergence is itself a useful diagnostic that the posterior is
non-Gaussian and that the percentile should be trusted. Whichever you adopt,
state it explicitly when publishing — the two are not interchangeable.

```python
data_nd   = fits.getdata("examples/data/example1_f200w_diff.fits")
err_nd    = fits.getdata("examples/data/example1_f200w_diff_error.fits")
kernel_nd = fits.getdata("examples/data/example1_f200w_cov_kernel.fits")
psf_nd    = fits.getdata("examples/PSF/example1_f200w_PSF_4_c.fits")

stamp_nd     = data_nd[cy-4:cy+5, cx-4:cx+5]
stamp_err_nd = err_nd[cy-4:cy+5, cx-4:cx+5]

# ── Run 1: broad priors (detection check) ─────────────────────────────
fit_nd_broad = jpm.prepare_for_fitting(data=stamp_nd, err=stamp_err_nd,
                                       psf_os=psf_nd, cov_kernel=kernel_nd)
sampler_nd_broad = jpm.run_mcmc(**fit_nd_broad, nsteps=2000, ncores=4,
                                progress=True)

# Confirm non-detection: flux median consistent with zero within 1σ
summary_broad = jpm.summarize_emcee(sampler_nd_broad, burnin=500, thin=4)
is_absent = abs(summary_broad['flux']['median']) < summary_broad['flux']['plus_1sigma']
print(f'Non-detection confirmed: {is_absent}')

# ── Run 2: tight priors (upper limit) ─────────────────────────────────
fit_nd_tight = jpm.prepare_for_fitting(data=stamp_nd, err=stamp_err_nd,
                                       psf_os=psf_nd, cov_kernel=kernel_nd,
                                       pos_prior_half_width=0.1)  # ±0.1 pix
sampler_nd_tight = jpm.run_mcmc(**fit_nd_tight, nsteps=2000, ncores=4,
                                progress=True)

flat_flux = sampler_nd_tight.get_chain(discard=500, thin=4, flat=True)[:, 0]

# Convention A — posterior quantile (distribution-free)
ul_percentile = float(np.percentile(flat_flux, 99.7))

# Convention B — Gaussian-equivalent, median + 3 x sigma
ul_3sigma = float(np.median(flat_flux) + 3.0 * np.std(flat_flux, ddof=1))

print(f"3-sigma upper limit (99.7th percentile): {ul_percentile:.4f}")
print(f"3-sigma upper limit (median + 3*sigma):  {ul_3sigma:.4f}")
```

---

## Full Worked Examples

Both notebooks run end to end against the bundled example data. Clone the
repository to use them — the FITS files live in `examples/` and are not shipped
inside the installed package.

### Covariance kernel estimation

See **[`examples/demo_covariance_kernel.ipynb`](examples/demo_covariance_kernel.ipynb)**
for a step-by-step demonstration of measuring the pixel-to-pixel covariance of a
JWST difference image (F444W example):

- Source masking and locating source-free sky regions (`get_source_mask`,
  `find_zero_squares`)
- Autocorrelation-based kernel estimation (`estimate_cov_kernel`)
- Cosine-bell windowing (`SplitCosineBellWindow`) and the Fourier power spectrum
  (`kernel_power_spectrum`)
- Whitening the sky patch (`whiten_image`) and confirming what it does: because
  the kernel spectrum has unit mean, the RMS is preserved (ratio 1.0002), while
  the nearest-neighbour pixel correlation falls from 0.61 to 0.03

### PSF photometry with MCMC

See **[`examples/demo_psf_photometry.ipynb`](examples/demo_psf_photometry.ipynb)** for a
step-by-step notebook covering both regimes, using two epochs of the same source:

- Loading and inspecting the JWST cutout data
- Estimating and visualising the covariance kernel and its power spectrum
- **Non-detection** (example1) — the two-run workflow, and a 3σ upper limit
  reported under both the 99.7th-percentile and `median + 3σ` conventions
- **Detection** (example2) — flux posterior, detection significance, and the
  total flux corrected for PSF energy falling outside the stamp
  (`summarize_flux_from_chain`); in this example ~23 % of the PSF lies outside
  the 9×9 cutout, so the correction matters
- Diagnosing convergence via chain traces and autocorrelation times
- Producing corner plots and residual triptychs
- Saving and reloading posteriors (`save_emcee_results` / `load_emcee_results`)

---

## Module Overview

| Module | Contents |
|--------|----------|
| `jwst_psfmc.psf` | PSF shifting, downsampling, model evaluation |
| `jwst_psfmc.covariance` | Covariance kernel estimation, power spectrum, whitening, Fourier-space pre-computation |
| `jwst_psfmc.mcmc` | MCMC preparation, log-prob, `run_mcmc`, posterior summaries |
| `jwst_psfmc.io` | Save/load `.npz` results |
| `jwst_psfmc.plot` | Triptych, chain traces, corner plot |

---

## Citation

If you use `jwst_psfmc` in your research, please cite:

> Zhuang et al. (arXiv:xxxxxx[]), *NEXUS: Transient Searches and First Results from Year One Observations*

---

## License

MIT — see [LICENSE](LICENSE).
