# Validation log

## 2026-11-18 — M1 / PR 1A

Base commit `38f0834d5fdd977c9c761907d0ab59a82bc5ae90` on `master`, with
pre-existing M0 files, ignore rules and a comment-only covariance diff in the
working tree. M1 follows section 4 of the roadmap in
`AI_md_files/jwst_psfmc` and M0 ADR 003. The M1 numerical tests use
synthetic parameter definitions and a scalar likelihood. The combined
regression suite also exercises the M0 FITS/PSF/kernel fixtures.

### Parameter contract

`ParameterSet` uses declaration order for `free_names` and resolves equality
ties to a shared free or fixed root. Physical constraints apply to the entire
root prior support, and ties require matching unit labels. An all-fixed set
uses an empty vector and has log prior zero.

```python
from jwst_psfmc import Parameter, ParameterSet, Uniform, spawn_seed_streams

parameters = ParameterSet([
    Parameter("agn.dx", 0, "pixel", prior=Uniform(-1, 1)),
    Parameter("host.dx", unit="pixel", tie="agn.dx"),
    Parameter("host.re", 2, "pixel", prior=Uniform(0.2, 8), transform="log"),
    Parameter("host.n", fixed=1),
    Parameter("host.pa", 0, "deg", prior=Uniform(-90, 90), period=180, origin=-90),
])
physical_values = parameters.decode(parameters.initial)
log_prior = parameters.log_prior_in_sampling_space(parameters.initial)
sample = parameters.prior_transform([0.5, 0.5, 0.5])

streams = spawn_seed_streams(42)
seed_metadata = {name: child.state for name, child in streams.items()}
worker_streams = streams["sampler"].spawn(4)
```

- The log coordinate adds the log-Jacobian to the physical log density. The
  cube transform applies the physical inverse CDF and then encodes the result.
  A future nested sampler using this transform supplies the likelihood alone.
  A Normal prior requires cube coordinates strictly between zero and one.
- Periodic PA values use a canonical half-open interval. Decoding wraps
  angles, and log-density evaluation is supported on only the canonical
  interval. Periodic sampling and circular posterior summaries belong with
  the future generic sampler and diagnostics.
- `to_schema()` records versioned JSON-compatible definitions. Full fit
  serialization and seed provenance capture are part of generic inference.
- A physical domain can accept signed difference flux or enforce nonnegative
  host flux via explicit priors and bounds. M1 verifies zero host flux and
  circular shape at the parameter level. Identifiability and comparisons to
  a model with the host removed require the later model and inference tests.

### Validation

Python 3.12.11, NumPy 2.5.3, SciPy 1.18.1. The isolated M0 environment
uses pytest 7.4.2. The local `py312` environment uses pytest 9.2.0 and
Nautilus 1.0.6.

| Check | Acceptance and result |
| --- | --- |
| Parameter order, units, fixed values and chained ties | Exact order and resolution, one prior factor per free root, invalid configurations rejected |
| Normalization and change of variables | All four physical densities integrated to one within `1e-11`, including a Gaussian truncated to 9–12 standard deviations |
| Same physical prior in linear and log coordinates | Numerical evidence, posterior mean and CDF agree to absolute tolerance `1e-12` for a Gaussian likelihood in physical radius |
| Unit-cube transform | Independent CDF recovered for analytic quantiles and 32 draws with seed 790, within absolute tolerance `2e-12` |
| PA, circular shape, signed and zero flux, all-fixed | Boundary and deterministic-vector checks pass |
| RNG | Root seed 18 replays from child state. Initialization, noise, sampler and worker keys differ |
| M1 test file | 35 passed |
| Full suite in `.venv-sersic` | 156 passed, 1 skipped (optional Nautilus), 5 warnings in 47.63 s |
| Full suite in `py312` with Nautilus | 157 passed, 5 warnings in 70.28 s, including the legacy Nautilus end-to-end test |

**M1 gate passed.** The five suite warnings are the existing photutils
`npixels` deprecations and blank-sky no-detection warnings.
Actual new sampler behavior and stochastic posterior calibration are M5 or
later. Existing demo notebook execution was outside this M1 test run.

## 2026-11-18 — M2 / PR 1B

Base commit `38f0834d5fdd977c9c761907d0ab59a82bc5ae90` on `master`, with
local M0 and M1 changes. M2 follows section 5 of the roadmap in
`AI_md_files/jwst_psfmc` and M0 ADR 002. The likelihood conditions on a
fixed covariance. Its default includes the Gaussian log determinant and
normalization constant.

| Backend | Covariance | Missing pixels |
| --- | --- | --- |
| `diagonal` | Per-pixel `err**2` | Observed pixels only |
| `dense` | Explicit covariance in image units, or `err` and a kernel with an explicit finite or periodic boundary | Exact observed principal submatrix |
| `fft_periodic` | `err` and a kernel under periodic boundaries | Fully observed grid required |

Boolean masks use True to exclude a pixel. Nonfinite data also exclude a
pixel. An observed pixel with nonfinite or nonpositive `err` raises an error.
Noise metadata identifies the boundary, Gaussian normalization, error
provenance, invalid pixel positions and weight handling. Exactness refers to
the declared covariance. The periodic boundary and fixed estimated kernel
are assumptions about the data. Soft weights remain available through the
legacy API. The dense guard defaults to a 128 MiB estimated preparation
workspace budget, using six float64 observed-matrix allocations as its
estimate. Caller-owned covariance is outside that budget.

Opposite kernel lags within `1e-7` on the unit correlation scale are averaged,
with the maximum absolute correction recorded. This accommodates the small
asymmetry measured in both shipped kernels. Larger asymmetries or a
nonpositive periodic spectrum raise an error. A periodic kernel must fit on
the output grid. Use an explicit finite boundary for a small stamp with a
larger kernel when that matches the covariance model.

### Deterministic model evaluation

```python
import numpy as np
from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet,
    evaluate_model_fit, prepare_model_fitting,
)

data = np.full((3, 4), 0.12)
model = CompositeModel([ConstantBackground("sky")])
parameters = ParameterSet([Parameter("sky.level", fixed=0.10)])
fit = prepare_model_fitting(data, model, parameters, noise_options={"err": np.full_like(data, 0.03)})
result = evaluate_model_fit(fit)
```

`PointSource` delegates to a supplied unit point response in `RenderContext`.
An analytic pixel-integrated Gaussian test checks odd/even geometry and
subpixel offsets. Production ePSF and pre-pixel response operators belong to
M4. The legacy adapter directly calls the existing PSF renderer. The legacy
likelihood now calls a residual-only helper with its original standardization,
mask, soft weights and relative normalization.

### Independent checks

The synthetic oracles use direct pairwise lag construction and dense
`solve`, `slogdet`, Cholesky and eigendecomposition. The periodic audit uses
seed 20261118 and the masked audit uses seed 46. There are 23 noise tests and
10 model and legacy bridge tests.

| Verification | Result |
| --- | --- |
| FFT quadratic and log likelihood on 5×5, 4×6 and 5×4 heterogeneous-error grids | Maximum absolute difference from the independent dense oracle was `7.11e-15` for each |
| FFT log determinant | Maximum absolute difference was `2.66e-15` |
| FFT whitening versus the symmetric inverse square root of the correlation matrix applied to standardized residuals | Maximum elementwise absolute difference was `3.22e-15` |
| Masked finite and periodic dense noise | Matches direct observed-submatrix likelihood. Zeroing masked residuals in the full solve gives a different quadratic in both cases |
| Legacy image and likelihood | Exact equality on six fixed-theta, shape and phase cases, including signed flux and invalid/soft-weighted pixels. Eight M0 regressions also pass |

Both shipped F200W kernels were checked with the corresponding difference
image and error map. The finite test used a central 9×9 cutout with its
central pixel masked, producing 80 observed pixels in each case. The
periodic test used a full 15×15 cutout.

| Input | Finite masked log determinant | Periodic raw minimum power | Maximum kernel correction |
| --- | ---: | ---: | ---: |
| `example1_f200w` | -635.43185 | 0.287467 | `2.54e-8` |
| `example2_f200w` | -581.11275 | 0.274414 | `3.73e-9` |

These FITS checks establish fixed-covariance preparation on those stamps.
They use the supplied uncertainty maps and kernels as inputs. Source
rendering, kernel estimation uncertainty and noise from bright sources
require their own validation.

**M2 gate passed.** The full suite ran in `py312` with Python 3.12.11,
NumPy 2.5.3, SciPy 1.18.1, pytest 9.2.0 and Nautilus 1.0.6. All 190 tests
passed, including the legacy Nautilus end-to-end test. There were five
existing photutils warnings. Suite wall time was 56.96 s and peak process
RSS was 368.4 MiB. The deterministic evaluation example above also ran
successfully.

## 2026-11-18 — M3 / PR 2A

Base commit `38f0834d5fdd977c9c761907d0ab59a82bc5ae90` on `master`, with
local M0–M2 changes. M3 follows the coordinate and flux conventions in
sections 2 and 6 of the roadmap in `AI_md_files/jwst_psfmc`. Tests use
deterministic synthetic intrinsic profiles.

`SersicProfile` normalizes to nonnegative total flux over the infinite
intrinsic profile. `re` is the semimajor-axis half-light radius in native
pixels and `re_circ` is `re * sqrt(q)`. PA rotates from +x toward +y in
array coordinates. Native and oversampled integrals return flux per cell.
A finite stamp retains its aperture losses.

```python
from jwst_psfmc import SersicProfile, render_intrinsic_sersic

host = SersicProfile(flux=1.0, re=0.7, n=2.0, q=0.65, pa=35.0)
native = render_intrinsic_sersic(host, (7, 8), dx=0.25, dy=-0.5)
subcells = render_intrinsic_sersic(host, (7, 8), dx=0.25, dy=-0.5, oversamp=2)
rebinned = subcells.reshape(7, 2, 8, 2).sum(axis=(1, 3))
```

`integrate_sersic_grid` also accepts nonuniform cell edges relative to the
source. The reference integrator evaluates the radial flux analytically
through the regularized gamma functions. It integrates the remaining angle
adaptively, with breaks at corner rays, axis directions, and intersections
of cell edges with enclosed-flux ellipses. The last set resolves narrow
angular transitions when the source lies close to a pixel boundary.
Default absolute tolerance is `1e-10` of total flux per cell and default
relative tolerance is `1e-9`. Quadrature reports failed cells with their
indices. Tolerances apply to individual cells.

### Independent checks

| Verification | Result |
| --- | --- |
| Rotated Gaussian brightness | Matches an independently constructed Gaussian covariance and normalization |
| Gaussian pixel flux | Matches products of erf integrals on odd, even and rectangular grids, with phase offsets and a 90° rotation |
| Brightness at `n = 1, 4` | Matches Astropy after converting total flux to intensity at `re` using independent constants |
| Pixel flux at `n = 1, 4` | Matches independent Cartesian quadrature for narrow, rotated profiles, including a cusp at `n = 4`, `re = 0.18` pixel |
| Total and half-light flux | Direct radial integrals agree within `5e-10` at `n = 0.5, 1, 2, 4`. Integer-shape gamma CDF series independently checks the curve of growth |
| Grid and tolerance refinement | Native results match the tightened reference and 2×/4× integrated subcells rebinned to native pixels within `2e-10`. Cases include `n = 4`, `re = 0.03` pixel |
| Flux and extent | Exact linear flux scaling and matching overlapping pixels on centered crop sizes. Zero flux and circular PA degeneracy pass |

The near-corner Gaussian stress sweep covered three radii from 0.03 to 0.8
pixel and distances from the pixel corner of 0.01 to `1e-10` pixel along each
axis. The largest absolute difference from the analytic pixel flux was
`2.08e-12` at unit total flux.

### Accuracy and runtime

On a 9×9 native stamp with unit total flux, the reference renderer took
54–74 ms per stamp (median of three runs per case). A 16× uniform midpoint
sum took 0.52–0.55 ms. Its errors depend strongly on the profile and phase.
Differences below are in units of total intrinsic flux.

| n | re (pixel) | q | PA (deg) | (dx, dy) | 16× midpoint maximum absolute pixel error | 16× midpoint stamp flux error |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0.5 | 1.2 | 0.55 | 0 | (0.25, -0.5) | 0.000051 | +0.000000049 |
| 1 | 0.7 | 0.44 | 26 | (0.25, 0) | 0.00038 | -0.00010 |
| 2 | 0.7 | 0.74 | 108 | (0.49, -0.5) | 0.00054 | -0.00121 |
| 4 | 0.18 | 0.39 | 61 | (0, 0) | 0.09005 | -0.09031 |
| 4 | 0.03 | 0.30 | 49 | (0.49, -0.5) | 0.23527 | -0.31613 |

**M3 gate passed.** All 224 tests passed, including 34 new M3 tests
and the legacy Nautilus end-to-end test. The suite ran in `py312` with
Python 3.12.11, NumPy 2.5.3, SciPy 1.18.1, pytest 9.2.0 and Nautilus 1.0.6.
There were five existing photutils warnings. Suite wall time was 60.28 s
and peak process RSS was 359.4 MiB. M4 uses these intrinsic reference
results to validate PSF convolution, response support and the rendering
domain for science fitting.

## 2026-11-21 — M4 / PR 2B

Base commit `38f0834d5fdd977c9c761907d0ab59a82bc5ae90` on `master`, with
local M0–M3 changes. See [M4 validated render domain](M4_RENDER_DOMAIN.md)
for the response conventions, discrete cases, references, runtime, and
real-data gate.

**The synthetic rendering gate passed.** Point and Sérsic sources share
the same response operator for pre-pixel density and ePSF samples. The
largest measured noise-weighted host error across the synthetic audit
was 0.01607 at total flux 1 and per-pixel sigma 0.02, below the roadmap
target of 0.1. Independent discrete, analytic and continuous comparisons
cover source subcells, crop and finite support.

All 251 tests passed, including 27 new M4 tests and the existing Nautilus
end-to-end test. The suite ran in `py312` with Python 3.12.11, NumPy 2.5.3,
SciPy 1.18.1, pytest 9.2.0 and Nautilus 1.0.6. The five warnings are the
existing photutils warnings. Wall time was 62.62 s and peak process RSS
was 351.4 MiB. The real PSFEx gate is pending matching star cutouts and
verification of export spacing, origin, and flux calibration.

## 2026-11-21 — M5 / PR 3

Base commit `38f0834d5fdd977c9c761907d0ab59a82bc5ae90` on `master`, with
local M0–M4 changes. See [M5 single-Sérsic inference](M5_INFERENCE.md)
for the API, convergence criteria, sample archive, and benchmark figures.

**The synthetic M5 gate passed.** Fixed-morphology white and correlated
single-Sérsic posteriors agree with an independent GLS reference. A
fixed-n, seven-parameter noiseless fit recovers the synthetic model. The
white-noise nonlinear audits include a numerical posterior reference for
radius and a seven-parameter Nautilus posterior that meets its weighted
ESS target. Insufficient and numerical-failure results also round-trip.

All 262 automated tests passed in 156.35 s, including 11 new M5 tests
and the legacy Nautilus test. The run used the local Python 3.12.11
`py312` environment with NumPy 2.5.3, SciPy 1.18.1, pytest 9.2.0 and
Nautilus 1.0.6. The five warnings are the existing photutils warnings.
Real stellar cutouts and PSFEx calibration metadata are pending for M4.

## 2026-11-21 — Legacy point-source follow-up

### Mask-consistent priors and initialization

The automatic flux upper bound and walker initialization use the effective
valid-pixel mask in the prepared likelihood. A test replaces the values of
four excluded pixels with values up to four million. The default flux
interval remains (-3, 6), and the seeded initial ensemble is identical.
An explicit `flux_prior_bounds=(low, high)` overrides the two legacy flux
heuristics. Walkers are sampled strictly inside the open intervals, and an
ensemble large enough for emcee must pass its full-rank check. Tests cover
a narrow, entirely negative flux interval and reproducible emcee sampling
with independent initialization and sampling streams.

### Equivalent PSF FFT caching

The legacy emcee and Nautilus samplers and the stamp-flux summary compute
the fixed forward PSF transform once per operation. Their model synthesis
shares the same Fourier shift, downsampling, crop and flux bookkeeping
as the direct model. Each fit uses a local transform of its supplied PSF.

A single-process microbenchmark used the shipped F200W detection and its
9×9 data stamp. Values are medians over five batches of 25 fixed parameters
covering signed flux and different centroid phases. All 50 direct and
cached likelihood values agreed bitwise across the two PSF sizes.

| Oversampled PSF array | Direct likelihood per evaluation | Cached likelihood per evaluation |
| --- | ---: | ---: |
| 44×44, default 11×11 native PSF | 0.217 ms | 0.194 ms |
| 532×532, full odd native PSF | 13.787 ms | 9.965 ms |

A macOS smoke test also produced bitwise-identical chains and log
posteriors with the same seed in serial emcee and emcee with two workers.
The timings above measure individual serial likelihood evaluations.

The full suite passed all 271 tests in 134.96 s, including the M0
goldens and both legacy sampler paths. The five warnings are the
existing photutils warnings. A separate targeted run passed all nine
new regressions against the final source.

## 2026-11-22 — M6 / PR 4

Base commit `38f0834d5fdd977c9c761907d0ab59a82bc5ae90` on `master`, with
local M0–M5 and legacy follow-up changes. See the
[M6 decomposition report](M6_DECOMPOSITION.md) for the diagnostics API,
controls and measured sensitivity.

The synthetic reduction and decomposition gate passed. Zero-flux
components and direct removal agree exactly on images and masked dense
likelihoods. The audit covers a resolved host, compact degeneracy,
wrong centroid ties, high contrast, and pure-star PSF mismatch. The
10:1 stress case and mismatched PSF and background models remain outside
the measured accuracy target.

All 280 automated tests passed in 133.08 s in the local Python 3.12.11
environment. The five warnings are the existing photutils warnings.
The nine M6 tests passed again against the final diagnostics. Real
stellar null tests and continuous free-n inference remain pending.

## 2026-11-22 — M7 synthetic validation

See [M7 validation and empirical inputs](M7_VALIDATION.md). The run design,
case-level manifest, numerical results, SBC ranks, and diagnostic plot are
under `m7/`. The final source-checkout run completed in 59.86 s and
used 268.11 MiB peak process memory.

The declared synthetic core targets passed across independent image
comparisons, posterior calibration and mismatch audits. The 8 core
render cases meet the 0.1 error budget, and 3 out of 4 boundary/stress
render cases exceed it. All 17 rank tests, all 3 core fixed-truth cases,
and all 4 sampler spot checks passed. Twenty-four boundary realizations
from the initial rejection budget were completed with exact conditional
sampling, preserving the target distribution. The final output contains
5,892 result rows and 2,040 rank rows with verified script, design and
reference hashes.

The full package regression suite passed all 280 tests in 144.57 s,
with five existing photutils warnings. The empirical gate still
requires original-grid real stars, matching ePSF exports, and relevant
blank-sky cutouts.

## 2026-11-22 — Empirical ep01 validation

See [the blank-sky and stellar validation report](EMPIRICAL_PSF_SKY.md)
for the matched input grids, held-out nulls, paired injections and
real-star signed host diagnostics. Across 288 held-out positions,
finite covariance brings the per-product standardized null scatter
to 1.05–1.16. The F200W mosaic stellar fits are sensitive to delivered
ERR weighting. OBS 002, 012 and 014 have archive-based shape checks.
The blank-sky calibration covers the ep01 direct mosaics.

All 59 targeted noise, rendering and decomposition regression tests
passed in 23.88 s.

The [Tile 12 variance follow-up](EMPIRICAL_PSF_SKY.md#f200w-tile-12-variance-component-follow-up)
compares the original F200W ERR to Poisson plus read noise and uniform
weights on 19 matched stellar stamps. Omitting the flat-field variance
reduces the fitted false host fraction in 18 of 19 stars with centroids
refitted. The median false host fraction remains 31.9%.

The [Tile 12 ePSF comparison](EMPIRICAL_PSF_SKY.md#f200w-tile-12-leave-one-out-epsf)
uses 19 local leave-one-out folds. With full diagonal ERR, the median
absolute false host fraction is 35.7% for archive OBS 001, 6.2% for the
local ePSF and 2.3% for archive OBS 002. The Tile 12 FITS header
identifies observation 002. The archive PSFs have training overlap
with the evaluation stars.

The [central workflow PSF check](EMPIRICAL_PSF_SKY.md#ep01-central-workflow-psf-check)
identifies the ep01 templates for all three products as identical to
archive OBS 002. On matched central stars, the median absolute false
host fraction is 2.9% for F200W 30 mas and 14.8% for F444W 60 mas.
It is 29.7% for F444W 30 mas, which needs further compact-host
validation.

The [F444W paired PSF follow-up](EMPIRICAL_PSF_SKY.md#f444w-paired-sampling-and-stellar-nulls)
tests common sky positions, stamp and host sizes, radial residual
projection, leave-one-out PSFs, and exchanged centroids. The largest
negative compact-host projection at 30 mas comes from 90–180 mas
and persists at fixed centroids under the sky-covariance weighting.
The leave-one-out templates improve the false-host median but worsen
the point-only residual. The workflow-PSF blank injections were also
repeated at all 288 held-out sky positions.

## 2026-11-22 — M8 results, examples, and packaging

See [M8 validation](M8_DELIVERY.md). The generic posterior NPZ and
FITS snapshots were read back and used to recreate diagnostic plots.
The six object 88 difference stamps agree with the old likelihood
to 2.2e-14 through the exact bridge. The change from the legacy to
effective PSF was quantified on a common aperture scale against a
deterministic posterior reference. Four generic MCMC runs retain
the insufficient-sampling status. The resolved-host direct mock
converged, recovered both injected fluxes within the 16th–84th
percentile intervals, and reproduced the component sum exactly.
The wheel, sdist and installed-wheel archive smoke checks completed.
The full suite passed 281 tests.

## 2026-11-22 — M9 joint epoch fitting

See [M9 joint fitting](M9_JOINT_FITTING.md). The prepared joint API
shares global roots and their priors while keeping per-frame PSFs,
noise, geometry and registration provenance. Independent inputs use
summed likelihoods. Reused exposures require an explicit block
covariance. The shared-reference oracle reproduces the cross terms
and the resulting nonvanishing common noise contribution.

The three-epoch direct mock converged, reproduced the GLS optimum
and posterior center, and round-tripped through the generic NPZ
archive. The full test suite passed 286 cases, with separate emcee
spawn and Nautilus smoke checks.

## 2026-11-22 — M10 real-epoch validation pilot

See [M10 real-epoch validation](M10_REAL_EPOCHS.md). Object 88 F200W
direct cutouts provide 54 training positions and 55 held-out sky
positions across nine epochs, with the corresponding workflow PSFs.
Finite covariance and training-only ERR scaling give shared-host
normalized scatter of 1.00 and 53/55 nominal 95% coverage for a
matched-template injection. Relative translations from the local
field are window-dependent. The exposure manifest and adopted
astrometry are still needed for a calibrated target fit.

## M0–M9 审查整改

2026-11-23. Baseline HEAD remains `38f0834` with pending changes.

- Joint preparation checks declared local fields and resolved free roots.
  The two-frame alias regression gives ΔlogL = 8 and an unchanged early
  model. A misspelled late-frame key or an unused free root fails at
  preparation. Reference IDs are stripped before duplicate checks.
- Dense preparation checks the observed joint dimension before any
  frame factorization. The shared-reference builder checks allocation
  before eigenvalue and projection operations. The 243-pixel review
  example is rejected under the 315,000-byte limit. Hash conversion
  uses bounded chunks and preserves the existing hash format.
- [Render accuracy](M5_INFERENCE.md#render-accuracy-and-build-provenance)
  is a separate posterior field. The a03 control passes. The a11
  fit warns before sampling with witnessed ε ≈ 0.66 and preserves
  `over_budget` in its NPZ even though the sampler converges.
  Configurations without an applicable reference remain `unvalidated`.
- New posteriors contain the running package fingerprint, key dependency
  versions and checkout information when available. An unmodified old
  M8 and M9 NPZ could be read. An offline wheel installed outside the
  checkout also produced its own package fingerprint and read the old
  joint archive.
- M7 passed all 33 automatic core checks with the default design.
  Tightening the parity limit from 5 to 1e-6 rejected all four spot
  checks, returned exit code 1 and retained both CSVs, the diagnostic
  figure and the failure manifest. The [default manifest](m7/manifest.json)
  records each criterion. M9's real-target follow-up remains paused.

Final regression on the updated worktree passed 292 tests in 130.56 s
with five existing Photutils warnings. The M9 three-epoch example
converged and its new archive preserved source and dependency metadata.

### M10 real-target pairwise follow-up

Recovered the adopted F200W shifts from the pre-explosion coadd
and aligned late-epoch FITS headers. Ran 168 diagnostic fits on
the object 88 target position, with distinct pre-explosion
single-epoch host templates and three stamp sizes. At 15×15,
42 pre-event pairs are within 1.67 nominal sigma of zero.
Late-epoch fluxes remain sensitive to position and host template,
particularly in ep09. The full F200W exposure manifest is still
needed for a joint science fit. See [M10](M10_REAL_EPOCHS.md).

### M10 footprint audit and real-target joint fit

Audited F200W observation groups using the supplied program 5105
footprint, the ep01 tile HDRTAB, the ep07 catalog name and ep08/09
single-exposure cutouts. The six early groups have 96 detector
records each. The supplied footprint lacks program 9263, so the
nine-frame fit uses the reduction owner's confirmation of disjoint
inputs. The M9 API likelihood check covers the six epochs with full
observation-level footprint lists.

Fit a shared static host on the nine native direct epochs using
per-frame PSFs, finite pixel covariance and recorded alignment.
The seven pre-transient holdouts span −4.74 to +4.30 nJy. The
baseline six direct fits give 26.2–29.0 nJy in ep08 and 8.0–10.1
nJy in ep09 at fixed position. All hit the 12-pixel host radius
limit. Raising that limit to 30 pixels in the 21×21 clump model
changes the two late fluxes by 0.16 and 0.20 nJy in absolute value.
See [M10](M10_REAL_EPOCHS.md#joint-fit-results) for the
model comparison and the remaining provenance and host limitations.
Cached the immutable intrinsic Sérsic cell grid across matching
PSF renders. A nine-frame warm/cold comparison was array-identical.
The full regression suite passed 292 tests.

### M10 local reference, ERR and late-epoch sensitivity follow-up

The F200W wide-radius direct model was profiled over a common
transient position and tested against registration jitter and a
late ePSF swap. With the original ePSFs and 0.15 pixel illustrative
jitter, the ep08 and ep09 measurements are 30.9 ± 3.5 and
8.4 ± 4.0 nJy. The prior width is not an astrometry measurement.

Verified the saved pre-SN difference construction for both late
epochs in F200W and F444W 30/60 mas. Propagated the common
reference into the pixel covariance blocks and checked the
predicted flux contrast errors on held-out sky. The newer
round-2 ep08, ep09 and DDT products have a separate empirical
flux covariance check. Audited five detector products' variance
planes and tested ERR against known exposure counts in the
earlier OBS 002 controls. The full program 9263 detector
inventory and drizzled F444W variance decomposition remain open.
See [M10 local follow-up](M10_REAL_EPOCHS.md#local-product-follow-up).
All 59 targeted tests passed.

The reduction owner confirmed matching exposure times, dithers
and observing setup for programs 9263 and 5105. The late
`aligned_images/` products are direct SCI/ERR cutouts on the
wide_ep01 grid. Their anchor offsets agree with the saved
`diff_preSN` construction for ep08 and ep09. The distinction
between that product and the newer round-2 subtraction was
verified separately.

### M10 three aligned late direct epochs and recovered 60 mas block

Cross-convolved ep08/09/DDT aligned direct cutouts against three
single pre-SN epochs in F200W and F444W at both samplings. The
ep06 F444W 30 mas control has insufficient common sky positions
and is recorded as skipped. F200W DDT is consistent with zero,
and F444W shows substantial sensitivity to the chosen template.

The round-2 F444W 60 mas reference can be recovered from the
aligned and difference images up to a sky constant. A three-epoch
pixel block estimated from this reconstruction gives held-out
contrast scatters 0.89, 1.03 and 0.94. The two 30 mas products
fail the analogous reference closure. See [the aligned three-epoch
tests](M10_REAL_EPOCHS.md#three-late-direct-cutouts-including-ddt).
All 32 joint and noise regression tests passed.

### Post-M10 release preflight

Built the 0.3.0 source distribution from the current worktree in
a temporary source directory, then built the wheel from the
unpacked source distribution. Installed the wheel offline in a
separate directory and verified the import location and package
version. The installed wheel read the legacy four-parameter emcee
archive and the M8 and M9 model archives. Its three-frame joint
ML solution matched the closed-form GLS reference within 3e-6
per free parameter.

The full Python 3.12 regression suite passed 292 tests in 129 s,
with five existing Photutils warnings. Setuptools warned about
the existing license metadata syntax. The Python 3.10/3.11
matrix is part of GitHub CI. The changelog has an Unreleased
section for the upgrade. No version bump, commit, tag or upload
was part of this local check.

### M7 and M9 follow-up review fixes

M7 now atomically replaces the acceptance manifest at startup and
on completion. An invalid design replaces a previous pass with an
error record. An interruption leaves a nonpassing in-progress record,
and concurrent runs cannot write the same output directory. The
default rerun passed all 33 acceptance checks. Reusing its output
directory with a stricter sampler-parity threshold returned exit
status 1, reported four failed checks, and retained the current
diagnostics and failure manifest.

M9 now applies the top-level dense workspace cap to independent
frames and honors stricter frame limits. Regressions exercise both
explicit matrices and finite correlation kernels, including a masked
stamp whose observed submatrix fits the lower budget. All 296 tests
passed on Python 3.12 with the five existing Photutils warnings.

### Source-distribution and M7 provenance follow-up

The source distribution now includes the M7 runner and synthetic
design, along with the package markers for test imports. From a
freshly unpacked archive, pytest collected without import errors and
finished with 165 passing tests and five expected checkout-only
module skips. The complete M7 gate also passed in the unpacked
archive without Git metadata. No FITS or NPZ files were present in
the archive.

Regenerated the checkout M7 manifest, trial table, rank table and
diagnostic figure. All 33 acceptance checks passed. The completed
manifest's runner, source, design and independent reference
fingerprints match the current files.

### Final review follow-up

Removed the deprecated Photutils keyword at the call site by using
the shared positional argument. The only filtered warning is the
expected no-source message inside the source-mask helper, which
already returns an empty mask in this case. The three source-mask
tests passed with warnings treated as errors. A focused check also
passed with Photutils 1.11.0. The full suite passed 296 tests without
warnings in the Python 3.12 and Photutils 3.0.0 environment.

M7 now uses the normal package source fingerprint, with Git commit
and dirty state recorded independently. The checkout and a newly
unpacked source distribution produced the same source fingerprint
and each passed all 33 M7 acceptance checks. The checkout M7 outputs
were regenerated.
