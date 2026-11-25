# M7 validation and empirical PSF inputs

The later [empirical PSF validation](EMPIRICAL_PSF_SKY.md) records real
stars, leave-one-out reconstruction and independent blank-sky positions.

## Reproduction

Run `python -m scripts.run_m7` from a source checkout or an unpacked source
distribution with the development environment. The JSON design fixes the case parameters, seeds, realization
counts and acceptance criteria. The manifest records the design and code
digests, runtime and per-case measurements. The source digest depends
on package contents. Git commit and dirty status are recorded separately
when the runner is invoked from a checkout. The CSV files preserve trial-level
posteriors and rank data separately. The runner saves diagnostics before
returning exit status 1 for a failed core criterion. The manifest
records expected and completed numerical, SBC and fixed-truth checks,
radius-grid refinement and production sampler parity. The declared
Nautilus spot check is required for the complete gate. Use
`--design PATH --output DIR` to test thresholds without overwriting
the baseline run. Boundary and mismatch results retain their individual
diagnostics.

The runner replaces the manifest at startup with `status: in_progress`
and `acceptance.passes: false`. Execution errors mark it `error` and
record the error. A completed run writes `status: completed` with its
acceptance result, including when a criterion fails. Treat the diagnostics
as current only after completion. One run at a time can write a given
output directory.

- [Design](m7/design.json)
- [Run manifest and diagnostics](m7/manifest.json)
- [Trial-level results](m7/results.csv)
- [SBC ranks](m7/sbc_ranks.csv)
- [Diagnostic plots](m7/diagnostics.png)

The inputs are locally shift-invariant synthetic Gaussian PSFs. Point
injections use analytic pixel integrals. Host references integrate an analytic
Gaussian pixel response against the intrinsic profile, independently of the
production subcell convolution. The M4 asymmetric-response test remains part
of the regression suite.

## Thresholds and inference distinctions

A compares the inverse-covariance-weighted image norm against the 0.1
renderer target. The independent continuous host reference is refined from
512 to 1,024 angular nodes and must agree within 0.005 in the same noise
units. Unit host flux sets the reference known-shape S/N. Background and a
point source can reduce the marginalized host significance.

B tests the actual prepared and whitened template likelihood against
independently assembled covariance/template algebra. The fixed-shape
posterior is sampled with independent draws from a rectangularly truncated
multivariate Gaussian. Each realization uses 512 independent posterior draws.
Rank histograms are checked with a familywise-corrected chi-square test
across the declared observables. Rank observables include physical flux,
aperture flux and relative log likelihood. The radius-only fixed-n test
uses a refined 1D physical-grid CDF and adds radius, circularized radius,
and aperture flux ranks. These calculations test the likelihood and numerical
reference. The emcee and Nautilus runs separately test selected sampler paths
against the exact posterior.

The first exact-rejection sweep exhausted its proposal budget in 24 boundary
realizations, including 22 very compact-source trials and two zero-host
trials. A mathematically equivalent fallback first draws the most truncated
Gaussian marginal within its own bounds, then draws the remaining parameters
conditionally and rejects proposals outside the original prior. Accepted
draws from both methods have the same target. The manifest lists uses of
the fallback and any final failures. Science criteria and seed streams were
held fixed.

Fixed-truth coverage uses central 68% and 95% intervals and 95% Wilson
uncertainty intervals. Core acceptance uses the declared familywise binomial
tests and a maximum absolute standardized mean bias of 0.30. A nonnegative
host prior with true zero flux is assessed via the conditional 95th
posterior percentile. Central interval coverage at that exact boundary is
reported separately. It cannot be interpreted using the same criterion as
the core cases.

C pairs noise realizations across finite and periodic covariance, masks,
uncertainty scaling and controlled model mismatch. The PSF width, wings,
ellipticity and centroid perturbations are synthetic stress settings.
The finite cropped covariance is computed from stationary nearest-neighbor
lags. The kernel estimation experiment additionally generates 65×65
stationary noise fields and estimates the kernel from 33×33 sky cutouts.
The posterior predictive check uses matched paired discrepancies for
whitened norm, standardized residual lag-1 correlations and radial means.
The small stress batches are sensitivity checks.

## Synthetic acceptance

All eight core render cases passed. The maximum core weighted error
was 0.0251. One additional compact boundary case passed at 0.0293.
Three stress cases exceed the original renderer error budget.

| Case | Configuration | Baseline error | Source grid at 32× | PSF spacing at 1/32 pixel | Both refinements |
| --- | --- | ---: | ---: | ---: | ---: |
| a10 | Pre-pixel, n=4, re=3, 2:1 contrast | 0.222 | 0.214 | 0.063 | 0.055 |
| a11 | ePSF, 10:1 contrast, off lattice | 0.660 | 0.650 | 0.197 | 0.190 |
| a12 | Pre-pixel, compact n=4, known-shape S/N 300 | 0.731 | 0.672 | 0.244 | 0.185 |

These post-test refinements are diagnostic. They keep the original cases
and thresholds intact. Expanding PSF support from ±4 to ±5 pixels at
unchanged spacing moves the weighted errors by less than one part in
a million in absolute noise units. These cases are more sensitive to
the tabulated PSF sampling than to source lattice refinement. PSF spacing
refinement here is feasible because the underlying PSF is analytic.
Interpolation of a fixed empirical table cannot recover missing samples.

All 17 SBC rank checks passed across 480 simulations. The radius posterior
CDF changes by at most 0.000057 between the 181- and 361-point grids in
eight predetermined checks. All four sampler checks meet their own
diagnostics, and the largest posterior-mean deviation from an independent
reference is 1.87 combined Monte Carlo standard errors.

| Core fixed-truth case | Trials | 68% interval coverage and 95% CI | 95% interval coverage and 95% CI | Mean bias / median posterior SD |
| --- | ---: | --- | --- | ---: |
| b01 point, diagonal | 120 | 71.7% [63.0%, 79.0%] | 96.7% [91.7%, 98.7%] | +0.075 |
| b02 host, finite correlated | 120 | 65.0% [56.1%, 72.9%] | 96.7% [91.7%, 98.7%] | −0.088 |
| b03 decomposition, finite correlated | 120 | 67.5% [58.7%, 75.2%] | 96.7% [91.7%, 98.7%] | +0.078 |

All 600 fixed-truth realizations and all 112 paired stress realizations
completed after the exact conditional fallback. The unresolved b05 host
has a median 68% interval width of 2.03 at true host flux 1. The zero-host
b04 posterior has median upper 95th percentile 0.972 in its stated flux
scale and central coverage zero at the exact nonnegative boundary.

The paired Gaussian width perturbation changes the mean pure-point host
posterior median from 0.132 to 0.284 in eight paired trials. This
illustrates why the host median is an unsuitable detection rule at a
nonnegative boundary. The eight estimated kernels were accepted under
both finite and periodic backends. Relative to the known finite
covariance, periodic estimates yielded host-flux uncertainty ratios
0.991–1.032 and sky uncertainty ratios 0.976–1.113. One scalar
error correction would give the same ratio to both parameters. For the
illustrative matched posterior predictive check the whitened-norm
paired tail fraction is 0.348, with remaining statistics in the manifest.

The synthetic numerical, calibration and robustness evidence reaches
the declared core targets. High-S/N stress, unresolved decomposition,
and the empirical star and blank-sky validation remain explicit limits.

## Empirical stellar inputs

Start with **15–20 clean stars in one F200W 30 mas mosaic epoch**. Aim for
phase and field coverage, with several stars in each quarter of the
subpixel phase plane. Include a usable range of bright unsaturated stars.
If F444W 60 mas is used scientifically, repeat with **15–20 stars from its
matching mosaic and PSF**. The same stars in both products are useful for
paired comparisons. Other filters and alternate pixel scales only need
separate validation when their ePSFs are used in the intended fits.
An ePSF from the 30 mas product cannot calibrate the 60 mas response.

Send science and error cutouts on their original mosaic grids, any bad-pixel
or neighbor masks, and full-mosaic coordinates, filter, epoch and output
pixel scale. Include the exact PSF export for each configuration and its
export spacing, reference center and normalization aperture. Use a stamp
large enough to test both core and wings and keep the original subpixel
phase.

Training stars can establish deterministic rendering consistency. For
science validation extend to the full usable sample, typically 40–50 in a
deep epoch or 100–200 in a wide epoch from the project provenance, and
prefer held-out stars or spatial/phase-stratified retraining folds. With
20 independent nulls and zero events, the one-sided 95% binomial upper
bound is still about 14%. The pilot is too small to constrain a rare
false-host rate. Real blank-sky injection needs corresponding blank-sky
cutouts with the same noise provenance.
