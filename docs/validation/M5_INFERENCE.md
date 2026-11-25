# M5 single-Sérsic inference

M5 validates single-Sérsic inference on synthetic PSFs. It includes
multistart optimization, emcee and Nautilus posterior sampling, and
versioned result archives. Real-star validation remains covered by the
separate [M4 gate](M4_RENDER_DOMAIN.md).

## Use with a prepared model

```python
import numpy as np
from scipy.special import ndtr
from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet, PSFOperator,
    PSFSpec, RenderContext, SersicProfile, SersicSource, Uniform,
    load_model_results, optimize_model_fit, prepare_model_fitting,
    run_model_mcmc, save_model_results, summarize_model_results,
)

axis = np.arange(-64, 65) / 16
response = ndtr((axis + .5) / .7) - ndtr((axis - .5) / .7)
spec = PSFSpec(np.outer(response, response), "effective", 1/16, (64, 64),
               normalization_reference="analytic total flux", source="Gaussian")
operator = PSFOperator(spec, oversamp=8, max_offset=.5)
context = RenderContext((5, 6), psf_operator=operator)
model = CompositeModel([SersicSource("host"), ConstantBackground("sky")])
parameters = ParameterSet([
    Parameter("host.flux", 1, "Jy", prior=Uniform(.2, 2.2)),
    Parameter("host.re", fixed=.7, unit="pixel"),
    Parameter("host.n", fixed=1),
    Parameter("host.q", fixed=.55),
    Parameter("host.pa", fixed=26, unit="deg"),
    Parameter("host.dx", fixed=.25, unit="pixel"),
    Parameter("host.dy", fixed=-.2, unit="pixel"),
    Parameter("sky.level", .01, "Jy / pixel", prior=Uniform(-.09, .13)),
])
sigma = .02
data = operator.render_sersic(SersicProfile(1.1, .7, 1, .55, 26), .25, -.2, context)
data += .018 + sigma * np.random.default_rng(18).normal(size=context.shape)
fit = prepare_model_fitting(data, model, parameters, context=context,
                            noise_options={"err": np.full(context.shape, sigma)},
                            data_unit="Jy / pixel")
optimum = optimize_model_fit(fit, objective="ml")
result = run_model_mcmc(fit, initial=optimum, seed=44,
                        sampling_options={"nsteps": 4000, "burn_in": 500})
save_model_results("posterior.npz", result)
restored = load_model_results("posterior.npz")
if restored.status == "converged":
    summary = summarize_model_results(restored)
```

The Nautilus backend uses the prepared prior and its unit-cube transform.
Pass `sampler="nautilus"`, omit `initial`, and specify for example
`sampling_options={"n_live": 110, "n_eff": 1800}`. The one-dimensional backend
uses an independent uniform auxiliary coordinate internally, integrating it
out of the reported samples. Evidence values and host-detection Bayes factors
are outside this stage. The archive records likelihood normalization.

## Diagnostics and scope

- Fixed morphology caches whitened unit-flux and background templates.
  The Gaussian normalization and mask semantics come from the same prepared
  noise operator as direct model evaluation.
- `objective="ml"` maximizes likelihood within prior support. The default
  `"map"` maximizes the density in sampling coordinates, including its
  Jacobian. A feasible whitened least-squares start is included for
  linear amplitudes. Each attempt records its value, status and message.
- emcee preserves the walker and step axes. At the default convergence
  thresholds, every post-burn-in marginal must have at least 50
  autocorrelation times, approximate ESS of at least 400, and an
  autocorrelation-time estimate within 20% of the first-half estimate.
  These within-ensemble checks are complemented by independent ensembles
  in the validation below. Acceptance fraction is recorded without a
  cutoff. The reported MCSE is an estimate for posterior means.
- Nautilus discards exploration draws and records normalized weights,
  weighted ESS, approximate mean MCSE and likelihood calls. Sampling
  that stops before its target retains an insufficient-sampling status.
  Draws may be unavailable if the run stopped before exploration finished.
- Periodic axes use a wrapped symmetric proposal mixed with emcee's
  stretch move. Nautilus receives the periodic unit-cube indices.
  Quantiles use an arc centered on the estimated mean direction and the
  summary records resultant length. A short resultant signals that
  phase intervals are poorly localized.
- Runtime numerical failures inside prior support return a
  `numerical_failure` result with the reason. Invalid configuration
  raises. All-fixed models use deterministic evaluation. The current
  Sérsic sampler requires fixed n pending a validated continuous n
  domain. A fixed morphology yields conditional flux uncertainties.
- The NPZ contains numeric arrays and JSON metadata with pickle disabled.
  It preserves sampling coordinates and physical free draws, the full
  parameter specification, PSF/noise settings, data and noise input
  digests, seed streams, backend state and sampling diagnostics. Source
  inputs are referenced by digests, so reproduction also needs the
  original arrays. The generic archive has its own schema version.

## Numerical validation

The fixed-morphology single Sérsic fixture independently evaluates the
Gaussian posterior for flux and sky background using generalized least
squares. The 5×6 stamp uses a synthetic sampled Gaussian ePSF and
heterogeneous errors. The correlated case adds finite-boundary
nearest-neighbor correlation of 0.12. Independent noise uses seed 18.
Each reported emcee ensemble is independently initialized. The numerical
checks cover likelihood normalization, linear and logarithmic sampling
coordinates, the periodic seam, and macOS spawn for both backends.

| Fixed-morphology audit | Analytic flux mean | emcee flux mean (seed 44) | Nautilus flux mean (seed 44) | Largest backend difference in combined MCSE, across flux and sky |
| --- | ---: | ---: | ---: | ---: |
| Diagonal noise | 1.270695 | 1.266354 | 1.270496 | 1.30 |
| Correlated noise | 1.308153 | 1.309951 | 1.307498 | 0.69 |

The fixed-shape optimized ML vector agrees with independent GLS to
`2e-7` in both cases. At 4,000 steps with burn-in 500, both seed-44 emcee
ensembles and the correlated seed-45 ensemble meet the checks. The
white-noise seed-45 ensemble initially reported insufficient sampling
because the sky autocorrelation estimate changed 30.1% between the two
windows. A repeat with the same seed and 10,000 steps reduced that change
to 4.6%. The two white-noise ensemble means agree within 1.25 combined
MCSE over the two parameters. The Nautilus weighted ESS values were
1,889 and 1,817 for the diagonal and correlated cases.

The fixed-n nonlinear case uses a 7×8 stamp, n=0.5, flux 1.1, semimajor
half-light radius 1.32 native pixels, q=0.52, PA 38°, centroid offset
(0.19, -0.22), and a background of 0.012. A noiseless optimizer run
recovers all seven free parameters. The tolerances are 0.012° for PA
and 0.0002 in the relevant physical units for all other parameters.
White-noise data use an independent continuous Gaussian convolution
from the M4 oracle, sigma 0.006 and noise seed 912.

| White-noise nonlinear audit | Result |
| --- | --- |
| Only radius free, log sampling coordinate, uniform physical radius prior | The physical-grid reference mean is 1.29018 pixels, and the Nautilus weighted mean is 1.28881 pixels, within 0.75 MCSE. Weighted ESS 513 |
| Seven free parameters, fixed n | Nautilus seed 52, 80 live points and three spawn workers. It met the ESS target with weighted ESS 588 after 5,202 likelihood calls in 57.9 s. The radius median and 16th/84th percentiles are 1.268 and [1.200, 1.341] pixels |

These tests cover deterministic optima, conditional Gaussian and
one-dimensional posterior references, and a single seven-parameter
noise realization. Population coverage, unresolved hosts and PSF
mismatch require later science validation.

Run `python -m pytest tests/test_model_inference.py` for the automated
checks. The synthetic fixtures are `linear_sersic_fit` and
`free_shape_sersic_fit` in that test module.

## Optimization review through M5

The priority is to protect measured fluxes and uncertainties, then reduce
the cost per likelihood evaluation. The following measurements use the
local Python 3.12.11 environment.

| Area | Finding and recommended change |
| --- | --- |
| Legacy prior and walker preparation | A sole masked pixel changed the flux upper bound from 6 to 30 million when its data value changed from 0.2 to one million. The fixed-parameter likelihood stayed exactly the same. In a dedicated compatibility change, derive initialization statistics from the actual fit pixels, allow explicit priors and seeds, and initialize an interior ensemble by rejection. |
| Legacy covariance | A masked residual set to zero inside the full periodic inverse has a different quadratic from the exact observed marginal. The M2 dense path supports that marginal for small stamps. In one 32×32 patch with about 12% missing pixels, two consecutive estimates differed by 0.034 in peak-normalized kernel amplitude. Adding infinite pixels produced a nonfinite kernel. Handle all nonfinite pixels explicitly, benchmark observed-pair normalization, and report overlap and spectrum diagnostics. |
| Real ePSF readiness | Confirm PSFEx origin, spacing and calibrated finite-aperture scale before plotting or fitting the new response against individual stars. With a 535×535 table and nominal 4× spacing, the default 8× source quadrature for a 9×9 stamp would require 1.37 million source cells, above its 500,000-cell guard. Calibrate and verify a fixed support crop before routine use. |
| M3/M4 source integration | For 20 renders of the 7×8, n=0.5 synthetic host, source-mass calculation used 96% of cumulative time, including 75% for adaptive core integration. The FFT used 3%. Reuse unit-flux source masses when morphology stays fixed and the centroid varies. For free morphology, benchmark a vectorized core rule against the adaptive and analytic oracles. |
| Legacy repeated PSF shift | On the distributed F200W examples, caching the forward PSF FFT reduced the 532×532 shift from 10.86 to 6.92 ms and the 36×36 shift from 0.090 to 0.054 ms. The cached and uncached output arrays matched exactly at the checked phase. Reuse the same cache in the likelihood and posterior aperture summaries. The benchmark times the shift component. |
| Sampling and I/O | The fixed-morphology likelihood is already cached, and convergence failures remain explicit. A checkpoint and resume interface would let an insufficient ensemble continue without repeating its early draws. Passing precomputed state once per spawn worker would reduce repeated serialization for large PSF tables. Benchmark worker counts with the actual stamp sizes. |

Legacy behavior changes and production renderer performance changes are
separate follow-up work. Keep the M0 goldens for the compatibility path and
use the M2/M4 reference tests to validate any new default.


## Render accuracy and build provenance

**Sampling status and image accuracy are separate.** A fit prepared with
independent unit-flux `render_references` and a `render_reference_label`
compares the rendered templates at the same fixed geometry and phase.
Use the same output pixels and amplitude normalization. The prepared noise
operator supplies whitening. The result records the measured witness and
the conservative upper bound over the supported flux ranges. A witnessed
threshold exceedance warns during preparation. The three states are
`within_budget`, `over_budget`, and `unvalidated`. Free morphology,
missing references and joint fits without an independent domain remain
unvalidated. Image discrepancy is distinct from bias in science parameters.

New posteriors record this status, a package-source SHA-256 fingerprint,
dependency versions, and checkout commit and dirty status when available.
Existing archives can still be read.
