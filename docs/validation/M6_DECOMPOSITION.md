# M6 synthetic PSF+Sérsic decomposition

The generic model, parameter catalog, PSF operator and likelihood are
shared with M4–M5. The new diagnostics are experimental. They use the
declared model, PSF calibration, priors and noise covariance.

## Diagnostics

Use `CompositeModel([PointSource("point"), SersicSource("host"),
ConstantBackground("sky")])` and the M5 fitting workflow. A shared
center is explicit in the parameter catalog, for example
`Parameter("host.dx", unit="pixel", tie="point.dx")` with the same
unit for the point root. Use independent centers for an offset-source
hypothesis. The model definitions support 11 free parameters with
independent centers and 9 with tied centers. Fixing n leaves 10 and 8,
respectively. M5 inference currently uses fixed n. Define a PSF-only
null by omitting the host component and its parameters.

```python
from jwst_psfmc import evaluate_model_fit, summarize_decomposition

images = evaluate_model_fit(fit, sample_vector, decompose=True)
host_image = images["components"]["host"]
standardized = images["standardized_residual"]
whitened = images["whitened_residual"]
summary = summarize_decomposition(fit, result, aperture=aperture_mask)
```

The decomposition uses a single sampling-space vector for all images. It
returns data, total and named components, residuals and the observed mask.
The standardized image divides by marginal pixel sigma. The whitened
residual is a vector in the same C-order as the valid pixels, since dense
Cholesky whitening mixes pixels. Data and residual images display masked
pixels as NaN.

The summary uses post-burn-in draws or normalized nested posterior weights.
It computes total and aperture flux, host fractions, and circularized
radius on corresponding draws. The aperture defaults to the full rendered
stamp and can be selected independently of the likelihood mask. Fixed shapes
reuse a unit aperture template. For variable shapes the aperture is evaluated
at each draw. A signed point contribution is retained in flux summaries.
The host fraction is defined on draws with nonnegative source components
and positive combined flux. The result reports the probability that this
ratio is defined.

The summary reports physical-parameter weighted correlations, including
point and host flux, size, n and constant background. A fixed parameter
has undefined correlation. The boundary diagnostic reports mass in the
outer 5% of each finite nonperiodic prior interval by default and flags
an edge above 25% mass. Thresholds are configurable. The resolution flag
requires a caller-calibrated semimajor radius threshold in native pixels
and reports when radius is fixed. The near-circular diagnostic reports
mass above q=0.95 and whether q is fixed. A radius fixed below a threshold
describes the fit assumption. The posterior archive preserves fixed and
tied values and permits regeneration of this summary with the original
prepared fit. The sampler status accompanies the summary.

## Synthetic experiments

The independent reference evaluates Gaussian pixel integrals for the
point and continuous radial/angle quadrature for the host. Except where
specified, the synthetic case uses a 7×8 stamp, background 0.02,
per-pixel sigma 0.006, a synthetic Gaussian ePSF with widths (0.68, 0.81)
native pixels, and a host with n=0.5, re=1.4, q=0.67 and PA 32°.
These injections are noiseless with positive likelihood weights.

### Reduction and contrast

A zero host or point component gives identical images and likelihoods
to direct removal in the automated masked, dense-covariance test. The
component images sum to the total. At fixed shape and true host flux
0.7, independent injection and ML fitting gave the following results.

| True point / host ratio | Recovered host flux | Recovered point flux | Whitened residual squared norm |
| ---: | ---: | ---: | ---: |
| 0 | 0.70006 | 0.00036 | 0.000072 |
| 0.5 | 0.69952 | 0.35082 | 0.000152 |
| 2 | 0.69790 | 1.40221 | 0.000853 |
| 10 | 0.68923 | 7.00959 | 0.01631 |

The 10:1 case has a 1.54% low host-flux estimate at the stated noise
scale. Its residual norm of 0.128 exceeds the M4 0.1 numerical target.
This stress case stays outside the claimed accuracy budget.

### Centers, stamp and model mismatch

The offset experiment varies both x centers, host radius, and amplitudes,
with dy fixed. With coincident injection, tying x gives essentially the
same optimum. With a true 0.51-pixel x separation, tying x inflates re
from 1.399 to 1.529 pixels and increases the squared whitened residual
from 0.00076 to 18.0. Allowing independent x centers recovers the injected
separation to about 0.0022 pixel.

Enlarging the stamp from 7×8 to 11×12 changes the matched 2:1 total
host flux from 0.69790 to 0.69785, and increases stamp host flux from
0.6930 to 0.6978. The total and stamp quantities track different
aperture coverage.

Adding a background gradient of 0.003 per x pixel while fitting a
constant sky gives host flux 0.743 with a broad background interval
and 0.720 with the interval [0.0199, 0.0201]. The residual norms
remain about 8.5. Adding a neighboring point of flux 0.15 at
(-2, 1) gives host flux 0.734 and squared residual norm 65.7 for
the baseline model. Inspect the spatial residuals before changing the
background bounds. A background component with a gradient, an explicit
neighbor, or a justified mask with an appropriate noise backend can
address these mismatches.

Under the gradient, fixed-index alternatives n=0.5, 1 and 4 give
total host flux 0.743, 0.913 and 1.629. For n=4, only 1.326 falls on
the stamp. The slightly improved residual objective leaves large
profile/sky sensitivity and an extrapolated wing contribution.
Continuous free-n inference is outside the currently measured renderer
domain.

### Unresolved and pure point-source controls

The compact test fixes re=0.03 pixel and q=0.99. Nearly equal point
and host templates show flux anticorrelation near -0.9995. The
decomposition reports the analysis threshold of 0.25 pixel and marks
the radius as fixed.

The pure-star experiment keeps host flux exactly zero in the injected
image. For the matched control, the ML host flux is essentially zero.
Broadening the injected point PSF in quadrature by 0.22 native pixel
in each axis and fitting with the original narrower ePSF yields an
ML host flux of 0.448, about 32% of the fitted source total. The host
template is fixed with re=0.5, q=1 and n=0.5 in both fits. The false
host in this controlled example measures sensitivity to this specific
PSF mismatch. Real PSFEx null calibration needs matching stellar
cutouts and calibration metadata.

### Posterior audit

The initial emcee runs registered insufficient sampling under the M5
criteria, so these summaries use separate Nautilus runs with seed 43,
100 live points, and a weighted ESS target of 900. All four runs met
that target. Shape and center assumptions are fixed within each run.

| Case | Weighted ESS | Host total flux median [16th, 84th] | Point–host flux correlation |
| --- | ---: | --- | ---: |
| Resolved 2:1 | 984 | 0.699 [0.616, 0.778] | -0.8995 |
| Compact with re=0.03 | 1008 | 1.037 [0.286, 1.726] | -0.9995 |
| Pure point, matched PSF | 989 | 0.115 [0.035, 0.238] | -0.9814 |
| Pure point, mismatched PSF | 929 | 0.440 [0.280, 0.618] | -0.9918 |

The matched null has 76.3% host posterior mass in the lower 5% of its
[0, 4] prior support. Its positive posterior median reflects the
nonnegative prior and measurement uncertainty at the zero boundary.
For this case its conditional 95th percentile is 0.363. The mismatched
null has just 7.3% near that boundary and gives median host fraction
0.314. These are conditional synthetic results for a single controlled
PSF pair. Real-star cutouts are required to measure the empirical
false-host distribution.

Run `python -m pytest tests/test_decomposition.py` for the automated
checks. The `synthetic_case` fixture constructs the independent model
and the noiseless synthetic data.
