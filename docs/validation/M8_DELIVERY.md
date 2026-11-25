# M8 result archives and end-to-end validation

## Saving a result

Use `save_model_results` and `load_model_results` for the versioned NPZ
posterior. The legacy emcee and Nautilus archive functions are still
available. Use `save_model_snapshot` to evaluate a particular vector
and persist a FITS bundle for exact image inspection and plotting.
The snapshot contains data, total, components, the valid-pixel mask,
the image-space residual and standardized residual, and the whitened
vector in operator output order. A snapshot label identifies the
chosen vector. The snapshot reproduces that evaluation; a refit uses
the demo configuration and the original inputs referenced in the
posterior metadata.

Pass the WCS of the fitted stamp to the FITS writer. The real-data
demo slices the original WCS with the same region used for SCI and ERR
and verifies the saved stamp coordinates to 1e-10 degrees. Figure
generation uses only the reloaded snapshot. The residual map shows
data minus model. The standardized residual uses the marginal pixel
standard deviation; the whitened result is plotted as a vector.

```python
from jwst_psfmc import load_model_snapshot, plot_model_fit

snapshot = load_model_snapshot("direct.fits")
figure, axes = plot_model_fit(snapshot)
figure.savefig("direct_replot.png")
```

## Two executable examples

Run from the source checkout. Set `MPLBACKEND=Agg` for a headless run.

```bash
python -m examples.demo_model_difference ../cutouts ../obj_39/out docs/validation/m8/difference
python -m examples.demo_model_difference ../cutouts ../obj_39/out docs/validation/m8/difference --grid-only
python -m examples.demo_model_decomposition docs/validation/m8/direct
python -m scripts.benchmark_m8 docs/validation/m8/benchmark.json
```

The difference demo expects the six object 88 SCI, ERR and covariance
kernel triplets from deep ep02–04 and the matching epoch PSF FITS
files. It takes 9×9 pixels around the catalog position and converts
SCI and ERR from MJy/sr to µJy per output pixel using the WCS pixel
area. The inputs precede discovery. The centroid posterior can be
broad and is conditional on the catalog-centered position prior.

The strict legacy check uses the same renderer through the new
composition interface and an explicit covariance derived from the
legacy periodic power spectrum. The migration comparison then samples
an effective ePSF with the same noise operator and physically aligned
position priors. Both fluxes are reported on the same centered 9×9
phase-zero aperture scale, and the legacy crop origin is accounted
for. The model choice is intentionally different in the migration
comparison. Its shift in flux or centroid does not measure a change
in the noise operator. Each output records a separately checked
generic posterior round trip and an archived legacy chain round trip.

The deterministic reference integrates the uniform centroid prior on
33×33 and 65×65 grids. At each centroid it integrates the truncated
bivariate Gaussian in point flux and sky level. Both orders of the
conditional Gaussian integral are checked. Grid refinement changed any
reported quantile by at most 0.12% of the legacy interval half-width.
On the common aperture scale the effective ePSF shifted the median flux
by at most 0.17 legacy interval half-widths, reduced the 16th–84th
percentile interval width by 2.9–6.1%, and shifted either centroid
coordinate by at most 0.031 native pixel. These are weak difference
stamps, and the changes in centroid medians are relative to broad
posteriors.

All six likelihood bridge comparisons agree to at most 2.2e-14 in
log likelihood. In the 5,200-step runs, two generic ensembles met
all autocorrelation and ESS criteria and four retained the
`insufficient_sampling` status. The saved 16th, 50th and 84th MCMC
percentiles differ from the same-model grid reference by at most
0.108 interval half-widths across the legacy and generic runs.
Use the deterministic reference for this comparison of weak sources.
The saved sampler statuses remain meaningful for use of the chains.
The [run comparison](m8/difference/comparison.json) records the
posterior summaries and timings. The
[quadrature results](m8/difference/quadrature.json) record both grid
resolutions and both response models.

The direct demo is a fixed-geometry mock, made with the example F200W
ePSF, a point source, and a resolved exponential host on a 25×25
stamp. It fits the two amplitudes and constant sky under finite
correlated noise. The input amplitudes of 1.10 and 0.80 µJy are both
inside the sampled 16th–84th percentile intervals. The recovered
medians are 1.03 and 0.93 µJy, respectively. The loaded components
sum exactly to the total image. Geometry, PSF support and aperture
choices are recorded in the run.

## Performance benchmark

The [timing JSON](m8/benchmark.json) covers stationary Gaussian errors
for 25, 51 and 101 pixel square stamps, plus single-core point+Sérsic
renders at n=1 and n=4 with fixed PSF support and phases. The default
dense timing is limited to 25 pixels per axis. The 101-pixel renderer
case meets the benchmark only with an explicit source-grid budget of
1,250,000 cells. The default budget rejects the 1,058,841-cell case
and that rejection is recorded. Approximate warm render times at
25, 51 and 101 pixels were 0.17–0.19 s, 0.47–0.48 s and 1.38–1.62 s.
The benchmark process peaked at about 175 MiB RSS. These measurements
apply to the recorded runtime and fixture. Sampling timings are
recorded separately in the demo outputs.

## Packaging check

Built the wheel and sdist from a clean temporary source directory.
Installed the wheel outside the checkout and verified that it can load
an existing legacy NPZ, load a generic posterior, and regenerate a
component figure from a FITS snapshot. The sdist was used to build
a second wheel offline. Setuptools reported its deprecation warnings
for the existing license metadata. The full test suite passed all
281 tests with five existing photutils warnings. The snapshot
round-trip test also passed after tightening WCS validation.

## Empirical scope

M7 found PSF and covariance sensitivity for bright point sources
with compact weak hosts in ep01 F444W 30 mas. See the
[F444W stellar validation](EMPIRICAL_PSF_SKY.md#f444w-paired-sampling-and-stellar-nulls).
The direct mock establishes a reproducible example at fixed
morphology. The real difference test checks signed point-source
photometry through both entry points. Calibration of difference-image
upper limits still needs held-out difference blanks from the same
subtraction workflow.
