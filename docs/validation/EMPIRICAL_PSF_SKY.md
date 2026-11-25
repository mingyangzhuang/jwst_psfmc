# NEXUS ep01 blank-sky and stellar PSF pilot

The original central stellar pilot used the archive OBS 001 template.
The [central workflow PSF check](#ep01-central-workflow-psf-check)
identifies and tests the actual templates used by the central workflow.

## Inputs and design

The matched blank-field inputs are the nine `nexus_central_deep_ep01_*`
FITS files in `blank_field_test`, spanning F200W 30 mas and F444W 30/60
mas. SCI, ERR and exposure counts agree in shape and WCS within each
product. This is a validation of direct imaging. Difference-image noise
from a reference and subtraction pipeline needs the corresponding
difference-image SCI and ERR.

The exposure-map values include a FITS BZERO offset and were decoded
before selecting depths. The analysis maps the mosaics read-only, keeping
work arrays to postage-stamp size. No mosaic contents are copied into
the validation deliverables.

The blank pilot uses distinct nonoverlapping training and held-out
positions within each product. It stratifies uniform exposure counts
1, 2 and 3 and tests mixed exposure boundaries separately. Source vetoes
use the other filter on the same sky coordinates. No clipping on the
target science pixels is applied when selecting held-out stamps.
This supports a signed null test. Subthreshold or single-band sources
can remain. No segmentation catalog was supplied. The F444W 30 mas
3-exposure stratum used 33×33 sky patches after the 49×49 search found no
homogeneous sampled patches of the required size.

For each training bank, the kernel uses valid-pair-normalized
autocorrelation of local background-subtracted SCI/ERR, a 6-sigma
training-only clip and a separable Bartlett lag taper. The original ERR
sets the marginal pixel variance. Photometry uses 13×13 stamps at 30
mas and 9×9 at 60 mas. The finite covariance is positive definite in the
measured groups. Mixed boundaries use an explicitly approximate pooled
correlation and spatially varying ERR. Both fits retain all the pixels
in each selected valid stamp.

The blank-sky fit uses signed linear amplitudes conditional on the
injection position, one constant sky level, and fixed morphology for
the host experiments. The 4× ePSF injects sources at three pixel phases
on the observed background. Nominal 5 and 20 S/N targets use marginalized
GLS amplitude uncertainty, and host injections use a 2:1 point-to-host
ratio. The fixture host has n=1, semimajor re=0.07 arcsec, q=0.67 and
array PA of 32 degrees. Reported noise intervals are analytic Gaussian
intervals for the linear amplitudes. This tests the real background and
covariance with known geometry. The matched-injection recovery is
conditional on that same renderer and template. In the mismatched
experiment the point renderer changes and the host template stays fixed.

The PSFs use the nominal 4× sample spacing and explicit geometric
midpoint. Both paths use a common 0.36-arcsec phase-zero aperture
normalization. The legacy reference includes the 1/8-native-pixel
centering adjustment associated with the even oversampled crop, so the
fixed-position injection comparison uses consistent physical positions.
The resulting flux units follow the supplied image values summed over
pixels in this normalization. The sample geometry and aperture
normalization are declared pilot conventions.

## Stellar checks

All archived OBS groups are fit on their own native cutout grids with
the matching OBS-tagged PSF. This preserves each OBS orientation.
The stored cutout centers provide starting coordinates and the
two-dimensional centroid is then optimized. Those tests use unweighted
relative image residuals.

The same OBS 001 catalog coordinates are used to extract new stamps from
the ep01 central mosaic and attach that mosaic's ERR and coverage.
Only the subset in the downloaded footprint is available. The archived
and mosaic images were verified to be different at these coordinates,
even after integer registration. The raw central-mosaic fits are
tested with ERR weighting and also with uniform weights as a control.

The central-mosaic stellar null fits a signed host plus an unresolved
point after the point-only centroid fit. Each point response is
evaluated at its own fitted centroid. The host uses the fixed fixture
above, also tied to that center. Bright-star interpretation is
conditional on the supplied ERR and local background covariance.
No stellar shot-noise component was synthesized or double-counted.
Reported signed host fractions describe profile mismatch under the
test model and do not specify a host-detection confidence.

## Reproduce

From the source checkout run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.validate_empirical_psf ../blank_field_test ../PSF_test
```

The [manifest](empirical/manifest.json) records training and held-out
counts, training coordinates, covariance estimates and diagnostics.
The per-product CSVs contain signed recovery results, star fits and
stellar null fits. See the [diagnostic figure](empirical/diagnostics.png)
for the coverage, weighting and injection comparisons.

## Blank-sky results

Each product has 24 independent training positions and 24 held-out positions
per uniform depth, plus 24 held-out mixed-coverage positions. This gives
96 held-out positions per product. Each position supplies all injection and
likelihood arms, so those arms are paired. None of the twelve finite matrices
required diagonal shrinkage. The smallest estimated correlation eigenvalue
was 0.0365 for the F444W 30 mas 3-exposure group.

| Product | Null SD, diagonal | Null SD, finite covariance | Finite nominal 95% interval contains zero |
| --- | ---: | ---: | ---: |
| F200W 30 mas | 1.70 | 1.16 | 90/96 |
| F444W 30 mas | 2.48 | 1.12 | 89/96 |
| F444W 60 mas | 1.42 | 1.05 | 90/96 |

The counts are descriptive. The respective Wilson 95% ranges are
[87.0%, 97.1%], [85.7%, 96.4%] and [87.0%, 97.1%]. The overall
scatter improves markedly. Its remaining dependence on depth and boundary
argues for keeping the strata separate.

| Product | 1 exp, finite SD | 2 exp | 3 exp | Mixed |
| --- | ---: | ---: | ---: | ---: |
| F200W 30 mas | 1.42 | 0.92 | 0.63 | 1.48 |
| F444W 30 mas | 0.97 | 1.21 | 1.33 | 0.99 |
| F444W 60 mas | 1.16 | 1.02 | 0.98 | 0.99 |

### Paired injection recovery

| Product | Point S/N 20 median fractional error, matched ePSF | Point S/N 20, legacy point response | Host S/N 20, matched ePSF | Host S/N 20, legacy point response |
| --- | ---: | ---: | ---: | ---: |
| F200W 30 mas | −0.01% | +4.96% | −0.38% | −36.0% |
| F444W 30 mas | +0.34% | +2.00% | −0.04% | −23.8% |
| F444W 60 mas | −0.38% | +4.30% | −1.03% | −45.0% |

The finite backend is used in this table. These specific host biases
are a deliberately mismatched experiment in which only the fitted point
response changes. They show sensitivity to the point template in the
specified decomposition and use a single host morphology and contrast.
The same-response linear null and point-injection standardized residuals
are algebraically identical, as expected.

## Stellar results

The table uses the same successful, positive-flux stars in the archived
cutout, the ep01 central cutout with uniform weight, and the ep01 central
cutout weighted by ERR. The residual is the unweighted L2 residual divided
by the L2 of the background-subtracted data. Values are medians across
stars.

| Product | Matched stars | Archive ePSF | Central ePSF uniform | Central ePSF ERR | Central legacy ERR |
| --- | ---: | ---: | ---: | ---: | ---: |
| F200W 30 mas | 31 | 10.0% | 11.8% | 49.8% | 50.3% |
| F444W 30 mas | 30 | 9.9% | 15.5% | 16.2% | 16.1% |
| F444W 60 mas | 23 | 6.4% | 9.6% | 15.2% | 16.9% |

The median ERR ratio between the inner two native pixels and the outer
ring is 45.5 for F200W 30 mas and about 7.9 for each F444W product.
The data establish this weighting difference. The provenance of the ERR
components needs separate verification. Excluding F200W ePSF stars with
fitted centroid offsets within 0.1 pixel of a bound leaves a 45.2%
ERR-weighted median residual, so the centroid bound alone does not
explain the result. Uniform weighting is used here as a shape control.
It has no calibrated photometric error interpretation.

Archived stars for OBS 002, 012 and 014 are included in the per-product
tables. Their local ERR and count maps were unavailable, so these fits
provide shape checks at the matching OBS orientation. The results do
not transfer the ep01 exposure strata to those epochs.

### False host fits on real stars

The entries are median signed host fractions in the same compact,
fixed-profile test. Counts are stars with successful fits and positive
total flux in the paired renderer comparison.

| Product | Stars per weighting | ePSF ERR + covariance | Legacy ERR + covariance | ePSF uniform | Legacy uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| F200W 30 mas | 30 | +45.2% | +43.7% | +1.2% | −11.7% |
| F444W 30 mas | 29 | −5.1% | −13.9% | +12.9% | +11.9% |
| F444W 60 mas | 23 | +28.7% | +23.5% | +4.2% | −11.5% |

The F200W and F444W 60 mas false host fractions are sensitive to how
the core and wings are weighted. A negative host amplitude is allowed
in this diagnostic and measures the direction of the profile correction.
These fitted fractions depend on the specified template, aperture and
centroid and should be interpreted as PSF mismatch diagnostics.

## F200W Tile 12 variance-component follow-up

The original ep01 Tile 12 i2d product shares the central mosaic pixel
grid up to an integer translation. Of the 30 central ePSF stellar-null
positions, 19 have a complete, valid 13×13 stamp in Tile 12 across SCI,
ERR, WHT and all three variance planes. This sample comprises all valid
overlap stars, including stars associated with neighboring catalog
tiles. The original Tile 12 SCI samples enter the fit directly.

The median inner-to-outer ERR ratio for these stars is 42.9 in the
original Tile 12 and 39.3 in the central mosaic. Within radius 2 pixels
in Tile 12, the median component fractions of total variance are 58.3%
Poisson, 41.3% flat-field and 0.3% read noise. The outer-ring fractions
at radius at least 5 pixels are 87.1%, 11.3% and 1.4%, respectively.
Each fraction is summarized separately across stars and pixels.

The controlled fits use the same OBS 001 ePSF, fixed host profile
(n=1, re=0.07 arcsec, q=0.67 and array PA 32 degrees), pixels and
constant background under each diagonal weighting. Full error uses
the native ERR, the flat-field ablation uses
sqrt(VAR_POISSON + VAR_RNOISE), and the uniform control gives every
pixel equal weight. Centroids are estimated with the point-only model.
The host is fitted at that centroid with a signed amplitude. A second
pass fixes all arms to each star's uniform-weight centroid to isolate
the effect of weights on amplitudes. Brightness is estimated from the
uniform-weight point-only flux on the same tile. These are conditional
profile diagnostics with a shared epoch ePSF and diagonal weights.
Tile-specific pixel covariance and a held-out local PSF are additional
requirements for a calibrated bright-source inference test.

| Diagonal weighting | Median signed host fraction, refitted centroid | Median signed host fraction, common centroid | Median point-only relative residual, refitted centroid |
| --- | ---: | ---: | ---: |
| Full pipeline ERR | +35.7% | +41.5% | 49.5% |
| Poisson + read noise | +31.9% | +34.4% | 43.9% |
| Uniform | +6.9% | +6.9% | 11.9% |

Removing the flat-field term decreases the signed fraction in 18 of 19
stars with the centroid refitted, with a paired median change of −4.6
percentage points. With a common centroid the fraction decreases in all
19 stars, with a paired median change of −7.3 points. The false host
signal persists at +32–34% under Poisson plus read-noise weighting.
The nine stars cataloged in Tile 12 show the same ordering, with
refitted-centroid medians of +40.6%, +36.5% and +8.3%. Excluding all
stars near a centroid bound in any refitted arm leaves 16 stars and
medians of +36.6%, +32.5% and +5.4%.

The Spearman coefficients between brightness and signed host fraction
are −0.28, −0.35 and +0.17 in the three refitted arms. Replacing fitted
brightness with sky-subtracted peak SCI gives −0.16, −0.23 and +0.06.
The data give no evidence for a positive brightness trend in the full
ERR arm within this small overlap sample. These results measure
sensitivity to pixel weights in the presence of profile mismatch.
Removing the flat-field term and uniform weighting are diagnostics.

Run from the checkout root with

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.validate_tile_variance \
    ../blank_field_test/nexus_deep_ep01_f200w_NEXUS-Tile-12_002_i2d.fits \
    ../blank_field_test/nexus_central_deep_ep01_f200w_i2d_error.fits ../PSF_test
```

The [per-star CSV](empirical/F200W_30mas_tile12_variance.csv) records
both centroid choices, fitted fluxes and relative image residuals.

## F200W Tile 12 leave-one-out ePSF

For each of the 19 overlap stars above, the local ePSF is rebuilt from
the other Tile 12 stars. Both training and evaluation use SCI from the
original i2d file. Training uses 25×25 stamps, an outer-ring background,
a data-derived core centroid, and uniform weights on valid pixels.
Target IDs and any intersecting training stamps are excluded. The
builder is Photutils 3.0.0 EPSFBuilder with a 4× grid, quartic
smoothing and up to eight iterations. The local ePSF is evaluated
through the same effective-response renderer and normalized to the
same centered 0.36-arcsec aperture as the epoch PSF.

Evaluation uses identical 13×13 pixels and full diagonal pipeline ERR
for each pair. A uniform-weight control is also reported. Each ePSF
is used for both the point response and the convolution of the fixed
host profile used above. The primary comparison refits the point-only
centroid independently under each weighting and ePSF. The common
centroid check uses the epoch ePSF's uniform-weight centroid for both
templates. Residuals are measured in image space. The epoch template
is the archived PSFEx model. It was not rebuilt with the target
excluded, and builder and training sample differ between templates.
The comparison tests these two templates on Tile 12, rather than
isolating spatial variation from the construction method.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.validate_tile_epsf \
    ../blank_field_test/nexus_deep_ep01_f200w_NEXUS-Tile-12_002_i2d.fits ../PSF_test
```

The [paired stellar results](empirical/F200W_30mas_tile12_epsf_loo.csv)
record the training IDs, fitted centroids, fluxes and residuals.

All 19 folds retained all 18 eligible training stars, with zero
target IDs in their respective training lists. With original ERR and
refitted centroids, the median signed host fraction drops from +35.7%
for OBS 001 to +4.1% for the local ePSF. The median absolute fraction
drops from 35.7% to 6.2%. Both the absolute host fraction and image
residual decrease in all 19 pairs. The median relative image residual
drops from 49.5% to 26.6%. Holding the centroid common gives
+41.5% and +13.0%, and 18 of 19 absolute fractions improve.
Excluding sources near a centroid bound in either full-ERR fit
retains 16 pairs and gives +36.6% and +3.2%.

With uniform weights and refitted centroids, the signed medians are
+6.9% and −3.9%, the absolute medians are 10.0% and 13.2%, and the
relative residuals are 11.9% and 11.4%. The local improvement over
OBS 001 is specific to the full-ERR weighting. The surviving
full-ERR residuals show that a local model does not account for all
of the star-to-model shape differences.

### Archive OBS 002 control

The original Tile 12 i2d header has `OBSERVTN='002'`. For observation
matching, the archive OBS 002 ePSF was evaluated on the identical
stamps. The two archive star tables mark 19 of the tested IDs included
for OBS 001 and 17 for OBS 002. Only the local ePSF has explicit
leave-one-out membership here.

| Template | Median signed host fraction, full ERR | Median absolute host fraction, full ERR | Median image residual, full ERR |
| --- | ---: | ---: | ---: |
| Archive OBS 001 | +35.7% | 35.7% | 49.5% |
| Local leave-one-out | +4.1% | 6.2% | 26.6% |
| Archive OBS 002 | −1.1% | 2.3% | 18.0% |

Holding centroids common, the OBS 002 signed median is −0.9% and its
median absolute fraction is 2.5%. The OBS 002 template also has a
smaller full-ERR point-only objective than the local model in all
19 refitted pairs. Its absolute host fraction is smaller in 12 of 19
pairs. The OBS 002 comparison supports checking observation-level
PSF selection before interpreting the high false-host fraction from
OBS 001. An independent OBS 002 validation would require a PSF built
with the individual evaluation stars held out. This control covers the
original Tile 12 stamps and does not determine which PSF applies
throughout the central mosaic.

To repeat the [OBS 002 control](empirical/F200W_30mas_tile12_epsf_archive_002.csv),
rerun the command above with `--check-obs-002`.

## ep01 central workflow PSF check

The original flat FITS exports for the central SCI, ERR and exposure
counts carry no per-pixel observation identity. The central workflow
PSF files under `obj_39/out` resolve which templates the workflow
uses. All three ep01 4× PSF arrays match the respective archive OBS
002 arrays element for element. This is a check of the workflow
template choice, independent of any assumptions about which inputs
contributed to each mosaic pixel.

The follow-up keeps the original OBS 001 catalog coordinates, stamp
selection, central SCI and ERR, sky kernels and host profile. It
changes the central point template and host convolution to OBS 002,
refitting the centroid for each star. The archive catalog label
identifies the source of the star positions. The workflow PSF match
identifies the template used to fit them.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.validate_central_psf \
    ../blank_field_test ../PSF_test ../obj_39/out
```

Each row uses the successful, finite pairs shared by OBS 001 and OBS
002 for the corresponding metric. The point-only residual is the
unweighted relative image norm after the ERR-weighted fit. Signed host
fractions use full central ERR and the original finite sky covariance.

| Product | Stars in null comparison | Median signed host fraction, 001 → 002 | Median absolute host fraction, 001 → 002 | Point-only residual, 001 → 002 |
| --- | ---: | ---: | ---: | ---: |
| F200W 30 mas | 30 | +45.2% → −0.6% | 45.2% → 2.9% | 49.8% → 21.6% (31 stars) |
| F444W 30 mas | 28 | −6.0% → −23.1% | 18.5% → 29.7% | 15.6% → 14.7% (30 stars) |
| F444W 60 mas | 23 | +28.7% → −1.3% | 28.7% → 14.8% | 15.2% → 10.4% (23 stars) |

The absolute fraction improves in 30 of 30 F200W, 10 of 28 F444W
30 mas and 19 of 23 F444W 60 mas pairs. F444W 30 mas retains a
negative signed profile correction at the median despite the modest
improvement in the point-only residual. Its compact-host
decomposition therefore needs further validation at this sampling.
The F444W 60 mas signed median is small, with a 14.8% typical
absolute fraction. The comparison uses real stars with the archived
workflow templates and does not establish independent training
membership. The `*_central_obs002_stars.csv` and
`*_central_obs002_null.csv` files contain the per-star results. This
follow-up did not rerun the earlier blank-sky injections.

## Workflow PSF injection recheck

For all 12 archived filter, sampling and observation combinations, the
native PSF export agrees with center-aligned every-fourth-sample
evaluation of the 4× export to a maximum relative L2 difference of
3.6e-16 after 0.36-arcsec aperture normalization. This directly checks
the centered native response convention of the supplied files.

The 288 held-out sky positions and their training banks were recovered
from the original manifest and CSVs. The regenerated kernels agree with
the stored kernels to numerical precision. Only the PSF changes to
the verified OBS 002 workflow template. The host injection uses the
same fixed morphology, phase sequence and 2:1 point-to-host contrast.
The legacy comparison changes only the fitted point response.

| Product | Full-covariance zero-flux SD | Nominal 95% interval contains zero | S/N 20 host bias, matched ePSF | S/N 20 host bias, legacy point model |
| --- | ---: | ---: | ---: | ---: |
| F200W 30 mas | 1.15 | 90/96 | −0.3% | −35.4% |
| F444W 30 mas | 1.15 | 87/96 | −0.4% | −81.8% |
| F444W 60 mas | 1.06 | 90/96 | −0.8% | −45.4% |

The low matched-ePSF biases apply to self-consistent injections with
fixed morphology and position. The renderer mismatch is a controlled
stress test. The result demonstrates sensitivity of the compact
decomposition to the point response even at matched coordinates.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.recheck_ep01_injections \
    ../blank_field_test ../PSF_test
```

The new per-position results are in `*_obs002_blank.csv`.

## F444W paired sampling and stellar nulls

The F444W follow-up matches source IDs from the OBS 002 included
catalogs at both pixel scales and reads fresh stamps from the central
mosaics. The fit compares square stamps with half-widths of 0.18,
0.24 and 0.36 arcsec. Exposure count selects the original finite
sky correlation, with a mixed-coverage kernel at boundaries. The
central ERR sets the pixel variance. Diagonal ERR and equal weights
are controls. Point-only centroids are reoptimized per arm and held
fixed for the signed host amplitude at Sérsic n=1, q=0.67, array
PA 32 degrees, and re of 40, 70 or 140 mas.

The reported radial contribution is the sum of the GLS host
projection inside 90 mas, from 90 to 180 mas, and outside 180 mas,
measured from the fitted point centroid. The point and constant
background templates are projected out of the host template first.
With correlated noise, each contribution uses the full precision
applied to the residual, so the bins identify where the conditional
host signal is assigned and are not independent annular measurements.
The three bins are checked numerically to sum to the fitted host
amplitude in every row.

A separate run rebuilds 4× ePSFs directly from the central SCI with
Photutils in leave-one-out folds for each sampling. The reference
sample is drawn from all coverage-valid common catalog positions
across the mosaic. Each fold holds out its source ID and any
intersecting training stamp. Training uses 0.48-arcsec half-width
stamps and no ERR weighting, and tests use finite covariance at
0.24-arcsec half-width. It is a sampling-specific mosaic ePSF
reconstruction, with archive OBS 002 as a separate comparison.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.validate_f444w_psf \
    ../blank_field_test ../PSF_test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.validate_f444w_psf \
    ../blank_field_test ../PSF_test --loo
```

### Stamp, host size and weighting

There are 27 paired stars with valid stamps at all three half-widths.
With archive OBS 002, a 70 mas host, and finite covariance, the
median signed fraction at 30 mas changes from −29.0% to −19.4% to
−13.6% as the stamp grows. The corresponding 60 mas medians are
−7.4%, −6.4% and −5.4%. At 0.24 arcsec, reducing host re to 40 mas
gives −34.5% and −10.6%, while increasing re to 140 mas gives −11.1%
and −3.6%.

At the 0.24-arcsec half-width, the median radial contributions at
30 mas are −4.9, −18.2 and +2.5 percentage points for the core,
inner wings and outer pixels. The three separately computed medians
need not sum to the total median. The 90–180 mas wings account for
the strongest negative projection at both pixel scales. For 30 mas
the median signed fraction changes from −5.7% with diagonal ERR and
−0.9% with equal weights to −19.4% with ERR plus sky covariance.
The 60 mas medians are −5.4%, −6.9% and −6.4%, respectively.

### Leave-one-out reconstruction

The 0.24-arcsec run has 30 common evaluation stars. Each fold retains
31 training stars at each scale, and membership is verified against
the evaluation ID. Use this paired comparison for the 70 mas host
under full ERR plus covariance.

| Sampling | Absolute fraction, archive → leave-one-out | Improvement count | Relative point residual, archive → leave-one-out |
| --- | ---: | ---: | ---: |
| 30 mas | 22.1% → 10.4% | 21/30 | 30.4% → 41.3% |
| 60 mas | 9.8% → 7.3% | 18/30 | 11.1% → 14.3% |

The leave-one-out reconstruction reduces the typical projection onto
the chosen host, while the image residual worsens. It also produces
individual outliers, including a signed-amplitude magnitude above
100% when signed point and host amplitudes partially cancel. These
are conditional mismatch diagnostics. The mosaic-wide reconstruction
does not justify replacing the archived workflow ePSF.

See the [paired diagnostics](empirical/F444W_paired_diagnostics.png),
[radial results](empirical/F444W_paired_radial.csv), and
[leave-one-out results](empirical/F444W_paired_loo.csv).

### Centroid control

At 0.24 arcsec and host re=70 mas, the baseline centroids from the
dense and diagonal fits were exchanged while all other inputs stayed
fixed. Reported values are median signed host fractions for the same
27 stars.

| Sampling | Diagonal weights, diagonal centroid | Dense weights, diagonal centroid | Diagonal weights, dense centroid | Dense weights, dense centroid |
| --- | ---: | ---: | ---: | ---: |
| 30 mas | −5.7% | −16.3% | −4.1% | −19.4% |
| 60 mas | −5.4% | −6.3% | −5.2% | −6.4% |

The median displacement between the dense and diagonal point
centroids is 5.9 mas for 30 mas data and 0.7 mas for 60 mas data.
At 30 mas the negative projection persists with a fixed centroid
as the likelihood weighting changes. Its dependence on the adopted
covariance does not establish that diagonal uncertainties are
better calibrated. The blank-sky tests provide the independent
calibration for background-dominated stamps. Separation of
source-variance contributions in the bright-star fits requires the
original F444W variance planes or independent exposures.

The [centroid results](empirical/F444W_paired_centroid.csv) are
reproduced with `--centroid-control` after the radial run.
All 59 targeted noise, rendering and decomposition regression
tests passed.
