# M10 real-epoch validation pilot

The development roadmap ended at M9. M10 extends the joint-fitting
validation to real NEXUS direct-image backgrounds. The pilot uses
the nine object 88 F200W cutout pairs, with the saved central
workflow PSFs for ep01 through ep09. The recorded fit is a
flux-recovery experiment at blank positions around the target.
The real transient and its host are outside that evaluation region.
The follow-up below adds pairwise tests at the real target.

The report and validation scripts are included in the repository. The
observing inputs and generated results under `docs/validation/m10/` are
local files. Links into that directory refer to outputs produced by the
reproduction commands below.

## Input audit

- Each F200W SCI and ERR pair has the same 401×401 sampling and a
  common WCS across epochs to within 1e-5 pixel at the tested points.
  Both are converted from MJy/sr to µJy per output pixel from the WCS
  pixel area. WCS agreement checks the grid, while residual physical
  registration at the target requires its own check.
- The nine 4× PSFs match their corresponding 1× files when evaluated
  at aligned native pixel centers and normalized to the same
  0.36-arcsec aperture. The largest aperture-vector L2 difference
  is 6.9e-17. File names and this sampling test establish the
  supplied PSF pairing and convention.
- No exposure-level identifiers are retained in the direct cutout
  FITS headers. The older individual-exposure diagnostic covers
  selected other products and does not establish the input list
  for these nine direct F200W cutouts. The fit below is conditional
  on block-diagonal epoch noise. Mosaic labels are not substituted
  for exposure identities in the M9 API.

## Tests on held-out sky

A buffered source mask is built from F444W on the same grid.
The central 70-pixel radius around object 88 is excluded. Complete,
nonoverlapping 25×25 patches are selected at common coordinates
across all nine epochs, then split reproducibly into 54 training
and 55 evaluation patches. F200W science values are left unclipped
in the held-out selection. Faint sources beyond the F444W veto
can remain.

The epoch-specific finite covariance kernels are estimated from the
training F200W patches only. Fits use their central 15×15 pixels.
An optional second training step takes the scatter of signed
point-only flux divided by its estimated error as an epoch-specific
ERR multiplier. These factors are applied to held-out data only
after training. The baseline diagonal and unscaled covariance
results are retained.

The injected scene has a shared 0.160 µJy Sérsic host with
n=1, semimajor re=2.3 pixels, q=0.72 and array PA=34 degrees.
The simulated point is offset from the host. The point amplitudes
are zero in ep01–07, 0.040 µJy in ep08 and 0.019 µJy in ep09.
The GLS fit allows signed point amplitudes in all epochs and one
constant sky level per epoch. Host geometry and the four pixel
phases are known by construction. The host and point templates
are injected and fitted with the corresponding epoch ePSFs.
This tests recovery and uncertainty on real backgrounds, conditional
on the known morphology and PSFs and the adopted covariance.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.validate_m10_epochs \
    ../cutouts ../obj_39/out '../interesting objs/2026abck/PSFs' docs/validation/m10
```

The [summary](m10/summary.json) records the input paths, coordinates,
noise estimates and the output tables. The [per-position CSV](m10/held_out.csv)
contains signed recovery and point-only null results.

## Held-out results

The table reports the standard deviation of (recovered − true)
flux divided by its fitted uncertainty. The Gaussian target is 1.

| Parameter | ERR only | Finite covariance | Finite + training ERR scaling |
| --- | ---: | ---: | ---: |
| Shared host | 1.44 | 1.04 | 1.00 |
| ep01 point | 1.43 | 0.96 | 1.15 |
| ep03 point | 1.86 | 1.32 | 0.94 |
| ep08 point | 1.77 | 1.31 | 1.17 |
| ep09 point | 1.70 | 1.24 | 1.06 |

For the shared host, nominal 95% intervals contain truth in 45,
51 and 53 of 55 positions, respectively. With training ERR scales,
the median flux errors are −0.31 nJy for the shared host, −0.47 nJy
for ep08 and +0.14 nJy for ep09. Individual epochs can respond
differently to the training calibration, including the degradation
in ep01. The 55 common positions give descriptive coverage, with
calibration uncertainties from the finite sky banks.

With finite covariance, the median cross-epoch correlation of
held-out point-only standardized errors across 36 pairs is
0.002, and the maximum absolute correlation is 0.30. This
describes forced fluxes in blank positions. It cannot establish
exposure independence or the pixel cross blocks near the host.

See the [calibration figure](m10/calibration.png) for all epochs.

## Local registration check

Each epoch and ep01 were cross-convolved with the other's centered
PSF. A robust translation fit then used four annuli around object
88, allowing a photometric scale and constant sky. The central
6–9 pixels were excluded even in the innermost fits. Pixels near
missing data were excluded before interpolation.

The two inner annuli, 6–22 and 9–28 pixels, give shifts differing
by 0.53–1.88 pixels across epochs. All fits in the innermost annulus
are inside the bounds. Thirteen of the 16 outer-annulus fits reach
the ±1.5 pixel limit in at least one coordinate. These shifts are
window-dependent. They do not give a stable astrometric correction
for the real target. See the [registration control](m10/registration.csv).
Run the same command with `--registration-only` to repeat it.

## Target follow-up with recorded registration

The pre-explosion coadd FITS header records all seven Deep
epoch shifts and the `wide_ep01` reference grid in its `EPOCHnn`
and `SHIFTnn` cards. The ep08 and ep09 aligned FITS headers
identify the same reference and record visually inspected
anchor offsets. The coadd contributes registration metadata only.
Each host template is a single direct pre-explosion epoch.

The check reapplies the recorded shifts to the native cutouts
with cubic interpolation. Science and template are each convolved
with the other's sampled and shifted PSF. The point response is
forwarded through the same interpolation and convolution. Fits
recover a signed point amplitude and a constant sky term on 11,
15 and 19 pixel squares. This is a cross-convolved diagnostic
with empirical host templates and an alternative resampling
method to the original Fourier registration pipeline.

The two propagated variance maps provide a spatial scale proxy.
Finite covariance and a scalar aperture calibration are measured
after matching on separate off-target training patches for each
pair. At least 27 training and 31 independent-of-training control
positions remain per fit. The median control scatter over all
pair and stamp-size combinations is 0.99, with a range of
0.73–1.55. The resulting uncertainties are conditional on the
background model and do not capture all host-local mismatch.

There are 42 ordered pre-explosion pairs in the 15×15 test.
All contain zero within their nominal 95% flux interval.
The largest absolute standardized residual is 1.67. Pairwise
tests share input epochs, so these counts do not constitute
42 independent trials.

| Science epoch | Fixed-position median | Floating-position median | Floating range across seven templates | Median fixed-position sky uncertainty |
| --- | ---: | ---: | ---: | ---: |
| ep08 | 29.2 nJy | 32.4 nJy | 28.0–33.0 nJy | 4.0 nJy |
| ep09 | 10.4 nJy | 13.6 nJy | 7.9–16.0 nJy | 4.1 nJy |

The comparison uses 15×15 stamps and allows offsets up to one
pixel per axis. One ep09 template hits the position limit.
The floating position changes the median flux by about 3 nJy
in both epochs. The existing adopted host-subtraction values
before extinction and aperture corrections are 37.8 and
16.6 nJy. These values come from different host subtraction,
image weighting and rendering choices, so the difference is
a method sensitivity check. The pairwise ranges share the
same science image and should not be combined as independent
flux estimates.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.validate_m10_target \
    ../cutouts ../obj_39/out '../interesting objs/2026abck' docs/validation/m10
```

See the [paired fluxes](m10/target_pairs.csv) and
[target summary](m10/target_summary.json).

## Direct-image joint fit after the exposure audit

The supplied `nexus_cal_footprint_coverage.fits` contains 3,712
detector records from program 5105 across all filters. For F200W,
the audit assigns `wide_ep01` to OBS 001/014/015 and Deep ep01–06
to OBS 002/004/006/008/010/012. The six Deep groups each have 96
distinct detector records. The local ep01 Tile-12 `HDRTAB` gives
exactly the same 96 normalized detector IDs as the footprint table
for 5105/002. These are observation-wide sets and can contain
detectors outside the target stamp.

The ep07 catalog name and the local ep08/09 detector files identify
program 9263 OBS 002/004/006, respectively. The last three epochs are
outside the supplied footprint table. Program and OBS together
distinguish them from 5105 observations reusing those OBS numbers.
The reduction owner confirms that program 9263 uses the same exposure
times, dither pattern and observing setup as the corresponding program
5105 observations. This supports matching the nominal observing
design. The realized local depth still follows each image's ERR map,
and detector-file IDs are used for the exposure inventory.
The nine-frame direct fit uses the survey reduction owner's
confirmation of disjoint inputs. The report distinguishes that
confirmation from the complete local detector inventory available
for ep01–06. The M9 API is used for a pixel likelihood cross-check
on those six epochs. With a local copy of
`nexus_cal_footprint_coverage_9263.fits`, pass `--coverage-9263` to
include its detector IDs and cross-check all covered epochs through
the M9 API. The wide image supplies the registration grid and stays
outside this likelihood.

The target fit uses the native direct SCI and ERR stamps, a
separate PSF, finite pixel covariance and sky level in each epoch,
and one shared static host. The recorded translations are applied
to the forward model in each native frame, so the data and noise
pixels stay on their original grids. Covariance and scalar ERR
calibration are measured on the same off-target training bank used
in the sky test. The two transient amplitudes are signed, at fixed
reference RA/Dec, and ep01–07 have zero transient flux in the joint
fit. Host model comparisons use a single exponential, a single
Sérsic with free index, and an exponential plus three PSF-shaped
static clumps. The clump positions come from the existing F200W
host notebook. The amplitudes are fitted here. Fluxes use a
0.36-arcsec phase-zero PSF aperture convention and no extinction
correction. Local likelihood errors are evaluated from a linearized
joint fit and are conditional on the selected host model, PSFs,
and registration.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -m scripts.validate_m10_joint \
    ../cutouts ../obj_39/out '../interesting objs/2026abck' \
    ../nexus_cal_footprint_coverage.fits docs/validation/m10 \
    --ep01-tile ../blank_field_test/nexus_deep_ep01_f200w_NEXUS-Tile-12_002_i2d.fits
```

The [joint fit summary](m10/joint_target_summary.json) records the
exposure groups and detector IDs, calibration and field controls,
model fits, and signed pre-transient fluxes from withheld epochs.
For the withheld fits, model geometry and amplitudes are fitted to
the other six pre-transient images. The static clump coordinates
are preselected from the existing analysis and the quoted null
errors are conditional on the fitted host. This checks the
transfer of the fitted model between epochs.

### Joint fit results

The quoted errors are local statistical uncertainties in nJy.
The table has no correction for extinction or for throughput
outside the PSF calibration aperture.

| Stamp | Static host | ep08 (nJy) | ep09 (nJy) | χ² / d.o.f. |
| --- | --- | ---: | ---: | ---: |
| 15×15 | Exponential | 28.34 ± 3.02 | 10.12 ± 3.81 | 1.074 |
| 15×15 | Free-n Sérsic | 26.24 ± 3.03 | 8.05 ± 3.82 | 1.060 |
| 15×15 | Exponential and static clumps | 28.11 ± 3.14 | 9.67 ± 3.91 | 1.009 |
| 21×21 | Exponential | 28.46 ± 2.98 | 9.58 ± 3.72 | 1.091 |
| 21×21 | Free-n Sérsic | 26.83 ± 2.98 | 7.97 ± 3.72 | 1.075 |
| 21×21 | Exponential and static clumps | 29.03 ± 3.10 | 10.02 ± 3.82 | 1.059 |

In the held-out sky, standardized forced flux scatter spans
0.83–1.18 across the two stamp sizes and nine epochs. At the
target, the pre-transient leave-one-epoch-out fluxes span
−4.74 to +4.30 nJy, with the largest absolute offset at ep07
(1.963 conditional standard deviations). The host training
sets overlap, so these tests are correlated.

All six fits reach the 12-pixel half-light radius limit.
A [radius control](m10/joint_radius_check.json) on the 21×21
clump model raises that limit to 30 pixels. The solution has
`re = 22.93` pixels and `q = 0.25` at its lower bound. The
ep08 and ep09 fluxes shift by −0.16 and −0.20 nJy, respectively,
to 28.86 ± 3.10 and 9.83 ± 3.82 nJy. Repeat the control with
`--radius-check` appended to the command above. The host structure
is weakly constrained in these local stamps, while the measured
point fluxes are stable against this radius-bound change.
The M9 API reproduces the six footprint-covered epochs' residual
χ² at the 15×15 free-index solution to a relative tolerance of
1e-11. The fits support a late-epoch signal in ep08. The ep09
measurement is more sensitive to host, registration and point
position, and this validation does not revise the adopted photometry.

The renderer reuses the last two intrinsic integration grids
when geometry and padded sampling agree. The PSF convolution
remains per frame. A nine-frame render with different pixel
phases produced array-identical output with and without grid
reuse. The local microbenchmark took 0.27 s with reuse and
1.21 s clearing the cache before each frame. The full regression
suite passed 292 tests.

## Local product follow-up

The direct F200W control repeats the 21×21 wide-radius solution
with the host geometry held fixed and the host, static clump,
sky and transient amplitudes refitted. It integrates a common
transient position over a uniform ±1 pixel per-axis grid.
The independent Gaussian registration widths are illustrative.
They enter the pixel covariance through finite-difference scene
gradients. The two late ePSFs are exchanged in a separate
stress test. This tests sensitivity to the supplied registration
and PSFs. It does not calibrate an astrophysical registration
or PSF prior.

| F200W direct model | ep08 (nJy) | ep09 (nJy) |
| --- | ---: | ---: |
| Recorded fixed position and registration | 28.9 ± 3.1 | 9.8 ± 3.8 |
| Common free position | 29.1 ± 3.2 | 9.2 ± 3.9 |
| Free position and 0.15-pixel jitter | 30.9 ± 3.5 | 8.4 ± 4.0 |
| Free position and 0.30-pixel jitter | 32.1 ± 3.7 | 7.9 ± 4.0 |
| Free position, 0.15-pixel jitter and swapped late ePSFs | 30.1 ± 3.2 | 8.7 ± 3.9 |

No change is made to the adopted photometry. See [position and
registration results](m10/systematics.json).

### Shared reference on real differences

The object directory's `aligned_images/` files are direct SCI
cutouts resampled onto the `wide_ep01` grid, with corresponding
ERR cutouts. The nine SCI/ERR pairs for ep08, ep09 and DDT in
the three tested products all carry verified anchor registration
cards. The F200W native-grid sensitivity check uses the original
cutouts and takes the recorded offsets from these headers.

For F200W and both F444W scales, the two stored `diff_preSN`
images match the corresponding aligned science minus the same
pre-SN reference and a per-image constant sky term to better
than 4e-7 of the pixel ERR. The stored ERR agrees with the
science and reference ERR quadrature sum to better than 6e-8
relative. The reference operator in the stored pixel frame is
the identity for each difference. The reference has nine F200W
inputs or eight F444W inputs. Their epoch names are read from
the FITS headers. Object 88 was excluded from the training and
held-out sky patches.

Finite covariance for the aligned science and reference is
trained separately. The block model propagates the same
reference covariance into both differences under the disjoint
science/reference input assumption. The per-epoch ERR maps set
the local scales, with an aperture multiplier fitted on the
training bank. The block and independent-marginal comparison
uses the same pixel diagonals and marginal blocks.

| Product | Training / held-out patches | Flux correlation at target | σ(ep09−ep08), block / diagonal across epochs (nJy) | Block held-out contrast scatter |
| --- | ---: | ---: | ---: | ---: |
| F200W | 67 / 68 | 0.063 | 4.31 / 4.45 | 0.83 |
| F444W 30 mas | 58 / 59 | 0.073 | 9.22 / 9.57 | 1.08 |
| F444W 60 mas | 167 / 167 | 0.068 | 9.34 / 9.66 | 1.01 |

The F200W difference fit gives 31.2 ± 2.7 and 11.5 ± 3.5 nJy for
ep08 and ep09. The fitted contrast is −19.7 ± 4.3 nJy. The
positive cross-epoch correlation reduces the contrast error by
about 3–4 per cent in these products. This percentage depends
on the reference depth and flux estimator. For the half of
held-out controls with nearest relative ERR, the standardized
contrast scatters are 0.91, 1.31 and 1.15 in table order. Most
of the F444W blank controls have smaller ERR than the center
of the galaxy stamp. Correlated sky and ERR calibration do not
measure the error from host residuals. The diagnostics use the
supplied ePSFs on the aligned grid and a 0.36 arcsec phase-zero
aperture convention.

The two F444W cutout scales come from the same measurements and
are correlated. The 30 mas forced target fluxes are lower than
the 60 mas diagnostic results. They should not be combined as
independent estimates. The [shared-reference report](m10/shared_reference.json)
keeps the flux estimates, actual ERR ranges and sky-control results.

### ERR as relative depth and detector variance

In the known-count OBS 002 controls used earlier, 24 held-out
stamps per coverage bin show decreasing median ERR with increasing
counts. Using the same positions as the archived blank CSVs,
the median of the per-stamp ERR medians for one, two and three
exposures is as follows.

| Product | One exposure | Two exposures | Three exposures |
| --- | ---: | ---: | ---: |
| F200W 30 mas | 0.02076 | 0.01521 | 0.01250 |
| F444W 30 mas | 0.01073 | 0.00807 | 0.00688 |
| F444W 60 mas | 0.00898 | 0.00667 | 0.00558 |

Values are in MJy/sr. The [depth controls](m10/err_vs_coverage.json)
include 10th to 90th percentile ranges. This supports using ERR
as a relative depth proxy within a product. The footprint and
detector IDs still supply exposure identity.

All five full detector products in the object directory were
audited around the target. Their ERR squared agrees with the
sum of the supplied variance planes. For the local ep09 F444W
detector, the Poisson fraction is about 94% within five native
pixels of the target and 93% in the selected off-target annulus.
The Poisson plane includes the background. This is one detector
measurement and does not give the separate source-only or
drizzled-mosaic variance. The F444W mosaics' component planes
and full program 9263 input list are still unavailable among
the tested local products. See the [detector audit](m10/individual_exposures.json).

For a point-only 250 nJy injection on the F444W held-out
background, fitting with the matching ePSF gives zero incremental
70 mas host amplitude to numerical precision. Injecting the
other epoch's ePSF and refitting the point-only centroid gives
median incremental signed host fractions of +36% at 30 mas
and +16% at 60 mas. Holding the centroid fixed gives +76% and
+19%, respectively. The injected signal intentionally swaps PSFs
across epochs and adds no source variance. These figures measure
sensitivity to a specific shape change. They do not estimate the
mismatch of the correctly paired ePSF or a host detection.

### Round-2 differences and DDT

The newer `difference_images_realign` products share a different
six-epoch coadd within each band. The six member names match
across ep08, ep09 and ddt_ep01. The available aligned images
and the original pre-SN coadds do not fully reproduce those
round-2 products, so the round-2 check estimates covariance
empirically in flux space. For each epoch, the finite kernel is
estimated from the round-2 difference ERR and blank stamps.
At paired training positions, the covariance of signed flux
divided by its local fitted error is measured, then rescaled by
the errors at each target or held-out position. This measures
empirical cross-epoch flux covariance, without identifying how
much comes from reference noise or scene residuals.

| Product | Training / held out | Held-out contrast scatter, with empirical cross terms | Held-out contrast scatter, diagonal in epoch |
| --- | ---: | --- | --- |
| F200W | 99 / 100 | 1.07, 0.88, 1.15 | 1.03, 0.83, 1.08 |
| F444W 30 mas | 91 / 92 | 0.89, 0.91, 0.76 | 0.84, 0.78, 0.79 |
| F444W 60 mas | 191 / 192 | 1.10, 1.07, 0.99 | 1.01, 0.93, 0.94 |

Contrasts in table order are ep09−ep08, DDT−ep08, and DDT−ep09.
The finite held-out banks do not establish a uniform calibration
improvement. See the [round-2 covariance results](m10/realign_flux_covariance.json).
The direct-cutout follow-up below recovers the reference for the
60 mas product and quantifies which other cases need the original
`preSN_realign` reference and alignment products.

### Reproduce the local follow-up

```bash
python -m scripts.validate_m10_systematics \
    ../cutouts ../obj_39/out '../interesting objs/2026abck' \
    docs/validation/m10/joint_radius_check.json docs/validation/m10/systematics.json
python -m scripts.validate_m10_shared_reference '../interesting objs/2026abck' \
    docs/validation/m10/shared_reference.json
python -m scripts.audit_m10_individual_exposures '../interesting objs/2026abck' \
    docs/validation/m10/individual_exposures.json
```

The joint, noise and rendering regressions passed all 59 targeted tests.

### Three late direct cutouts including DDT

Read the saved direct cutouts for ep08, ep09 and DDT. Each late
frame was cross-convolved against the recorded-shift ep02,
ep04 or ep06 single-epoch pre-SN image, then fitted for a
fixed-position signed point and constant sky. The workflow ePSFs
are sampled on the aligned grid as an approximation. Matching
retains pixels with at least 98% of the convolution weight
present and normalizes by the surviving weight. The propagated
ERR is a scale proxy. The actual aperture covariance, including
cross-epoch terms, is calibrated empirically on paired training
stamps and evaluated on separate sky positions. Convolution
can correlate neighboring sky stamps. Ep06 at 30 mas has only
8 training and 8 held-out common patches and is omitted.

Flux ranges across accepted templates for the nominal 210 mas
half-width are given in nJy. The actual pixel-rounded half-width
is 210 mas at 30 mas sampling and 240 mas at 60 mas sampling.
Templates share the late science exposures. The ranges describe
template sensitivity and are not independent measurements.

| Product | ep08 | ep09 | DDT | Held-out contrast scatter |
| --- | ---: | ---: | ---: | ---: |
| F200W | 30.9–33.3 | 10.9–13.4 | −0.1–2.4 | 0.86–1.02 |
| F444W 30 mas | 9.5–22.5 | 15.6–29.4 | 17.1–30.0 | 0.80–1.23 |
| F444W 60 mas | 14.5–25.2 | 21.8–32.0 | 20.4–31.9 | 0.84–1.17 |

For the round-2 products, subtracting each difference from its
aligned direct image should recover the same reference up to
a constant. F444W 60 mas passes at the stored numerical
precision over the central 98% of the residual distribution
(under 8.1e-8 of the nominal pixel ERR in both comparisons).
The implied reference variance from ep09 and DDT agrees with
the ep08 reconstruction to 1.75% and 0.54%, respectively, at
the 99th percentile. Using the ep08 reconstruction, the real
three-epoch pixel-block likelihood gives F444W 60 mas forced
fluxes of 18.6 ± 5.1, 22.7 ± 8.3 and 25.0 ± 3.7 nJy. The
three held-out contrast scatters are 0.89, 1.03 and 0.94 over
117 sky positions. This noise check is conditional on the
recovered reference and the per-epoch input independence.

At 30 mas, the same closure has tails up to 1.97 nominal
pixel errors for F200W and 0.79 for F444W after removal of a
constant. The corresponding round-2 original reference and
exact science alignment remain needed to rebuild those pixel
blocks. These tests leave the adopted photometry unchanged.

```bash
python -m scripts.validate_m10_aligned_epochs '../interesting objs/2026abck' \
    ../cutouts ../obj_39/out docs/validation/m10/aligned_three_epochs.json
```

See the [three-epoch results](m10/aligned_three_epochs.json).
All 32 joint and noise regression tests passed.
