# M4 validated render domain

The numerical gate covers two responses for locally shift-invariant,
calibrated synthetic PSFs. The measured cases below bound discretization
error at the specified noise level. The later [empirical PSF validation](EMPIRICAL_PSF_SKY.md) includes real stars, leave-one-out PSF reconstruction and independent blank-sky positions.

## Response and flux conventions

**Pre-pixel density.** The operator uses the bilinear interpolant of samples
in inverse square native pixels. It integrates that interpolant over a
uniform native pixel, with analytic integrals of the linear basis. For an
extended source, linearity lets the convolution use this pixel-integrated
response directly. The pixel response is applied once.

**Effective PSF.** Samples represent the response of one native pixel to a
unit-flux point at the specified offset. The operator interpolates this
response at native pixel centers. For an extended source it integrates the
intrinsic flux over source subcells and convolves their masses with the
effective response.

Both branches use the same operator for point sources and galaxies.
The source lattice follows the centroid. The central 3×3 subcells use the
M3 adaptive integrator, and other cells use a four-point Gauss–Legendre
rule per axis. Each integrated mass is assigned to its subcell center for
the convolution. A full linear FFT convolution with explicit phase and
crop indices returns the native stamp. The source grid extends beyond
the stamp according to the response support and allowed offsets, so it
includes flux scattered into the stamp from beyond its edges.

The sample array is zero outside its tabulated rectangle. A crop must
preserve the declared scale and update the origin for the new sample
indices. With nonzero response at a table edge, this zero extension has a
discontinuity. Test support and interpolation for each new PSF. The
optional `support_fraction` describes an independently known fraction of
the normalization reference. Unknown fractions are represented by `None`.
The reported sample hash is SHA256 of the raw float64, little-endian,
C-order payload, prior to calibration. No phase- or stamp-dependent flux
rescaling is applied. A component's stamp flux is the sum of its rendered
pixel values.

The operator requires an explicit response kind, spacing in native pixels,
sample-index origin and independent normalization reference. Pre-pixel
inputs use sampled densities. Array cell masses need a separately defined
response and are rejected as a response kind. The padded source grid has a
configurable 500,000-cell default limit. Large PSF tables and fine source
grids may exceed it. Any truncation should be checked by enlarging the
support under the same calibration.

## Measured domain

The Gaussian PSF has native standard deviations of 0.68 in x and 0.81 in y.
The pre-pixel table samples the Gaussian density. The ePSF table samples
the analytic Gaussian integral over a unit square pixel. Each table spans
[-4, 4] native pixels on both axes, with sample spacing 1/16 native pixel.
The source-integrated results use 16 subcells per native pixel per axis.
The table specifies discrete validated synthetic profiles and phases.

The primary metric is the square root of `delta.T @ inv(C) @ delta`, where
`delta` is the image difference against the independent reference. The
test budget is 0.1. The unit-flux tests use a uniform error of 0.02 per
native pixel. Max pixel and stamp flux errors below are expressed as a
fraction of total intrinsic flux.

| Scenario | n | re (native px) | q | PA (deg) | (dx, dy) native px | Pre-pixel / ePSF diagonal weighted error | Pre-pixel / ePSF time (ms) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Tilted Gaussian, 5×6 | 0.5 | 1.2 | 0.45 | 37 | (0.17, -0.23) | 0.00788 / 0.00924 | 34.4 / 30.3 |
| Exponential, 5×6 | 1 | 0.7 | 0.44 | 26 | (0.25, 0) | 0.01186 / 0.00395 | 33.0 / 29.4 |
| Elliptical host, 5×6 | 2 | 0.7 | 0.74 | 108 | (0.49, -0.5) | 0.01035 / 0.00731 | 31.5 / 28.4 |
| Compact high-n host, 5×6 | 4 | 0.18 | 0.39 | 61 | (0, 0) | 0.01490 / 0.00441 | 35.1 / 32.1 |
| Near-edge high-n host, 5×6 | 4 | 0.03 | 0.30 | 49 | (0.49, -0.5) | 0.01607 / 0.01037 | 32.4 / 29.4 |

Times are medians of three local executions, including source integration.
The maximum absolute host pixel error was `2.17e-4` and the maximum
absolute host stamp flux error was `9.03e-5`. An additional periodic noise
audit uses correlations of 0.10 and 0.09 at the two horizontal and two
vertical nearest neighbors, respectively. The maximum weighted host error
for that covariance was 0.01573. A change in flux or noise level changes
the weighted error.

The independent extended-source reference integrates the analytic
Gaussian pixel response against the Sérsic radial flux measure and samples
orbital angle independently of the source lattice. Increasing the angular
grid from 256 to 512 samples changed the five reference images by at most
`5.56e-17` per pixel. Tests also compare an axis-aligned Gaussian host
against the analytic convolved Gaussian on 7×7, 6×8 and 7×8 stamps at
phases (0, 0), (0.25, -0.5) and (0.5, 0.25). Point sources are compared
against the analytic pixel integrals. On the 5×6 benchmark, including
the off-lattice phase (0.17, -0.23), the worst point-source weighted
error was 0.01551. Median point render time was 0.31–0.36 ms for pre-pixel
and 0.058–0.062 ms for ePSF.

## Refinement and boundary checks

- In the offset n=4, re=0.18 case, increasing the source grid from 4×
  to 8× to 16× reduced the ePSF residual image L2 norm from
  `1.05e-3` to `5.61e-4` to `3.60e-4`. The source phase for this
  check is (0.17, -0.23).
- Changing the ePSF half-width from 2 to 3 to 4 native pixels, at fixed
  grid spacing and source quadrature, changed the image by `8.01e-3`
  and then `5.83e-6` in L2 norm. Comparing successive supports isolates
  the truncation change. Comparing with the continuous reference also
  includes the remaining interpolation and subcell errors.
- Refining the pre-pixel table spacing from 1/4 to 1/8 to 1/16 pixel
  reduces the point-source error against the analytic reference.
- A finite asymmetric ePSF with a fractional origin matches a direct sum
  of independently integrated Gaussian source cells within `5e-12` per
  pixel. This tests kernel orientation and the full linear crop.
- A smooth-source narrow pre-pixel response recovers the unblurred M3
  pixel integrals within `1.5e-4` per pixel. Both synthetic response
  branches match the point limit at n=4, re=`1e-6` pixel and off-lattice
  phase within `2e-7` per pixel.
- Shifted and centered crops preserve overlapping pixel values within
  `2e-15`. Constant calibration, finite support loss, signed point flux,
  and point+host+sky model evaluation are covered by regression tests.

## Real F200W validation

The two distributed F200W PSF files are 535×535 samples. The producer
confirmed that they are mosaic-level PSFEx ePSFs. The FITS headers contain
star counts and structural fields, with no export spacing, origin or
normalization reference metadata. The corresponding native-grid cutouts
of individual stars and training positions were not available for this
validation. The supplied example observation cutouts are transient
difference images.

The real-data gate requires the export spacing and origin, calibration
reference, and matching native-mosaic star stamps with uncertainty data.
Compare both paths at the same finite-aperture normalization and report
flux, centroid and residual changes by phase and field position. Training
star recovery measures renderer consistency. Independent or cross-validated
stars are required to measure generalization. The synthetic benchmarks
above leave these empirical errors unmeasured.

## Usage

```python
import numpy as np
from scipy.special import ndtr
from jwst_psfmc import PSFOperator, PSFSpec, RenderContext, SersicProfile

axis = np.arange(-64, 65) / 16
wx = ndtr((axis + 0.5) / 0.68) - ndtr((axis - 0.5) / 0.68)
wy = ndtr((axis + 0.5) / 0.81) - ndtr((axis - 0.5) / 0.81)
spec = PSFSpec(np.outer(wy, wx), "effective", 1/16, (64, 64),
               normalization_reference="analytic total flux", source="Gaussian")
operator = PSFOperator(spec, oversamp=16, max_offset=0.5)
context = RenderContext((5, 6), psf_operator=operator)
point = operator.render_point(0.25, -0.5, context)
host = operator.render_sersic(SersicProfile(1, 0.7, 1, q=0.44, pa=26),
                              0.25, -0.5, context)
```

For model evaluation, put `PointSource` and `SersicSource` in a
`CompositeModel` with the same context and namespaced physical parameters.
`evaluate_model_fit` records the PSF metadata alongside the noise metadata.
The validation is reproducible with `python -m pytest tests/test_rendering.py`.
