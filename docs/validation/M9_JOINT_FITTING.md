# M9 joint fitting of registered epochs

## Independent direct images

Create one `JointObservation` per registered image. Supply a
`CompositeModel`, native render context and PSF, uncertainty and mask,
the observational band, and stable exposure identifiers. Give each
frame a description of its fixed astrometric registration and flux
calibration. The input images must already be on a common physical
flux scale. The renderer works in native output pixels. Use frame
aliases for known pixel offsets, and share a radius in pixels only on
a common pixel scale. The preparation checks radius sharing when the
PSF metadata supply the scale.

Create one joint `ParameterSet`. A local field such as `host.flux`
can use the same root in every frame. Frame aliases such as
`{"point.flux": "ep02.flux", "sky.level": "ep02.sky"}` select
epoch-specific roots. A root contributes its prior once. Preparation rejects unknown local fields
and roots that are unused by the full observation list. Custom joint
components declare a tuple of local field suffixes in `parameter_fields`. A fixed zero
for a pre-transient flux is an explicit modeling choice. The API
does not infer such a constraint from filenames or dates.

```python
from jwst_psfmc import (
    evaluate_joint_model_fit, optimize_model_fit, prepare_joint_model_fitting,
    run_model_mcmc, save_model_results,
)

joint = prepare_joint_model_fitting(observations, parameters)
optimum = optimize_model_fit(joint, objective="ml")
result = run_model_mcmc(joint, initial=optimum, seed=9021)
save_model_results("joint.npz", result)
images = evaluate_joint_model_fit(joint, optimum.best.sample_vector, decompose=True)
```

Independent frames keep their native noise operators. Their likelihoods
sum after rendering the same decoded parameter vector in each frame.
Fixed-geometry linear components reuse a single prepared joint design.
Unfixed geometric quantities use the same render path as the
single-frame API. The posterior NPZ preserves the global parameter
schema, frame aliases, input hashes, exposure identities, PSF and noise
metadata. The likelihood is conditional on the user-supplied
registration and calibration. It does not propagate uncertainty in the
registration or photometric solution automatically.

## Difference images with reused reference exposures

The observation list must consist of direct images or difference images.
For a difference frame, set `kind="difference"`, list its science and
reference exposure identities, and use a forward model on the
difference image. The ordinary static Sérsic component is excluded
from these models because it belongs on the direct imagery. A signed
point amplitude measures change relative to the reference.

If differences share inputs, pass `joint_covariance` in observation
order, with C-order pixels within each observation, including masked
pixels. Exposure IDs have surrounding whitespace removed before overlap
checks. The matrix provides diagonal blocks and cross blocks. Each
observation may still supply a mask, and the dense likelihood uses the
observed principal submatrix. With a block covariance, per-frame
diagnostic log likelihoods are explicitly labeled marginal. Only the
joint log likelihood includes correlations between frames.

When the science images and the common reference are mutually
independent, `shared_reference_covariance` takes their covariances and
the known linear reference operators. For
$D_t=S_t-A_tR$, it adds $A_t C_R A_u^T$ between frames, including
the reference contribution on the diagonal. For other dependencies,
supply the full covariance directly. Dense blocks are intended for
small stamps. Construction preflights the full matrix workspace; fitting
preflights the observed joint pixel count before any frame factorization.
For independent observations, the top-level `max_dense_bytes` caps
each dense workspace using the frame's observed pixel count. A frame
can set a stricter local cap in `noise_options`.

## Validation

Run the portable mock example from the source checkout.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.demo_joint_epochs docs/validation/m9
```

The three synthetic F200W direct frames use different PSFs, stamp
sizes, masks, correlated background levels and fixed translations.
The shared host geometry is fixed to the synthetic
truth. The joint fit estimates the host amplitude and an independent
sky level for each epoch, plus transient amplitudes in the rise
and late epochs. The transient amplitude is explicitly zero in the
mock pre epoch. A closed-form Gaussian solve checks the joint
optimization and posterior center. The result archive is reopened
and the images are decomposed frame by frame.

An independent oracle checks the covariance of three differences with
unit science variance and reference variance of two. The uncertainty
of their common mean is $\sqrt{2+1/3}$ with the cross-frame terms,
and 1 when the differences are treated as independent. The unit
tests also cover a reference resampling operator, unequal stamp sizes,
pixel masks, likelihood normalization, prior counting, provenance
guards and NPZ round trips.

The joint mock reached the within-chain tau and ESS criteria. The
minimum approximate ESS was 2,133. The shared host had input flux
1.50 µJy and posterior median 1.464 µJy with a 16th–84th percentile
interval of 1.393–1.534 µJy. The conditional Gaussian host uncertainty
was 0.091 µJy in the pre epoch alone and 0.073 µJy jointly.
The fitted rise and late point medians were 0.926 and 0.363 µJy,
compared with inputs of 0.840 and 0.400 µJy. All posterior medians
agree with the same-noise GLS solution to within 0.15 model standard
deviations. See the [run summary](m9/summary.json) for the full
intervals and seed-based results.

The full regression suite passed 286 tests with five existing
Photutils warnings. An emcee spawn smoke test and a Nautilus run
both reproduced the deterministic likelihood at a saved draw.
