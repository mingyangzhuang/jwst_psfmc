# Legacy issues: local issue records, separate from Sérsic feature work

These are repository-local issue specifications, not claims of filed GitHub
issues or submitted PRs. No remote issue/PR was created in M0. Production source
is intentionally unchanged. Each fix should be reviewed separately.

## L001 — even native shape drops unreported PSF flux

Reproduce: run `scripts/m0_baseline.py --capture`; inspect requested 10/11/12
and actual 9/11/11 in psf-audit.json. Measurement uses both actual PSFs.
Downsampling forces odd dimensions, but preparation fraction does not include
this loss. Do not describe the measured retention as a universal fit flux bias.
Fix acceptance: reject unsupported configurations or account for every crop;
test even/odd rectangular grids, shifts, flux fractions and origin explicitly.
New API default: reject the unsupported legacy shape, per ADR 001.

## L002 — unseeded preparation and NaN infill

Evidence: prepare_for_fitting uses default_rng() with no argument;
estimate_cov_kernel uses unseeded random noise for nonfinite infill.
run_mcmc's seed is only used by the Nautilus branch. Fixed theta is reproducible;
the legacy end-to-end stochastic workflow is not controlled by one seed.
Fix acceptance: explicit RNG propagation and tests of preparation, NaN handling
and both samplers. Existing inputs retain their defaults for compatibility.

## L003 — masked FFT residuals are an approximation

Reproduce: noise-audit.json compares the actual masked legacy likelihood with
an independently marginalized dense covariance. New generic exact likelihood
must pass the dense reference; document legacy approximation separately.

## L004 — PSF scientific provenance unavailable

Blocked on the producer's reconstruction/response/calibration recipe (ADR 001).
Header audit and SHA256 identify the audited bytes; they cannot supply missing
physical semantics. Do not double-integrate an ePSF or infer infinite flux from
normalization of a finite array.
