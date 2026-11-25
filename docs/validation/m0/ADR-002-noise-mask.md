# ADR 002: noise, masks and likelihood normalization

Status: accepted design; generic likelihood implementation belongs to M2.

Legacy likelihood evaluates -0.5/N sum(|FFT(r)|²/P), with
r=(data-model)/err * sqrt(fit_weight), invalid pixels zeroed, and a spectrum
floor of 1e-8. It omits the Gaussian normalization. For fixed covariance the
omitted term is constant with respect to source parameters; absolute evidence
and comparisons across different covariance/mask choices need the full term.

`noise-audit.json` and the tests compare white noise with direct chi-square
and an independently constructed 25×25 periodic matrix with the FFT quadratic.
The single masked center pixel explicitly demonstrates that zeroing residuals
does not marginalize missing observations. This toy difference is not a bound
on general scientific bias. Periodic boundaries are part of this oracle and
must not be confused with a nonperiodic finite-image covariance.

New Gaussian likelihoods must explicitly declare exact marginal or approximate
mask handling. Exact treatment subsets C to observed pixels and includes
logdet(C_observed) when normalization is needed. A fractional fit_weight is an
altered objective, not automatically a physical covariance model. Generic
noise models must report boundary assumptions, PSD/floor diagnostics, valid
pixel count and approximation status. Validate before introducing samplers.

Real example F200W kernels are audited separately from the toy; see measured
9×9 minima and floor fractions. No inference that real data have clipping
problems follows from pathological synthetic kernels. Noise hyperparameters,
nonstationary covariance and cross-epoch reference covariance are out of M0.
