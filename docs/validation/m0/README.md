# M0 baseline and decision record — 2026-11-17

**Follow-up verification.** The [full extras and notebook run](full-run-01/commands.json)
completed both notebooks (14/14 and 28/28 code cells). The
[user-confirmed PSF provenance](psf-provenance-user.json) identifies the
example F200W arrays as PSFEx ePSFs from the corresponding mosaics.
The original environment and provenance limitations are preserved below.

In the original run, M0's numerical baseline, provenance audit, regression protection and
design decisions were implemented. **Full-environment verification remains incomplete**:
Nautilus is unavailable and both original notebooks are blocked at kernel
startup by the execution sandbox. These are recorded failures, not successful
executions. Real PSF response semantics are explicitly **unknown**, so analytic
development may proceed; real-JWST Sérsic science rendering is not validated.

## Results

| Verification | Observed result | Evidence |
| --- | --- | --- |
| Original suite (excluding M0 additions) | 113 passed, 1 skipped, 0 failed; 5 warnings | original-suite.xml / .log |
| New M0 regressions | 8 passed, 0 skipped, 0 failed | m0-suite.xml / .log |
| Earlier combined run before two real-data regressions were added | 119 passed, 1 skipped | baseline-suite.xml / .log |
| Online `.[dev,nautilus,notebooks]` installation | exit 1; PyPI DNS/network unavailable at build-dependency installation | install.log |
| Offline editable install, `--no-build-isolation --no-deps` | exit 0 | offline-install.log |
| pip check (installed package requirements) | exit 0 | pip-check.log |
| demo_covariance_kernel, original code and parameters | exit 1; 0/14 code cells; local port bind denied | notebooks/demo_covariance_kernel/execution.json |
| demo_psf_photometry, original code and parameters | exit 1; 0/28 code cells; local port bind denied | notebooks/demo_psf_photometry/execution.json |
| White-noise FFT vs direct logL | difference 0 in this calculation | noise-audit.json |
| Periodic 5×5 FFT vs independent dense matrix | relative quadratic difference 2.96e-16 | noise-audit.json |
| Masked center: legacy zero residual vs exact marginal | 24.0378191974 vs 23.9904658104 | noise-audit.json |
| Real F200W kernels on 9×9 grid | minima 0.298760675 / 0.289055275; floor fraction 0 in both | psf-audit.json |

The skipped original test is `test_nautilus_end_to_end`, because Nautilus is not
installed. pip check does not establish that optional extras are installed.
The five suite warnings are photutils `npixels` deprecations (3) and expected
NoDetectionsWarnings (2), not test failures. Notebook failure occurs before any
science cell; it is not evidence of a notebook code defect or posterior failure.
No stochastic-chain agreement or posterior calibration claim is made.
`command-results.json` records command exits; `preservation-check.json` confirms
unchanged production sources, original notebooks, input hashes and user patch.

## Roadmap checklist

- [x] Capture baseline identity, dirty patch, dependency versions and real-input hashes.
- [ ] Install all requested extras and verify Nautilus: blocked by network; offline fallback recorded.
- [x] Freeze four synthetic fixed-theta cases and two real-data regressions.
- [x] Exclude uncontrolled chains from deterministic references.
- [x] Attempt both original notebooks and retain failures and reasons (neither passed).
- [x] Audit PSF provenance; explicitly mark unsupported response facts unknown.
- [x] Record PSF/units, noise/mask and parameter/prior decisions.
- [x] Record legacy issues locally and the future shape-rejection requirement; no remote issues/PRs filed.

Full extras and successful notebook execution remain outstanding verification.

## Baseline identity and environment

HEAD: `38f0834d5fdd977c9c761907d0ab59a82bc5ae90`, branch `master`, package 0.3.0.
`initial-dirty.patch` preserves the user's three removed comment lines in
covariance.py; no production `.py` file or original example notebook was changed
by M0. `initial-status.txt` also contains the newly created validation directory,
because recording started after creating it. There were no other preexisting
source edits. The baseline is not described as a clean checkout.

`manifest.json` contains SHA256 for every real FITS/PSF/kernel input, all package
source files and the original notebooks, with dependency versions and platform.
`head.txt`, `branch.txt`, `requirements-freeze.txt` retain the raw identity records.
The captured status is a snapshot during implementation, not a final diff.

The initial install attempted Python 3.13.1 and failed before package installation.
The numerical runs use a private Python **3.12.11** venv, with an offline copy of
the locally installed science/Jupyter stack and pure-Python pytest dependencies.
No system environment was modified. This is an offline fallback, **not** a
successful resolution of all requested extras from PyPI. NumPy 2.5.3, SciPy
1.18.1, Astropy 8.0.1, photutils 3.0.0, emcee 3.1.6 and pytest 7.4.2 were used.
Exact dependency versions are in the manifest; this run does not certify the
package's whole supported Python/dependency range.

Offline inputs were copied from the local conda `py312` science site-packages;
pytest, _pytest, pluggy, iniconfig, py and attrs were copied from `py39` as
Python sources plus distribution metadata, never as Python-3.9 extension modules.
An initial pytest bootstrap failed because the compatibility `py` dependency
had not yet been copied. It was corrected before either reported suite run.
The standard editable install was then performed offline. Plugin autoload was
disabled to avoid unrelated installed plugins; coverage was not measured.

## Fixed-theta fixtures

`tests/baselines/m0/` contains four small synthetic cases: bright (flux 80),
faint (0.8), non-detection (0), signed flux (-8), plus two real F200W stamp
regressions. NPZ files contain model images, fixed theta, logL, prior bounds,
and preparation/match/total fractions. Synthetic cases additionally record all
eight open prior endpoints and an interior probe. Random synthetic input noise
uses `default_rng(20261117)`. Unseeded legacy walker positions are discarded.

Tests compare stored values at rtol=2e-11 / atol=2e-12 and never regenerate the
fixtures. These tolerances allow small floating-point changes across machines;
material scientific algorithm changes must be reviewed, not hidden by updating
goldens. `metadata.json` records fixture hashes and generating HEAD. The fixtures
are compatibility references, not proof that legacy PSF physics is correct.

## PSF audit and decisions

Both PSF headers report n_good_stars=n_all_stars=29, suggesting star-based
provenance but not documenting how those stars were reconstructed. README and
callers specify factor 4; no generation recipe establishes the physical response.
Both 535×535 arrays have unknown pixel/drizzle response and normalization
reference beyond the supplied finite array. Unknown fields are blockers in
ADR-001, not invented defaults.

| PSF | Requested native | Actual downsampled | Retained prepared-array sum |
| --- | --- | --- | --- |
| example1 | 10 / 11 / 12 | 9 / 11 / 11 | 96.703% / 100% / 98.663% |
| example2 | 10 / 11 / 12 | 9 / 11 / 11 | 97.411% / 100% / 98.847% |

These are unshifted finite-array losses, not universal photometric biases.
Decisions are in ADR-001 (PSF/units), ADR-002 (noise/mask), ADR-003
(parameters/priors). Separate local issue records are in LEGACY-ISSUES.md.
No GitHub issue or PR was filed. There is no new rendering API in M0; rejecting
unsupported legacy dimensions is an explicit requirement for the future API.

## Reproduce and finish the outstanding verification

From this checkout, in an environment with PyPI access and local Jupyter ports:

```bash
python3.12 -m venv .venv-m0-full
source .venv-m0-full/bin/activate
python -m pip install -e '.[dev,nautilus,notebooks]'
python scripts/run_m0.py --output docs/validation/m0/full-environment-run
```

Choose a fresh output directory for each run. The runner preserves every command
and exit, continues to the second notebook after a failure, and does not update
goldens. Notebooks run unmodified (including 5000-step chains and eight workers)
in private folders with links to the original data. Their written NPZ outputs
do not overwrite the example archives. Review convergence and Monte Carlo error
separately after successful execution. Do not promote notebook output to golden
chains. Obtain the PSF producer's response recipe before real-PSF validation.

To recheck only numerical tests with the current offline environment:

```bash
MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sersic/bin/python -m pytest -q
```

Maintainer-only baseline regeneration, after reviewing an intentional change:

```bash
MPLBACKEND=Agg .venv-sersic/bin/python scripts/m0_baseline.py --capture --write-goldens
```

This overwrites capture/fixture files; keep the original report when creating a
new baseline. Ordinary tests and the verification runner never update goldens.
