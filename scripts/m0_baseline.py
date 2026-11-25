"""Capture the legacy numerical baseline, without changing production behavior.

Run from the repository root with the editable package installed. Goldens are
never updated by pytest. Use --write-goldens only after reviewing a baseline
change; --capture writes an independent run manifest and PSF/noise audits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from astropy.io import fits

from jwst_psfmc.covariance import kernel_power_spectrum, prepare_covariance_terms
from jwst_psfmc.mcmc import log_likelihood_cov_prepared, log_prior, prepare_for_fitting
from jwst_psfmc.psf import downsample_psf, prepare_psf_for_oversamp, psf_model

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests/baselines/m0"
SEED = 20261117
CASES = ("bright", "faint", "non_detection", "signed_flux")
REAL_CASES = ("example1", "example2")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analytic_psf():
    y, x = np.mgrid[:65, :65]
    psf = np.exp(-((x - 32.0)**2 + (y - 32.0)**2) / (2 * 4.1**2))
    return psf / psf.sum()


def evaluate_case(name):
    """Evaluate fixed inputs; deliberately discard unseeded walker positions."""
    flux = dict(bright=80.0, faint=0.8, non_detection=0.0, signed_flux=-8.0)[name]
    theta = np.array([flux, 0.23, -0.31, 0.012])
    psf = analytic_psf()
    cropped, native, fraction = prepare_psf_for_oversamp(
        psf, 4, native_shape=(11, 11), return_fraction=True)
    model, meta = psf_model(theta[:3], cropped, 4, output_shape=(9, 9),
                           native_shape=native, psf_prepare_fraction=fraction,
                           return_meta=True)
    rng = np.random.default_rng(SEED)
    err = rng.uniform(0.03, 0.07, size=(9, 9))
    data = model + theta[3] + rng.normal(size=(9, 9)) * err
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0
    kernel[0, 1] = kernel[2, 1] = kernel[1, 0] = kernel[1, 2] = 0.1
    fit = prepare_for_fitting(data, err, psf, kernel, native_shape=(11, 11),
                              flux_prior_min=-100.0)
    likelihood_keys = ("prepared", "psf_os", "oversamp", "native_shape",
                       "psf_prepare_fraction")
    logl = log_likelihood_cov_prepared(theta, **{k: fit[k] for k in likelihood_keys})
    bounds = np.array([fit["prior_bounds"][k] for k in ("flux", "dx", "dy", "bkg")])
    # Every open endpoint must be rejected, independently of the case theta.
    midpoint = bounds.mean(axis=1)
    boundary_prior = []
    for i in range(4):
        for endpoint in bounds[i]:
            probe = midpoint.copy()
            probe[i] = endpoint
            boundary_prior.append(log_prior(probe, fit["prior_bounds"]))
    return dict(theta=theta, data=data, err=err, kernel=kernel, psf_os=cropped,
                image=model, log_likelihood=np.array(logl), prior_bounds=bounds,
                boundary_log_prior=np.array(boundary_prior),
                interior_log_prior=np.array(log_prior(midpoint, fit["prior_bounds"])),
                theta_log_prior=np.array(log_prior(theta, fit["prior_bounds"])),
                fractions=np.array([meta[k] for k in (
                    "psf_prepare_fraction", "model_match_fraction", "total_encircled_fraction")]))


def noise_audit():
    rng = np.random.default_rng(SEED)
    data = rng.normal(size=(9, 9))
    err = rng.uniform(0.5, 2.0, size=(9, 9))
    prep = prepare_covariance_terms(data, err, np.ones((1, 1)))
    psf, native, fraction = prepare_psf_for_oversamp(
        analytic_psf(), 4, (11, 11), return_fraction=True)
    white = log_likelihood_cov_prepared([0, 0, 0, 0], prep, psf, 4, native, fraction)
    direct = -0.5 * np.sum((data / err)**2)
    residual = rng.normal(size=25)
    # Independent periodic covariance matrix, constructed from neighbors.
    covariance = np.eye(25)
    for y in range(5):
        for x in range(5):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                covariance[y*5+x, ((y+dy) % 5)*5 + (x+dx) % 5] = 0.1
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1
    kernel[0, 1] = kernel[2, 1] = kernel[1, 0] = kernel[1, 2] = 0.1
    power = kernel_power_spectrum(kernel, (5, 5))
    fft_q = np.sum(abs(np.fft.fft2(residual.reshape(5, 5)))**2 / power) / 25
    dense_q = residual @ np.linalg.solve(covariance, residual)
    keep = np.arange(25) != 12
    masked_data = residual.reshape(5, 5).copy()
    masked_data[2, 2] = np.nan
    prep_mask = prepare_covariance_terms(masked_data, np.ones((5, 5)), kernel)
    zeroed_q = -2 * log_likelihood_cov_prepared(
        [0, 0, 0, 0], prep_mask, psf, 4, native, fraction)
    marginal_q = residual[keep] @ np.linalg.solve(covariance[np.ix_(keep, keep)], residual[keep])
    return dict(seed=SEED, white_logl=white, direct_white_logl=direct,
                white_logl_difference=white-direct, fft_quadratic=float(fft_q),
                dense_quadratic=float(dense_q),
                relative_difference=float(abs(fft_q-dense_q)/abs(dense_q)),
                masked_zero_residual_quadratic=float(zeroed_q),
                masked_marginal_quadratic=float(marginal_q),
                interpretation="Periodic toy oracle only; masked zero residuals are not marginalization.")


def evaluate_real_case(name):
    """Freeze fixed-theta behavior on the shipped F200W stamps and real PSFs."""
    data_root = ROOT / "examples/data"
    raw = fits.getdata(ROOT / f"examples/PSF/{name}_f200w_PSF_4_c.fits")
    data = fits.getdata(data_root / f"{name}_f200w_diff.fits").astype(float)
    err = fits.getdata(data_root / f"{name}_f200w_diff_error.fits").astype(float)
    cy, cx = np.array(data.shape) // 2
    sl = (slice(cy-4, cy+5), slice(cx-4, cx+5))
    data, err = data[sl], err[sl]
    kernel = fits.getdata(data_root / f"{name}_f200w_cov_kernel.fits")
    fit = prepare_for_fitting(data, err, raw, kernel)
    theta = np.array([0.5 if name == "example1" else 12.3, 0.17, -0.21, 0.003])
    image, meta = psf_model(theta[:3], fit["psf_os"], 4, output_shape=data.shape,
                           native_shape=fit["native_shape"],
                           psf_prepare_fraction=fit["psf_prepare_fraction"], return_meta=True)
    logl = log_likelihood_cov_prepared(theta, **{k: fit[k] for k in (
        "prepared", "psf_os", "oversamp", "native_shape", "psf_prepare_fraction")})
    return dict(theta=theta, image=image, data=data, err=err,
                log_likelihood=np.array(logl),
                prior_bounds=np.array([fit["prior_bounds"][k] for k in ("flux", "dx", "dy", "bkg")]),
                fractions=np.array([meta[k] for k in (
                    "psf_prepare_fraction", "model_match_fraction", "total_encircled_fraction")]))


def psf_audit():
    provenance_path = ROOT / "docs/validation/m0/psf-provenance-user.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    result = {}
    for path in sorted((ROOT / "examples/PSF").glob("*.fits")):
        raw, header = fits.getdata(path, header=True)
        cases = []
        for n in (10, 11, 12):
            cropped, native, fraction = prepare_psf_for_oversamp(raw, 4, (n, n), return_fraction=True)
            native_image = downsample_psf(cropped, 4)
            cases.append(dict(requested_native=list(native), actual_native=list(native_image.shape),
                              preparation_fraction=float(fraction),
                              block_sum_retention=float(native_image.sum()/cropped.sum())))
        relative = str(path.relative_to(ROOT))
        reported = relative in provenance.get("applies_to", [])
        result[relative] = dict(
            sha256=sha256(path), shape=list(raw.shape), raw_sum=float(raw.sum()),
            header=header.tostring(sep="\n"), assumed_oversampling_for_legacy_checks=4,
            response_semantics=provenance["response_semantics"] if reported else "unknown",
            generation_method=provenance["generation_method"] if reported else "unknown",
            semantics_evidence=provenance.get("evidence_level") if reported else "unavailable",
            response_validation_status="real-star renderer validation pending",
            center_definition="unknown", normalization_reference="finite input array sum in legacy code",
            pixel_response="included in empirical mosaic response; exact kernel unverified" if reported else "unknown",
            drizzle_kernel="unknown", drizzle_pixfrac="unknown",
            drizzle_scale="unknown", even_native_checks=cases)
    spectra = {}
    for path in sorted((ROOT / "examples/data").glob("*cov_kernel.fits")):
        p = kernel_power_spectrum(fits.getdata(path), (9, 9))
        spectra[str(path.relative_to(ROOT))] = dict(minimum=float(p.min()), maximum=float(p.max()),
                                                   floor_fraction=float(np.mean(p <= 1e-8)))
    return dict(psfs=result, example_kernel_spectra=spectra)


def capture(output):
    output.mkdir(parents=True, exist_ok=True)
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    files = sorted((ROOT / "examples").rglob("*.fits"))
    sources = sorted((ROOT / "jwst_psfmc").glob("*.py"))
    manifest = dict(schema_version=1, seed=SEED, head=git("rev-parse", "HEAD"),
                    branch=git("branch", "--show-current"), status=git("status", "--short"),
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    packages={d.metadata["Name"]: d.version for d in metadata.distributions()},
                    input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in files},
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                    notebooks_sha256={str(p.relative_to(ROOT)): sha256(p)
                                      for p in (ROOT / "examples").glob("*.ipynb")})
    for name, value in (("manifest", manifest), ("noise-audit", noise_audit()), ("psf-audit", psf_audit())):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--write-goldens", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.capture:
        if args.output is None:
            parser.error("--capture requires --output pointing to the documentation directory")
        capture(args.output)
    if args.write_goldens:
        GOLDENS.mkdir(parents=True, exist_ok=True)
        for name in CASES:
            np.savez_compressed(GOLDENS / f"{name}.npz", **evaluate_case(name))
        for name in REAL_CASES:
            np.savez_compressed(GOLDENS / f"{name}.npz", **evaluate_real_case(name))
        (GOLDENS / "metadata.json").write_text(json.dumps(dict(
            seed=SEED, head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            note="Legacy numerical compatibility fixtures, not a physical PSF validation. No MCMC chains.",
            files={p.name: sha256(p) for p in sorted(GOLDENS.glob("*.npz"))}), indent=2) + "\n")
