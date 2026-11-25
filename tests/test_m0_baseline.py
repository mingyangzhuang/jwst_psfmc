"""Fixed-theta legacy regressions and independent dense noise reference."""

from pathlib import Path
import importlib.util

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if not (_ROOT / "scripts/m0_baseline.py").is_file() or not (_ROOT / "tests/baselines/m0/bright.npz").is_file():
    pytest.skip("M0 validation tools and fixtures require the source checkout", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("m0_baseline", _ROOT / "scripts/m0_baseline.py")
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


@pytest.mark.parametrize("name", baseline.CASES)
def test_fixed_theta_golden(name):
    actual = baseline.evaluate_case(name)
    with np.load(_ROOT / "tests/baselines/m0" / f"{name}.npz", allow_pickle=False) as expected:
        assert set(actual) == set(expected.files)
        for key, value in actual.items():
            np.testing.assert_allclose(value, expected[key], rtol=2e-11, atol=2e-12,
                                       err_msg=f"{name}: {key}")


def test_white_and_periodic_dense_reference():
    audit = baseline.noise_audit()
    assert abs(audit["white_logl_difference"]) < 1e-12
    assert audit["relative_difference"] < 1e-12


def test_mask_zeroing_is_not_marginalization():
    audit = baseline.noise_audit()
    # Documents the approximation; do not assert equality to this in a new API.
    assert abs(audit["masked_zero_residual_quadratic"] - audit["masked_marginal_quadratic"]) > 1e-3


@pytest.mark.parametrize("name", baseline.REAL_CASES)
def test_real_fixed_theta_golden(name):
    if not (_ROOT / f"examples/PSF/{name}_f200w_PSF_4_c.fits").is_file():
        pytest.skip("Real example inputs are available only in the source checkout")
    with np.load(_ROOT / "tests/baselines/m0" / f"{name}.npz", allow_pickle=False) as expected:
        actual = baseline.evaluate_real_case(name)
        assert set(actual) == set(expected.files)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-11, atol=2e-12,
                                       err_msg=f"{name}: {key}")
