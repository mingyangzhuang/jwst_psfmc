"""Independent Gaussian noise oracles on small image grids."""

import math

import numpy as np
import pytest

from jwst_psfmc.noise import prepare_noise


def neighbor_covariance(shape, err, *, periodic):
    """Define R by pairs of adjacent pixels, independently of kernel embedding."""
    ny, nx = shape
    coords = list(np.ndindex(shape))
    correlation = np.eye(len(coords))
    for i, (y1, x1) in enumerate(coords):
        for j, (y2, x2) in enumerate(coords):
            dy, dx = y1 - y2, x1 - x2
            if periodic:
                dy, dx = (dy + ny // 2) % ny - ny // 2, (dx + nx // 2) % nx - nx // 2
            if (abs(dy), dx) == (1, 0):
                correlation[i, j] = 0.16
            elif (dy, abs(dx)) == (0, 1):
                correlation[i, j] = 0.09
            elif abs(dy) == abs(dx) == 1:
                correlation[i, j] = 0.05 if dy == dx else 0.03
    sigma = err.ravel()
    return sigma[:, None] * correlation * sigma[None, :]


KERNEL = np.array([[0.05, 0.16, 0.03], [0.09, 1.0, 0.09], [0.03, 0.16, 0.05]])


def assert_against_matrix(noise, residual, covariance):
    observed = residual[noise.valid]
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    quadratic = observed @ np.linalg.solve(covariance, observed)
    assert noise.quadratic(residual) == pytest.approx(quadratic, rel=2e-13, abs=2e-13)
    assert noise.logdet == pytest.approx(logdet, rel=2e-13, abs=2e-13)
    expected_logl = -0.5 * (quadratic + logdet + observed.size * math.log(2 * math.pi))
    assert noise.log_likelihood(residual) == pytest.approx(expected_logl, rel=2e-13, abs=2e-13)
    white = noise.whiten(residual)
    assert white.shape == observed.shape
    assert white @ white == pytest.approx(quadratic, rel=2e-13, abs=2e-13)


def test_white_likelihood_and_normalization():
    data = np.arange(24).reshape(4, 6) / 7
    err = np.linspace(0.5, 2, data.size).reshape(data.shape)
    diagonal = prepare_noise(data, err)
    fft = prepare_noise(data, err, backend="fft_periodic", kernel=[[1]])
    relative = prepare_noise(data, err, normalization="relative")
    covariance = np.diag(err.ravel() ** 2)
    for noise in (diagonal, fft):
        assert_against_matrix(noise, data, covariance)
        np.testing.assert_allclose(noise.whiten(data), (data / err).ravel(), atol=1e-14)
    assert relative.log_likelihood(data) == -0.5 * relative.quadratic(data)
    assert relative.metadata["normalization"] == "relative"


@pytest.mark.parametrize("shape", [(5, 5), (4, 6), (5, 4)])
def test_periodic_likelihood_and_whitening_against_dense(shape):
    rng = np.random.default_rng(20261118)
    residual = rng.normal(size=shape)
    err = rng.uniform(0.5, 2, size=shape)
    covariance = neighbor_covariance(shape, err, periodic=True)
    dense = prepare_noise(residual, err, backend="dense", kernel=KERNEL, boundary="periodic")
    fft = prepare_noise(residual, err, backend="fft_periodic", kernel=KERNEL)
    for noise in (dense, fft):
        assert_against_matrix(noise, residual, covariance)
        assert noise.metadata["boundary"] == "periodic"
    np.testing.assert_allclose(
        dense.whiten(residual), np.linalg.solve(np.linalg.cholesky(covariance), residual.ravel()),
        rtol=2e-13, atol=2e-13,
    )

    correlation = neighbor_covariance(shape, np.ones(shape), periodic=True)
    eigenvalues, vectors = np.linalg.eigh(correlation)
    standardized = (residual / err).ravel()
    expected = vectors @ ((vectors.T @ standardized) / np.sqrt(eigenvalues))
    np.testing.assert_allclose(fft.whiten(residual), expected, rtol=2e-13, atol=2e-13)
    assert fft.metadata["spectrum"]["raw_min"] == pytest.approx(eigenvalues.min())
    assert fft.metadata["spectrum"]["clipped_fraction"] == 0


@pytest.mark.parametrize("boundary", ["periodic", "finite"])
def test_masked_dense_is_marginal_distribution(boundary):
    rng = np.random.default_rng(46)
    data = rng.normal(size=(4, 5))
    err = rng.uniform(0.5, 2, size=data.shape)
    covariance = neighbor_covariance(data.shape, err, periodic=boundary == "periodic")
    mask = np.zeros_like(data, dtype=bool)
    mask[0, 0] = mask[1, 2] = True
    data[2, 2] = np.nan
    valid = ~mask & np.isfinite(data)
    marginal = covariance[np.ix_(valid.ravel(), valid.ravel())]
    err[~valid] = np.nan
    kernel_fit = prepare_noise(data, err, backend="dense", kernel=KERNEL,
                               boundary=boundary, mask=mask, error_kind="background_only")
    # Missing rows/columns of a supplied full covariance can be placeholders.
    supplied = covariance.copy()
    supplied[~valid.ravel(), :] = supplied[:, ~valid.ravel()] = np.nan
    explicit = prepare_noise(data, backend="dense", covariance=supplied, mask=mask)
    zeroed = np.where(valid, data, 0).ravel()
    wrong_quadratic = zeroed @ np.linalg.solve(covariance, zeroed)
    for noise in (kernel_fit, explicit):
        assert_against_matrix(noise, data, marginal)
        assert abs(noise.quadratic(data) - wrong_quadratic) > 1e-3
        assert noise.metadata["n_invalid"] == 3
        assert noise.metadata["invalid_pixels"] == [[0, 0], [1, 2], [2, 2]]
        assert noise.metadata["mask_mode"] == "exact_marginal"
        assert noise.metadata["weight_mode"] == "none"
        assert noise.metadata["approximate"] is False
    assert kernel_fit.metadata["error_kind"] == "background_only"
    assert explicit.metadata["boundary"] == "explicit"
    diagonal = prepare_noise(data, err, mask=mask)
    assert_against_matrix(diagonal, data, np.diag(err[valid] ** 2))


def test_fft_refuses_masked_pixels():
    image = np.ones((4, 5))
    mask = np.zeros_like(image, dtype=bool)
    mask[2, 2] = True
    with pytest.raises(ValueError, match="fully observed"):
        prepare_noise(image, image, backend="fft_periodic", kernel=KERNEL, mask=mask)
    data = image.copy()
    data[0, 0] = np.nan
    with pytest.raises(ValueError, match="fully observed"):
        prepare_noise(data, image, backend="fft_periodic", kernel=KERNEL)


@pytest.mark.parametrize(("options", "message"), [
    ({"err": np.ones((2, 3))}, "err.*shape"),
    ({"err": np.zeros((4, 5))}, "positive.*err|err.*positive"),
    ({"err": np.full((4, 5), np.nan)}, "finite"),
    ({"err": np.ones((4, 5)), "mask": np.ones((4, 5), dtype=bool)}, "no valid"),
    ({"err": np.ones((4, 5)), "mask": np.zeros((4, 5))}, "boolean"),
    ({"backend": "dense", "covariance": np.ones((20, 20))}, "positive definite"),
    ({"backend": "dense", "covariance": np.eye(19)}, "covariance.*shape"),
    ({"backend": "dense", "covariance": np.eye(20), "max_dense_bytes": 1}, "dense.*budget"),
    ({"backend": "dense", "covariance": np.eye(20), "err": np.ones((4, 5))}, "covariance.*err"),
    ({"backend": "dense", "err": np.ones((4, 5)), "kernel": KERNEL}, "boundary"),
    ({"backend": "fft_periodic", "err": np.ones((4, 5)), "kernel": [[0.7, 1, 0.7]]}, "positive definite"),
    ({"backend": "fft_periodic", "err": np.ones((4, 5)), "kernel": [[0.0, 1, 0.2]]}, "centrosymmetric"),
    ({"backend": "fft_periodic", "err": np.ones((4, 5)), "kernel": [[2]]}, "zero lag"),
    ({"backend": "fft_periodic", "err": np.ones((4, 5)), "kernel": np.ones((3, 7))}, "kernel.*grid"),
])
def test_invalid_covariance_and_noise_inputs(options, message):
    with pytest.raises(ValueError, match=message):
        prepare_noise(np.zeros((4, 5)), **options)


def test_prepared_noise_owns_inputs_and_checks_residuals():
    data = np.ones((4, 5))
    err = np.full((4, 5), 2.0)
    kernel = KERNEL.copy()
    noise = prepare_noise(data, err, backend="fft_periodic", kernel=kernel)
    expected = noise.log_likelihood(data)
    err[:] = 3
    kernel[:] = np.nan
    assert noise.log_likelihood(data) == expected
    with pytest.raises(ValueError, match="read-only"):
        noise.valid[0, 0] = False
    with pytest.raises(ValueError, match="shape"):
        noise.quadratic(np.ones((5, 4)))
    data[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        noise.whiten(data)


def test_roundoff_asymmetry_is_corrected_and_recorded():
    image = np.arange(25, dtype=float).reshape(5, 5) / 5
    err = np.ones_like(image)
    kernel = KERNEL.copy()
    kernel[0, 0] += 4e-8
    kernel[-1, -1] -= 4e-8
    original = prepare_noise(image, err, backend="fft_periodic", kernel=KERNEL)
    corrected = prepare_noise(image, err, backend="fft_periodic", kernel=kernel)
    assert corrected.log_likelihood(image) == pytest.approx(original.log_likelihood(image))
    assert corrected.metadata["kernel_max_abs_adjustment"] == pytest.approx(4e-8)
    assert original.metadata["kernel_max_abs_adjustment"] == 0.0
