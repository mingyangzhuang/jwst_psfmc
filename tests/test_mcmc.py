"""
Tests for jwst_psfmc.mcmc prior construction.
"""

import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jwst_psfmc.mcmc import (
    prepare_for_fitting,
    log_likelihood_cov_prepared,
    log_prob_prepared,
    log_prior,
    run_mcmc,
    summarize_emcee,
    summarize_flux_from_chain,
)
from jwst_psfmc.plot import plot_psf_fit_triptych

_EXAMPLES = Path(__file__).parent.parent / "examples"
_DATA = _EXAMPLES / "data"
_PSF = _EXAMPLES / "PSF"

# See the note in test_psf.py: examples/ is not shipped in the distribution.
if not _DATA.is_dir():
    pytest.skip(
        "example data not available (examples/ is not shipped in the "
        "distribution) - clone the repository to run these tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def fit_kwargs():
    diff = fits.getdata(_DATA / "example1_f200w_diff.fits").astype(np.float64)
    err = fits.getdata(_DATA / "example1_f200w_diff_error.fits").astype(np.float64)
    kernel = fits.getdata(_DATA / "example1_f200w_cov_kernel.fits").astype(np.float64)
    psf = fits.getdata(_PSF / "example1_f200w_PSF_4_c.fits").astype(np.float64)
    cy, cx = diff.shape[0] // 2, diff.shape[1] // 2
    sl = (slice(cy - 4, cy + 5), slice(cx - 4, cx + 5))
    return dict(data=diff[sl], err=err[sl], psf_os=psf, cov_kernel=kernel,
                oversamp=4, native_shape=(11, 11))


class TestFluxPriorMin:
    def test_default_is_minus_three(self, fit_kwargs):
        """The default must reproduce the previously hard-coded value."""
        fit = prepare_for_fitting(**fit_kwargs)
        assert fit["prior_bounds"]["flux"][0] == -3.0

    @pytest.mark.parametrize("fmin", [-3.0, -50.0, -500.0])
    def test_lower_bound_is_honoured(self, fit_kwargs, fmin):
        fit = prepare_for_fitting(**fit_kwargs, flux_prior_min=fmin)
        assert fit["prior_bounds"]["flux"][0] == fmin

    def test_upper_bound_is_independent_of_it(self, fit_kwargs):
        """flux_prior_scale governs the upper bound; flux_prior_min must not."""
        a = prepare_for_fitting(**fit_kwargs)["prior_bounds"]["flux"][1]
        b = prepare_for_fitting(**fit_kwargs, flux_prior_min=-500.0)["prior_bounds"]["flux"][1]
        assert a == b

    @pytest.mark.parametrize("fmin", [-3.0, -50.0, -500.0])
    def test_walkers_start_inside_the_prior(self, fit_kwargs, fmin):
        """A wide lower bound must not push any walker to -inf log-prior."""
        fit = prepare_for_fitting(**fit_kwargs, flux_prior_min=fmin)
        bounds = fit["prior_bounds"]
        assert all(np.isfinite(log_prior(w, bounds)) for w in fit["pos"])


def test_excluded_pixels_leave_default_flux_prior_and_initialization_unchanged(fit_kwargs):
    data = np.full((9, 9), .2)
    err = np.ones_like(data)
    err[0, 1] = 0
    err[0, 2] = np.nan
    weight = np.ones_like(data)
    weight[0, 0] = 0
    weight[0, 3] = np.nan
    data[0, 4] = np.nan
    options = dict(fit_kwargs, data=data, err=err, fit_weight=weight, seed=22)
    first = prepare_for_fitting(**options)
    modified = data.copy()
    modified[0, :4] = [1e6, 2e6, 3e6, 4e6]
    second = prepare_for_fitting(**(options | {"data": modified}))
    assert first["prior_bounds"]["flux"] == (-3.0, 6.0)
    assert first["prior_bounds"] == second["prior_bounds"]
    np.testing.assert_array_equal(first["pos"], second["pos"])


def test_explicit_flux_bounds_and_narrow_intervals_support_signed_flux(fit_kwargs):
    bounds = (-2.002, -2.001)
    fit = prepare_for_fitting(**fit_kwargs, flux_prior_bounds=bounds,
                              pos_prior_half_width=.003, bkg_prior_half_width=.002,
                              flux_prior_scale=500, flux_prior_min=-900, seed=14)
    assert fit["prior_bounds"]["flux"] == bounds
    assert len(np.unique(fit["pos"], axis=0)) == fit["nwalkers"]
    assert all(log_prior(row, fit["prior_bounds"]) == 0 for row in fit["pos"])
    assert np.linalg.matrix_rank(fit["pos"] - fit["pos"].mean(axis=0)) == 4
    lower, upper = np.array(list(fit["prior_bounds"].values())).T
    assert not np.any((fit["pos"] == lower) | (fit["pos"] == upper))


@pytest.mark.parametrize("bounds", [(-2, -2), (2, -2), (-2, np.inf), (-2,), ("bad", 1)])
def test_invalid_explicit_flux_bounds_raise(fit_kwargs, bounds):
    with pytest.raises(ValueError, match="bounds"):
        prepare_for_fitting(**fit_kwargs, flux_prior_bounds=bounds)


def test_legacy_emcee_seed_replays(fit_kwargs):
    def sample(seed):
        fit = prepare_for_fitting(**fit_kwargs, nwalkers=8, seed=seed)
        return run_mcmc(**fit, nsteps=12, ncores=1, progress=False, seed=seed).get_chain()

    first = sample(12)
    np.testing.assert_array_equal(first, sample(12))
    assert not np.array_equal(first, sample(13))


@pytest.fixture(scope="module")
def det_fit():
    """prepare_for_fitting on the detection example."""
    diff = fits.getdata(_DATA / "example2_f200w_diff.fits").astype(np.float64)
    err = fits.getdata(_DATA / "example2_f200w_diff_error.fits").astype(np.float64)
    kernel = fits.getdata(_DATA / "example2_f200w_cov_kernel.fits").astype(np.float64)
    psf = fits.getdata(_PSF / "example2_f200w_PSF_4_c.fits").astype(np.float64)
    cy, cx = diff.shape[0] // 2, diff.shape[1] // 2
    sl = (slice(cy - 4, cy + 5), slice(cx - 4, cx + 5))
    fit = prepare_for_fitting(data=diff[sl], err=err[sl], psf_os=psf,
                              cov_kernel=kernel, oversamp=4, native_shape=(11, 11))
    return fit, diff[sl], err[sl]


@pytest.fixture(scope="module")
def det_sampler(det_fit):
    fit, _, _ = det_fit
    return run_mcmc(**fit, nsteps=400, ncores=1, progress=False)


class TestRunMcmcRejectsTypos:
    """run_mcmc used to accept **_extra, silently absorbing misspellings."""

    @pytest.mark.parametrize("bad", [{"nstep": 5000}, {"n_cores": 8},
                                     {"nwalker": 64}, {"progres": False}])
    def test_unknown_keyword_raises(self, det_fit, bad):
        fit, _, _ = det_fit
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            run_mcmc(**fit, **bad, nsteps=10, ncores=1, progress=False)

    def test_splatting_the_prepared_dict_still_works(self, det_sampler):
        assert det_sampler.get_chain().shape[-1] == 4


def test_nautilus_end_to_end(det_fit, tmp_path, monkeypatch):
    pytest.importorskip("nautilus")
    import jwst_psfmc as jpm
    from jwst_psfmc import mcmc

    fit, _, _ = det_fit
    transform = mcmc._psf_fft
    calls = []

    def counted_fft(psf):
        calls.append(psf.shape)
        return transform(psf)

    monkeypatch.setattr(mcmc, "_psf_fft", counted_fft)
    result = run_mcmc(
        **fit, sampler="nautilus", n_live=100, n_eff=100, ncores=1,
        seed=7, progress=False,
    )
    assert calls == [fit["psf_os"].shape]
    sample = result["samples"][0]
    fields = ("prepared", "psf_os", "oversamp", "native_shape", "psf_prepare_fraction")
    assert result["log_l"][0] == log_likelihood_cov_prepared(sample, **{k: fit[k] for k in fields})
    path = tmp_path / "posterior.v1"
    jpm.save_nautilus_results(path, result)
    loaded = jpm.load_nautilus_results(path)
    assert loaded["labels"] == ("flux", "dx", "dy", "bkg")
    for key in ("samples", "log_w", "log_l"):
        np.testing.assert_array_equal(loaded[key], result[key])
    assert loaded["n_eff"] == result["n_eff"] >= 100
    assert loaded["n_like"] == result["n_like"] >= len(loaded["samples"])
    assert np.exp(loaded["log_w"]).sum() == pytest.approx(1)

    flux = loaded["samples"][:, 0]
    order = np.argsort(flux)
    cdf = np.cumsum(np.exp(loaded["log_w"][order]))
    summary = jpm.summarize_posterior(loaded)
    assert summary["flux"]["median"] == flux[order][np.searchsorted(cdf, 0.5)]
    fig, axes = jpm.plot_corner(loaded)
    try:
        assert axes.size == 16
        assert axes[-1].get_xlabel() == "bkg"
    finally:
        plt.close(fig)


def test_legacy_emcee_and_flux_summary_each_compute_one_psf_fft(det_fit, monkeypatch):
    from jwst_psfmc import mcmc

    fit, _, _ = det_fit
    transform = mcmc._psf_fft
    calls = []

    def counted_fft(psf):
        calls.append(psf.shape)
        return transform(psf)

    monkeypatch.setattr(mcmc, "_psf_fft", counted_fft)
    sampler = run_mcmc(**fit, nsteps=12, ncores=1, progress=False, seed=21)
    assert calls == [fit["psf_os"].shape]
    result = summarize_flux_from_chain(sampler, burnin=2, thin=2, fit=fit)
    assert calls == [fit["psf_os"].shape] * 2
    fields = ("prepared", "psf_os", "oversamp", "prior_bounds", "native_shape", "psf_prepare_fraction")
    point = sampler.get_chain()[-1, 0]
    assert sampler.get_log_prob()[-1, 0] == log_prob_prepared(point, **{k: fit[k] for k in fields})
    assert np.isfinite(result["stamp_flux_sum"]["median"])


class TestPostFitNormalisation:
    """The post-fit helpers defaulted to geometry that silently mis-normalises."""

    def test_summarize_flux_warns_without_output_shape(self, det_fit, det_sampler):
        fit, _, _ = det_fit
        with pytest.warns(RuntimeWarning, match="output_shape is None"):
            res = summarize_flux_from_chain(det_sampler, burnin=100, thin=4,
                                            psf_os=fit["psf_os"])
        # the giveaway: a constant fraction of exactly 1 with no spread
        assert res["total_encircled_fraction"]["median"] == pytest.approx(1.0)
        assert res["total_encircled_fraction"]["plus_1sigma"] == pytest.approx(0.0)

    def test_summarize_flux_with_fit_is_correct_and_quiet(self, det_fit, det_sampler):
        fit, data, _ = det_fit
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = summarize_flux_from_chain(det_sampler, burnin=100, thin=4, fit=fit)
        explicit = summarize_flux_from_chain(
            det_sampler, burnin=100, thin=4, psf_os=fit["psf_os"], oversamp=4,
            native_shape=fit["native_shape"], output_shape=data.shape,
            psf_prepare_fraction=fit["psf_prepare_fraction"])
        for key in ("total_flux", "stamp_flux_sum", "total_encircled_fraction"):
            assert got[key]["median"] == pytest.approx(explicit[key]["median"])
        assert got["total_encircled_fraction"]["median"] < 1.0

    def test_explicit_arguments_beat_fit(self, det_fit, det_sampler):
        fit, data, _ = det_fit
        res = summarize_flux_from_chain(det_sampler, burnin=100, thin=4, fit=fit,
                                        output_shape=data.shape)
        assert res["stamp_flux_sum"]["median"] > 0

    def test_triptych_warns_without_native_shape(self, det_fit, det_sampler):
        fit, data, err = det_fit
        summary = summarize_emcee(det_sampler, burnin=100, thin=4)
        with pytest.warns(RuntimeWarning, match="native_shape is None"):
            plot_psf_fit_triptych(data, err, fit["psf_os"], 4, summary=summary)
        plt.close("all")

    def test_triptych_with_fit_is_quiet(self, det_fit, det_sampler):
        fit, data, err = det_fit
        summary = summarize_emcee(det_sampler, burnin=100, thin=4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fig, _ = plot_psf_fit_triptych(data, err, summary=summary, fit=fit)
        plt.close("all")

    def test_triptych_requires_psf_or_fit(self, det_fit, det_sampler):
        _, data, err = det_fit
        summary = summarize_emcee(det_sampler, burnin=100, thin=4)
        with pytest.raises(ValueError, match="psf_os is required"):
            plot_psf_fit_triptych(data, err, summary=summary)

    def test_triptych_annotation_uses_mathtext_for_negatives(self, det_fit):
        """Parameter values are wrapped in $...$ so a negative renders with a
        true minus sign instead of a hyphen, and the mathtext must parse."""
        from matplotlib import mathtext

        fit, data, err = det_fit
        param = np.array([12.345, -0.271, -0.038, -0.0042])
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # a mathtext parse problem raises
            fig, axes = plot_psf_fit_triptych(data, err, param=param, fit=fit)
            fig.canvas.draw()                   # forces the text to be laid out
        text = [t.get_text() for a in axes for t in a.texts][0]
        plt.close("all")

        assert "$-0.271$" in text and "$-0.038$" in text
        assert "$12.345$" in text
        parser = mathtext.MathTextParser("agg")
        for frag in ("$12.345$", "$-0.271$", "$-0.0042$"):
            parser.parse(frag)
