"""
Tests for jwst_psfmc.io — the save/load round trip that persists posteriors.

A short MCMC chain is run against the bundled example data so the round trip
is exercised on a real sampler object rather than a stand-in.
"""

import numpy as np
import pytest
from pathlib import Path
from astropy.io import fits

from jwst_psfmc.io import save_emcee_results, load_emcee_results
from jwst_psfmc.mcmc import prepare_for_fitting, run_mcmc, summarize_emcee

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

_BURNIN, _THIN, _NSTEPS = 100, 2, 400


def _stamp(img, half=4):
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    return img[cy - half : cy + half + 1, cx - half : cx + half + 1]


@pytest.fixture(scope="module")
def sampler():
    """A short but genuine chain on the detection example."""
    diff = fits.getdata(_DATA / "example2_f200w_diff.fits").astype(np.float64)
    err = fits.getdata(_DATA / "example2_f200w_diff_error.fits").astype(np.float64)
    kernel = fits.getdata(_DATA / "example2_f200w_cov_kernel.fits").astype(np.float64)
    psf = fits.getdata(_PSF / "example2_f200w_PSF_4_c.fits").astype(np.float64)
    prepared = prepare_for_fitting(
        data=_stamp(diff), err=_stamp(err), psf_os=psf, cov_kernel=kernel,
        oversamp=4, native_shape=(11, 11), nwalkers=48,
    )
    return run_mcmc(**prepared, nsteps=_NSTEPS, ncores=1, progress=False)


@pytest.fixture(scope="module")
def roundtrip(sampler, tmp_path_factory):
    path = tmp_path_factory.mktemp("io") / "result"
    payload = save_emcee_results(path, sampler, burnin=_BURNIN, thin=_THIN)
    return payload, load_emcee_results(str(path) + ".npz")


class TestSaveLoadRoundTrip:
    def test_file_is_created(self, sampler, tmp_path):
        path = tmp_path / "res"
        save_emcee_results(path, sampler, burnin=_BURNIN, thin=_THIN)
        assert (tmp_path / "res.npz").is_file()

    def test_keys_survive(self, roundtrip):
        _, loaded = roundtrip
        for key in ("chain", "flat_chain", "summary", "labels", "tau",
                    "acceptance_fraction", "burnin", "thin"):
            assert key in loaded, f"missing key: {key}"

    def test_chain_values_are_preserved(self, sampler, roundtrip):
        _, loaded = roundtrip
        expected = sampler.get_chain(discard=_BURNIN, thin=_THIN, flat=True)
        np.testing.assert_allclose(loaded["flat_chain"], expected, rtol=0, atol=0)

    def test_summary_matches_a_fresh_summarize(self, sampler, roundtrip):
        """The persisted summary must equal one recomputed from the sampler."""
        _, loaded = roundtrip
        fresh = summarize_emcee(sampler, burnin=_BURNIN, thin=_THIN)
        for label in fresh:
            for stat in ("median", "minus_1sigma", "plus_1sigma"):
                assert loaded["summary"][label][stat] == pytest.approx(
                    fresh[label][stat], rel=1e-12
                ), f"{label}.{stat} changed across the round trip"

    def test_metadata_is_preserved(self, sampler, roundtrip):
        _, loaded = roundtrip
        assert int(loaded["burnin"]) == _BURNIN
        assert int(loaded["thin"]) == _THIN
        assert list(loaded["labels"]) == ["flux", "dx", "dy", "bkg"]
        np.testing.assert_allclose(
            loaded["acceptance_fraction"], sampler.acceptance_fraction, rtol=1e-12
        )

    def test_returned_payload_matches_loaded(self, roundtrip):
        payload, loaded = roundtrip
        np.testing.assert_allclose(payload["flat_chain"], loaded["flat_chain"])

    @pytest.mark.parametrize("name", [
        "run1",                 # no suffix
        "run1.npz",             # explicit suffix
        "sn2023abc_v1.2",       # dot in the name - would previously fail
        "out.dat",              # unrelated suffix
    ])
    def test_round_trip_for_any_path_spelling(self, sampler, tmp_path, name):
        """save then load with the same string must work.

        numpy appends .npz unless the name already ends in it, so
        "sn2023abc_v1.2" is written as "sn2023abc_v1.2.npz". Testing
        `path.suffix` instead of the ".npz" ending made load_emcee_results
        raise FileNotFoundError for a file it had just written.
        """
        path = tmp_path / name
        save_emcee_results(path, sampler, burnin=_BURNIN, thin=_THIN)
        loaded = load_emcee_results(path)
        assert loaded["flat_chain"].shape[1] == 4

    def test_load_accepts_the_written_filename_too(self, sampler, tmp_path):
        """The on-disk name (with .npz) also loads."""
        save_emcee_results(tmp_path / "sn.1", sampler, burnin=_BURNIN, thin=_THIN)
        assert (tmp_path / "sn.1.npz").is_file()
        assert load_emcee_results(tmp_path / "sn.1.npz")["thin"] == _THIN

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_emcee_results(tmp_path / "does_not_exist")


class TestAutocorrelationTime:
    def test_tau_excludes_burnin(self):
        """tau must be measured on the post-burn-in chain.

        Including the burn-in leaves the transient in the series and inflates
        tau by orders of magnitude, which wrongly clears tau_ok.
        """
        import emcee

        class _TransientSampler:
            """Chain with a strong 300-step transient, then white noise."""

            def __init__(self):
                rng = np.random.default_rng(1)
                burn = (np.linspace(50, 0, 300)[None, :, None]
                        + rng.standard_normal((48, 300, 4)))
                good = rng.standard_normal((48, 700, 4))
                self.c = np.concatenate([burn, good], axis=1)
                self.acceptance_fraction = np.full(48, 0.5)

            def get_chain(self, discard=0, thin=1, flat=False):
                c = self.c[:, discard::thin, :]
                return c.reshape(-1, 4) if flat else np.swapaxes(c, 0, 1)

            def get_autocorr_time(self, discard=0, thin=1, **kw):
                x = np.swapaxes(self.c[:, discard::thin, :], 0, 1)
                return thin * emcee.autocorr.integrated_time(x, quiet=True)

            def get_log_prob(self, **kw):
                raise RuntimeError("no log prob")

        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "res")
            save_emcee_results(path, _TransientSampler(), burnin=500, thin=4)
            loaded = load_emcee_results(path + ".npz")

        # post-burn-in the chain is white noise: tau ~ 1, not ~100
        assert np.all(loaded["tau"] < 10), f"tau not deburned: {loaded['tau']}"
        assert loaded["tau_ok"]
