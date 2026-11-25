"""M1 contracts for sampler-independent parameter definitions."""

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

from jwst_psfmc import (
    LogUniform,
    Normal,
    Parameter,
    ParameterSet,
    TruncatedNormal,
    Uniform,
    spawn_seed_streams,
)


def test_ordered_round_trip_and_ties_share_a_single_prior():
    parameters = ParameterSet([
        Parameter("host.dx", unit="pixel", tie="shared.dx", bounds=(-2, 2)),
        Parameter("host.n", unit="", fixed=1, bounds=(0.3, 8)),
        Parameter("host.re", 2, "pixel", prior=Uniform(0.2, 8), transform="log"),
        Parameter("shared.dx", unit="pixel", tie="agn.dx"),
        Parameter("agn.dx", -0.4, "pixel", prior=Uniform(-1, 1)),
        Parameter("neighbor.n", tie="host.n"),
    ])
    assert parameters.free_names == ("host.re", "agn.dx")
    assert parameters.ndim == 2
    values = parameters.decode(parameters.initial)
    assert tuple(values) == ("host.dx", "host.n", "host.re", "shared.dx", "agn.dx", "neighbor.n")
    assert values == dict(zip(values, [-0.4, 1, 2, -0.4, -0.4, 1]))
    np.testing.assert_allclose(parameters.encode(values), [math.log(2), -0.4])
    assert parameters.log_prior_in_sampling_space(parameters.initial) == pytest.approx(
        -math.log(7.8) + math.log(2) - math.log(2)
    )
    schema = parameters.to_schema()
    assert schema == json.loads(json.dumps(schema, allow_nan=False))
    assert schema["parameters"][0]["tie"] == "shared.dx"
    assert schema["parameters"][2]["prior"] == {"kind": "Uniform", "lo": 0.2, "hi": 8.0}
    assert schema["parameters"][2]["unit"] == "pixel"
    with pytest.raises(ValueError, match="conflicts"):
        parameters.encode({**values, "host.dx": 1})
    with pytest.raises(ValueError, match="conflicts"):
        parameters.encode({**values, "host.n": 2})
    with pytest.raises(ValueError, match="free names"):
        parameters.encode({"host.re": 2})
    with pytest.raises(ValueError, match="only known"):
        parameters.encode({**values, "typo": 1})


@pytest.mark.parametrize(("definitions", "message"), [
    ([Parameter("a", fixed=1), Parameter("a", fixed=1)], "duplicate"),
    ([Parameter("a", tie="b")], "missing tie target"),
    ([Parameter("a", tie="a")], "cyclic"),
    ([Parameter("a", tie="b"), Parameter("b", tie="a")], "cyclic"),
    ([Parameter("a", unit="deg", tie="b"), Parameter("b", unit="pixel", fixed=1)], "matching units"),
    ([Parameter("a", fixed=1, tie="b")], "exactly one"),
    ([Parameter("a", initial=1, tie="b")], "only for free"),
    ([Parameter("a", initial=3, prior=Uniform(0, 2))], "initial value"),
    ([Parameter("a", initial=np.nan, prior=Uniform(0, 2))], "finite"),
    ([Parameter("a", initial=1, prior=Uniform(0, 2), transform="log")], "positive"),
    ([Parameter("a", fixed=np.inf)], "finite"),
    ([Parameter("a", fixed=2, bounds=(0, 1))], "outside physical bounds"),
    ([Parameter("a", initial=1, prior=Uniform(0, 2), bounds=(0, 1))], "prior support"),
    ([Parameter("a", tie="b", bounds=(0, 1)), Parameter("b", 1, prior=Uniform(0, 2))], "tie bounds"),
    ([Parameter("a", tie="b", bounds=(0, 1)), Parameter("b", fixed=2)], "tie bounds"),
    ([Parameter("a", fixed=0, period=-180)], "periodic interval"),
    ([Parameter("a", fixed=0, origin=-90)], "origin requires"),
    ([Parameter("a", 0, prior=Uniform(-45, 45), period=180, origin=-90)], "canonical interval"),
    ([Parameter("a", tie="b", period=180), Parameter("b", fixed=0)], "inherits periodicity"),
])
def test_invalid_parameter_definitions(definitions, message):
    with pytest.raises(ValueError, match=message):
        ParameterSet(definitions)


def test_invalid_vector_support_and_cube():
    parameters = ParameterSet([Parameter("x", 0, prior=Uniform(-3, 2))])
    for vector in ([0, 0], [[0]], 0, [1 + 2j], ["1"]):
        for method in (parameters.decode, parameters.prior_transform, parameters.log_prior_in_sampling_space):
            with pytest.raises(ValueError, match="real vector"):
                method(vector)
    for vector in ([np.nan], [np.inf], [-3.01], [2.01]):
        with pytest.raises(ValueError):
            parameters.decode(vector)
        assert parameters.log_prior_in_sampling_space(vector) == -np.inf
    for cube in ([np.nan], [np.inf], [-0.01], [1.01]):
        with pytest.raises(ValueError, match="unit-cube"):
            parameters.prior_transform(cube)
    assert parameters.decode(parameters.prior_transform([0])) == {"x": -3}
    assert parameters.decode(parameters.prior_transform([1])) == {"x": 2}


def test_normalized_jacobian_and_posterior_invariant_to_coordinates():
    lo, hi = 0.125, 10.0
    prior = Uniform(lo, hi)
    linear = ParameterSet([Parameter("re", 1, prior=prior)])
    logarithmic = ParameterSet([Parameter("re", 1, prior=prior, transform="log")])
    for x in (lo, 0.4, 2.1, hi):
        z = logarithmic.encode({"re": x})
        assert logarithmic.decode(z)["re"] == pytest.approx(x)
        assert logarithmic.log_prior_in_sampling_space(z) == pytest.approx(
            linear.log_prior_in_sampling_space([x]) + math.log(x)
        )

    def integrate(parameters, a, b, likelihood, power):
        def density(t):
            x = parameters.decode([t])["re"]
            return math.exp(parameters.log_prior_in_sampling_space([t])) * likelihood(x) * x**power
        return quad(density, a, b, epsabs=1e-12)[0]

    log_lo, log_hi = math.log(lo), math.log(hi)
    assert integrate(logarithmic, log_lo, log_hi, lambda _: 1, 0) == pytest.approx(1, abs=1e-12)

    def likelihood(x):
        return math.exp(-0.5 * ((x - 2.2) / 0.6)**2)

    evidence_linear = integrate(linear, lo, hi, likelihood, 0)
    evidence_log = integrate(logarithmic, log_lo, log_hi, likelihood, 0)
    assert evidence_log == pytest.approx(evidence_linear, abs=1e-12)
    mean_linear = integrate(linear, lo, hi, likelihood, 1) / evidence_linear
    mean_log = integrate(logarithmic, log_lo, log_hi, likelihood, 1) / evidence_log
    assert mean_log == pytest.approx(mean_linear, abs=1e-12)
    cdf_linear = integrate(linear, lo, 2, likelihood, 0) / evidence_linear
    cdf_log = integrate(logarithmic, log_lo, math.log(2), likelihood, 0) / evidence_log
    assert cdf_log == pytest.approx(cdf_linear, abs=1e-12)

    log_uniform = ParameterSet([Parameter("re", 1, prior=LogUniform(lo, hi), transform="log")])
    assert log_uniform.log_prior_in_sampling_space([0]) == pytest.approx(-math.log(log_hi - log_lo))
    assert log_uniform.log_prior_in_sampling_space([1]) == pytest.approx(
        log_uniform.log_prior_in_sampling_space([0])
    )


def normal_cdf(x):
    return (1 + erf(x / math.sqrt(2))) / 2


@pytest.mark.parametrize(("prior", "initial", "transform", "cdf"), [
    (Uniform(-2, 6), 0, "linear", lambda x: (x + 2) / 8),
    (Uniform(0.2, 30), 1, "log", lambda x: (x - 0.2) / 29.8),
    (LogUniform(0.1, 20), 0.2, "log", lambda x: math.log(x / 0.1) / math.log(200)),
    (Normal(3, 2), 3, "linear", lambda x: normal_cdf((x - 3) / 2)),
    (TruncatedNormal(0, 1, 1, 3), 2, "log",
     lambda x: (normal_cdf(x) - normal_cdf(1)) / (normal_cdf(3) - normal_cdf(1))),
])
def test_cube_quantiles_against_independent_cdf(prior, initial, transform, cdf):
    parameters = ParameterSet([Parameter("x", initial, prior=prior, transform=transform)])
    uniforms = np.concatenate(([1e-5, 0.16, 0.5, 0.84, 1 - 1e-5], np.random.default_rng(790).random(32)))
    for u in uniforms:
        physical = parameters.decode(parameters.prior_transform([u]))["x"]
        assert cdf(physical) == pytest.approx(u, abs=2e-12)


@pytest.mark.parametrize(("prior", "start", "interval"), [
    (Uniform(-3, 8), 1, (-3, 8)),
    (LogUniform(0.05, 70), 1, (0.05, 70)),
    (Normal(1, 2), 1, (-np.inf, np.inf)),
    (TruncatedNormal(0, 1, 9, 12), 10, (9, 12)),
])
def test_prior_density_is_normalized(prior, start, interval):
    parameters = ParameterSet([Parameter("x", start, prior=prior)])
    integral = quad(lambda x: math.exp(parameters.log_prior_in_sampling_space([x])), *interval)[0]
    assert integral == pytest.approx(1, abs=1e-11)


def test_unbounded_cube_endpoints_require_interior_quantiles():
    parameters = ParameterSet([Parameter("x", 0, prior=Normal(0, 1))])
    for endpoint in (0, 1):
        with pytest.raises(ValueError, match="finite"):
            parameters.prior_transform([endpoint])


def test_periodic_pa_round_trip_and_boundary():
    parameters = ParameterSet([
        Parameter("host.pa", 90, "deg", prior=Uniform(-90, 90), period=180, origin=-90),
        Parameter("neighbor.pa", unit="deg", tie="host.pa"),
        Parameter("host.q", 0.9999, prior=Uniform(0.1, 1)),
    ])
    assert parameters.initial.tolist() == [-90, 0.9999]
    assert parameters.decode([91, 1]) == {"host.pa": -89, "neighbor.pa": -89, "host.q": 1}
    assert parameters.decode([-91, 1])["host.pa"] == 89
    assert parameters.encode({"host.pa": 91, "neighbor.pa": -89, "host.q": 1}).tolist() == [-89, 1]
    assert parameters.encode({"host.pa": 0.1, "neighbor.pa": 180.1, "host.q": 1}).tolist() == [0.1, 1]
    with pytest.raises(ValueError, match="conflicts"):
        parameters.encode({"host.pa": 0.1, "neighbor.pa": 180.10001, "host.q": 1})
    assert parameters.prior_transform([1, 1]).tolist() == [-90, 1]
    assert np.isfinite(parameters.log_prior_in_sampling_space([-90, 1]))
    assert parameters.log_prior_in_sampling_space([90, 1]) == -np.inf


def test_fixed_circular_zero_host_and_signed_difference_flux():
    fixed = ParameterSet([
        Parameter("host.pa", fixed=181, unit="deg", period=180),
        Parameter("host.q", fixed=1, bounds=(0, 1)),
        Parameter("host.flux", fixed=0, bounds=(0, None)),
    ])
    assert fixed.ndim == 0
    assert fixed.initial.shape == fixed.encode({}).shape == fixed.prior_transform([]).shape == (0,)
    assert fixed.decode([]) == {"host.pa": 1, "host.q": 1, "host.flux": 0}
    assert fixed.log_prior_in_sampling_space([]) == 0
    signed = ParameterSet([Parameter("diff.flux", -1, prior=Uniform(-4, 2))])
    assert signed.decode(signed.initial) == {"diff.flux": -1}
    assert signed.decode([0]) == {"diff.flux": 0}


def test_seed_streams_replay_from_provenance_and_workers_are_distinct():
    streams = spawn_seed_streams(18)
    replay = spawn_seed_streams(18)
    assert tuple(streams) == ("initialization", "noise", "sampler")
    draws = []
    for index, (name, child) in enumerate(streams.items()):
        assert child.state["entropy"] == 18
        assert child.state["spawn_key"] == (index,)
        assert child.state == replay[name].state
        recorded = np.random.SeedSequence(**child.state)
        draw = np.random.default_rng(child).integers(0, 2**32, size=8).tolist()
        assert draw == np.random.default_rng(recorded).integers(0, 2**32, size=8).tolist()
        draws.append(tuple(draw))
    assert len(set(draws)) == 3
    workers = streams["sampler"].spawn(2)
    assert [worker.spawn_key for worker in workers] == [(2, 0), (2, 1)]
    assert not np.array_equal(workers[0].generate_state(4), workers[1].generate_state(4))
    for seed in (-1, 1.5, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            spawn_seed_streams(seed)
