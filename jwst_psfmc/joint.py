"""Joint fixed-covariance inference across registered observations."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import block_diag

from .fitting import PreparedModelFit, _array_digest, _evaluate_image, prepare_model_fitting
from .models import CompositeModel, SersicSource, _COMPONENT_FIELDS
from .noise import PreparedNoise, _check_dense_budget, _valid_pixels, prepare_noise
from .parameters import ParameterSet
from .rendering import RenderContext


@dataclass(frozen=True, kw_only=True, eq=False)
class JointObservation:
    """One registered single-filter image and its model.

    Aliases map component field names in this frame to names in the joint
    ParameterSet. Unmapped names are read directly. Exposures must be stable
    identities of original science inputs. A difference must also list its
    reference inputs. Images must already use a common flux calibration and
    unit. Registration and calibration describe these fixed assumptions.
    Offsets and sizes used by the model are native output pixel quantities.
    A shared Sérsic radius requires a common pixel scale. For other grids,
    specify frame-specific native geometry parameters.
    """

    name: str
    band: str
    data: np.ndarray = field(repr=False)
    model: CompositeModel
    exposures: tuple[str, ...]
    registration: str
    calibration: str
    context: RenderContext | None = None
    aliases: Mapping[str, str] = field(default_factory=dict)
    noise_options: Mapping = field(default_factory=dict)
    kind: str = "direct"
    reference_exposures: tuple[str, ...] = ()
    data_unit: str = "flux per output pixel"


@dataclass(frozen=True, eq=False)
class _PreparedObservation:
    name: str
    fit: PreparedModelFit
    aliases: dict[str, str]
    provenance: dict

    def local_values(self, values):
        return {**values, **{local: values[target] for local, target in self.aliases.items()}}


@dataclass(frozen=True, eq=False)
class PreparedJointFit:
    """Prepared observations, one joint parameter set and optional block covariance."""

    observations: tuple[_PreparedObservation, ...]
    parameters: ParameterSet
    noise: PreparedNoise | None
    provenance: dict

    @property
    def metadata(self):
        """Return ordered frame metadata and the joint likelihood convention."""
        observations = []
        for obs in self.observations:
            fit = obs.fit
            observations.append({
                **obs.provenance,
                **fit.provenance,
                "name": obs.name,
                "aliases": dict(obs.aliases),
                "components": [{"name": c.name, "type": type(c).__name__} for c in fit.model.components],
                "geometry": {"shape": list(fit.data.shape), "center_xy": list(fit.context.center),
                             "coordinate_convention": "array[y, x]; offsets in native pixels"},
                "psf": None if fit.context.psf_operator is None else fit.context.psf_operator.metadata,
                "noise": fit.noise.metadata,
            })
        return {
            **self.provenance,
            "observations": observations,
            "noise": {
                "backend": "independent_observations" if self.noise is None else "block_dense",
                "normalization": self.provenance["likelihood_normalization"],
                "block_noise": None if self.noise is None else self.noise.metadata,
            },
        }


def _ids(items, name, *, required):
    if items is None or isinstance(items, str):
        raise ValueError(f"{name} must contain exposure identifiers")
    ids = tuple(items)
    if (required and not ids) or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError(f"{name} requires unique, nonempty exposure identifiers")
    normalized = tuple(i.strip() for i in ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} requires unique, nonempty exposure identifiers")
    return normalized


def _pack(images):
    return np.concatenate([image.ravel(order="C") for image in images])[None, :]


def _validate_observation(obs, parameters):
    if not isinstance(obs, JointObservation):
        raise ValueError("every observation must be a JointObservation")
    if any(not isinstance(value, str) or not value.strip()
           for value in (obs.name, obs.band, obs.registration, obs.calibration, obs.data_unit)):
        raise ValueError("observation name, band, registration, calibration and data unit are required")
    if obs.kind not in ("direct", "difference"):
        raise ValueError("observation kind must be direct or difference")
    science = _ids(obs.exposures, "science exposures", required=True)
    reference = _ids(obs.reference_exposures, "reference exposures", required=obs.kind=="difference")
    if obs.kind == "direct" and reference:
        raise ValueError("direct observations cannot have a reference")
    if obs.kind == "difference" and any(isinstance(c, SersicSource) for c in obs.model.components):
        raise ValueError("a static Sérsic host belongs on direct images; supply a difference response model")
    aliases = dict(obs.aliases)
    if any(not isinstance(key, str) or not key.strip() or not isinstance(target, str)
           or target not in parameters._roots for key, target in aliases.items()):
        raise ValueError(f"{obs.name}: aliases require nonempty local names and known joint parameters")
    fields = set()
    for component in obs.model.components:
        suffixes = _COMPONENT_FIELDS.get(type(component))
        if suffixes is None:
            suffixes = getattr(component, "parameter_fields", None)
            if (not isinstance(suffixes, (tuple, list))
                    or any(not isinstance(field, str) or not field.strip() for field in suffixes)):
                raise ValueError(f"{obs.name}: component {component.name} must declare parameter_fields")
        fields.update(f"{component.name}.{field}" for field in suffixes)
    unused = aliases.keys() - fields
    if unused:
        alias = sorted(unused)[0]
        raise ValueError(f"{obs.name}: component {alias.rsplit('.', 1)[0]} has invalid alias {alias!r}")
    targets = {aliases.get(field, field) for field in fields}
    missing = targets - parameters._roots.keys()
    if missing:
        raise ValueError(f"{obs.name}: missing model parameter {sorted(missing)[0]}")
    return science, reference, aliases, {parameters._roots[name].name for name in targets}


def prepare_joint_model_fitting(observations, parameters, *, joint_covariance=None,
                                normalization="absolute", max_dense_bytes=64*1024**2):
    """Prepare independent frames or a full Gaussian block covariance.

    A full covariance uses observation order, then C-order pixels within
    each frame, including masked pixels. It supplies all marginal and
    cross-frame covariances; only per-observation masks may accompany it.
    Reused exposure identities require a block covariance. Direct and
    difference frames are separate likelihoods because their forward
    models and shared inputs differ. Effects of registration and flux
    calibration are conditional on the supplied inputs.

    `max_dense_bytes` bounds the workspace of the full block or of each
    independent dense fit. An independent frame may specify a lower
    limit through its noise options.
    """
    obs = tuple(observations)
    if not obs or not isinstance(parameters, ParameterSet):
        raise ValueError("provide observations and one joint ParameterSet")
    details = [_validate_observation(item, parameters) for item in obs]
    if len({item.name for item in obs}) != len(obs):
        raise ValueError("observation names must be unique")
    if any(len({getattr(item, attr) for item in obs}) != 1 for attr in ("band", "data_unit", "kind")):
        raise ValueError("joint observations must share one band, data unit and image kind")
    unused = set(parameters.free_names).difference(*(detail[3] for detail in details))
    if unused:
        raise ValueError(f"joint observations do not use free parameters: {', '.join(sorted(unused))}")
    seen = set()
    for science, reference, _, _ in details:
        used = set(science + reference)
        if joint_covariance is None and (seen & used or set(science) & set(reference)):
            raise ValueError("overlapping exposures require an explicit joint covariance")
        seen.update(used)
    radius_scales = {}
    for item, (_, _, aliases, _) in zip(obs, details):
        operator = None if item.context is None else item.context.psf_operator
        scale = None if operator is None else operator.spec.native_pixel_scale
        if scale is None:
            continue
        for component in item.model.components:
            name = f"{component.name}.re"
            root = parameters._roots.get(aliases.get(name, name))
            if isinstance(component, SersicSource) and root is not None:
                prior_scale = radius_scales.setdefault(root.name, scale)
                if not np.isclose(scale, prior_scale, rtol=1e-8, atol=0):
                    raise ValueError("a shared Sérsic radius requires a common native pixel scale")

    data = [np.asarray(item.data) for item in obs]
    if any(image.ndim != 2 or image.dtype.kind not in "iuf" for image in data):
        raise ValueError("observation data must be real two-dimensional images")
    if joint_covariance is not None:
        if any(item.noise_options.keys() - {"mask"} for item in obs):
            raise ValueError("only per-observation masks may accompany a joint covariance")
        n_valid = sum(np.count_nonzero(_valid_pixels(image, item.noise_options.get("mask")))
                      for image, item in zip(data, obs))
        _check_dense_budget(n_valid, max_dense_bytes)
        covariance = np.asarray(joint_covariance)
        n_pixels = sum(image.size for image in data)
        if covariance.shape != (n_pixels, n_pixels):
            raise ValueError("joint covariance must cover all observation pixels in C order")
    prepared = []
    offset = 0
    for item, image, (science, reference, aliases, _) in zip(obs, data, details):
        options = dict(item.noise_options)
        if joint_covariance is not None:
            indices = slice(offset, offset+image.size)
            options.update(backend="dense", covariance=covariance[indices, indices],
                           max_dense_bytes=max_dense_bytes, error_kind="joint_covariance_marginal")
            offset += image.size
        elif options.get("backend") == "dense":
            _check_dense_budget(0, max_dense_bytes)
            frame_budget = options.get("max_dense_bytes", max_dense_bytes)
            _check_dense_budget(0, frame_budget)
            options["max_dense_bytes"] = min(max_dense_bytes, frame_budget)
        if options.get("normalization", normalization) != normalization:
            raise ValueError("observations must use the joint likelihood normalization")
        options["normalization"] = normalization
        single = prepare_model_fitting(image, item.model, parameters, context=item.context,
                                       noise_options=options, data_unit=item.data_unit)
        prepared.append(_PreparedObservation(item.name, single, aliases, {
            "band": item.band, "kind": item.kind, "exposures": list(science),
            "reference_exposures": list(reference), "registration": item.registration,
            "calibration": item.calibration,
        }))
    provenance = {
        "render_accuracy": {"status": "unvalidated",
                            "reason": "no independent reference for the joint fitting domain"},
        "joint_fit": True, "covariance_mode": "independent" if joint_covariance is None else "explicit_block",
        "likelihood_normalization": normalization,
    }
    noise = None
    if joint_covariance is not None:
        noise = prepare_noise(_pack(data), backend="dense", covariance=covariance,
                              mask=~_pack(item.fit.noise.valid for item in prepared),
                              normalization=normalization, max_dense_bytes=max_dense_bytes,
                              error_kind="joint_covariance")
        provenance["joint_covariance_sha256"] = _array_digest(covariance, "<f8")
    return PreparedJointFit(tuple(prepared), parameters, noise, provenance)


def evaluate_joint_model_fit(fit, sample_vector=None, *, decompose=False):
    """Evaluate one shared prior and either independent or block frame likelihoods.

    With block covariance, frame diagnostics show marginal frame errors.
    Only the top-level likelihood and whitened vector include cross-frame
    correlations. Frame marginal log likelihoods cannot be added in this case.
    """
    vector = fit.parameters.initial if sample_vector is None else sample_vector
    prior = fit.parameters.log_prior_in_sampling_space(vector)
    if not np.isfinite(prior):
        raise ValueError("sample vector is outside prior support")
    values = fit.parameters.decode(vector)
    frames = {obs.name: _evaluate_image(obs.fit, obs.local_values(values), decompose)
              for obs in fit.observations}
    if fit.noise is None:
        logl = sum(frame["log_likelihood"] for frame in frames.values())
    else:
        residual = _pack(obs.fit.data-frames[obs.name]["image"] for obs in fit.observations)
        logl = fit.noise.log_likelihood(residual)
        for frame in frames.values():
            frame["marginal_log_likelihood"] = frame.pop("log_likelihood")
    result = dict(values=values, log_prior=prior, log_likelihood=logl,
                  log_posterior=logl+prior, observations=frames, metadata=fit.metadata)
    if decompose and fit.noise is not None:
        result["whitened_residual"] = fit.noise.whiten(residual)
    return result


def shared_reference_covariance(science_covariances, reference_covariance, reference_operators, *,
                                max_dense_bytes=64*1024**2):
    """Construct Cov(D) for D_t = science_t - A_t reference.

    The science frames and reference must be mutually independent.
    A_t maps reference pixels into difference-frame pixels in C order.
    For other input dependencies, supply the full covariance directly.
    max_dense_bytes bounds the estimated full output and workspace.
    """
    science = tuple(np.asarray(c) for c in science_covariances)
    operators = tuple(np.asarray(a) for a in reference_operators)
    reference = np.asarray(reference_covariance)
    if not science or len(science) != len(operators):
        raise ValueError("provide a reference operator for each science covariance")
    for cov in (*science, reference):
        if cov.ndim != 2 or not cov.shape[0] or cov.shape[0] != cov.shape[1]:
            raise ValueError("input covariance matrices must be nonempty and square")
    size = sum(c.shape[0] for c in science)
    # Guard both the full block and temporary reference projections.
    _check_dense_budget(max(size, reference.shape[0]), max_dense_bytes,
                        extra_bytes=2 * np.dtype(float).itemsize * size * reference.shape[0])
    for cov in (*science, reference):
        if (cov.dtype.kind not in "iuf"
                or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1e-12, atol=0)):
            raise ValueError("input covariance matrices must be nonempty, finite and symmetric")
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[0] < -1e-12*np.max(np.abs(eigenvalues)):
            raise ValueError("input covariance matrices must be positive semidefinite")
    if any(a.dtype.kind not in "iuf" or a.shape != (c.shape[0], reference.shape[0]) or not np.all(np.isfinite(a))
           for c, a in zip(science, operators)):
        raise ValueError("each reference operator must map the reference into its science frame")
    combined = np.vstack(operators).astype(float, copy=False)
    cross = combined @ reference @ combined.T
    return block_diag(*science) + (cross+cross.T)/2
