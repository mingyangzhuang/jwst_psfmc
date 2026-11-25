"""Prepare and evaluate image models with fixed Gaussian covariance."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib

import numpy as np

from ._render_accuracy import assess_rendering
from .models import CompositeModel, _sum_components
from .noise import PreparedNoise, prepare_noise
from .parameters import ParameterSet
from .rendering import RenderContext


@dataclass(frozen=True, eq=False)
class PreparedModelFit:
    """Image, parameters, geometry and noise cached for model evaluation.

    Use :func:`prepare_model_fitting` to construct this object.
    """

    data: np.ndarray = field(repr=False)
    model: CompositeModel
    parameters: ParameterSet
    context: RenderContext
    noise: PreparedNoise
    provenance: dict = field(default_factory=dict)


def _array_digest(array, dtype):
    values = np.asarray(array)
    digest = hashlib.sha256(str(values.shape).encode())
    if values.flags.c_contiguous and values.dtype == np.dtype(dtype):
        digest.update(memoryview(values).cast("B"))
    else:
        # A mostly masked covariance may be large even when its observed
        # submatrix is small. Hash conversions in bounded C-order chunks.
        with np.nditer(values, flags=["external_loop", "buffered", "zerosize_ok"],
                       op_dtypes=[dtype], order="C", casting="unsafe") as chunks:
            for chunk in chunks:
                digest.update(np.ascontiguousarray(chunk).tobytes())
    return digest.hexdigest()


def prepare_model_fitting(
    data, model: CompositeModel, parameters: ParameterSet, *,
    context: RenderContext | None = None, noise_options: Mapping | None = None,
    data_unit: str = "flux per output pixel",
    render_references: Mapping | None = None, render_reference_label: str | None = None,
    render_epsilon_max: float = .1,
) -> PreparedModelFit:
    """Prepare a model for deterministic image and likelihood evaluation.

    Parameters
    ----------
    data : array-like, shape (ny, nx)
        Image in flux per output pixel.
    model : CompositeModel
        Image model. A single component can be wrapped in a composite.
    parameters : ParameterSet
        Physical parameter definitions used by the model.
    context : RenderContext, optional
        Render geometry. Defaults to the shape of the data.
    noise_options : mapping, optional
        Keyword arguments to :func:`~jwst_psfmc.noise.prepare_noise`. Supply
        err here, or an explicit covariance for the dense backend.
    data_unit : str, optional
        Unit label for provenance. No unit conversion is applied.
    render_references : mapping, optional
        Independent unit-flux reference images keyed by source component,
        at the same fixed geometry and phase. Each image uses the same PSF
        normalization and native pixel units as its fitted component.
    render_reference_label : str, optional
        Provenance of the supplied references.
    render_epsilon_max : float, optional
        Budget on the whitened norm of the image difference. A witnessed
        exceedance warns before fitting and is separate from sampler status.

    Returns
    -------
    PreparedModelFit
        Fit with owned data and cached fixed covariance.

    Raises
    ------
    ValueError
        If the noise input or the render geometry is invalid.
    """
    if not isinstance(data_unit, str) or not data_unit.strip():
        raise ValueError("data_unit must be a nonempty unit label")
    options = {} if noise_options is None else noise_options
    noise = prepare_noise(data, **options)
    shape = noise.valid.shape
    if context is None:
        context = RenderContext(shape)
    elif context.shape != shape:
        raise ValueError(f"render context shape must match the data shape {shape}")
    digests = {"data": _array_digest(data, "<f8"), "valid": _array_digest(noise.valid, "u1")}
    for key in ("err", "kernel", "covariance"):
        if options.get(key) is not None:
            digests[key] = _array_digest(options[key], "<f8")
    data = np.where(noise.valid, np.asarray(data, dtype=float), 0.0)
    data.setflags(write=False)
    provenance = {"data_unit": data_unit, "input_sha256": digests}
    fit = PreparedModelFit(data, model, parameters, context, noise, provenance)
    accuracy = assess_rendering(fit, render_references, render_reference_label, render_epsilon_max)
    if render_references is not None:
        accuracy["reference_sha256"] = {name: _array_digest(image, "<f8")
                                        for name, image in render_references.items()}
    provenance["render_accuracy"] = accuracy
    return fit


def evaluate_model_fit(fit: PreparedModelFit, sample_vector=None, *, decompose: bool = False) -> dict:
    """Evaluate a prepared model, Gaussian likelihood, and sampling-space prior.

    Parameters
    ----------
    fit : PreparedModelFit
        Prepared model and fixed covariance.
    sample_vector : array-like, optional
        One vector in `fit.parameters.free_names` order and sampling
        coordinates. Defaults to the initial vector. All-fixed models
        use an empty vector.
    decompose : bool, optional
        Include component images, data, residuals and a valid-pixel mask.
        Residual and standardized-residual images use NaN at masked pixels.
        The whitened residual is an observed-order vector since dense
        whitening mixes pixels.

    Returns
    -------
    dict
        Physical `values`, rendered `image`, `log_likelihood`, `log_prior`,
        `log_posterior`, and `metadata` describing the noise calculation.

    Raises
    ------
    ValueError
        If the sampling vector is outside support or the render is invalid.
    KeyError
        If the model requests an undefined physical parameter.
    """
    if sample_vector is None:
        sample_vector = fit.parameters.initial
    log_prior = fit.parameters.log_prior_in_sampling_space(sample_vector)
    if not np.isfinite(log_prior):
        raise ValueError("sample vector is outside prior support")
    values = fit.parameters.decode(sample_vector)
    result = _evaluate_image(fit, values, decompose)
    result.update(values=values, log_prior=log_prior,
                  log_posterior=result["log_likelihood"] + log_prior)
    return result


def _evaluate_image(fit, values, decompose):
    if decompose:
        parts = fit.model.render_components(values, fit.context)
        image = _sum_components(parts, fit.context.shape)
    else:
        image = fit.model.render(values, fit.context)
    residual = fit.data - image
    log_likelihood = fit.noise.log_likelihood(residual)
    metadata = {"noise": fit.noise.metadata, "render_accuracy": fit.provenance["render_accuracy"]}
    if fit.context.psf_operator is not None:
        metadata["psf"] = fit.context.psf_operator.metadata
    result = {
        "image": image,
        "log_likelihood": log_likelihood,
        "metadata": metadata,
    }
    if decompose:
        valid = fit.noise.valid
        standardized = np.full(fit.context.shape, np.nan)
        standardized[valid] = fit.noise.standardize(residual)
        result.update(
            components=parts, data=np.where(valid, fit.data, np.nan), valid=valid,
            residual=np.where(valid, residual, np.nan), standardized_residual=standardized,
            whitened_residual=fit.noise.whiten(residual),
        )
    return result
