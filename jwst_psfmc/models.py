"""Additive image components with named physical parameters."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .psf import psf_model
from .rendering import RenderContext
from .sersic import SersicProfile


def _point_parameters(name, values):
    return values[f"{name}.flux"], values[f"{name}.dx"], values[f"{name}.dy"]


def _sum_components(parts, shape):
    image = np.zeros(shape, dtype=float)
    for component in parts.values():
        image += component
    if not np.all(np.isfinite(image)):
        raise ValueError("sum of component images is nonfinite")
    return image


@dataclass(frozen=True)
class PointSource:
    """Point source evaluated using the response in a render context.

    Parameters
    ----------
    name : str
        Component namespace for `<name>.flux`, `<name>.dx`, and `<name>.dy`.
        Offsets are relative to the stamp center in native pixels. Flux
        follows the response's declared unit-flux convention.
    """

    name: str

    def render(self, values: Mapping[str, float], context: RenderContext) -> np.ndarray:
        """Render a point source.

        Parameters
        ----------
        values : mapping
            Physical flux and offsets in the component namespace.
        context : RenderContext
            Native output geometry and unit point response.

        Returns
        -------
        ndarray
            Point source image in flux per output pixel.

        Raises
        ------
        ValueError
            If the context has no point response or it returns an invalid image.
        KeyError
            If a required physical parameter is missing.
        """
        point_response = context.point_response
        if context.psf_operator is not None:
            point_response = context.psf_operator.render_point
        if point_response is None:
            raise ValueError("PointSource requires a point_response in RenderContext")
        flux, dx, dy = _point_parameters(self.name, values)
        response = np.asarray(point_response(dx, dy, context))
        if response.shape != context.shape or response.dtype.kind not in "iuf":
            raise ValueError(f"point_response must return a real image of shape {context.shape}")
        return flux * response.astype(float, copy=False)


@dataclass(frozen=True)
class SersicSource:
    """PSF-convolved Sérsic component with named physical parameters.

    Parameters
    ----------
    name : str
        Namespace for flux, dx, dy, re, n, q and pa. Offsets are relative
        to the stamp center. re is in native pixels and pa is in degrees.
    """

    name: str

    def render(self, values: Mapping[str, float], context: RenderContext) -> np.ndarray:
        """Render a Sérsic source with the context's shared PSF response.

        Parameters
        ----------
        values : mapping
            Physical parameters in the component namespace.
        context : RenderContext
            Native stamp geometry with a PSFOperator.

        Returns
        -------
        ndarray
            Convolved flux per native pixel.

        Raises
        ------
        ValueError
            If the response is absent or the source values are invalid.
        KeyError
            If a required physical parameter is missing.
        RuntimeError
            If intrinsic quadrature fails to converge.
        """
        if context.psf_operator is None:
            raise ValueError("SersicSource requires a psf_operator in RenderContext")
        flux, dx, dy = _point_parameters(self.name, values)
        re, n, q, pa = (values[f"{self.name}.{key}"] for key in ("re", "n", "q", "pa"))
        return context.psf_operator.render_sersic(SersicProfile(flux, re, n, q, pa), dx, dy, context)


@dataclass(frozen=True)
class ConstantBackground:
    """Constant additive background.

    Parameters
    ----------
    name : str
        Component namespace for `<name>.level` in flux per output pixel.
    """

    name: str

    def render(self, values: Mapping[str, float], context: RenderContext) -> np.ndarray:
        """Render a constant background.

        Parameters
        ----------
        values : mapping
            Physical parameter values including the background level.
        context : RenderContext
            Native output geometry.

        Returns
        -------
        ndarray
            Constant image in flux per output pixel.

        Raises
        ------
        KeyError
            If the background level is missing.
        """
        return np.full(context.shape, values[f"{self.name}.level"], dtype=float)


@dataclass(frozen=True)
class _LegacyPointSourceAdapter:
    """Render a prepared legacy PSF using its exact shift and flux convention."""

    name: str
    psf_os: np.ndarray
    oversamp: int
    native_shape: tuple[int, int] | None = None
    psf_prepare_fraction: float = 1.0

    def render(self, values, context):
        return psf_model(
            _point_parameters(self.name, values), self.psf_os, self.oversamp,
            output_shape=context.shape, native_shape=self.native_shape,
            psf_prepare_fraction=self.psf_prepare_fraction,
        )


_COMPONENT_FIELDS = {
    PointSource: ("flux", "dx", "dy"),
    SersicSource: ("flux", "dx", "dy", "re", "n", "q", "pa"),
    ConstantBackground: ("level",),
    _LegacyPointSourceAdapter: ("flux", "dx", "dy"),
}


@dataclass(frozen=True, init=False)
class CompositeModel:
    """Ordered sum of named image components.

    Parameters
    ----------
    components : iterable
        Each component supplies a unique `name` and a
        `render(values, context)` method. At most one may be a
        `ConstantBackground`.

    Raises
    ------
    ValueError
        If components are missing, invalid, have duplicate names, or include
        multiple constant backgrounds.
    """

    components: tuple

    def __init__(self, components: Iterable):
        components = tuple(components)
        if not components:
            raise ValueError("CompositeModel requires at least one component")
        names = set()
        for component in components:
            name = getattr(component, "name", None)
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ValueError("component name must be a nonempty, trimmed string")
            if name in names:
                raise ValueError(f"duplicate component name: {name}")
            if not callable(getattr(component, "render", None)):
                raise ValueError(f"component {name!r} must implement render")
            names.add(name)
        if sum(isinstance(component, ConstantBackground) for component in components) > 1:
            raise ValueError("CompositeModel supports at most one constant background")
        object.__setattr__(self, "components", components)

    def render_components(self, values: Mapping[str, float], context: RenderContext) -> dict[str, np.ndarray]:
        """Render the component images individually.

        Parameters
        ----------
        values : mapping
            Physical parameter values keyed by name.
        context : RenderContext
            Native output geometry and any required point response.

        Returns
        -------
        dict of str to ndarray
            Images in component order, all float64 and context-shaped.

        Raises
        ------
        ValueError
            If a component image has the wrong shape or invalid values.
        KeyError
            If a component parameter is missing.
        """
        parts = {}
        for component in self.components:
            image = np.asarray(component.render(values, context))
            if image.shape != context.shape or image.dtype.kind not in "iuf":
                raise ValueError(f"component {component.name!r} must render a real image of shape {context.shape}")
            image = image.astype(float, copy=False)
            if not np.all(np.isfinite(image)):
                raise ValueError(f"component {component.name!r} rendered nonfinite values")
            parts[component.name] = image
        return parts

    def render(self, values: Mapping[str, float], context: RenderContext) -> np.ndarray:
        """Render the sum of the components.

        Parameters
        ----------
        values : mapping
            Physical parameter values keyed by name.
        context : RenderContext
            Native output geometry and any required point response.

        Returns
        -------
        ndarray
            Float64 image with the native output shape.

        Raises
        ------
        ValueError
            If a component image is invalid or their sum is nonfinite.
        KeyError
            If a component parameter is missing.
        """
        return _sum_components(self.render_components(values, context), context.shape)
