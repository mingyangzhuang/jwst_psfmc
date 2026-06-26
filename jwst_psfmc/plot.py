"""
Visualisation utilities for PSF photometry diagnostics.

All plotting functions return ``(fig, axes)`` without saving or closing the
figure, letting the caller decide on output format and file path.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee

from .psf import psf_model


def plot_psf_fit_triptych(
    data: np.ndarray,
    error: np.ndarray,
    psf_os: np.ndarray,
    oversamp: int,
    native_shape: tuple[int, int] | None = None,
    psf_prepare_fraction: float = 1.0,
    param: np.ndarray | None = None,
    summary_median: np.ndarray | None = None,
    summary: dict | None = None,
    cmap: str = "viridis",
    resid_cmap: str = "coolwarm",
    fig_title: str = "",
    return_meta: bool = False,
) -> tuple:
    """Plot data / PSF model / normalised residual side by side.

    Exactly one of *param*, *summary_median*, or *summary* must be provided.

    Parameters
    ----------
    data : ndarray, 2-D
        Observed difference-image stamp.
    error : ndarray, 2-D
        Per-pixel 1-σ uncertainty, same shape as *data*.
    psf_os : ndarray
        Pre-cropped oversampled PSF.
    oversamp : int
        Integer oversampling factor.
    native_shape : (ny, nx) or None, optional
        Native PSF shape; forwarded to :func:`~jwst_psfmc.psf.psf_model`.
    psf_prepare_fraction : float, optional
        Encircled-energy fraction; forwarded to
        :func:`~jwst_psfmc.psf.psf_model`.
    param : array-like of length 4 or None, optional
        Best-fit ``[flux, dx, dy, bkg]`` to use directly.
    summary_median : array-like of length 4 or None, optional
        Posterior medians in the same order, used when *param* is not given.
    summary : dict or None, optional
        Output of :func:`~jwst_psfmc.mcmc.summarize_emcee`; used when neither
        *param* nor *summary_median* is given.
    cmap : str, optional
        Colormap for the data and model panels. Default ``'viridis'``.
    resid_cmap : str, optional
        Colormap for the residual panel. Default ``'coolwarm'``.
    fig_title : str, optional
        Overall figure suptitle. Default ``''``.
    return_meta : bool, optional
        If *True*, also return a metadata dictionary.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes, shape (3,)
    meta : dict
        Only returned when *return_meta* is *True*. Keys:
        ``'param'``, ``'model'``, ``'residual'``, ``'model_meta'``.

    Raises
    ------
    ValueError
        If none of *param*, *summary_median*, *summary* are provided,
        or if *data* contains no finite pixels.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")

    # --- resolve best-fit parameter vector ---
    if param is not None:
        best_param = np.asarray(param, dtype=np.float64)
    elif summary_median is not None:
        best_param = np.asarray(summary_median, dtype=np.float64)
    elif summary is not None:
        try:
            best_param = np.array(
                [
                    summary["flux"]["median"],
                    summary["dx"]["median"],
                    summary["dy"]["median"],
                    summary["bkg"]["median"],
                ],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise ValueError(
                f"summary dict must contain keys flux, dx, dy, bkg; missing: {exc}"
            ) from exc
    else:
        raise ValueError("Provide one of param, summary_median, or summary")

    if best_param.shape != (4,):
        raise ValueError(f"best-fit parameter must have shape (4,), got {best_param.shape}")

    bkg_val = float(best_param[3])
    model, model_meta = psf_model(
        best_param[:3],
        psf_os,
        oversamp=oversamp,
        output_shape=data.shape,
        native_shape=native_shape,
        psf_prepare_fraction=psf_prepare_fraction,
        return_meta=True,
    )
    model = model + bkg_val
    residual = (data - model) / error

    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        raise ValueError("data contains no finite pixels")

    vmin = float(np.nanpercentile(finite_data, 5))
    vmax = float(np.nanpercentile(finite_data, 99))
    rmax = float(np.nanpercentile(np.abs(residual[np.isfinite(residual)]), 99))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = [
        (data, "Data", cmap, vmin, vmax),
        (model, "Model", cmap, vmin, vmax),
        (residual, "Residual (σ)", resid_cmap, -rmax, rmax),
    ]
    for ax, (img, title, cm, v0, v1) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", cmap=cm, vmin=v0, vmax=v1)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[1].text(
        0.5, 0.04,
        (
            f"flux = {best_param[0]:.3f},  bkg = {best_param[3]:.3f}\n"
            f"dx = {best_param[1]:.3f},  dy = {best_param[2]:.3f}"
        ),
        ha="center", va="bottom",
        transform=axes[1].transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    if fig_title:
        fig.suptitle(fig_title, y=1.02, fontsize=15)

    if return_meta:
        return fig, axes, {
            "param": best_param,
            "model": model,
            "residual": residual,
            "model_meta": model_meta,
        }
    return fig, axes


def plot_chains(
    sampler: emcee.EnsembleSampler,
    labels: tuple[str, ...] = ("flux", "dx", "dy", "bkg"),
    burnin: int = 500,
    title: str = "",
) -> tuple:
    """Plot walker traces for each parameter.

    A vertical dashed red line marks the burn-in boundary.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Completed sampler.
    labels : tuple of str, optional
        Parameter names. Default ``('flux', 'dx', 'dy', 'bkg')``.
    burnin : int, optional
        Step index at which to draw the burn-in boundary. Default ``500``.
    title : str, optional
        Figure title. Default ``''``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes, shape (ndim,)
    """
    samples = sampler.get_chain()
    ndim = samples.shape[-1]

    fig, axes = plt.subplots(ndim, figsize=(10, 2.5 * ndim), sharex=True)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        ax.plot(samples[:, :, i], color="k", alpha=0.25, linewidth=0.5)
        ax.axvline(burnin, color="r", linestyle="--", linewidth=1.2)
        ax.set_ylabel(labels[i] if i < len(labels) else f"param {i}")
        ax.set_xlim(0, samples.shape[0])

    axes[-1].set_xlabel("Step number")
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_corner(
    sampler: emcee.EnsembleSampler,
    labels: tuple[str, ...] = ("flux", "dx", "dy", "bkg"),
    burnin: int = 500,
    thin: int = 4,
    title: str = "",
    **corner_kwargs,
) -> tuple:
    """Plot a ``corner.py`` corner plot of the posterior samples.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Completed sampler.
    labels : tuple of str, optional
        Parameter names. Default ``('flux', 'dx', 'dy', 'bkg')``.
    burnin : int, optional
        Burn-in steps to discard. Default ``500``.
    thin : int, optional
        Thinning factor. Default ``4``.
    title : str, optional
        Figure title displayed via ``fig.suptitle``. Default ``''``.
    **corner_kwargs
        Additional keyword arguments forwarded to :func:`corner.corner`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
    """
    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    corner_kwargs.setdefault("show_titles", True)
    corner_kwargs.setdefault("title_fmt", ".3f")
    corner_kwargs.setdefault("quantiles", [0.16, 0.5, 0.84])

    fig = corner.corner(flat, labels=list(labels), **corner_kwargs)
    if title:
        fig.suptitle(title, y=1.01)
    axes = np.array(fig.axes)
    return fig, axes
