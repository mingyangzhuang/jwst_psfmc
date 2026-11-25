"""Save a rendered fit with its diagnostics in a portable FITS file."""

import json

from astropy.io import fits
import numpy as np

from .fitting import evaluate_model_fit

_IMAGES = {
    "data": "DATA", "image": "MODEL", "residual": "RESIDUAL",
    "standardized_residual": "STDRES", "valid": "VALID",
    "whitened_residual": "WHITENED",
}


def save_model_snapshot(path, fit, sample_vector=None, *, label, wcs=None, overwrite=False):
    """Save one evaluated model, components and residuals as a FITS snapshot.

    Save posterior samples separately with save_model_results. Label the
    evaluated vector with its meaning, such as "maximum likelihood" or
    "posterior draw". If supplied, the two-dimensional WCS must already
    describe the fitted stamp, including the offset from the parent image.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError("a model snapshot needs a nonempty label")
    result = evaluate_model_fit(fit, sample_vector, decompose=True)
    if wcs is not None and (wcs.pixel_n_dim != 2 or
                            wcs.array_shape not in (None, fit.context.shape)):
        raise ValueError("snapshot WCS must describe the 2-D fit stamp")
    if wcs is not None and any(getattr(wcs, name) is not None
                               for name in ("cpdis1", "cpdis2", "det2im1", "det2im2")):
        raise ValueError("snapshot WCS does not support external distortion tables")
    primary = fits.PrimaryHDU()
    primary.header["MODSCHEM"] = 1
    unit = fit.provenance["data_unit"]
    metadata = {
        "label": label, "data_unit": unit, "provenance": fit.provenance,
        "components": list(result["components"]),
        **{key: result[key] for key in ("values", "log_likelihood", "log_prior",
                                       "log_posterior", "metadata")},
    }
    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
    hdus = [primary, fits.ImageHDU(np.frombuffer(encoded, dtype=np.uint8), name="METADATA")]
    header = wcs.to_header(relax=True) if wcs is not None else fits.Header()
    images = [(name, result[key], key in ("data", "image", "residual"))
              for key, name in _IMAGES.items()]
    images.extend((f"COMP{index}", data, True)
                  for index, data in enumerate(result["components"].values()))
    for name, data, has_unit in images:
        image = fits.ImageHDU(data.astype(np.uint8) if name == "VALID" else data,
                              header=header if data.ndim == 2 else None, name=name)
        if has_unit and unit.isascii():
            image.header["BUNIT"] = unit
        hdus.append(image)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite, checksum=True)


def load_model_snapshot(path):
    """Read a FITS snapshot into owned arrays for plotting or inspection."""
    with fits.open(path, memmap=False, checksum=True) as hdus:
        if hdus[0].header.get("MODSCHEM") != 1:
            raise ValueError("unsupported model snapshot schema")
        try:
            result = json.loads(hdus["METADATA"].data.tobytes().decode("utf-8"))
            result.update({key: hdus[name].data.copy() for key, name in _IMAGES.items()})
            names = result["components"]
            if (not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names)
                    or not isinstance(result["label"], str) or not isinstance(result["data_unit"], str)):
                raise ValueError("malformed model snapshot metadata")
            result["components"] = {name: hdus[f"COMP{index}"].data.copy()
                                    for index, name in enumerate(names)}
            result["wcs_header"] = hdus["DATA"].header.copy()
        except (AttributeError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed model snapshot") from exc
    result["valid"] = result["valid"].astype(bool)
    shape = result["data"].shape
    if (len(shape) != 2 or any(result[key].shape != shape for key in _IMAGES
                               if key != "whitened_residual")
            or result["whitened_residual"].shape != (np.count_nonzero(result["valid"]),)
            or not names or len(names) != len(result["components"])
            or any(part.shape != shape for part in result["components"].values())):
        raise ValueError("inconsistent model snapshot image shapes")
    return result
