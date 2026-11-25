"""Round-trip a masked, correlated model and its cropped WCS."""

import matplotlib
matplotlib.use("Agg")

from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet,
    evaluate_model_fit, load_model_snapshot, plot_decomposition, plot_model_fit,
    plot_residual_diagnostics, prepare_model_fitting, save_model_snapshot,
)


def test_snapshot_round_trip_and_diagnostic_panels(tmp_path):
    shape = (5, 7)
    noise = np.eye(np.prod(shape))
    noise[0, 2] = noise[2, 0] = .1
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    data[0, 1] = np.nan
    fit = prepare_model_fitting(
        data, CompositeModel([ConstantBackground("sky")]),
        ParameterSet([Parameter("sky.level", fixed=2.)]),
        noise_options={"backend": "dense", "covariance": noise}, data_unit="uJy/pixel",
    )
    parent = WCS(naxis=2)
    parent.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    parent.wcs.crval = [268., 65.]
    parent.wcs.crpix = [20., 21.]
    parent.wcs.cdelt = [-.03/3600, .03/3600]
    stamp = parent[10:15, 9:16]
    path = tmp_path/"fit.fits"
    with pytest.raises(ValueError, match="nonempty"):
        save_model_snapshot(path, fit, label="")
    save_model_snapshot(path, fit, label="Saved draw", wcs=stamp)
    snapshot = load_model_snapshot(path)
    expected = evaluate_model_fit(fit, decompose=True)
    for name in ("data", "image", "residual", "standardized_residual", "whitened_residual", "valid"):
        np.testing.assert_array_equal(snapshot[name], expected[name])
    np.testing.assert_array_equal(snapshot["components"]["sky"], expected["components"]["sky"])
    np.testing.assert_allclose(
        WCS(snapshot["wcs_header"]).pixel_to_world_values(3, 2),
        parent.pixel_to_world_values(12, 12), atol=1e-12, rtol=0,
    )
    for function, panels in ((plot_model_fit, 3), (plot_decomposition, 1),
                             (plot_residual_diagnostics, 2)):
        figure, axes = function(snapshot)
        assert len(axes) == panels
        plt.close(figure)
