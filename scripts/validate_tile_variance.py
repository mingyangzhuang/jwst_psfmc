"""Compare fixed-profile stellar null fits under Tile 12 variance ablations."""

from argparse import ArgumentParser
import csv
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from scripts.validate_empirical_psf import RenderContext, SersicProfile, linear_fit, models, write_csv


def compare_star(data, errors, initial, render, operator, normalization):
    context = RenderContext(data.shape)
    sky = np.ones(data.shape)
    profile = SersicProfile(1, 70 / 30, 1, .67, 32)
    offsets = {}
    rows = []
    for name, error in errors.items():
        def residual(offset):
            point = render("epsf", *offset, context)
            return linear_fit(data, [point, sky], error)[2]

        fit = least_squares(residual, initial, bounds=(-1.2, 1.2), max_nfev=80)
        if not fit.success:
            raise RuntimeError(f"{name} centroid optimization failed: {fit.message}")
        offsets[name] = fit.x
    for name, error in errors.items():
        for centroid, offset in (("refit", offsets[name]), ("uniform", offsets["uniform"])):
            point = render("epsf", *offset, context)
            host = operator.render_sersic(profile, *offset, context) / normalization
            (point_flux, background), _, residual = linear_fit(data, [point, sky], error)
            (point_with_host, host_flux, _), _, _ = linear_fit(data, [point, host, sky], error)
            rows.append(dict(weight=name, centroid=centroid, dx=offset[0], dy=offset[1],
                             point_flux=point_flux, host_flux=host_flux,
                             signed_host_fraction=host_flux / (point_with_host + host_flux)
                             if point_with_host + host_flux > 0 else np.nan,
                             relative_residual=np.linalg.norm(residual.reshape(data.shape) * error)
                             / np.linalg.norm(data - background),
                             near_bound=np.max(np.abs(offset)) > 1.1))
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tile", type=Path)
    parser.add_argument("central_error", type=Path)
    parser.add_argument("psfs", type=Path)
    parser.add_argument("--results", type=Path, default=Path("docs/validation/empirical"))
    args = parser.parse_args()
    render, operator, norm = models(args.psfs, "F200W", "001", 30)
    reference_wcs = WCS(fits.getheader(args.central_error)).celestial
    with (args.results / "F200W_30mas_stellar_null.csv").open() as handle:
        stars = [row for row in csv.DictReader(handle) if row["renderer"] == "epsf"]
    radius = 6
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    ring = np.hypot(xx, yy) >= 5
    results = []
    with fits.open(args.tile, memmap=True, do_not_scale_image_data=True) as hdus:
        tile_wcs = WCS(hdus["SCI"].header).celestial
        height, width = hdus["SCI"].shape
        for star in stars:
            sky = reference_wcs.pixel_to_world_values(int(star["x"]), int(star["y"]))
            x, y = np.rint(tile_wcs.world_to_pixel_values(*sky)).astype(int)
            if not (radius <= x < width-radius and radius <= y < height-radius):
                continue
            stamp = np.s_[y-radius:y+radius+1, x-radius:x+radius+1]
            data, err, wht, poisson, read, flat = (
                hdus[name].data[stamp].astype(float)
                for name in ("SCI", "ERR", "WHT", "VAR_POISSON", "VAR_RNOISE", "VAR_FLAT")
            )
            if not (np.all(np.isfinite([data, err, wht, poisson, read, flat]))
                    and np.all((err > 0) & (wht > 0) & (poisson + read > 0))):
                continue
            np.testing.assert_allclose(err**2, poisson + read + flat, rtol=5e-7, atol=0)
            yy_peak, xx_peak = np.unravel_index(
                np.argmax(np.where(np.hypot(xx, yy) <= 2, data, -np.inf)), data.shape
            )
            initial = np.clip([xx_peak-radius, yy_peak-radius], -1., 1.)
            errors = dict(full=err, no_flat=np.sqrt(poisson + read), uniform=np.ones_like(err))
            rows = compare_star(data, errors, initial, render, operator, norm)
            brightness = next(row["point_flux"] for row in rows if row["weight"]=="uniform")
            results.extend(dict(source_id=star["source_id"], catalog_tile=star["tile"], x=x, y=y,
                                brightness=brightness, peak=data.max()-np.median(data[ring]), **row)
                           for row in rows)
    write_csv(args.results / "F200W_30mas_tile12_variance.csv", results)
    for centroid in ("refit", "uniform"):
        for weight in ("full", "no_flat", "uniform"):
            rows = [r for r in results if r["centroid"] == centroid and r["weight"] == weight]
            fraction = np.array([r["signed_host_fraction"] for r in rows])
            brightness = [r["brightness"] for r in rows]
            print(centroid, weight, "N", len(rows), "host quartiles", np.quantile(fraction,[.25,.5,.75]),
                  "brightness rho", spearmanr(brightness,fraction).statistic,
                  "residual", np.median([r["relative_residual"] for r in rows]),
                  "near bound", sum(r["near_bound"] for r in rows))


if __name__ == "__main__":
    main()
