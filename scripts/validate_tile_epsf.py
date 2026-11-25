"""Compare a leave-one-out local ePSF with the F200W epoch ePSF."""

from argparse import ArgumentParser
import csv
from pathlib import Path

from astropy.io import fits
import numpy as np
from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars
from scipy.optimize import least_squares

from jwst_psfmc import PSFOperator, PSFSpec, RenderContext, SersicProfile
from scripts.validate_empirical_psf import linear_fit, models, write_csv


def training_stamp(data, valid):
    radius = data.shape[0] // 2
    y, x = np.indices(data.shape) - radius
    distances = np.hypot(x, y)
    background = np.median(data[valid & (distances >= radius - 2)])
    signal = np.where(valid, data - background, 0.)
    core = np.where(distances <= 3, np.maximum(signal, 0.), 0.)
    center = (radius + np.sum(x * core) / core.sum(), radius + np.sum(y * core) / core.sum())
    return signal, valid, center


def build_local(training, mas=30):
    stars = EPSFStars([
        EPSFStar(data.copy(), weights=valid.astype(float), cutout_center=center, id_label=source_id)
        for source_id, (data, valid, center) in training.items()
    ])
    model, fitted = EPSFBuilder(oversampling=4, maxiters=8, progress_bar=False)(stars)
    retained = [str(star.id_label) for star in fitted.all_good_stars]
    if set(retained) - set(training) or len(retained) < 8:
        raise ValueError("invalid local ePSF training membership")
    spec = PSFSpec(model.data, "effective", .25, tuple(model.origin),
                   "nominal 0.36 arcsec aperture", source=f"held-out ePSF at {mas} mas")
    operator = PSFOperator(spec, oversamp=8, max_offset=1.3)
    radius = 360 // mas
    reference = operator.render_point(0, 0, RenderContext((2*radius+1,)*2))
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    norm = reference[np.hypot(xx, yy) <= radius].sum()
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid local ePSF aperture flux")
    return operator, norm, retained


def estimator(data, point, error, initial):
    sky = np.ones(data.shape)

    def solve(offset):
        return linear_fit(data, [point(offset), sky], error)

    fit = least_squares(lambda offset: solve(offset)[2], initial,
                        bounds=(-1.2, 1.2), max_nfev=80)
    if not fit.success:
        raise RuntimeError(f"stellar centroid fit failed: {fit.message}")
    return fit.x


def summarize_fit(data, point, host, error):
    sky = np.ones(data.shape)
    (point_flux, background), _, residual = linear_fit(data, [point, sky], error)
    (point_plus_host, host_flux, _), _, with_host = linear_fit(data, [point, host, sky], error)
    total = point_plus_host + host_flux
    return dict(point_flux=point_flux, host_flux=host_flux, total_flux=total,
                signed_host_fraction=host_flux/total if total > 0 else np.nan,
                relative_residual=np.linalg.norm(residual.reshape(data.shape) * error)
                / np.linalg.norm(data-background),
                point_objective=residual @ residual, delta_objective=residual @ residual - with_host @ with_host)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tile", type=Path)
    parser.add_argument("psfs", type=Path)
    parser.add_argument("--results", type=Path, default=Path("docs/validation/empirical"))
    parser.add_argument("--check-obs-002", action="store_true",
                        help="Evaluate the archive OBS 002 template as a separate control")
    args = parser.parse_args()
    with (args.results / "F200W_30mas_tile12_variance.csv").open() as handle:
        catalog = {r["source_id"]: r for r in csv.DictReader(handle)
                   if r["weight"] == "full" and r["centroid"] == "refit"}
    render, epoch_operator, epoch_norm = models(args.psfs, "F200W", "001", 30)
    if args.check_obs_002:
        alternate_render, alternate_operator, alternate_norm = models(args.psfs, "F200W", "002", 30)
    context = RenderContext((13, 13))
    profile = SersicProfile(1, 70/30, 1, .67, 32)
    inner = np.s_[6:19, 6:19]
    training, test = {}, {}
    positions = {}
    with fits.open(args.tile, memmap=True, do_not_scale_image_data=True) as hdus:
        height, width = hdus["SCI"].shape
        for source_id, star in catalog.items():
            x, y = int(star["x"]), int(star["y"])
            if not (12 <= x < width-12 and 12 <= y < height-12):
                raise ValueError(f"star {source_id} too close to array edge")
            sl = np.s_[y-12:y+13, x-12:x+13]
            data, err, wht = [hdus[k].data[sl].astype(float) for k in ("SCI", "ERR", "WHT")]
            valid = np.isfinite(data) & np.isfinite(err) & (err > 0) & (wht > 0)
            if valid.mean() < .90 or not np.all(valid[inner]):
                raise ValueError(f"insufficient valid pixels for star {source_id}")
            training[source_id] = training_stamp(data, valid)
            test[source_id] = data[inner], err[inner]
            positions[source_id] = np.array([x, y])
    rows = []
    for source_id, (data, err) in test.items():
        def point_epoch(offset):
            return render("epsf", *offset, context)

        if args.check_obs_002:
            def point_002(offset):
                return alternate_render("epsf", *offset, context)

            templates = [("archive_002", point_002, alternate_operator, alternate_norm)]
            retained = []
        else:
            # Reject overlapping stamps as well as the target ID.
            train = {key: value for key, value in training.items()
                     if np.max(np.abs(positions[key] - positions[source_id])) >= 25}
            local, local_norm, retained = build_local(train)

            def point_local(offset):
                return local.render_point(*offset, context) / local_norm

            templates = [("epoch", point_epoch, epoch_operator, epoch_norm),
                         ("local_loo", point_local, local, local_norm)]
        initial = np.clip(np.array(training[source_id][2])-12, -1., 1.)
        common = estimator(data, point_epoch, np.ones_like(err), initial)
        for weight, error in (("full", err), ("uniform", np.ones_like(err))):
            for label, point, operator, norm in templates:
                offset = estimator(data, point, error, initial)
                for centroid, center in (("refit", offset), ("epoch_uniform", common)):
                    host = operator.render_sersic(profile, *center, context) / norm
                    result = summarize_fit(data, point(center), host, error)
                    rows.append(dict(source_id=source_id, catalog_tile=catalog[source_id]["catalog_tile"],
                                     renderer=label, weight=weight, centroid=centroid,
                                     dx=center[0], dy=center[1], n_training=len(retained),
                                     training_ids="|".join(retained), **result))
        print(source_id, len(retained), flush=True)
    suffix = "archive_002" if args.check_obs_002 else "loo"
    write_csv(args.results / f"F200W_30mas_tile12_epsf_{suffix}.csv", rows)


if __name__ == "__main__":
    main()
