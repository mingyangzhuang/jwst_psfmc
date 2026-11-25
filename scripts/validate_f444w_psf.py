"""Compare F444W PSF residuals and compact stellar nulls on common sky positions."""

from argparse import ArgumentParser
from contextlib import ExitStack
import csv
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from jwst_psfmc import RenderContext, SersicProfile
from scripts.validate_empirical_psf import linear_fit, models, noise_factor, open_mosaic, write_csv
from scripts.validate_tile_epsf import build_local, training_stamp


def catalog(path, mas):
    suffix = "_60mas" if mas == 60 else ""
    with fits.open(path / "F444W" / f"Stars_OBS_002{suffix}.fits", memmap=True) as hdus:
        return {str(row["SOURCE_ID"]): (row["skycoord.ra"], row["skycoord.dec"], row["tilename"].strip())
                for row in hdus[1].data if row["include"]}


def stamp(mosaic, ra, dec, half):
    x, y = mosaic.wcs.world_to_pixel_values(ra, dec)
    ix, iy = np.rint([x, y]).astype(int)
    height, width = mosaic.sci.shape
    if not (half <= ix < width-half and half <= iy < height-half):
        return None
    data, err, coverage = mosaic.region(ix, iy, half)
    valid = np.isfinite(data) & np.isfinite(err) & (err > 0) & (coverage > 0)
    return data, err, coverage, valid, np.array([ix, iy]), np.clip([x-ix, y-iy], -.6, .6)


def diagnostics(data, err, factor, point, operator, norm, mas, initial,
                *, center=None, sizes=(40, 70, 140)):
    sky = np.ones(data.shape)

    def point_fit(offset):
        return linear_fit(data, [point(offset), sky], err, factor)

    if center is None:
        fit = least_squares(lambda offset: point_fit(offset)[2], initial, bounds=(-1.2, 1.2),
                            max_nfev=80)
        if not fit.success:
            return ()
        center = fit.x
    dx, dy = center
    pt = point(center)
    (flux, bkg), _, residual = point_fit(center)
    image_residual = data-flux*pt-bkg
    y, x = np.indices(data.shape) - data.shape[0]//2
    distance = np.hypot((x-dx)*mas, (y-dy)*mas).ravel()
    partitions = (distance <= 90, (distance > 90) & (distance <= 180), distance > 180)
    if factor is not None:
        backproject = solve_triangular(factor.T, residual, lower=False, check_finite=False)
    else:
        backproject = residual
    output = []
    for re in sizes:
        host = operator.render_sersic(SersicProfile(1, re/mas, 1, .67, 32), dx, dy,
                                     RenderContext(data.shape))/norm
        fit_host, error, _ = linear_fit(data, [pt, host, sky], err, factor)
        nuisance, _, host_whitened = linear_fit(host, [pt, sky], err, factor)
        host_projected = host - nuisance[0]*pt - nuisance[1]
        contribution = (host_projected/err).ravel()*backproject*error[1]**2
        np.testing.assert_allclose(contribution.sum(), fit_host[1], rtol=1e-7, atol=1e-7)
        total = fit_host[0]+fit_host[1]
        output.append(dict(dx=dx, dy=dy, near_bound=max(abs(dx),abs(dy))>1.1, re_mas=re,
                           point_flux=flux, host_flux=fit_host[1], total_flux=total,
                           signed_fraction=fit_host[1]/total if total>0 else np.nan,
                           residual=np.linalg.norm(image_residual)/np.linalg.norm(data-bkg),
                           objective=residual@residual,
                           host_core=contribution[partitions[0]].sum(),
                           host_inner_wing=contribution[partitions[1]].sum(),
                           host_outer_wing=contribution[partitions[2]].sum(),
                           host_identifiability=np.linalg.norm(host_whitened)))
    return output


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("blank_dir", type=Path)
    parser.add_argument("psfs", type=Path)
    parser.add_argument("--results", type=Path, default=Path("docs/validation/empirical"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--loo", action="store_true", help="run the paired held-out ePSF comparison")
    mode.add_argument("--centroid-control", action="store_true",
                      help="swap the dense and diagonal fit centroids at fixed host size")
    args = parser.parse_args()
    with (args.results/"manifest.json").open() as handle:
        manifest = json.load(handle)["products"]
    catalogs = {mas: catalog(args.psfs, mas) for mas in (30, 60)}
    common = sorted(catalogs[30].keys() & catalogs[60].keys())
    controls = {}
    if args.centroid_control:
        with (args.results/"F444W_paired_radial.csv").open() as handle:
            for row in csv.DictReader(handle):
                if (row["half_width_mas"],row["re_mas"]) == ("240","70"):
                    key = row["source_id"], int(row["mas"]), row["weight"]
                    controls[key] = (float(row["dx"]),float(row["dy"]))
        common = [sid for sid in common if (sid,30,"dense") in controls
                  and (sid,60,"dense") in controls]
    radius_list = (240,) if args.loo or args.centroid_control else (180, 240, 360)
    stamps, training = {}, {}
    with ExitStack() as stack:
        for mas in (30, 60):
            mosaic = open_mosaic(stack, args.blank_dir, "F444W", mas)
            for source_id in common:
                ra, dec, _ = catalogs[mas][source_id]
                if source_id not in stamps:
                    stamps[source_id], training[source_id] = {}, {}
                full_stamp = stamp(mosaic, ra, dec, 480//mas)
                if full_stamp is None:
                    continue
                data, _, _, valid, position, _ = full_stamp
                if valid.mean() >= .9:
                    training[source_id][mas] = training_stamp(data, valid), position
                radii = {}
                for r in radius_list:
                    region = stamp(mosaic, ra, dec, r//mas)
                    if region is not None and np.all(region[3]):
                        radii[r] = region
                stamps[source_id][mas] = radii
    paired = [sid for sid in common if all(set(stamps[sid].get(m, ())) == set(radius_list)
                                          for m in (30, 60))]
    output = []
    for mas in (30, 60):
        render, op, norm = models(args.psfs, "F444W", "002", mas)
        kernels = manifest[f"F444W_{mas}mas"]["kernels"]
        factors = {}
        for source_id in paired:
            for radius in radius_list:
                data, err, coverage, _, position, initial = stamps[source_id][mas][radius]
                context = RenderContext(data.shape)
                depth = str(int(coverage.min())) if coverage.min()==coverage.max() else "edge"
                if depth not in kernels:
                    raise ValueError(f"unsupported coverage {depth}")
                key = depth, data.shape
                if key not in factors:
                    factors[key] = noise_factor(np.asarray(kernels[depth]["kernel"]), data.shape)[0]
                if args.loo:
                    train = {sid: triple[mas][0] for sid, triple in training.items()
                             if mas in triple and
                             np.max(np.abs(triple[mas][1]-position)) > (480+radius)//mas}
                    local, local_norm, retained = build_local(train, mas)
                    if source_id in retained:
                        raise ValueError("test star retained in a training fold")
                    templates = (("archive", op, norm, ()), ("mosaic_loo", local, local_norm, retained))
                    weights = (("dense",err,factors[key]),)
                elif args.centroid_control:
                    templates = (("archive",op,norm,()),)
                    weights = (("dense_at_diagonal",err,factors[key]),("diagonal_at_dense",err,None))
                else:
                    templates = (("archive",op,norm,()),)
                    weights = (("dense",err,factors[key]),("diagonal",err,None),
                               ("uniform",np.ones_like(err),None))
                for label, operator, normalization, ids in templates:
                    def point(offset):
                        return (render("epsf", *offset, context) if label=="archive"
                                else operator.render_point(*offset, context)/normalization)

                    for weight, error, factor in weights:
                        options = {}
                        if args.centroid_control:
                            reference = weight.split("_at_")[1]
                            options = dict(center=controls[source_id,mas,reference],sizes=(70,))
                        output.extend(dict(source_id=source_id, catalog_tile=catalogs[mas][source_id][2],
                                           mas=mas, half_width_mas=radius, depth=depth, weight=weight,
                                           psf=label, training_ids="|".join(ids), **row)
                                      for row in diagnostics(data,error,factor,point,operator,
                                                             normalization,mas,initial,**options))
            print(mas,source_id,len(output),flush=True)
    suffix = "loo" if args.loo else "centroid" if args.centroid_control else "radial"
    write_csv(args.results/f"F444W_paired_{suffix}.csv",output)


if __name__ == "__main__":
    main()
