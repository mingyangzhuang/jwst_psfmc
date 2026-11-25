"""Fit the object 88 F200W direct epochs with a shared static Sérsic host."""

from argparse import ArgumentParser
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import re

from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from jwst_psfmc import (
    CompositeModel, ConstantBackground, JointObservation, Parameter, ParameterSet,
    PointSource, RenderContext, SersicProfile, SersicSource,
    evaluate_joint_model_fit, prepare_joint_model_fitting,
)
from scripts.validate_empirical_psf import linear_fit, measure_kernel, noise_factor
from scripts.validate_m10_epochs import EPOCHS, RA_DEC, epoch_inputs, positions
from scripts.validate_m10_target import adopted_alignment, audit_exposures, window


GROUPS = dict(zip(("wide_ep01", *EPOCHS), (
    (5105, ("001", "014", "015")),
    *((5105, (f"{i:03}",)) for i in range(2, 13, 2)),
    (9263, ("002",)), (9263, ("004",)), (9263, ("006",)),
)))
EXPOSURE = re.compile(r"^(jw(\d{5})(\d{3})\d{3}_\d{5}_\d{5}_nrc[a-b](?:[1-4]|long))")
GEOMETRY = ("dx", "dy", "re", "q", "pa", "n")
# Fixed 1-based reference-grid coordinates from the existing F200W host
# notebook's knot sensitivity fit. The amplitudes are re-fit on direct images.
CLUMPS = {"c0": (205.22, 197.58), "c1": (206.83, 202.42), "site": (200.94, 200.97)}


def audit(coverage, coverage_9263, products, tile):
    tables = [Table.read(path) for path in (coverage, coverage_9263) if path]
    groups = {name: set() for name in GROUPS}
    partial = audit_exposures(products)
    for table in tables:
        for row in table[table["band"] == "F200W"]:
            match = EXPOSURE.match(row["filename"])
            if not match or (int(match[2]), match[3]) != (row["propid"], row["NOBS"]):
                raise ValueError(f"inconsistent exposure identity: {row['filename']}")
            for name, (program, observations) in GROUPS.items():
                if int(match[2]) == program and match[3] in observations:
                    groups[name].add(match[1])
    ep01_tile = None
    if tile:
        header_rows = {EXPOSURE.match(filename)[1] for filename in fits.getdata(tile, "HDRTAB")["FILENAME"]}
        if not header_rows or not header_rows <= groups["ep01"]:
            raise ValueError("ep01 tile HDRTAB must be covered by the ep01 footprint list")
        ep01_tile = {"path": str(tile), "unique_detector_records": len(header_rows),
                     "matches_full_ep01_set": header_rows == groups["ep01"]}
    records = {}
    evidence = {}
    for name, (program, observations) in GROUPS.items():
        known = set()
        for filename in partial.get(name, ()):
            match = EXPOSURE.match(filename)
            if not match or int(match[2]) != program or match[3] not in observations:
                raise ValueError(f"unexpected local exposure for {name}: {filename}")
            known.add(match[1])
        covered = {EXPOSURE.match(filename)[3] for filename in groups[name]}
        complete = covered == set(observations)
        if complete and not known <= groups[name]:
            raise ValueError(f"footprint list disagrees with local exposure cutouts for {name}")
        evidence[name] = groups[name] | known
        records[name] = dict(program=program, observations=observations, coverage_complete=complete,
                             coverage_detector_ids=sorted(groups[name]), local_detector_ids=sorted(known))
    for first, second in combinations(GROUPS, 2):
        if evidence[first] & evidence[second]:
            raise ValueError(f"shared exposure between {first} and {second}")
    return dict(groups=records, footprint_paths=[str(p) for p in (coverage, coverage_9263) if p],
                ep01_hdrtab=ep01_tile, group_pairs_compared=len(tuple(combinations(GROUPS, 2))),
                known_exposure_overlap=False,
                basis="User confirms all mosaic inputs are disjoint; footprint and available detector files corroborate observation grouping.",
                late_group_sources={"ep07": "../ID_DR1_43433.ipynb (OBS 002 catalog)",
                                    "ep08": "local single-exposure F200W cutouts (OBS 004)",
                                    "ep09": "local single-exposure F200W cutout (OBS 006)"})


def frame_inputs(arrays, offsets, wcs, train, held, half):
    x, y = wcs.world_to_pixel_values(*RA_DEC)
    ix, iy = np.rint([x, y]).astype(int)
    phase = np.array([x-ix, y-iy])
    context = RenderContext((2*half+1,)*2)
    frames = {}
    for epoch, (data, err, op) in arrays.items():
        bank = [(u, v, data[window(u, v, 12)], err[window(u, v, 12)], None) for u, v in train]
        kernel, _, _ = measure_kernel(bank)
        factor, _, shrinkage = noise_factor(kernel, context.shape)
        native_offset = phase - offsets[epoch]["xy"]
        op = replace(op, max_offset=9.)
        point = op.render_point(*native_offset, context)
        sky = np.ones(context.shape)
        def blank_z(coords):
            values = []
            for u, v in coords:
                stamp = window(u, v, half)
                coef, sigma, _ = linear_fit(data[stamp], [point, sky], err[stamp], factor)
                values.append(coef[0]/sigma[0])
            return np.array(values)

        scale = np.std(blank_z(train), ddof=1)
        region = window(ix, iy, half)
        image, sigma = data[region], err[region]*scale
        if not (np.isfinite(image).all() and np.isfinite(sigma).all() and (sigma > 0).all()):
            raise ValueError(f"invalid target pixels or noise calibration in {epoch}")

        def whiten(value, factor=factor, sigma=sigma):
            return solve_triangular(factor, (value/sigma).ravel(), lower=True, check_finite=False)

        clumps = {name: op.render_point(*(native_offset + np.array(xy) - [1+x, 1+y]), context)
                  for name, xy in CLUMPS.items()}
        frames[epoch] = dict(image=image, sigma=sigma, covariance_factor=factor, whiten=whiten,
                             op=op, context=context, native_offset=native_offset, point=point,
                             clumps=clumps, sky=whiten(sky), data=whiten(image), training_scale=scale,
                             held_scatter=np.std(blank_z(held)/scale, ddof=1), shrinkage=shrinkage)
    return frames


def host_vector(frame, geometry):
    dx, dy, re, q, pa, n = geometry
    native = frame["native_offset"] + [dx, dy]
    return frame["op"].render_sersic(SersicProfile(1., re, n, q, pa), *native, frame["context"])


def profile_fit(frames, epochs, active_points, initial, free_n, clumps=(), max_re=12.):
    names = ["host", *clumps, *(f"{epoch}.sky" for epoch in epochs),
             *(f"{epoch}.point" for epoch in active_points)]
    n_pixels = frames[epochs[0]]["data"].size
    data = np.concatenate([frames[epoch]["data"] for epoch in epochs])
    design = np.zeros((data.size, len(names)))
    for j, epoch in enumerate(epochs):
        frame = frames[epoch]
        rows = slice(j*n_pixels, (j+1)*n_pixels)
        design[rows, names.index(f"{epoch}.sky")] = frame["sky"]
        for name in clumps:
            design[rows, names.index(name)] = frame["whiten"](frame["clumps"][name])
        if epoch in active_points:
            design[rows, names.index(f"{epoch}.point")] = frame["whiten"](frame["point"])

    count = len(GEOMETRY) if free_n else len(GEOMETRY)-1
    low = np.array([-3., -6., .55, .25, -270., .4])[:count]
    high = np.array([8., 6., max_re, 1., 270., 4.])[:count]

    def matrix(vector):
        geometry = np.r_[vector, 1.] if not free_n else vector
        for j, epoch in enumerate(epochs):
            frame = frames[epoch]
            design[j*n_pixels:(j+1)*n_pixels, 0] = frame["whiten"](host_vector(frame, geometry))
        return design

    def residual(vector):
        current = matrix(vector)
        coef = np.linalg.lstsq(current, data, rcond=None)[0]
        return current @ coef - data

    fit = least_squares(residual, initial[:count], bounds=(low, high), diff_step=1e-4, max_nfev=85)
    if not fit.success:
        raise RuntimeError(f"host optimization failed: {fit.message}")
    current = matrix(fit.x)
    coef = np.linalg.lstsq(current, data, rcond=None)[0]
    if coef[0] <= 0:
        raise ValueError("fitted host has nonpositive flux")
    # The covariance follows the local linearization of the full joint fit.
    # Use unprofiled derivatives to retain the amplitude/geometry cross terms.
    jacobian = np.empty((data.size, count))
    for i in range(count):
        step = np.eye(count)[i]*1e-4
        upper = np.minimum(fit.x+step, high)
        lower = np.maximum(fit.x-step, low)
        plus = matrix(upper)[:, 0].copy()
        minus = matrix(lower)[:, 0].copy()
        jacobian[:, i] = coef[0]*(plus-minus)/(upper[i]-lower[i])
    joint = np.column_stack([matrix(fit.x), jacobian])
    covariance = np.linalg.pinv(joint.T @ joint, hermitian=True, rcond=1e-13)
    estimates = {name: dict(value=value, local_sigma=sigma)
                 for name, value, sigma in zip(names, coef, np.sqrt(np.diag(covariance)))}
    whitened = fit.fun.reshape(len(epochs), n_pixels)
    return dict(geometry=dict(zip(GEOMETRY, fit.x if free_n else np.r_[fit.x, 1.])),
                geometry_at_bound=[name for name, active in zip(GEOMETRY, fit.active_mask) if active],
                host=estimates["host"], clumps={name: estimates[name] for name in clumps},
                sky={e: estimates[f"{e}.sky"] for e in epochs},
                point={e: estimates[f"{e}.point"] for e in active_points},
                chi2_by_epoch=dict(zip(epochs, (whitened**2).sum(axis=1))),
                dof=data.size-len(names)-count, evaluations=fit.nfev)


def signed_holdout(frame, host_model):
    host = host_model["host"]["value"]*host_vector(frame, list(host_model["geometry"].values()))
    for name, estimate in host_model["clumps"].items():
        host += estimate["value"]*frame["clumps"][name]
    coef, sigma, residual = linear_fit(frame["image"]-host, [frame["point"], np.ones_like(host)],
                                      frame["sigma"], frame["covariance_factor"])
    return dict(flux_uJy=coef[0], conditional_sigma_uJy=sigma[0], z=coef[0]/sigma[0],
                chi2=residual@residual)


def check_m9(frames, result, provenance, offsets):
    observations, definitions = [], []
    model = CompositeModel([SersicSource("host"), PointSource("point"), ConstantBackground("sky")])
    for epoch, frame in frames.items():
        manifest = provenance["groups"][epoch]
        if not manifest["coverage_complete"]:
            continue
        geometry = result["geometry"].copy()
        geometry["dx"], geometry["dy"] = frame["native_offset"] + [geometry["dx"], geometry["dy"]]
        local = {f"host.{key}": value for key, value in geometry.items()}
        local.update({"host.flux": result["host"]["value"],
                      "point.flux": result["point"].get(epoch, {"value": 0.})["value"],
                      "point.dx": frame["native_offset"][0], "point.dy": frame["native_offset"][1],
                      "sky.level": result["sky"][epoch]["value"]})
        aliases = {key: f"{epoch}.{key}" for key in local}
        definitions.extend(Parameter(aliases[key], fixed=value) for key, value in local.items())
        factor = frame["covariance_factor"]*frame["sigma"].reshape(-1, 1)
        observations.append(JointObservation(
            name=epoch, band="F200W", data=frame["image"], model=model,
            context=replace(frame["context"], psf_operator=frame["op"]), aliases=aliases,
            exposures=tuple(manifest["coverage_detector_ids"]),
            registration=f"Recorded native-grid offsets {offsets[epoch]['xy']} to common frame",
            calibration="MJy/sr to uJy/pixel; phase-zero PSF normalized at 0.36 arcsec",
            data_unit="uJy/pixel", noise_options=dict(backend="dense", covariance=factor@factor.T),
        ))
    m9 = evaluate_joint_model_fit(
        prepare_joint_model_fitting(observations, ParameterSet(definitions)), decompose=True,
    )
    for observation in observations:
        residual = m9["observations"][observation.name]["residual"]
        whitened = frames[observation.name]["whiten"](residual)
        np.testing.assert_allclose(whitened@whitened, result["chi2_by_epoch"][observation.name],
                                   rtol=1e-11, atol=1e-10)
    return [observation.name for observation in observations]


def main():
    parser = ArgumentParser(description=__doc__)
    for label in ("cutouts", "early_psfs", "products", "coverage", "output"):
        parser.add_argument(label, type=Path)
    parser.add_argument("--coverage-9263", type=Path)
    parser.add_argument("--ep01-tile", type=Path)
    parser.add_argument("--radius-check", action="store_true",
                        help="Repeat the 21-pixel clump fit with a wider host radius bound.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    exposure = audit(args.coverage, args.coverage_9263, args.products, args.ep01_tile)
    arrays, records, veto, wcs = epoch_inputs(args.cutouts, args.early_psfs, args.products/"PSFs")
    reference, offsets = adopted_alignment(args.products, wcs, arrays["ep01"][0].shape)
    train, held = positions(arrays, veto, wcs)
    starts = ([4., 0., 3., .7, 90., 1.], [2., -1., 4., .6, 0., 1.])
    pre, late = EPOCHS[:7], EPOCHS[-2:]
    if args.radius_check:
        frames = frame_inputs(arrays, offsets, wcs, train, held, 10)
        result = profile_fit(frames, EPOCHS, late, [6., -1.3, 12., .32, -9.], False,
                             clumps=tuple(CLUMPS), max_re=30.)
        result.update(stamp_size=21, max_re_px=30., registration_reference=reference)
        (args.output/"joint_radius_check.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)+"\n",
        )
        print(result["geometry"], result["point"], flush=True)
        return
    fits_by_size = {}
    for half in (7, 10):
        print(f"Fitting {2*half+1} × {2*half+1} native stamps", flush=True)
        frames = frame_inputs(arrays, offsets, wcs, train, held, half)
        fixed = min((profile_fit(frames, EPOCHS, late, start, False) for start in starts),
                    key=lambda result: sum(result["chi2_by_epoch"].values()))
        print("  Fixed-n fits complete", flush=True)
        free = profile_fit(frames, EPOCHS, late, list(fixed["geometry"].values()), True)
        print("  Free-n fit complete", flush=True)
        clumps = profile_fit(frames, EPOCHS, late, list(fixed["geometry"].values()), False,
                             clumps=tuple(CLUMPS))
        print("  Fixed-n fit with clumps complete", flush=True)
        result = dict(exponential=fixed, free_n=free, exponential_with_clumps=clumps,
                      held_noise={epoch: {field: frame[field] for field in
                                  ("training_scale", "held_scatter", "shrinkage")}
                                  for epoch, frame in frames.items()})
        if half == 7:
            result["m9_likelihood_checked_epochs"] = check_m9(frames, free, exposure, offsets)
            result["pre_held_out"] = {}
            for epoch in pre:
                train_epoch = tuple(item for item in pre if item != epoch)
                training = profile_fit(frames, train_epoch, (), starts[0], False, clumps=tuple(CLUMPS))
                result["pre_held_out"][epoch] = signed_holdout(frames[epoch], training)
                print(f"  Withheld {epoch}: {result['pre_held_out'][epoch]['z']:.2f} sigma", flush=True)
        size = 2*half+1
        fits_by_size[size] = result
        for label, fit in (("exponential", fixed), ("free_n", free), ("exponential_with_clumps", clumps)):
            print(size, label, fit["geometry"], fit["point"], flush=True)
    report = dict(exposure_audit=exposure, registration_reference=reference, offsets=offsets,
                  source_inputs=records, train_patches=len(train), held_patches=len(held),
                  pre_point_flux_fixed_zero=list(pre), point_position="fixed at recorded RA/Dec",
                  conditional_on_disjoint_direct_inputs=True, fits=fits_by_size,
                  shared_host_clump_xy_fits=CLUMPS,
                  host_clump_source="../2026abck_coadd_host_model.ipynb, F200W knot sensitivity")
    (args.output/"joint_target_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
