"""Recheck central stellar stamps with the PSFs used by the ep01 workflow."""

from argparse import ArgumentParser
from contextlib import ExitStack
import json
from pathlib import Path

from astropy.io import fits
import numpy as np

from scripts.validate_empirical_psf import open_mosaic, star_tests, stellar_null_tests, write_csv


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("blank_dir", type=Path)
    parser.add_argument("star_dir", type=Path)
    parser.add_argument("central_psf_dir", type=Path)
    parser.add_argument("--results", type=Path, default=Path("docs/validation/empirical"))
    args = parser.parse_args()
    with (args.results / "manifest.json").open() as handle:
        kernel_sets = json.load(handle)["products"]
    for band, mas in (("F200W", 30), ("F444W", 30), ("F444W", 60)):
        key = f"{band}_{mas}mas"
        suffix = "_60mas" if mas == 60 else ""
        archive = args.star_dir / band / f"{band}_PSF_OBS_002_4_c{suffix}.fits"
        workflow = args.central_psf_dir / f"nexus_central_deep_ep01_{band.lower()}{suffix}_PSF_4_c.fits"
        np.testing.assert_array_equal(fits.getdata(archive), fits.getdata(workflow))
        with ExitStack() as stack:
            mosaic = open_mosaic(stack, args.blank_dir, band, mas)
            stars = star_tests(args.star_dir, band, mas, mosaic, mosaic_psf_obs="002",
                               archive_observations=("001",))
            stars = [row for row in stars if row["origin"] == "ep01_mosaic"]
            nulls = stellar_null_tests(args.star_dir, band, mas, mosaic, stars,
                                       kernel_sets[key]["kernels"], psf_obs="002")
        write_csv(args.results / f"{key}_central_obs002_stars.csv", stars)
        write_csv(args.results / f"{key}_central_obs002_null.csv", nulls)
        print(key, len(stars), len(nulls), flush=True)


if __name__ == "__main__":
    main()
