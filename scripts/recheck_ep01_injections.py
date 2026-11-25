"""Repeat the held-out ep01 injections with the verified workflow PSFs."""

from argparse import ArgumentParser
from collections import defaultdict
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import numpy as np

from scripts.validate_empirical_psf import blank_tests, models, open_mosaic, write_csv


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("blank_dir", type=Path)
    parser.add_argument("psfs", type=Path)
    parser.add_argument("--results", type=Path, default=Path("docs/validation/empirical"))
    args = parser.parse_args()
    with (args.results/"manifest.json").open() as handle:
        products = json.load(handle)["products"]
    for band, mas in (("F200W",30), ("F444W",30), ("F444W",60)):
        key = f"{band}_{mas}mas"
        product = products[key]
        held = defaultdict(list)
        with (args.results/f"{key}_blank.csv").open() as handle:
            for row in csv.DictReader(handle):
                if (row["case"], row["renderer"], row["backend"]) == ("zero","epsf","diagonal"):
                    held[row["depth"]].append((int(row["x"]), int(row["y"])))
        output = []
        banks = defaultdict(dict)
        with ExitStack() as stack:
            mosaic = open_mosaic(stack,args.blank_dir,band,mas)
            for depth, bank in product["kernels"].items():
                half = (bank["sky_patch_size"]-1)//2
                label = "edge" if depth=="edge" else int(depth)
                assert len(held[depth]) == bank["heldout"]
                positions = bank["training_xy"] + held[depth]
                banks[half][label] = [(x,y,*mosaic.region(x,y,half)) for x,y in positions]
            render,op,norm = models(args.psfs,band,"002",mas)
            for half, selected in banks.items():
                estimates,rows = blank_tests(
                    selected,half,(product["stamp_size"]-1)//2,render,op,norm
                )
                for depth, bank in estimates.items():
                    np.testing.assert_allclose(bank["kernel"],product["kernels"][depth]["kernel"],
                                               rtol=1e-10,atol=1e-12)
                output.extend(rows)
        if len(output) != 28*sum(map(len,held.values())):
            raise ValueError(f"incomplete held-out rows for {key}")
        write_csv(args.results/f"{key}_obs002_blank.csv",output)
        print(key,len(output),flush=True)


if __name__ == "__main__":
    main()
