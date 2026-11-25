"""Inventory local NIRCam detector files and separate their variance planes."""

from argparse import ArgumentParser
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from scripts.validate_m10_epochs import RA_DEC
from scripts.validate_m10_joint import EXPOSURE, GROUPS


PLANES = ("VAR_POISSON", "VAR_RNOISE", "VAR_FLAT")


def detector_report(path):
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        sci_header = hdul["SCI"].header
        shape = (sci_header["NAXIS2"], sci_header["NAXIS1"])
        x,y = WCS(sci_header,fix=False).celestial.world_to_pixel_values(*RA_DEC)
        ix,iy = np.rint([x,y]).astype(int)
        if not (24 <= ix < shape[1]-24 and 24 <= iy < shape[0]-24):
            raise ValueError(f"target outside variance audit region in {path}")
        region = np.s_[iy-24:iy+25,ix-24:ix+25]
        science = hdul["SCI"].data[region]
        sigma = hdul["ERR"].data[region].astype(float)
        dq = hdul["DQ"].data[region]
        variance = np.stack([hdul[key].data[region] for key in PLANES]).astype(float)
    total = variance.sum(axis=0)
    good = (dq==0)&np.isfinite(science+sigma+total)&(total>0)&(sigma>0)
    dy,dx = np.mgrid[-24:25,-24:25]
    radius = np.hypot(dx+ix-x,dy+iy-y)
    blank = good&(radius >= 13)&(radius <= 24)
    limit = np.percentile(science[blank],80)
    blank &= science < limit
    groups = {"target_radius_5_px":good&(radius<=5), "background_annulus":blank}
    summary = {}
    for name, mask in groups.items():
        if not np.any(mask):
            raise ValueError(f"no unflagged data for {path}: {name}")
        summary[name] = dict(pixels=int(mask.sum()),
            median_fraction=dict(zip(PLANES,np.median(variance[:,mask]/total[mask],axis=1))),
            median_err2_over_variance_sum=np.median(sigma[mask]**2/total[mask]),
            median_var_poisson=np.median(variance[0,mask]))
    return dict(path=str(path),band=header["FILTER"],date=header["DATE-OBS"],summary=summary)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("products",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    records = {}
    files = sorted(args.products.glob("jw*_tweakreg.fits"))
    for path in files:
        match = EXPOSURE.match(path.name)
        if match is None:
            raise ValueError(f"unexpected exposure name: {path}")
        program,obs = int(match[2]),match[3]
        epochs = [epoch for epoch,(p,visits) in GROUPS.items() if p==program and obs in visits]
        record = detector_report(path)
        record.update(program=program, obs=obs, epochs=epochs)
        records[match[1]] = record
    if not records:
        raise ValueError("no full detector products found")
    args.output.write_text(json.dumps(records,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
