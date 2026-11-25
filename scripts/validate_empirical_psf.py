"""Validate matched mosaic noise and archived NEXUS stellar PSF shapes.

Run with python -m scripts.validate_empirical_psf BLANK_DIR STAR_DIR.
Only small regions of the FITS mosaics are mapped into NumPy work arrays.
"""
from argparse import ArgumentParser
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.linalg import cholesky, solve_triangular
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.signal import correlate

from jwst_psfmc import PSFOperator, PSFSpec, RenderContext, SersicProfile, prepare_psf_for_oversamp
from jwst_psfmc.psf import _psf_fft, _psf_model_from_fft


@dataclass
class Mosaic:
    sci: object
    err: object
    exposure: object
    wcs: WCS
    mas: int

    def region(self, x, y, margin):
        ys, xs = slice(y-margin, y+margin+1), slice(x-margin, x+margin+1)
        h = self.exposure.header
        coverage = (self.exposure.data[ys,xs].astype(np.int16) * h.get("BSCALE", 1)
                    + h.get("BZERO", 0))
        return (np.asarray(self.sci.data[ys,xs], dtype=float),
                np.asarray(self.err.data[ys,xs], dtype=float), coverage)


def open_mosaic(stack, path, band, mas):
    suffix = "_60mas" if mas == 60 else ""
    prefix = path / f"nexus_central_deep_ep01_{band.lower()}{suffix}_i2d"
    entries = [stack.enter_context(fits.open(
        f"{prefix}_{role}.fits", memmap=True, do_not_scale_image_data=True
    ))[0] for role in ("data", "error", "expt")]
    wcs = [WCS(hdu.header).celestial for hdu in entries]
    sample_x, sample_y = np.array([0, 5000, 13000]), np.array([0, 5000, 11000])
    sky = wcs[0].pixel_to_world_values(sample_x, sample_y)
    for hdu, grid in zip(entries[1:], wcs[1:]):
        if hdu.shape != entries[0].shape:
            raise ValueError(f"mismatched mosaic shapes for {prefix}")
        np.testing.assert_allclose(grid.world_to_pixel_values(*sky), [sample_x,sample_y], atol=1e-5, rtol=0)
    pixel = np.sqrt(abs(np.linalg.det(wcs[0].pixel_scale_matrix))) * 3.6e6
    np.testing.assert_allclose(pixel, mas, rtol=1e-5)
    return Mosaic(*entries, wcs[0], mas)


def independent_veto(veto, sky, angular_margin):
    x, y = np.rint(veto.wcs.world_to_pixel_values(*sky)).astype(int)
    margin = round(angular_margin/veto.mas) + 3
    ny, nx = veto.sci.shape
    if x < margin or y < margin or x >= nx-margin or y >= ny-margin:
        return False
    data, err, cover = veto.region(x, y, margin)
    valid = np.isfinite(data) & np.isfinite(err) & (err > 0) & (cover > 0)
    if valid.mean() < .85:
        return False
    z = np.divide(data, err, where=valid, out=np.zeros_like(data))
    smoothed = gaussian_filter(z, .9)[3:-3,3:-3]
    center = np.median(smoothed)
    mad = 1.4826*np.median(abs(smoothed-center))
    return np.max(smoothed-center) < 5.5*max(.25,mad)


def select(mosaic, veto, rng, per_bank, *, margin=None, step=43, depths=(1,2,3,"edge"), excluded=()):
    ny,nx = mosaic.sci.shape
    if margin is None:
        margin = 24 if mosaic.mas == 30 else 16
    fit_half = 6 if mosaic.mas == 30 else 4
    h = mosaic.exposure.header
    sampled = (mosaic.exposure.data[margin:ny-margin:step,margin:nx-margin:step].astype(np.int16)
               + h.get("BZERO",0))
    selected = {depth: [] for depth in depths}
    used = list(excluded)
    for depth in selected:
        indices = np.argwhere(sampled == (2 if depth=="edge" else depth))
        target = per_bank if depth=="edge" else 2*per_bank
        rng.shuffle(indices)
        for iy, ix in indices:
            x, y = int(margin+ix*step), int(margin+iy*step)
            if any(max(abs(x-u),abs(y-v)) <= 2*margin for u,v in used):
                continue
            data, err, cover = mosaic.region(x,y,margin)
            core = np.s_[margin-fit_half:margin+fit_half+1,margin-fit_half:margin+fit_half+1]
            if depth == "edge":
                unique = np.unique(cover[core])
                if len(unique)!=2 or np.any(unique<=0) or np.max(unique)>3:
                    continue
                ok = np.isfinite(data[core]) & np.isfinite(err[core]) & (err[core]>0)
            else:
                if not np.all(cover == depth):
                    continue
                ok = np.isfinite(data) & np.isfinite(err) & (err>0)
            if not ok.all():
                continue
            sky = mosaic.wcs.pixel_to_world_values(x,y)
            if not independent_veto(veto,sky,margin*mosaic.mas if depth!="edge" else fit_half*mosaic.mas):
                continue
            selected[depth].append((x,y,data,err,cover))
            used.append((x,y))
            if len(selected[depth]) == target:
                break
    return selected, margin, fit_half


def measure_kernel(patches, cutoff=5):
    shape = (2*cutoff+1,)*2
    numerator, pairs = np.zeros(shape), np.zeros(shape)
    for _,_,data,err,_ in patches:
        normalized = (data-np.median(data))/err
        good = abs(normalized - np.median(normalized)) < 6
        normalized = np.where(good, normalized-np.median(normalized[good]), 0)
        ys,xs = [slice(n-1-cutoff,n+cutoff) for n in data.shape]
        numerator += correlate(normalized,normalized,method="fft")[ys,xs]
        pairs += correlate(good.astype(float),good.astype(float),method="fft")[ys,xs]
    kernel = numerator / pairs
    zero_variance = kernel[cutoff,cutoff]
    grid = np.arange(-cutoff,cutoff+1)
    window = (1 - abs(grid)/(cutoff+1))
    return kernel/zero_variance * np.outer(window,window), float(zero_variance), int(pairs[cutoff,cutoff])


def noise_factor(kernel, shape):
    coords = np.array(list(np.ndindex(shape)))
    deltas = coords[:,None]-coords[None,:]
    center = kernel.shape[0]//2
    dx,dy = deltas[...,1]+center,deltas[...,0]+center
    good = (dx>=0)&(dy>=0)&(dx<kernel.shape[1])&(dy<kernel.shape[0])
    corr = np.zeros_like(dx,dtype=float)
    corr[good] = kernel[dy[good],dx[good]]
    corr = (corr+corr.T)/2
    minimum = float(np.linalg.eigvalsh(corr)[0])
    # Transparent diagonal shrinkage only if the estimated finite matrix is
    # numerically indefinite.
    shrinkage = max(0.,(1e-6-minimum)/(1.-minimum)) if minimum < 1e-6 else 0.
    strength = (1.-shrinkage)*corr + shrinkage*np.eye(len(corr))
    return cholesky(strength, lower=True), minimum, shrinkage


def models(path, band, obs, mas):
    suffix = "_60mas" if mas==60 else ""
    data = fits.getdata(path/band/f"{band}_PSF_OBS_{obs}_4_c{suffix}.fits").astype(float)
    y,x = np.indices(data.shape)
    ox,oy = (data.shape[1]-1)/2,(data.shape[0]-1)/2
    # ±0.9 arcsec keeps the host convolution tractable. Preserve the input
    # scale and truncate both branches to similar angular support.
    native_radius = int(900/mas)
    sl = np.s_[int(oy)-4*native_radius:int(oy)+4*native_radius+1,
               int(ox)-4*native_radius:int(ox)+4*native_radius+1]
    cropped = data[sl].copy()
    spacing = .25
    reference_radius = 360/mas
    aperture = np.hypot((x-ox)*spacing,(y-oy)*spacing)<=reference_radius
    total = np.sum(data[aperture]) * spacing**2
    center = (4*native_radius,)*2
    aperture_spec = PSFSpec(cropped, "effective", spacing, center,
                           "nominal 0.36 arcsec aperture", scale=1/total)
    op = PSFOperator(aperture_spec, oversamp=8, max_offset=1.3)
    raw, native, _ = prepare_psf_for_oversamp(data,4,(2*native_radius-1,)*2,return_fraction=True)
    ft = _psf_fft(raw)
    cut_y, cut_x = (np.array(data.shape)-np.array(raw.shape))//2
    legacy_x = (ox - cut_x - (raw.shape[1]-1)/2)*spacing
    legacy_y = (oy - cut_y - (raw.shape[0]-1)/2)*spacing
    ref_shape = (2*native_radius-1,)*2
    yy,xx = np.indices(ref_shape)-(np.array(ref_shape)[:,None,None]-1)/2
    ref_mask = np.hypot(xx,yy)<=reference_radius
    # Both flux units are defined by the identical nominal native-pixel
    # aperture applied to a unit template at the centered phase.
    norm_e = op.render_point(0,0,RenderContext(ref_shape))[ref_mask].sum()
    norm_l = _psf_model_from_fft(
        [1,-legacy_x,-legacy_y],ft,4,output_shape=ref_shape,native_shape=native
    )[ref_mask].sum()
    def render(kind, dx,dy, context):
        if kind=="epsf":
            return op.render_point(dx,dy,context)/norm_e
        return _psf_model_from_fft(
            [1,dx-legacy_x,dy-legacy_y],ft,4,output_shape=context.shape,native_shape=native
        )/norm_l
    return render,op,norm_e


def linear_fit(data, templates, err, lower=None):
    flat = data.ravel()/err.ravel()
    matrix = np.column_stack([template.ravel()/err.ravel() for template in templates])
    if lower is not None:
        transformed = solve_triangular(lower,np.column_stack((flat,matrix)),lower=True,check_finite=False)
        flat, matrix = transformed[:,0],transformed[:,1:]
    normal = matrix.T @ matrix
    inverse = np.linalg.solve(normal,np.eye(len(normal)))
    mean = inverse @ (matrix.T @ flat)
    return mean,np.sqrt(np.diag(inverse)),flat-matrix@mean


def blank_tests(selected, margin, fit_half, render, op, norm_e):
    shape = (2*fit_half+1,)*2
    context = RenderContext(shape)
    definitions = {}
    rows = []
    kernels = {}
    for depth, entries in selected.items():
        if depth=="edge":
            if not kernels:
                continue
            kernel = np.mean(list(kernels.values()),axis=0)
            train_count,heldout,sigma,pairs = 0,entries,None,None
        else:
            train_count = len(entries)//2
            if train_count < 5:
                continue
            kernel,sigma,pairs = measure_kernel(entries[:train_count])
            kernels[depth] = kernel
            heldout = entries[train_count:]
        lower,minimum,shrinkage = noise_factor(kernel,shape)
        definitions[str(depth)] = dict(training=train_count,heldout=len(heldout),standardized_variance=sigma,
                                       pairs_at_zero_lag=pairs,min_eigenvalue=minimum,shrinkage=shrinkage,
                                       kernel=kernel.tolist(),sky_patch_size=2*margin+1,
                                       training_xy=[[int(x),int(y)] for x,y,*_ in entries[:train_count]])
        core = np.s_[margin-fit_half:margin+fit_half+1,margin-fit_half:margin+fit_half+1]
        phases = ((0.,0.),(.25,-.25),(-.49,.49))
        hosts = [op.render_sersic(SersicProfile(1,.07/(.03 if fit_half==6 else .06),1,.67,32),
                                 dx,dy,context)/norm_e for dx,dy in phases]
        ones = np.ones(shape)
        for i,(x,y,patch,unc,cover) in enumerate(heldout):
            data,err,nexp = patch[core],unc[core],cover[core]
            dx,dy = phases[i%len(phases)]
            host = hosts[i%len(phases)]
            p = render("epsf",dx,dy,context)
            l = render("legacy",dx,dy,context)
            _,sigma_p,_ = linear_fit(data,[p,ones],err,lower)
            _,sigma_h,_ = linear_fit(data,[p,host,ones],err,lower)
            for case in ("zero","point5","point20","host5","host20"):
                truth = np.zeros(2)
                if case.startswith("point"):
                    truth[0] = int(case[5:])*sigma_p[0]
                elif case.startswith("host"):
                    truth[1] = int(case[4:])*sigma_h[1]
                    truth[0] = 2*truth[1]
                injected = data+truth[0]*p+truth[1]*host
                for renderer,pt in (("epsf",p),("legacy",l)):
                    templates = [pt,host,ones] if case.startswith("host") else [pt,ones]
                    for backend,factor in (("diagonal",None),("dense",lower)):
                        measured,errors,res = linear_fit(injected,templates,err,factor)
                        for j,name in enumerate(("point","host") if case.startswith("host") else ("point",)):
                            estimate,sigma = measured[j],errors[j]
                            rows.append(dict(depth=depth,x=x,y=y,depth_min=nexp.min(),depth_max=nexp.max(),
                                case=case,component=name,renderer=renderer,backend=backend,
                                truth=truth[j],estimate=estimate,sigma=sigma,z=(estimate-truth[j])/sigma,
                                chi2=res@res,dof=len(res)-len(templates)))
    return definitions,rows


def fit_star(data, err, render, shape, initial):
    context = RenderContext(shape)
    valid = np.isfinite(data) & np.isfinite(err) & (err>0)
    if valid.sum()<.85*data.size:
        return None
    core_data, core_err = data[valid],err[valid]
    ones = np.ones(len(core_data))
    rows = []
    for kind in ("epsf","legacy"):
        def solve(offset):
            template = render(kind,*offset,context)
            coeff,_,res = linear_fit(core_data,[template[valid],ones],core_err)
            return coeff,res,template
        result = least_squares(lambda offset: solve(offset)[1], np.clip(initial,-1.15,1.15),
                               bounds=(-1.2,1.2),max_nfev=35)
        coeff,res,template = solve(result.x)
        norm = np.linalg.norm(core_data-coeff[1])
        fraction = np.linalg.norm(res*core_err)/norm if norm else np.nan
        rows.append(dict(renderer=kind,success=result.success,dx=float(result.x[0]),dy=float(result.x[1]),
                         flux=float(coeff[0]),relative_residual=float(fraction),
                         objective_per_dof=float(res@res/(len(res)-4))))
    return rows


def star_tests(path, band, mas, mosaic, *, mosaic_psf_obs="001", archive_observations=("001", "002", "012", "014")):
    suffix = "_60mas" if mas==60 else ""
    margin = 6 if mas==30 else 4
    out = []
    shape = (2*margin+1,)*2
    for obs in archive_observations:
        render,_,_ = models(path,band,obs,mas)
        mosaic_render = render
        if obs == "001" and mosaic_psf_obs != obs:
            mosaic_render,_,_ = models(path,band,mosaic_psf_obs,mas)
        with fits.open(path/band/f"Stars_OBS_{obs}{suffix}.fits",memmap=True) as hdus:
            table = hdus[1].data
            for row in table[table["include"]]:
                id_ = str(row["SOURCE_ID"])
                tile = row["tilename"].strip()
                center = row["center_cutout"]
                cutout = row["cutout_data_masked"]
                x,y = np.rint(center).astype(int)
                image = cutout[y-margin:y+margin+1,x-margin:x+margin+1]
                variants = [("archive",image,np.ones(shape),np.empty(0),center-[x,y])]
                if obs=="001":
                    mx,my = mosaic.wcs.world_to_pixel_values(row["skycoord.ra"],row["skycoord.dec"])
                    ix,iy = np.rint([mx,my]).astype(int)
                    ny,nx = mosaic.sci.shape
                    if margin<=ix<nx-margin and margin<=iy<ny-margin:
                        sci,err,coverage=mosaic.region(ix,iy,margin)
                        variants.append(("ep01_mosaic",sci,err,coverage,np.array([mx-ix,my-iy])))
                        variants.append(("ep01_uniform",sci,np.where(err>0,1.,np.nan),
                                         coverage,np.array([mx-ix,my-iy])))
                for origin,img,err,coverage,guess in variants:
                    if img.shape!=shape:
                        continue
                    tested = fit_star(img,err,render if origin=="archive" else mosaic_render,shape,guess)
                    if tested is not None:
                        for metrics in tested:
                            out.append(dict(obs=obs,tile=tile,source_id=id_,origin=origin,
                                            depth_min=int(coverage.min()) if coverage.size else None,
                                            depth_max=int(coverage.max()) if coverage.size else None,**metrics))
    return out


def stellar_null_tests(path, band, mas, mosaic, stars, kernels, *, origin="ep01_mosaic", psf_obs="001"):
    """Fit a fixed host under each point response in matched ep01 stamps."""
    suffix = "_60mas" if mas == 60 else ""
    margin = 6 if mas == 30 else 4
    context = RenderContext((2*margin+1,)*2)
    render, op, norm_e = models(path, band, psf_obs, mas)
    kernels = {key: np.asarray(value["kernel"]) for key,value in kernels.items()}
    factors = {key: noise_factor(value, context.shape)[0] for key,value in kernels.items()}
    profile = SersicProfile(1, 70/mas, 1, .67, 32)
    positions = {}
    with fits.open(path/band/f"Stars_OBS_001{suffix}.fits") as hdus:
        for row in hdus[1].data:
            key = (row["tilename"].strip(), str(row["SOURCE_ID"]))
            positions[key] = tuple(np.rint(mosaic.wcs.world_to_pixel_values(
                row["skycoord.ra"],row["skycoord.dec"])).astype(int))
    output = []
    for star in stars:
        if star["origin"] != origin or not star["success"] or star["flux"] <= 0:
            continue
        dx,dy = star["dx"],star["dy"]
        x,y = positions[(star["tile"],star["source_id"])]
        data,err,cover = mosaic.region(x,y,margin)
        # A PSF shape diagnostic on a bright source conditional on the
        # delivered ERR. No source shot-noise term is assumed or added.
        if not (np.isfinite(data).all() and np.all(np.isfinite(err)&(err>0)) and np.all(cover>0)):
            continue
        depth = str(int(cover.min())) if cover.min()==cover.max() else "edge"
        lower = factors.get(depth)
        if origin == "ep01_uniform":
            err, lower = np.ones_like(err), None
        point = render(star["renderer"],dx,dy,context)
        host = op.render_sersic(profile,dx,dy,context)/norm_e
        ones = np.ones(context.shape)
        fitted,uncertainty,res = linear_fit(data,[point,host,ones],err,lower)
        _,_,null = linear_fit(data,[point,ones],err,lower)
        total = float(fitted[0]+fitted[1])
        output.append(dict(
            tile=star["tile"],source_id=star["source_id"],x=x,y=y,depth_min=cover.min(),depth_max=cover.max(),
            renderer=star["renderer"],backend="dense" if lower is not None else "diagonal",
            point_flux=fitted[0],host_flux=fitted[1],host_sigma=uncertainty[1],
            signed_host_fraction=fitted[1]/total if total>0 else np.nan,
            delta_chi2=null@null-res@res,centroid_near_limit=max(abs(dx),abs(dy))>1.1,
        ))
    return output


def write_csv(path, rows):
    with path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_blank(rows):
    groups = {}
    for r in rows:
        key = "|".join(str(r[a]) for a in ("depth","case","component","renderer","backend"))
        groups.setdefault(key,[]).append(float(r["z"]))
    summary = {}
    for key,values in groups.items():
        v = np.asarray(values)
        summary[key] = dict(n=len(v),median=float(np.median(v)),mean=float(np.mean(v)),
                            sd=float(np.std(v,ddof=1)),coverage68=float(np.mean(abs(v)<=1)),
                            coverage95=float(np.mean(abs(v)<=1.96)))
    return summary


def summarize_stars(rows):
    groups = {}
    for r in rows:
        key = "|".join(str(r[k]) for k in ("obs","origin","renderer"))
        if r["success"] and r["flux"]>0 and np.isfinite(r["relative_residual"]):
            groups.setdefault(key,[]).append(r["relative_residual"])
    return {k:dict(n=len(v),median_residual=float(np.median(v)),q84_residual=float(np.quantile(v,.84)))
            for k,v in groups.items()}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("blank_dir",type=Path)
    parser.add_argument("stars_dir",type=Path)
    parser.add_argument("--bank-size",type=int,default=24)
    parser.add_argument("--seed",type=int,default=20261122)
    parser.add_argument("--output",type=Path,default=Path("docs/validation/empirical"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    begun=perf_counter()
    total={"seed":args.seed,"bank_size":args.bank_size,"products":{}}
    with ExitStack() as stack:
        keys=(("F200W",30),("F444W",30),("F444W",60))
        mosaics={key:open_mosaic(stack,args.blank_dir,*key) for key in keys}
        for index,(band,mas) in enumerate(keys):
            key=f"{band}_{mas}mas"
            mosaic=mosaics[(band,mas)]
            veto=mosaics[("F444W",30) if band=="F200W" else ("F200W",30)]
            rng=np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(index,)))
            selected,margin,fit_half=select(mosaic,veto,rng,args.bank_size)
            print(key,"selected",{str(k):len(v) for k,v in selected.items()},flush=True)
            render,op,norm=models(args.stars_dir,band,"001",mas)
            definitions,blank=blank_tests(selected,margin,fit_half,render,op,norm)
            if (band,mas)==("F444W",30) and "3" not in definitions:
                supplement,small,half=select(
                    mosaic,veto,np.random.default_rng(np.random.SeedSequence(args.seed,spawn_key=(1,3))),
                    args.bank_size,margin=16,step=23,depths=(3,),
                    excluded=[(x,y) for entries in selected.values() for x,y,*_ in entries],
                )
                added,rows=blank_tests(supplement,small,half,render,op,norm)
                definitions.update(added)
                blank.extend(rows)
            stars=star_tests(args.stars_dir,band,mas,mosaic)
            write_csv(args.output/f"{key}_blank.csv",blank)
            write_csv(args.output/f"{key}_stars.csv",stars)
            stellar_null=stellar_null_tests(args.stars_dir,band,mas,mosaic,stars,definitions)
            if stellar_null:
                write_csv(args.output/f"{key}_stellar_null.csv",stellar_null)
            uniform_null=stellar_null_tests(args.stars_dir,band,mas,mosaic,stars,definitions,
                                            origin="ep01_uniform")
            if uniform_null:
                write_csv(args.output/f"{key}_stellar_null_uniform.csv",uniform_null)
            counts=Counter((r["obs"],r["origin"],r["renderer"]) for r in stars)
            total["products"][key]=dict(kernels=definitions,blank=summarize_blank(blank),
                stars=summarize_stars(stars),tested_stars={"|".join(k):v for k,v in counts.items()},
                stamp_size=2*fit_half+1,sky_patch_size=2*margin+1,
                mosaic_files=[Path(hdu.fileinfo()["file"].name).name
                              for hdu in (mosaic.sci,mosaic.err,mosaic.exposure)],
                ep01_psf_sha256=op.spec.metadata["samples_sha256"],
                null_rows=len(stellar_null))
            print(key,"prepared",len(blank),"blank rows and",len(stars),"star rows",flush=True)
    total["wall_s"]=perf_counter()-begun
    (args.output/"manifest.json").write_text(json.dumps(total,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
