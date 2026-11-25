"""Measure prepared noise and PSF+Sérsic rendering at common stamp sizes."""

from argparse import ArgumentParser
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import numpy as np

from jwst_psfmc import PSFOperator, PSFSpec, RenderContext, SersicProfile, prepare_noise


def elapsed(operation, repeats=1):
    start = perf_counter()
    for _ in range(repeats):
        operation()
    return (perf_counter()-start)/repeats


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    axes = np.arange(-48,49)/4
    response = np.exp(-.5*((axes[:,None]/1.05)**2+(axes[None,:]/1.05)**2))
    kernel = np.array([[0.,.1,0.],[.1,1.,.1],[0.,.1,0.]])
    rows = []
    for size in (25,51,101):
        shape = (size,size)
        data = np.zeros(shape)
        err = np.ones(shape)
        for name in ("diagonal","fft_periodic","dense"):
            if name == "dense" and size > 25:
                continue
            settings = {"backend":name}
            if name != "diagonal":
                settings.update(kernel=kernel,boundary="periodic" if name=="fft_periodic" else "finite")
            start = perf_counter()
            noise = prepare_noise(data,err,**settings)
            rows.append(dict(size=size,backend=name,prepare_s=perf_counter()-start,
                             likelihood_s=elapsed(lambda: noise.log_likelihood(err), 100)))
        for n,re in ((1.,3.5),(4.,1.5)):
            start = perf_counter()
            spec = PSFSpec(response,"effective",.25,(48,48),"benchmark amplitude")
            op = PSFOperator(spec,oversamp=8)
            context = RenderContext(shape,psf_operator=op)
            profile = SersicProfile(1.,re,n,.72,28.)
            def scene():
                return .5*op.render_point(.25,-.17,context)+op.render_sersic(profile,.25,-.17,context)
            try:
                image = scene()
            except ValueError as exc:
                if "above max_source_cells" not in str(exc):
                    raise
                rows.append(dict(size=size,renderer="point+Sersic",n=n,re_pixels=re,
                                 default_guard=str(exc)))
                op = PSFOperator(spec,oversamp=8,max_source_cells=1_250_000)
                context = RenderContext(shape,psf_operator=op)
                start = perf_counter()
                image = scene()
            cold = perf_counter()-start
            assert image.shape == shape and np.all(np.isfinite(image))
            rows.append(dict(size=size,renderer="point+Sersic",n=n,re_pixels=re,point_to_host=.5,
                             max_source_cells=op.max_source_cells,cold_s=cold,warm_s=elapsed(scene,4)))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"platform":platform.platform(),"python":platform.python_version(),
              "peak_rss_bytes":int(rss if sys.platform=="darwin" else rss*1024),"metrics":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
