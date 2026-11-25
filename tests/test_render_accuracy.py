"""Image-error budgets remain separate from the sampler status."""

import json
from pathlib import Path

import numpy as np
import pytest

from jwst_psfmc import (
    CompositeModel, ConstantBackground, Parameter, ParameterSet, PointSource,
    SersicProfile, SersicSource, Uniform, load_model_results, prepare_model_fitting,
    run_model_mcmc, save_model_results,
)
from scripts.run_m7 import _context
from tests.test_rendering import analytic_gaussian, continuous_gaussian_convolution

DESIGN = Path(__file__).resolve().parents[1] / "docs/validation/m7/design.json"


@pytest.mark.parametrize(("case_id", "status"), [("a03", "within_budget"), ("a11", "over_budget")])
def test_m7_reference_status_survives_sampling_and_archive(case_id,status,tmp_path):
    config = json.loads(DESIGN.read_text())
    case = next(dict(zip(config["numeric_fields"],row)) for row in config["numeric"] if row[0]==case_id)
    context = _context(config["stamp_shape"],config,case["kind"])
    dx,dy = case["dx"],case["dy"]
    profile = SersicProfile(1.,case["re"],case["n"],case["q"],case["pa"])
    host = continuous_gaussian_convolution(profile,context,dx,dy,angles=1024)
    point = analytic_gaussian(context,dx,dy,config["psf"]["sigma_x"],config["psf"]["sigma_y"])
    noise = np.full(context.shape,np.linalg.norm(host)/case["host_sn"])
    params = ParameterSet([
        Parameter("host.flux", initial=1.,prior=Uniform(.95,1.05)),
        Parameter("sky.level", initial=0.,prior=Uniform(-.015,.015)),
        Parameter("point.flux", fixed=case["contrast"]),
        *[Parameter(f"{name}.{axis}",fixed=value)
          for name in ("host","point") for axis,value in (("dx",dx),("dy",dy))],
        *[Parameter(f"host.{name}",fixed=case[name]) for name in ("n","re","q","pa")],
    ])
    model = CompositeModel([PointSource("point"),SersicSource("host"),ConstantBackground("sky")])
    def prepare(**kwargs):
        return prepare_model_fitting(host+case["contrast"]*point,model,params,context=context,
                                     noise_options={"err":noise},**kwargs)
    options = dict(render_references={"point":point,"host":host},
                   render_reference_label=f"M7 {case_id} independent Gaussian")
    if status == "over_budget":
        with pytest.warns(RuntimeWarning,match="before fitting"):
            fit = prepare(**options)
    else:
        fit = prepare(**options)
    assert prepare().provenance["render_accuracy"]["status"] == "unvalidated"
    measured = fit.provenance["render_accuracy"]
    assert measured["status"] == status
    if case_id=="a11":
        assert measured["witness_epsilon"] == pytest.approx(.660,abs=.005)
    result = run_model_mcmc(fit,seed=9021,sampling_options={"nsteps":4300,"burn_in":600,"nwalkers":16})
    assert result.status == "converged"
    path = tmp_path/"fit.npz"
    save_model_results(path,result)
    loaded = load_model_results(path)
    assert loaded.metadata["render_accuracy"]==measured
    assert len(loaded.metadata["build"]["source_sha256"])==64
    assert "numpy" in loaded.metadata["dependency_versions"]
    assert {"git_commit","git_dirty"}<=loaded.metadata["build"].keys()
