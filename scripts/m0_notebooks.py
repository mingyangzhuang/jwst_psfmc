"""Execute original notebooks in an explicitly selected output folder.

Uses the current interpreter in a private kernel spec. Does not shorten MCMC,
alter notebook code, overwrite examples, or assert convergence from execution.
Outputs and errors are saved even when kernel startup or a cell fails.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def execute(name, output):
    output = output.resolve()
    run = output / name
    run.mkdir(parents=True, exist_ok=True)
    for folder in ("data", "PSF"):
        link = run / folder
        if not link.exists():
            link.symlink_to(os.path.relpath(ROOT / "examples" / folder, run), target_is_directory=True)
    # Jupyter writes only under the repository; no user kernel registration.
    config = ROOT / ".venv-sersic/m0-jupyter"
    spec = config / "kernels/m0"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps(dict(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="M0 baseline", language="python")))
    os.environ["JUPYTER_PATH"] = str(config)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(config / "runtime")
    os.environ["IPYTHONDIR"] = str(config / "ipython")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".venv-sersic/mplconfig")
    original = ROOT / "examples" / f"{name}.ipynb"
    nb = nbformat.read(original, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    started = time.monotonic()
    report = dict(notebook=str(original.relative_to(ROOT)), code_modified=False,
                  interpreter=sys.executable, status="running", stochastic=True,
                  comparison="Execution only; chains are not deterministic goldens.")
    try:
        NotebookClient(nb, timeout=1800, kernel_name="m0",
                       resources={"metadata": {"path": str(run)}},
                       record_timing=True).execute()
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        report["seconds"] = time.monotonic() - started
        report["executed_code_cells"] = sum(c.get("execution_count") is not None
                                              for c in nb.cells if c.cell_type == "code")
        report["total_code_cells"] = sum(c.cell_type == "code" for c in nb.cells)
        nbformat.write(nb, run / f"{name}.executed.ipynb")
        (run / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("demo_psf_photometry", "demo_covariance_kernel"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(0 if execute(args.name, args.output) else 1)
