"""Rerun M0 verification, preserving all exits and leaving goldens untouched.

Usage: python scripts/run_m0.py --output /path/to/documentation/m0/rerun
Install .[dev,nautilus,notebooks] in that interpreter before running. This runner
does not install dependencies or modify source/example inputs. Use a new output
directory for each run. Executed notebooks go in the selected run directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / ".venv-sersic/mplconfig"),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    commands = [
        ("freeze", [sys.executable, "-m", "pip", "freeze"]),
        ("dependency-check", [sys.executable, "-m", "pip", "check"]),
        ("requested-extras", [sys.executable, "-c",
            "from importlib.metadata import version\nfrom packaging.version import Version\n"
            "for n,v in [('pytest','7'),('pytest-cov','0'),('nautilus-sampler','1.0.6'),('jupyter','1')]:\n"
            " print(n, version(n))\n"
            " if Version(version(n)) < Version(v): raise RuntimeError(n + ' is too old')\n"]),
        ("capture", [sys.executable, "scripts/m0_baseline.py", "--capture", "--output", str(out)]),
        ("all-tests", [sys.executable, "-m", "pytest", "-q", f"--junitxml={out / 'suite.xml'}"]),
        ("demo_covariance_kernel", [sys.executable, "scripts/m0_notebooks.py", "demo_covariance_kernel",
                                    "--output", str(out / "notebooks")]),
        ("demo_psf_photometry", [sys.executable, "scripts/m0_notebooks.py", "demo_psf_photometry",
                                 "--output", str(out / "notebooks")]),
    ]
    result = []
    for name, command in commands:
        with (out / f"{name}.txt").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result.append(dict(name=name, command=command, exit_code=process.returncode))
        (out / "commands.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: exit {process.returncode}", flush=True)
    return int(any(item["exit_code"] for item in result))


if __name__ == "__main__":
    sys.exit(main())
