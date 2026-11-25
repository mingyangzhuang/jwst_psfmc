"""Record the running build and its numerical dependencies."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess


def runtime_provenance():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source = path.read_bytes()
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(len(source).to_bytes(8, "big"))
        digest.update(source)
    dependencies = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "astropy", "photutils", "emcee", "nautilus-sampler"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    build = {"source_sha256": digest.hexdigest(), "git_commit": None, "git_dirty": None}
    root = package.parent
    if (root / ".git").exists() and (root / "pyproject.toml").is_file():
        try:
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL,
                                               text=True, timeout=5).strip()
            build.update(git_commit=git("rev-parse", "HEAD"),
                         git_dirty=bool(git("status", "--porcelain", "--untracked-files=normal")))
        except (OSError, subprocess.SubprocessError):
            pass
    return {"build": build, "dependency_versions": dependencies}
