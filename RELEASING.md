# Releasing `jwst_psfmc`

Releases are published to PyPI with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so **no API
token is stored anywhere** — not in the repository, not in GitHub secrets, not
on a laptop. PyPI verifies a short-lived OpenID Connect identity issued by
GitHub Actions for this specific repository, workflow, and environment.

The workflow is `.github/workflows/publish.yml`.

---

## One-time setup

### 1. Create the two GitHub environments

In the repository: **Settings → Environments → New environment**.

Create one named `testpypi` and one named `pypi`. No secrets or variables are
needed — the names alone are what the trusted publisher configuration binds to.

Optionally add yourself as a **required reviewer** on the `pypi` environment so
a real release always pauses for a manual approval.

### 2. Register the pending publisher on TestPyPI

`jwst-psfmc` does not exist on either index yet, so use a *pending* publisher —
this both reserves the name and authorizes the first upload.

Go to <https://test.pypi.org/manage/account/publishing/> and add a new pending
publisher with exactly these values:

| Field | Value |
|---|---|
| PyPI Project Name | `jwst-psfmc` |
| Owner | `mingyangzhuang` |
| Repository name | `jwst_psfmc` |
| Workflow name | `publish.yml` |
| Environment name | `testpypi` |

### 3. Register the pending publisher on PyPI

Same thing at <https://pypi.org/manage/account/publishing/>:

| Field | Value |
|---|---|
| PyPI Project Name | `jwst-psfmc` |
| Owner | `mingyangzhuang` |
| Repository name | `jwst_psfmc` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Note the environment name differs between the two. Everything else matches.

---

## Publishing a release

### Dry run to TestPyPI

**Actions → Publish → Run workflow**, leaving the target as `testpypi`.

Then verify the upload installs cleanly. TestPyPI does not mirror the
scientific stack, so pull the dependencies from real PyPI:

```bash
python -m venv /tmp/verify && source /tmp/verify/bin/activate
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  jwst-psfmc
python -c "import jwst_psfmc; print(jwst_psfmc.__version__)"
```

### Real release to PyPI

A version number on PyPI can never be reused or overwritten, so only do this
once the TestPyPI run looks right.

1. Move the release notes in `CHANGELOG.md` from `[Unreleased]` into a new
   version section and set the date.
2. Bump the version in **both** `pyproject.toml` and
   `jwst_psfmc/__init__.py` (`__version__`) — they must agree.
3. Commit, then tag and push:

   ```bash
   git commit -am "Release v0.1.0"
   git tag v0.1.0
   git push origin master --tags
   ```

Pushing the tag triggers the publish job automatically. (The same job can also
be run manually from the Actions tab with the target set to `pypi`.)

### After publishing

```bash
pip install jwst-psfmc
```

and check the project page at <https://pypi.org/project/jwst-psfmc/>.

---

## If something goes wrong

* **`invalid-publisher` / OIDC rejected** — the five fields in the trusted
  publisher config must match the workflow exactly, including the environment
  name and the `publish.yml` filename. This is the usual cause.
* **`File already exists`** — that version was already uploaded. PyPI does not
  allow re-uploads; bump the version and release again.
* **Yanking a bad release** — a release can be yanked from the PyPI project
  settings, which hides it from new installs without breaking existing pins.
  It cannot be deleted and its version number cannot be reused.
