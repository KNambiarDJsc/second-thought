# Releasing to PyPI

Releases go through `.github/workflows/release.yml` via PyPI's **trusted publishing** (OIDC) — no
API token is ever generated, pasted, or stored anywhere, for this repo or in any chat. One-time
setup, done once by whoever owns the PyPI account:

## One-time setup (do this once, before the first release)

1. Log into [pypi.org](https://pypi.org) (create an account if you don't have one — no need to
   create the `second-thought` project first).
2. Go to **Your account → Publishing** → "Add a new pending publisher".
3. Fill in:
   - PyPI project name: `second-thought`
   - Owner: `KNambiarDJsc`
   - Repository name: `second-thought`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
4. Save. PyPI now trusts a `release.yml` run from this exact repo to publish as `second-thought`
   the first time it runs — nothing else is needed on the PyPI side.
5. In this repo's GitHub settings, create an environment named `pypi`
   (Settings → Environments → New environment) so the workflow's `environment: pypi` line has
   somewhere to attach to. Optionally add required reviewers on that environment if you want a
   manual approval gate before every publish.

## Cutting a release

1. Bump `version` in `pyproject.toml`.
2. Commit that bump.
3. Tag it to match exactly: `git tag v0.1.0 && git push origin v0.1.0`.
4. The workflow re-runs the full test suite (same gate as CI), builds the sdist/wheel, and
   publishes — only if the tag's version matches `pyproject.toml`'s, so a stray tag can't publish
   the wrong version.
5. Watch it at `gh run watch` or the Actions tab. If it fails before the `publish` job, nothing
   was uploaded — fix and re-tag.
