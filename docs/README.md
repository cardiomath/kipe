# Documentation

Built with Sphinx (reStructuredText and Markdown via MyST). API pages are generated
automatically from the package docstrings at build time (`sphinx.ext.apidoc` in `conf.py`).

## Build locally

```bash
pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

The `sphinx-build` pre-commit hook runs this same build on commits touching `docs/` or `src/`.

Open `docs/build/html/index.html` (or `python -m http.server -d docs/build/html`).

## GitHub Pages

The workflow `.github/workflows/docs.yml` builds and deploys on every push to `main`
(or manually via *Run workflow*). One-time setup: **Settings → Pages → Build and deployment →
Source: GitHub Actions**. The site is then at `https://cardiomath.github.io/kipe/`.
