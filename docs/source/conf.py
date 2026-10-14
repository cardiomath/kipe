"""Sphinx configuration for the kipe documentation."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path

from sphinx.ext.apidoc import main as apidoc_main

ROOT = Path(__file__).resolve().parents[2]

project = "kipe"
author = "kipe contributors"
try:
    release = _version("kipe")
except PackageNotFoundError:
    release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_apidoc/modules.rst"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["mermaid.css"]
mermaid_height = "auto"  # in proportion to the width, not a fixed box; see mermaid.css
autodoc_member_order = "bysource"
# compiled dependencies (MPI, FEniCSx) are not needed to render the API
autodoc_mock_imports = ["mpi4py", "dolfinx", "ufl", "petsc4py", "basix"]


def _generate_api(app):
    """Generate the API pages from the package, so none are maintained by hand."""
    out = Path(app.srcdir) / "_apidoc"
    apidoc_main([
        "--force",
        "--separate",
        "--module-first",
        "-o",
        str(out),
        str(ROOT / "src" / "kipe"),
    ])


def setup(app):
    """Register the API generation hook."""
    app.connect("builder-inited", _generate_api)
