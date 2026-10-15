"""Sphinx configuration for the kipe documentation."""

import sys
import typing
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.generate_api import generate_api  # ruff: ignore[module-import-not-at-top-of-file]

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
exclude_patterns = ["_apidoc/modules.rst", "_generated/*"]  # _generated: included by pages
html_theme = "sphinx_rtd_theme"
# \bs{x}: bold symbols, as in the article the theory page follows
mathjax3_config = {"tex": {"macros": {"bs": [r"\boldsymbol{#1}", 1]}}}
html_static_path = ["_static"]
html_css_files = ["mermaid.css", "tables.css"]
mermaid_height = "auto"  # in proportion to the width, not a fixed box; see mermaid.css
autodoc_member_order = "bysource"
# compiled dependencies (MPI, FEniCSx) are not needed to render the API
autodoc_mock_imports = ["mpi4py", "dolfinx", "ufl", "petsc4py", "basix"]


def _generate_api(app):
    """Generate the API pages from the package, so none are maintained by hand."""
    generate_api(Path(app.srcdir), ROOT / "src" / "kipe")


def _generate_study_reference(app):
    """Generate the study file reference from the options, so none is maintained by hand."""
    sys.path.insert(0, str(Path(__file__).parent))  # study_reference.py, next to this file
    from study_reference import write_reference

    write_reference(Path(app.srcdir) / "_generated" / "study_file.rst")


def _show_annotated_as_its_type(app):
    """Show ``Annotated[X, ...]`` as ``X`` in the API pages.

    The options use ``Annotated`` for pydantic's constraints, e.g., ``Field(ge=0)``, which
    autodoc would show as ``Annotated[float, FieldInfo(..., required=True, ...)]``, misleading
    for optional fields. The study file reference shows the constraints readably. Sphinx has no
    setting for this, so this wraps its ``stringify_annotation`` wherever a Sphinx module holds
    it; if none does (a renamed function), the build fails here.
    """
    import sphinx.util.typing

    original = sphinx.util.typing.stringify_annotation

    def stringify_annotation(annotation, *args, **kwargs):
        if typing.get_origin(annotation) is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
        return original(annotation, *args, **kwargs)

    modules = [
        module
        for name, module in list(sys.modules.items())
        if name.startswith("sphinx")
        and getattr(module, "stringify_annotation", None) is original
    ]
    if not modules:
        raise RuntimeError("sphinx.util.typing.stringify_annotation not found: update conf.py")
    for module in modules:
        module.stringify_annotation = stringify_annotation


def setup(app):
    """Register the API generation hook."""
    app.connect("builder-inited", _generate_api)
    app.connect("builder-inited", _generate_study_reference)
    app.connect("builder-inited", _show_annotated_as_its_type)
