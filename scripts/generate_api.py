"""Generate Sphinx API pages and add corresponding Mermaid class diagrams."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "kipe"
DIAGRAMS_DIR = ROOT / "docs" / "source" / "_static" / "diagrams"


def insert_class_diagram(page: Path, module: str, diagram: Path) -> None:
    """Insert a Mermaid class diagram before a generated module's API details."""
    content = page.read_text(encoding="utf-8")
    marker = f".. Generated class diagram for ``{module}``."
    if marker in content:
        return

    image_path = Path(os.path.relpath(diagram, page.parent)).as_posix()
    directive = (
        f"{marker}\n\n"
        f".. mermaid:: {image_path}\n"
        f"   :alt: UML class diagram for {module}, showing its classes and relationships.\n"
        f"   :caption: Class relationships in ``{module}``.\n\n"
    )
    api_details = content.find(".. automodule::")
    if api_details < 0:
        raise RuntimeError(f"Generated API page has no automodule directive: {page}")
    page.write_text(content[:api_details] + directive + content[api_details:], encoding="utf-8")


def generate_api(source_dir: Path, package_root: Path = PACKAGE_ROOT) -> None:
    """Generate the API pages and insert available class diagrams."""
    try:
        from sphinx.ext.apidoc import main as apidoc_main
    except ImportError as error:
        raise RuntimeError("Install the docs dependencies to generate the API pages.") from error

    api_dir = source_dir / "_apidoc"
    apidoc_main([
        "--force",
        "--separate",
        "--module-first",
        "-o",
        str(api_dir),
        str(package_root),
    ])

    for diagram in sorted(DIAGRAMS_DIR.glob("class-*.mmd")):
        module = diagram.stem.removeprefix("class-")
        page = api_dir / f"{module}.rst"
        if not page.is_file():
            raise RuntimeError(f"No generated API page for class diagram {diagram.name}.")
        insert_class_diagram(page, module, diagram)


def main() -> None:
    """Prepare API pages before the Sphinx HTML build."""
    generate_api(ROOT / "docs" / "source")


if __name__ == "__main__":
    main()
