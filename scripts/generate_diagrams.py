"""Generate source-derived UML diagrams for the documentation."""

import ast
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "kipe"
DIAGRAMS_DIR = ROOT / "docs" / "source" / "_static" / "diagrams"
IGNORED_DIRECTORIES = {"__pycache__", "_generated", "build", "dist", "tests"}


def module_name(source: Path, package_root: Path = PACKAGE_ROOT) -> str:
    """Return the dotted module name for a package source file."""
    relative = source.relative_to(package_root)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((package_root.name, *parts))


def class_modules(package_root: Path = PACKAGE_ROOT) -> list[tuple[str, Path]]:
    """Find Python modules containing class definitions without importing them.

    Returns:
        Module names paired with their source paths.
    """
    result = []
    for source in sorted(package_root.rglob("*.py")):
        if IGNORED_DIRECTORIES.intersection(source.relative_to(package_root).parts):
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        if any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)):
            result.append((module_name(source, package_root), source))
    return result


def _run_pyreverse(
    sources: list[Path], project: str, output_dir: Path, *, ignore_examples: bool = False
) -> None:
    command = [
        "pyreverse",
        "--only-classnames",
        "--all-ancestors",
        "--show-associated",
        "1",
        "--colorized",
        "--output",
        "svg",
        "--project",
        project,
        "--source-roots",
        str(SOURCE_ROOT),
        "--output-directory",
        str(output_dir),
    ]
    if ignore_examples:
        command.extend(["--ignore", "examples,__pycache__"])
    command.extend(str(source) for source in sources)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Pyreverse could not be started; install the docs dependencies."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Pyreverse failed while analyzing project {project!r}.") from error


def _require_tools() -> None:
    missing = [tool for tool in ("pyreverse", "dot") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(
            f"Missing required diagram tool(s): {', '.join(missing)}. "
            "Install the docs dependencies and Graphviz."
        )


def generate_diagrams(
    package_root: Path = PACKAGE_ROOT, output_dir: Path = DIAGRAMS_DIR
) -> list[Path]:
    """Generate class diagrams and core/backend module dependency diagrams.

    Returns:
        Paths to the generated SVG files.
    """
    _require_tools()
    sources = class_modules(package_root)
    core_sources = sorted(package_root.glob("*.py"))
    backend_root = package_root / "fenicsx"
    if not core_sources or not backend_root.is_dir():
        raise RuntimeError(f"Expected KIPE source package at {package_root}.")

    with tempfile.TemporaryDirectory(prefix="kipe-diagrams-") as temporary:
        temporary_dir = Path(temporary)
        generated: dict[str, Path] = {}

        for module, source in sources:
            project = f"class_{module.replace('.', '_')}"
            _run_pyreverse([source], project, temporary_dir)
            diagram = temporary_dir / f"classes_{project}.svg"
            if not diagram.is_file():
                raise RuntimeError(f"Pyreverse did not generate the class diagram for {module}.")
            generated[f"class-{module}.svg"] = diagram

        for project, package_sources, ignores in (
            ("kipe_core", core_sources, True),
            ("kipe_fenicsx", [backend_root], True),
        ):
            _run_pyreverse(package_sources, project, temporary_dir, ignore_examples=ignores)
            diagram = temporary_dir / f"packages_{project}.svg"
            if not diagram.is_file():
                raise RuntimeError(
                    f"Pyreverse did not generate the dependency diagram for {project}."
                )
            generated[f"dependencies-{project.removeprefix('kipe_')}.svg"] = diagram

        output_dir.mkdir(parents=True, exist_ok=True)
        expected_names = set(generated)
        for name, source in generated.items():
            shutil.copyfile(source, output_dir / name)
        for stale in output_dir.glob("*.svg"):
            if stale.name not in expected_names:
                stale.unlink()

    return [output_dir / name for name in sorted(generated)]


def main() -> None:
    """Generate all documentation diagrams."""
    diagrams = generate_diagrams()
    print(f"Generated {len(diagrams)} SVG diagrams in {DIAGRAMS_DIR}.")


if __name__ == "__main__":
    main()
