from pathlib import Path

from scripts.generate_diagrams import class_modules, generate_diagrams, module_name


def test_class_modules_are_discovered_without_irrelevant_directories(tmp_path: Path) -> None:
    package = tmp_path / "kipe"
    (package / "fenicsx").mkdir(parents=True)
    (package / "tests").mkdir()
    (package / "__pycache__").mkdir()
    (package / "fenicsx" / "points.py").write_text("class PointSampler:\n    pass\n")
    (package / "helpers.py").write_text("def helper():\n    pass\n")
    (package / "tests" / "test_module.py").write_text("class TestModule:\n    pass\n")
    (package / "__pycache__" / "cached.py").write_text("class Cached:\n    pass\n")

    assert class_modules(package) == [("kipe.fenicsx.points", package / "fenicsx" / "points.py")]


def test_package_initializer_maps_to_package_module(tmp_path: Path) -> None:
    package = tmp_path / "kipe"

    assert module_name(package / "fenicsx" / "__init__.py", package) == "kipe.fenicsx"


def test_mermaid_generation_replaces_stale_diagrams(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "kipe"
    backend = package / "fenicsx"
    backend.mkdir(parents=True)
    output = tmp_path / "diagrams"
    output.mkdir()
    (package / "options.py").write_text("class Options:\n    pass\n")
    (backend / "__init__.py").write_text("")
    (output / "old.svg").write_text("stale")
    (output / "old.mmd").write_text("stale")

    def fake_pyreverse(sources, project, output_dir, *, ignore=None):
        prefix = "classes" if project.startswith("class_") else "packages"
        (output_dir / f"{prefix}_{project}.mmd").write_text("classDiagram\n")

    monkeypatch.setattr("scripts.generate_diagrams._require_tools", lambda: None)
    monkeypatch.setattr("scripts.generate_diagrams._run_pyreverse", fake_pyreverse)

    diagrams = generate_diagrams(package, output)

    assert {diagram.name for diagram in diagrams} == {
        "class-kipe.options.mmd",
        "dependencies-core.mmd",
        "dependencies-fenicsx.mmd",
    }
    assert {diagram.name for diagram in output.iterdir()} == {
        "class-kipe.options.mmd",
        "dependencies-core.mmd",
        "dependencies-fenicsx.mmd",
    }
