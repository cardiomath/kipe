from pathlib import Path

from scripts.generate_diagrams import class_modules, module_name


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
