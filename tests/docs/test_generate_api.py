from pathlib import Path

from scripts.generate_api import insert_class_diagram


def test_class_diagram_insertion_is_relative_and_idempotent(tmp_path: Path) -> None:
    api_dir = tmp_path / "_apidoc"
    api_dir.mkdir()
    static_dir = tmp_path / "_static" / "diagrams"
    static_dir.mkdir(parents=True)
    page = api_dir / "kipe.options.rst"
    diagram = static_dir / "class-kipe.options.svg"
    page.write_text("kipe.options\n============\n\n.. automodule:: kipe.options\n")

    insert_class_diagram(page, "kipe.options", diagram)
    first_result = page.read_text()
    insert_class_diagram(page, "kipe.options", diagram)

    assert first_result == page.read_text()
    assert first_result.index(".. figure::") < first_result.index(".. automodule::")
    assert ".. figure:: ../_static/diagrams/class-kipe.options.svg" in first_result
    assert first_result.count(".. figure::") == 1
