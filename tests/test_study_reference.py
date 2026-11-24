"""The study reference renders the real options schema, including fixed coordinate tuples."""

import runpy
from pathlib import Path


def test_study_reference_with_plane_orientation_and_component_noise(tmp_path):
    generator = Path(__file__).parents[1] / "docs/source/reference/study_reference.py"
    namespace = runpy.run_path(str(generator))
    output = tmp_path / "study_file.rst"
    namespace["write_reference"](output)
    text = output.read_text()
    assert "normal_toward: <[number, number, number]" in text
    assert "number > 0 or list of number > 0" in text
