"""The analysis template of the FitzHugh-Nagumo example runs on the results of its study."""

import runpy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from kipe.cli import main
from kipe.options import dump_study, load_study

EXAMPLE = Path(__file__).parents[2] / "examples" / "fitzhugh_nagumo"


def test_analysis_template(tmp_path, monkeypatch, capsys):
    """With four realizations and a sensitivity analysis: the summary and the four figures."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)  # the study's paths are relative to the working directory

    study = load_study(EXAMPLE / "study_realizations.yaml")
    dump_study(replace(study, synthesis=replace(study.synthesis, realizations=4)), "study.yaml")
    for command in ("synthesis", "estimation", "sensitivity"):
        assert main([command, "study.yaml"]) == 0

    monkeypatch.setattr(sys, "argv", ["analysis.py", "study.yaml", "--save", "figures"])
    runpy.run_path(str(EXAMPLE / "analysis.py"), run_name="__main__")

    output = capsys.readouterr().out
    assert "4 realizations of study.yaml" in output
    assert "Cramér-Rao" in output  # the bounds, from the sensitivity analysis
    for name in ("estimates", "consistency", "correlations", "pairs"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
