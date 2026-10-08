"""Loading a study file, constructing the forward solver, and ``kipe list-parameters``."""

import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from kipe.cli import main
from kipe.forward_solver import ForwardSolverError, build_forward_solver
from kipe.options import (
    ForwardSolverOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    OutputOptions,
    ParameterPrior,
    ParametersOptions,
    StudyFileError,
    StudyOptions,
    TimeRange,
    dump_study,
    load_study,
)

STUDY = """\
output:
  path: results/fhn

forward_solver:
  factory: "kipe.examples.fitzhugh_nagumo:Solver"
  arguments: {dt: 0.05, c: 2.5}
"""


class _IncompleteSolver:
    def nominal_parameters(self) -> dict[str, float]:
        return {}


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "study.yaml"
    path.write_text(text)
    return path


def test_load_study(tmp_path):
    study = load_study(_write(tmp_path, STUDY))
    assert study.output.path == "results/fhn"
    assert study.output.log_level == "info"
    assert study.forward_solver.arguments == {"dt": 0.05, "c": 2.5}


@pytest.mark.parametrize(
    "text",
    [
        STUDY + "unknown_section: {}\n",  # unknown section
        STUDY.replace("path:", "pth:"),  # typo in a key
        STUDY.replace("kipe.examples.fitzhugh_nagumo:Solver", "kipe.examples.Solver"),  # no ':'
        "output: {path: results}\n",  # missing forward_solver
    ],
)
def test_load_study_rejects_invalid(tmp_path, text):
    with pytest.raises(ValidationError):
        load_study(_write(tmp_path, text))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- just a list\n", "mapping"),
        ("output: {path: results\n", "not valid YAML"),  # unclosed brace
    ],
)
def test_load_study_rejects_unreadable(tmp_path, text, message):
    with pytest.raises(StudyFileError, match=message):
        load_study(_write(tmp_path, text))


def test_load_study_reports_missing_file(tmp_path):
    with pytest.raises(StudyFileError, match=r"cannot read study file .*No such file"):
        load_study(tmp_path / "missing.yaml")


def test_build_forward_solver():
    options = ForwardSolverOptions("kipe.examples.fitzhugh_nagumo:Solver", {"c": 2.5})
    solver = build_forward_solver(options)
    assert solver.nominal_parameters()["c"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("factory", "arguments", "message"),
    [
        ("kipe.no_such_module:Solver", {}, "No module named"),
        ("kipe.examples.fitzhugh_nagumo:NoSuchSolver", {}, "has no attribute 'NoSuchSolver'"),
        ("kipe.examples.fitzhugh_nagumo:Solver", {"d": 1.0}, "unexpected keyword argument 'd'"),
    ],
)
def test_build_forward_solver_errors(factory, arguments, message):
    with pytest.raises(ForwardSolverError, match=message):
        build_forward_solver(ForwardSolverOptions(factory, arguments))


def test_build_forward_solver_rejects_nonconforming(monkeypatch):
    module = types.ModuleType("fake_solvers")
    module.IncompleteSolver = _IncompleteSolver
    monkeypatch.setitem(sys.modules, "fake_solvers", module)  # removed after the test

    with pytest.raises(ForwardSolverError) as excinfo:
        build_forward_solver(ForwardSolverOptions("fake_solvers:IncompleteSolver"))

    message = str(excinfo.value)
    for name in ["initial_state", "propagate", "state_spec"]:
        assert name in message
    assert "nominal_parameters" not in message


def test_list_parameters(tmp_path, capsys):
    assert main(["list-parameters", str(_write(tmp_path, STUDY))]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split() for line in lines] == [
        ["parameter", "nominal"],
        ["a", "0.2"],
        ["b", "0.2"],
        ["c", "2.5"],
    ]


def test_list_parameters_shows_selection(tmp_path, capsys):
    selection = """
parameters:
  reparameterization: multiplicative
  select:
    c: {relative_stddev: 0.5, initial: 3.0}
    a: {reparameterization: additive, stddev: 0.1}
"""
    assert main(["list-parameters", str(_write(tmp_path, STUDY + selection))]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "parameter", "nominal", "reparameterization", "initial", "stddev", "1σ", "range"
    ]  # fmt: skip
    assert lines[1].split() == ["a", "0.2", "additive", "0.2", "0.1", "[0.1,", "0.3]"]
    assert lines[2].split() == ["b", "0.2"]
    assert lines[3].split() == [
        "c", "2.5", "multiplicative", "3", "0.5", "(relative)", "[1.5,", "4.5]"
    ]  # fmt: skip


def test_list_parameters_reports_errors(tmp_path, capsys):
    study = _write(tmp_path, STUDY.replace("c: 2.5", "d: 2.5"))
    assert main(["list-parameters", str(study)]) == 1
    assert "unexpected keyword argument 'd'" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["list-parameters", "synthesis", "estimation", "plot"])
def test_cli_reports_invalid_options(tmp_path, capsys, command):
    study = _write(tmp_path, STUDY.replace("path:", "pth:"))
    assert main([command, str(study)]) == 1
    err = capsys.readouterr().err
    assert f"kipe: error: invalid study file {study}" in err
    assert "output.pth: Unexpected keyword argument" in err
    assert "errors.pydantic.dev" not in err


def test_load_study_reads_yaml_1_2(tmp_path):
    """YAML 1.2: ``1e-3`` is a float (YAML 1.1 reads a string), passed to the solver as is."""
    study = load_study(_write(tmp_path, STUDY.replace("dt: 0.05", "dt: 1e-3")))
    assert study.forward_solver.arguments["dt"] == pytest.approx(1e-3)


def test_dump_study_round_trip(tmp_path):
    """load_study reads back what dump_study writes, unions included (times, priors)."""
    study = StudyOptions(
        output=OutputOptions(path="results"),
        forward_solver=ForwardSolverOptions(factory="kipe.examples.fitzhugh_nagumo:Solver"),
        parameters=ParametersOptions(
            reparameterization="log",
            select={
                "a": ParameterPrior(initial=0.25, relative_stddev=0.1),
                "c": ParameterPrior(reparameterization="additive", stddev=0.5),
            },
        ),
        measurements={
            "v": MeasurementOptions(
                fields=["v"],
                times=[0.5, 1.0],
                data=NumpyDataOptions(type="numpy", path="v.npz"),
                noise=NoiseOptions(stddev=0.05, seed=0),
            ),
            "w": MeasurementOptions(
                fields=["w"],
                times=TimeRange(start=0.5, stop=2.0, step=0.5),
                data=NumpyDataOptions(type="numpy", path="w.npz"),
                noise=NoiseOptions(stddev=0.05, seed=1),
            ),
        },
    )
    path = tmp_path / "study.yaml"
    dump_study(study, path)
    assert load_study(path) == study
