"""``kipe plot``: reading the estimation history and plotting it."""

from pathlib import Path

import numpy as np
import pytest

from kipe.cli import main
from kipe.options import (
    EstimationOptions,
    ForwardSolverOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    OutputOptions,
    ParameterPrior,
    ParametersOptions,
    StudyOptions,
    dump_study,
)
from kipe.plot import PlotError, plot_histories, read_history, render_histories

pytest.importorskip("matplotlib")


def _study(tmp_path: Path, names: tuple[str, ...] = ("a", "c")) -> str:
    """Write and run a small FitzHugh-Nagumo estimation; return the study file."""
    study = StudyOptions(
        output=OutputOptions(path=str(tmp_path / "results")),
        forward_solver=ForwardSolverOptions(factory="kipe.examples.fitzhugh_nagumo:Solver"),
        parameters=ParametersOptions(
            reparameterization="log",
            priors={name: ParameterPrior(mean=0.25, relative_stddev=0.1) for name in names},
        ),
        measurements={
            "v": MeasurementOptions(
                fields=["v"],
                times=[0.5, 1.0],
                data=NumpyDataOptions(type="numpy", path=str(tmp_path / "v.npz")),
                noise=NoiseOptions(stddev=0.05, seed=0),
            )
        },
        estimation=EstimationOptions(iterations=2),
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "study.yaml"
    dump_study(study, path)
    assert main(["synthesis", str(path)]) == 0
    assert main(["estimation", str(path)]) == 0
    return str(path)


def _study_without_estimation(tmp_path: Path) -> Path:
    """Write a study file without parameters or measurements; return its path."""
    study = StudyOptions(
        output=OutputOptions(path=str(tmp_path / "results")),
        forward_solver=ForwardSolverOptions(factory="kipe.examples.fitzhugh_nagumo:Solver"),
    )
    path = tmp_path / "study.yaml"
    dump_study(study, path)
    return path


def test_read_history_matches_npz(tmp_path):
    _study(tmp_path)
    history = read_history(tmp_path / "results" / "estimation" / "history.csv")
    npz = np.load(tmp_path / "results" / "estimation" / "history.npz")

    assert history.names == ["a", "c"]
    np.testing.assert_array_equal(history.iteration, npz["iteration"])
    for key in ["parameters", "lower", "upper"]:
        np.testing.assert_allclose(getattr(history, key), npz[key])


def test_plot_saves_figure(tmp_path):
    study = _study(tmp_path)
    figure = tmp_path / "history.png"
    assert main(["plot", study, "--truth", "--save", str(figure)]) == 0
    assert figure.stat().st_size > 0


def test_plot_compares_studies(tmp_path):
    first = _study(tmp_path / "first")
    second = _study(tmp_path / "second")
    figure = tmp_path / "compare.png"
    assert main(["plot", first, second, "--save", str(figure)]) == 0
    assert figure.exists()


def test_plot_without_history(tmp_path, capsys):
    path = _study_without_estimation(tmp_path)
    assert main(["plot", str(path), "--save", str(tmp_path / "x.png")]) == 1
    assert "no estimation history" in capsys.readouterr().err


def test_plot_rejects_different_parameters(tmp_path, capsys):
    first = _study(tmp_path / "first", names=("a", "c"))
    second = _study(tmp_path / "second", names=("a",))
    assert main(["plot", first, second, "--save", str(tmp_path / "x.png")]) == 1
    assert "estimate different parameters" in capsys.readouterr().err


def test_read_history_without_steps(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("iteration,time,a,lower_a,upper_a,theta_a,stddev_theta_a\n")
    with pytest.raises(PlotError, match="no steps yet"):
        read_history(path)


def test_read_history_skips_partial_row(tmp_path):
    """The estimation may be writing the last row while the history is read."""
    path = tmp_path / "history.csv"
    header = "iteration,time,a,lower_a,upper_a,theta_a,stddev_theta_a\n"
    path.write_text(header + "0,0.0,1,0.9,1.1,0,0.1\n0,0.5,1.2,1.0")
    np.testing.assert_array_equal(read_history(path).parameters, [[1.0]])


def test_plot_terminal(tmp_path, capsys):
    study = _study(tmp_path)
    assert main(["plot", study, "--terminal", "--truth"]) == 0
    out = capsys.readouterr().out
    assert "a = " in out
    assert "c = " in out
    assert "truth 3" in out  # c of the FitzHugh-Nagumo solver
    assert "step of the current iteration" in out
    assert out.rstrip().splitlines()[-3:-1] == [f"━━ {study}", "━━ truth"]  # legend
    assert "1σ range" in out.splitlines()[-1]  # key to the lines
    assert "\x1b[" not in out  # no colors when not printing to a terminal


def test_plot_terminal_compares_studies(tmp_path, capsys):
    first = _study(tmp_path / "first")
    second = _study(tmp_path / "second")
    assert main(["plot", first, second, "--terminal"]) == 0
    out = capsys.readouterr().out
    assert first in out  # legend
    assert second in out


def test_plot_watch(tmp_path, capsys, monkeypatch):
    """One redraw, then Ctrl+C, which ends the command normally."""

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    study = _study(tmp_path)
    # after the run: this patches time.sleep everywhere, e.g., in subprocess.run with a timeout
    monkeypatch.setattr("kipe.plot.time.sleep", interrupt)
    assert main(["plot", study, "--watch"]) == 0
    assert "a = " in capsys.readouterr().out


def test_plot_watch_waits_for_history(tmp_path, capsys, monkeypatch):
    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("kipe.plot.time.sleep", interrupt)
    path = _study_without_estimation(tmp_path)
    assert main(["plot", str(path), "--watch"]) == 0
    assert "no estimation history" in capsys.readouterr().out


@pytest.mark.parametrize("option", ["--terminal", "--watch"])
def test_plot_terminal_excludes_save(tmp_path, capsys, option):
    with pytest.raises(SystemExit):
        main(["plot", "study.yaml", option, "--save", str(tmp_path / "x.png")])
    assert "--save writes a matplotlib figure" in capsys.readouterr().err


def test_plot_watch_with_terminal(tmp_path, capsys, monkeypatch):
    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    study = _study(tmp_path)
    # after the run: this patches time.sleep everywhere, e.g., in subprocess.run with a timeout
    monkeypatch.setattr("kipe.plot.time.sleep", interrupt)
    assert main(["plot", study, "--terminal", "--watch"]) == 0
    assert "a = " in capsys.readouterr().out


def test_render_histories_keeps_terminal_background(tmp_path):
    """Only foreground colors: a background color clashes with the terminal's theme."""
    _study(tmp_path)
    history = read_history(tmp_path / "results" / "estimation" / "history.csv")
    text = render_histories({"study": history}, truth={"a": 0.2, "c": 3.0})
    assert "\x1b[38;" in text
    assert "\x1b[48;" not in text


def test_plot_terminal_columns(tmp_path, capsys):
    """--cpr puts several plots in a row, so the output is shorter than one plot per row."""
    study = _study(tmp_path)
    assert main(["plot", study, "--terminal"]) == 0
    one = capsys.readouterr().out
    assert main(["plot", study, "--terminal", "--cpr", "2"]) == 0
    two = capsys.readouterr().out
    assert len(two.splitlines()) < len(one.splitlines())
    assert "a = " in two
    assert "c = " in two


def test_plot_terminal_unused_cells_are_empty(tmp_path, capsys):
    """Three parameters in rows of two: the fourth cell has no frame."""
    study = _study(tmp_path, names=("a", "b", "c"))
    assert main(["plot", study, "--terminal", "--cpr", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("┌") == 3


def test_plot_terminal_current_iteration_by_default(tmp_path, capsys):
    """Only the current outer iteration, unless --all: the key then has no iteration lines."""
    study = _study(tmp_path)
    assert main(["plot", study, "--terminal"]) == 0
    current = capsys.readouterr().out
    assert main(["plot", study, "--terminal", "--all"]) == 0
    everything = capsys.readouterr().out
    assert "step of the current iteration" in current
    assert "outer iterations" not in current
    assert "outer iterations" in everything


def test_read_history_reads_the_plan(tmp_path):
    """The plan from the run's provenance: 2 iterations of 2 times and the initial state."""
    _study(tmp_path)
    history = read_history(tmp_path / "results" / "estimation" / "history.csv")
    assert (history.steps_per_iteration, history.iterations) == (3, 2)
    assert history.steps == len(history.iteration) == 6
    assert history.iteration_starts() == [3]


def test_running_estimation_spans_the_whole_run(tmp_path):
    """While the estimation runs, the axis and the iteration starts cover the whole run."""
    _study(tmp_path)
    path = tmp_path / "results" / "estimation" / "history.csv"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))  # 2 steps
    history = read_history(path)
    assert len(history.iteration) == 2
    assert history.steps == 6
    assert history.iteration_starts() == [3]

    from matplotlib.figure import Figure

    figure = Figure()
    plot_histories(figure, {"study": history})
    assert figure.axes[0].get_xlim() == (0, 5)


def test_without_plan_the_axis_follows_the_history(tmp_path):
    """Older runs have no run.yaml: the steps and iteration starts come from the history."""
    _study(tmp_path)
    (tmp_path / "results" / "estimation" / "run.yaml").unlink()
    history = read_history(tmp_path / "results" / "estimation" / "history.csv")
    assert history.steps_per_iteration is None
    assert history.steps == 6
    assert history.iteration_starts() == [3]
