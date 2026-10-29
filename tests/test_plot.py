"""``kipe plot-estimation``: reading the estimation history and plotting it."""

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
from kipe.plot import PlotError, read_history, render_histories

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
    assert main(["plot-estimation", study, "--truth", "--save", str(figure)]) == 0
    assert figure.stat().st_size > 0


def test_plot_compares_studies(tmp_path):
    first = _study(tmp_path / "first")
    second = _study(tmp_path / "second")
    figure = tmp_path / "compare.png"
    assert main(["plot-estimation", first, second, "--save", str(figure)]) == 0
    assert figure.exists()


def test_plot_without_history(tmp_path, capsys):
    path = _study_without_estimation(tmp_path)
    assert main(["plot-estimation", str(path), "--save", str(tmp_path / "x.png")]) == 1
    assert "no estimation history" in capsys.readouterr().err


def test_plot_rejects_different_parameters(tmp_path, capsys):
    first = _study(tmp_path / "first", names=("a", "c"))
    second = _study(tmp_path / "second", names=("a",))
    assert main(["plot-estimation", first, second, "--save", str(tmp_path / "x.png")]) == 1
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
    assert main(["plot-estimation", study, "--terminal", "--truth"]) == 0
    out = capsys.readouterr().out
    assert "a = " in out
    assert "c = " in out
    assert "truth 3" in out  # c of the FitzHugh-Nagumo solver
    assert "starts of iterations" in out
    assert out.rstrip().splitlines()[-3:-1] == [f"━━ {study}", "━━ truth"]  # legend
    assert "━━ 1σ range" in out.splitlines()[-1]  # key to the lines
    assert "\x1b[" not in out  # no colors when not printing to a terminal


def test_plot_terminal_compares_studies(tmp_path, capsys):
    first = _study(tmp_path / "first")
    second = _study(tmp_path / "second")
    assert main(["plot-estimation", first, second, "--terminal"]) == 0
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
    assert main(["plot-estimation", study, "--watch"]) == 0
    assert "a = " in capsys.readouterr().out


def test_plot_watch_waits_for_history(tmp_path, capsys, monkeypatch):
    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("kipe.plot.time.sleep", interrupt)
    path = _study_without_estimation(tmp_path)
    assert main(["plot-estimation", str(path), "--watch"]) == 0
    assert "no estimation history" in capsys.readouterr().out


@pytest.mark.parametrize("option", ["--terminal", "--watch"])
def test_plot_terminal_excludes_save(tmp_path, capsys, option):
    figure = str(tmp_path / "x.png")
    assert main(["plot-estimation", "study.yaml", option, "--save", figure]) == 1
    assert "--save writes a matplotlib figure" in capsys.readouterr().err


def test_plot_watch_with_terminal(tmp_path, capsys, monkeypatch):
    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    study = _study(tmp_path)
    # after the run: this patches time.sleep everywhere, e.g., in subprocess.run with a timeout
    monkeypatch.setattr("kipe.plot.time.sleep", interrupt)
    assert main(["plot-estimation", study, "--terminal", "--watch"]) == 0
    assert "a = " in capsys.readouterr().out


def test_render_histories_keeps_terminal_background(tmp_path):
    """Only foreground colors: a background color clashes with the terminal's theme."""
    _study(tmp_path)
    history = read_history(tmp_path / "results" / "estimation" / "history.csv")
    text = render_histories({"study": history}, truth={"a": 0.2, "c": 3.0})
    assert "\x1b[38;" in text
    assert "\x1b[48;" not in text


def test_plot_terminal_columns(tmp_path, capsys):
    """--plots-per-row puts several plots in a row, so the output is shorter than
    one plot per row."""
    study = _study(tmp_path)
    assert main(["plot-estimation", study, "--terminal"]) == 0
    one = capsys.readouterr().out
    assert main(["plot-estimation", study, "--terminal", "--plots-per-row", "2"]) == 0
    two = capsys.readouterr().out
    assert len(two.splitlines()) < len(one.splitlines())
    assert "a = " in two
    assert "c = " in two


def test_plot_terminal_unused_cells_are_empty(tmp_path, capsys):
    """Three parameters in rows of two: the fourth cell has no frame."""
    study = _study(tmp_path, names=("a", "b", "c"))
    assert main(["plot-estimation", study, "--terminal", "--plots-per-row", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("┌") == 3


def test_plot_terminal_all_iterations_by_default(tmp_path, capsys):
    """--terminal draws all outer iterations, unless --no-all; --watch only the current one."""

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    study = _study(tmp_path)
    assert main(["plot-estimation", study, "--terminal"]) == 0
    everything = capsys.readouterr().out
    assert main(["plot-estimation", study, "--terminal", "--no-all"]) == 0
    current = capsys.readouterr().out
    assert "starts of iterations" in everything
    assert "step of iteration 2/2" in current
    assert "starts of iterations" not in current

    import pytest

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("kipe.plot.time.sleep", interrupt)
        assert main(["plot-estimation", study, "--watch"]) == 0
        watched = capsys.readouterr().out
        assert main(["plot-estimation", study, "--watch", "--all"]) == 0
        watched_all = capsys.readouterr().out
    assert "starts of iterations" not in watched
    assert "starts of iterations" in watched_all
