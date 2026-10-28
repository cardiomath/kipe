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
from kipe.plot import (
    History,
    PlotError,
    _Timeline,
    plot_histories,
    read_history,
    render_histories,
)

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
    np.testing.assert_allclose(history.time, npz["time"])
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
    assert "time" in out
    assert "iteration 2" in out  # the outer iterations, side by side
    assert out.rstrip().splitlines()[-2:] == [f"━━ {study}", "━━ truth"]  # legend
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


def test_read_history_reads_the_plan(tmp_path):
    """The plan from the run's provenance: 2 iterations up to the last time, 1.0."""
    _study(tmp_path)
    history = read_history(tmp_path / "results" / "estimation" / "history.csv")
    assert (history.end_time, history.iterations) == (1.0, 2)
    assert (history.start_time, history.span, history.iteration_count) == (0.0, 1.0, 2)


def test_running_estimation_spans_the_whole_run(tmp_path):
    """While the estimation runs, the axis covers the whole run: all outer iterations."""
    _study(tmp_path)
    path = tmp_path / "results" / "estimation" / "history.csv"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))  # 2 steps
    history = read_history(path)
    assert len(history.iteration) == 2
    assert (history.span, history.iteration_count) == (1.0, 2)

    from matplotlib.figure import Figure

    figure = Figure()
    plot_histories(figure, {"study": history})
    assert figure.axes[0].get_xlim() == (0.0, 2.0)


def test_without_plan_the_axis_follows_the_history(tmp_path):
    """Older runs have no run.yaml: the span and the iterations come from the history."""
    _study(tmp_path)
    path = tmp_path / "results" / "estimation" / "history.csv"
    (path.parent / "run.yaml").unlink()
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))  # 2 steps
    history = read_history(path)
    assert history.end_time is None
    assert (history.span, history.iteration_count) == (0.5, 1)


def _history(iteration: list[int], time: list[float], **plan: float) -> History:
    """Return a history of one parameter at the given steps; its values do not matter."""
    values = np.ones((len(time), 1))
    return History(
        ["a"], np.array(iteration), np.array(time), values, 0.9 * values, 1.1 * values, **plan
    )


def test_timeline_places_steps_at_their_times():
    """Irregular times, e.g., of asynchronous measurements, are placed by their time."""
    history = _history([0, 0, 0, 1, 1], [0.0, 0.3, 2.0, 0.0, 0.3], end_time=2.0, iterations=2)
    timeline = _Timeline.of({"study": history})
    assert (timeline.start, timeline.span, timeline.end) == (0.0, 2.0, 4.0)
    assert timeline.boundaries() == [2.0]

    (x0, _), (x1, rows1) = timeline.segments(history)
    np.testing.assert_allclose(x0, [0.0, 0.3, 2.0])
    np.testing.assert_allclose(x1, [2.0, 2.3])  # the time restarts in the second iteration
    np.testing.assert_array_equal(rows1, [False, False, False, True, True])


def test_timeline_ticks_restart_in_each_iteration():
    """The end of an iteration is labeled as the start of the next; the last end too."""
    timeline = _Timeline(start=0.0, span=13.0, iterations=3)
    assert timeline.ticks(bins=2) == (
        [0.0, 10.0, 13.0, 23.0, 26.0, 36.0, 39.0],
        ["0", "10", "0", "10", "0", "10", "13"],
    )
    assert timeline.labels() == (
        [6.5, 19.5, 32.5],
        ["iteration 1", "iteration 2", "iteration 3"],
    )


def test_timeline_ticks_skip_the_end_close_to_a_tick():
    timeline = _Timeline(start=0.0, span=10.2, iterations=1)
    assert timeline.ticks(bins=3) == ([0.0, 5.0, 10.0], ["0", "5", "10"])


def test_timeline_covers_all_histories():
    """Studies compared in one plot share the axis: the earliest start, the longest run."""
    short = _history([0, 0], [0.0, 1.0], end_time=1.0, iterations=3)
    long = _history([0, 0], [0.5, 2.0], end_time=4.0, iterations=2)
    assert _Timeline.of({"short": short, "long": long}) == _Timeline(0.0, 4.0, 3)


def test_render_histories_of_one_parameter():
    """plotext has no subplots in a 1 x 1 grid: the figure itself is the plot."""
    history = _history([0, 0, 1, 1], [0.0, 1.0, 0.0, 1.0], end_time=1.0, iterations=2)
    text = render_histories({"study": history}, color=False)
    assert "a = 1" in text
    assert "iteration 2" in text
