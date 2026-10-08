"""``kipe plot``: reading the estimation history and plotting it."""

from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from kipe.cli import main
from kipe.plot import PlotError, plot_histories, read_history, render_histories

pytest.importorskip("matplotlib")


def _study(tmp_path: Path, select: tuple[str, ...] = ("a", "c")) -> str:
    """Write and run a small FitzHugh-Nagumo estimation; return the study file."""
    study = {
        "output": {"path": str(tmp_path / "results")},
        "forward_solver": {"factory": "kipe.examples.fitzhugh_nagumo:Solver"},
        "parameters": {
            "reparameterization": "log",
            "select": {name: {"initial": 0.25, "relative_stddev": 0.1} for name in select},
        },
        "measurements": {
            "v": {
                "fields": ["v"],
                "times": [0.5, 1.0],
                "data": {"type": "numpy", "path": str(tmp_path / "v.npz")},
                "noise": {"stddev": 0.05, "seed": 0},
            }
        },
        "estimation": {"iterations": 2},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "study.yaml"
    YAML().dump(study, path)
    assert main(["synthesis", str(path)]) == 0
    assert main(["estimation", str(path)]) == 0
    return str(path)


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
    path = tmp_path / "study.yaml"
    YAML().dump(
        {
            "output": {"path": str(tmp_path / "results")},
            "forward_solver": {"factory": "kipe.examples.fitzhugh_nagumo:Solver"},
        },
        path,
    )
    assert main(["plot", str(path), "--save", str(tmp_path / "x.png")]) == 1
    assert "no estimation history" in capsys.readouterr().err


def test_plot_rejects_different_parameters(tmp_path, capsys):
    first = _study(tmp_path / "first", select=("a", "c"))
    second = _study(tmp_path / "second", select=("a",))
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
    assert "assimilation step" in out
    assert out.rstrip().splitlines()[-2:] == [f"━━ {study}", "━━ truth"]  # legend
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
    path = tmp_path / "study.yaml"
    YAML().dump(
        {
            "output": {"path": str(tmp_path / "results")},
            "forward_solver": {"factory": "kipe.examples.fitzhugh_nagumo:Solver"},
        },
        path,
    )
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
