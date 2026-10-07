"""``kipe plot``: reading the estimation history and plotting it."""

from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from kipe.cli import main
from kipe.plot import PlotError, read_history

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
