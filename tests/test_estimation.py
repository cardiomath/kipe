"""``kipe estimation``: the FitzHugh-Nagumo twin experiment and example, history and errors."""

import copy
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from kipe.cli import main
from kipe.estimation import Estimation
from kipe.examples.fitzhugh_nagumo import Solver
from kipe.measurements import DifferenceModel, Measurement, ObservationOperator, write_numpy
from kipe.options import EstimationOptions, ParameterPrior, ParametersOptions
from kipe.parameters import build_parameterization
from kipe.sampling import ArraySampler

TRUE = {"a": 0.2, "b": 0.2, "c": 3.0}  # nominal values of the solver: generate the data
INITIAL = {"a": 0.3, "b": 0.3, "c": 2.0}  # deliberately wrong initial estimate
TIMES = {"start": 0.5, "stop": 20.0, "step": 0.5}
EXAMPLE = Path(__file__).parents[1] / "examples" / "fitzhugh_nagumo" / "study.yaml"


def _study(
    tmp_path: Path,
    initial: dict[str, float] = INITIAL,
    *,
    particles: str = "simplex",
    iterations: int = 1,
    times: dict[str, float] | list[float] = TIMES,
    parameters: bool = True,
    relative_stddev: float = 0.2,
    noise: float = 0.05,
) -> Path:
    """Write a FitzHugh-Nagumo study file measuring v and w."""
    study = {
        "output": {"path": str(tmp_path / "results")},
        "forward_solver": {
            "factory": "kipe.examples.fitzhugh_nagumo:Solver",
            "arguments": {"dt": 0.05},
        },
        "measurements": {
            field: {
                "fields": [field],
                "times": copy.deepcopy(times),  # no YAML anchors for a shared object
                "data": {"type": "numpy", "path": str(tmp_path / f"{field}.npz")},
                "noise": {"stddev": noise, "seed": seed},
            }
            for field, seed in [("v", 0), ("w", 1)]
        },
        "estimation": {"particles": particles, "iterations": iterations},
    }
    if parameters:
        study["parameters"] = {
            "reparameterization": "log",
            "select": {
                name: {"initial": value, "relative_stddev": relative_stddev}
                for name, value in initial.items()
            },
        }
    path = tmp_path / "study.yaml"
    YAML().dump(study, path)  # round-trip dumper: keeps the order of the keys
    return path


def _history(tmp_path: Path):
    return np.load(tmp_path / "results" / "estimation" / "history.npz")


def test_twin_experiment_recovers_parameters(tmp_path):
    """Each estimate is within three standard deviations of the true value."""
    study = str(_study(tmp_path))
    assert main(["synthesis", study]) == 0
    assert main(["estimation", study]) == 0

    history = _history(tmp_path)
    names = list(history["names"])
    theta, P = history["theta"][-1], history["P_theta"][-1]
    theta_true = np.log2([
        TRUE[name] / INITIAL[name] for name in names
    ])  # log reparameterization
    error = np.abs(theta - theta_true)
    np.testing.assert_array_less(error, 3 * np.sqrt(np.diag(P)))


def test_iterations_converge_to_truth_on_noise_free_data(tmp_path):
    """Consistency of the whole pipeline: with (almost) noise-free data and a moderate
    sigma-point stencil, the outer iterations converge to the true parameters.

    A large stencil (relative_stddev 0.5) would converge to a biased fixed point instead: the
    mean of the propagated sigma points then deviates from the trajectory at the mean
    parameters.
    """
    study = str(_study(tmp_path, noise=1e-4, relative_stddev=0.1, iterations=5))
    assert main(["synthesis", study]) == 0
    _study(tmp_path, noise=0.05, relative_stddev=0.1, iterations=5)  # assumed noise
    assert main(["estimation", study]) == 0

    estimate = _history(tmp_path)["parameters"][-1]
    np.testing.assert_allclose(estimate, list(TRUE.values()), rtol=5e-3)


def test_unique_keeps_true_parameters(tmp_path):
    """unique: no correction; starting from the truth, the estimate stays there."""
    study = str(_study(tmp_path, TRUE, particles="unique"))
    assert main(["synthesis", study]) == 0
    assert main(["estimation", study]) == 0

    parameters = _history(tmp_path)["parameters"]
    np.testing.assert_allclose(parameters, np.tile(list(TRUE.values()), (len(parameters), 1)))


def test_iterations_restart_from_estimate(tmp_path):
    """The second pass starts from the first pass's estimate, with the initial uncertainty."""
    study = str(_study(tmp_path, iterations=2, times={"start": 0.5, "stop": 5.0, "step": 0.5}))
    assert main(["synthesis", study]) == 0
    assert main(["estimation", study]) == 0

    history = _history(tmp_path)
    first = history["iteration"] == 0
    second = history["iteration"] == 1
    assert first.sum() == second.sum() == 11  # initial estimate + 10 assimilation steps
    np.testing.assert_allclose(
        history["parameters"][second][0], history["parameters"][first][-1]
    )
    np.testing.assert_allclose(history["P_theta"][second][0], history["P_theta"][first][0])
    np.testing.assert_array_equal(history["theta"][second][0], 0.0)  # log: restart at theta = 0


def test_history_files(tmp_path):
    study = str(_study(tmp_path, times=[0.5, 1.0]))
    assert main(["synthesis", study]) == 0
    assert main(["estimation", study]) == 0

    lines = (tmp_path / "results" / "estimation" / "history.csv").read_text().splitlines()
    header = ["iteration", "time"]
    for name in "abc":
        header += [
            name,
            f"lower_{name}",
            f"upper_{name}",
            f"theta_{name}",
            f"stddev_theta_{name}",
        ]
    assert lines[0].split(",") == header
    assert len(lines) == 1 + 3  # header, initial estimate, two steps

    history = _history(tmp_path)
    np.testing.assert_allclose(history["time"], [0.0, 0.5, 1.0])
    for key in ["parameters", "lower", "upper", "theta"]:
        assert history[key].shape == (3, 3)
    assert history["P_theta"].shape == (3, 3, 3)
    # the physical 1σ range encloses the estimate
    assert np.all(history["lower"] < history["parameters"])
    assert np.all(history["parameters"] < history["upper"])


def test_times_select_from_the_data(tmp_path):
    """``times`` selects from the times in the data; the rest is not assimilated."""
    for field in ["v", "w"]:
        write_numpy(tmp_path / f"{field}.npz", np.array([0.5, 1.0, 1.5]), np.zeros((3, 1)))
    assert main(["estimation", str(_study(tmp_path, times=[1.0]))]) == 0
    np.testing.assert_allclose(_history(tmp_path)["time"], [0.0, 1.0])


@pytest.mark.parametrize(
    ("data_times", "times", "message"),
    [
        ([0.5, 1.0], [0.75], "time 0.75 is not in the data"),
        ([0.0, 0.5], [0.0, 0.5], "not after the solver's start time"),
    ],
)
def test_estimation_errors(tmp_path, capsys, data_times, times, message):
    for field in ["v", "w"]:
        write_numpy(tmp_path / f"{field}.npz", np.array(data_times), np.zeros((2, 1)))
    assert main(["estimation", str(_study(tmp_path, times=times))]) == 1
    assert message in capsys.readouterr().err


def test_estimation_needs_parameters(tmp_path, capsys):
    assert main(["estimation", str(_study(tmp_path, parameters=False))]) == 1
    assert "needs a parameters section" in capsys.readouterr().err


def test_measurements_must_share_times(tmp_path, capsys):
    write_numpy(tmp_path / "v.npz", np.array([0.5, 1.0]), np.zeros((2, 1)))
    write_numpy(tmp_path / "w.npz", np.array([0.5]), np.zeros((1, 1)))
    study = _study(tmp_path, times=[0.5, 1.0])
    yaml = YAML(typ="safe")
    options = yaml.load(study)
    options["measurements"]["w"]["times"] = [0.5]
    yaml.dump(options, study)
    assert main(["estimation", str(study)]) == 1
    assert "have different times" in capsys.readouterr().err


def test_estimation_from_python(tmp_path):
    """Estimation without a study file: from objects built in Python (here with unique, starting
    from the truth on noise-free data of v, so the estimate stays at the truth)."""
    solver = Solver()
    times = np.array([0.5, 1.0])
    t, state = solver.initial_state()
    values = []
    for t_next in times:
        state = solver.propagate(t, t_next, state, {})
        t = t_next
        values.append(state["v"].copy())

    operator = ObservationOperator(["v"], ArraySampler(["v"]), DifferenceModel())
    measurement = Measurement("v", operator, times, np.array(values), stddev=0.1)
    parameters = ParametersOptions(
        reparameterization="log", select={"c": ParameterPrior(relative_stddev=0.5)}
    )
    parameterization = build_parameterization(parameters, solver.nominal_parameters())

    estimation = Estimation(
        solver,
        parameterization,
        [measurement],
        EstimationOptions(particles="unique"),
        output=tmp_path / "estimation",
    )
    assert estimation.run() == pytest.approx({"c": 3.0})


def test_example_study(tmp_path, monkeypatch):
    """The FitzHugh-Nagumo example runs end to end and recovers the parameters accurately."""
    shutil.copy(EXAMPLE, tmp_path / "study.yaml")
    monkeypatch.chdir(tmp_path)  # paths in the study file are relative to the working directory
    assert main(["synthesis", "study.yaml"]) == 0
    assert main(["estimation", "study.yaml"]) == 0

    estimate = np.load("results/estimation/history.npz")["parameters"][-1]
    np.testing.assert_allclose(estimate, list(TRUE.values()), rtol=0.02)


def test_log_output(tmp_path, caplog):
    """Setup summary, one table row per assimilation step, estimate with its 1σ range."""
    study = str(_study(tmp_path, times=[0.5, 1.0]))
    assert main(["synthesis", study]) == 0
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="kipe"):
        assert main(["estimation", study]) == 0

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "ROUKF estimation"
    assert any(m.endswith("simplex (4 sigma points)") for m in messages)
    header = next(m for m in messages if "innovation" in m)
    assert header.split() == ["step", "time", "a", "b", "c", "innovation"]
    rows = [m.split() for m in messages if m.split()[:1] in (["1"], ["2"])]
    assert [row[:2] for row in rows] == [["1", "0.5"], ["2", "1"]]
    estimate = messages.index(next(m for m in messages if m.startswith("  estimate after")))
    assert messages[estimate + 1].split() == ["parameter", "estimate", "1σ", "range"]
    assert messages[estimate + 2].split()[0] == "a"


def test_log_level_option_overrides_study_file(tmp_path, caplog):
    """``--log-level debug`` shows the per-sigma-point details and where they were logged."""
    study = str(_study(tmp_path, times=[0.5]))
    assert main(["synthesis", study]) == 0
    caplog.clear()
    assert main(["estimation", study, "--log-level", "debug"]) == 0
    record = next(r for r in caplog.records if "sigma point 1/4" in r.getMessage())

    # at the debug level, the message shows where it was logged
    handler = logging.getLogger("kipe").handlers[0]
    assert handler.format(record).startswith("kipe: estimation:_propagate_and_observe:")


def test_solver_failure_reports_the_sigma_point(tmp_path, caplog, monkeypatch):
    """If the solver fails, the log says in which step and for which parameters."""
    study = str(_study(tmp_path, times=[0.5, 1.0]))
    assert main(["synthesis", study]) == 0

    calls = []
    propagate = Solver.propagate

    def failing_propagate(self, t0, t1, state, parameters):
        calls.append(1)
        if len(calls) == 7:  # step 2, sigma point 3 of 4
            raise RuntimeError("Newton did not converge")
        return propagate(self, t0, t1, state, parameters)

    monkeypatch.setattr(Solver, "propagate", failing_propagate)
    caplog.clear()
    with pytest.raises(RuntimeError, match="Newton did not converge"):
        main(["estimation", study])

    error = next(r for r in caplog.records if r.levelno == logging.ERROR).getMessage()
    assert error.startswith(
        "forward solver failed in step 2 (t = 0.5 → 1), sigma point 3/4: a = "
    )


def test_log_file_has_full_detail(tmp_path):
    """The log file gets every message, from the debug level on, whatever the console level."""
    study = str(_study(tmp_path, times=[0.5]))
    assert main(["synthesis", study]) == 0
    assert main(["estimation", study]) == 0  # console at the default level, info

    log = (tmp_path / "results" / "estimation" / "kipe.log").read_text()
    assert "ROUKF estimation" in log
    assert "DEBUG   estimation:_propagate_and_observe:" in log
    assert "sigma point 1/4" in log
    assert (tmp_path / "results" / "synthesis" / "kipe.log").exists()
