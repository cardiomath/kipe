"""``kipe estimation``: the FitzHugh-Nagumo twin experiment and example, history and errors."""

import hashlib
import io
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from kipe._yaml import write_yaml
from kipe.cli import main
from kipe.estimation import Estimation
from kipe.examples.fitzhugh_nagumo import Solver
from kipe.measurements import (
    DifferenceModel,
    Measurement,
    ObservationOperator,
    read_numpy,
    write_numpy,
)
from kipe.options import (
    EstimationOptions,
    ForwardSolverOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    OutputOptions,
    ParameterPrior,
    ParametersOptions,
    SigmaPoints,
    StudyOptions,
    SynthesisOptions,
    TimeRange,
    dump_study,
)
from kipe.parameters import build_parameterization
from kipe.sampling import ArraySampler

TRUE = {"a": 0.2, "b": 0.2, "c": 3.0}  # nominal values of the solver: generate the data
INITIAL = {"a": 0.3, "b": 0.3, "c": 2.0}  # deliberately wrong initial estimate
TIMES = TimeRange(start=0.5, stop=20.0, step=0.5)
EXAMPLE = Path(__file__).parents[1] / "examples" / "fitzhugh_nagumo" / "study.yaml"


def _options(
    tmp_path: Path,
    initial: dict[str, float] = INITIAL,
    *,
    sigma_points: SigmaPoints = "simplex",
    iterations: int = 1,
    times: TimeRange | list[float] = TIMES,
    parameters: bool = True,
    relative_stddev: float = 0.2,
    noise: float = 0.05,
    realizations: int = 1,
) -> StudyOptions:
    """Return the options of a FitzHugh-Nagumo study measuring v and w."""
    data = tmp_path / ("r{realization}" if realizations > 1 else "")
    parameters_options = ParametersOptions(
        reparameterization="log",
        select={
            name: ParameterPrior(initial=value, relative_stddev=relative_stddev)
            for name, value in initial.items()
        },
    )
    return StudyOptions(
        output=OutputOptions(path=str(tmp_path / "results")),
        forward_solver=ForwardSolverOptions(
            factory="kipe.examples.fitzhugh_nagumo:Solver", arguments={"dt": 0.05}
        ),
        parameters=parameters_options if parameters else None,
        measurements={
            field: MeasurementOptions(
                fields=[field],
                times=times,
                data=NumpyDataOptions(type="numpy", path=str(data / f"{field}.npz")),
                noise=NoiseOptions(stddev=noise, seed=seed),
            )
            for field, seed in [("v", 0), ("w", 1)]
        },
        synthesis=SynthesisOptions(realizations=realizations),
        estimation=EstimationOptions(sigma_points=sigma_points, iterations=iterations),
    )


def _study(tmp_path: Path, *args, **kwargs) -> Path:
    """Write the study file of :func:`_options` (same arguments); return its path."""
    path = tmp_path / "study.yaml"
    dump_study(_options(tmp_path, *args, **kwargs), path)
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
    study = str(_study(tmp_path, TRUE, sigma_points="unique"))
    assert main(["synthesis", study]) == 0
    assert main(["estimation", study]) == 0

    parameters = _history(tmp_path)["parameters"]
    np.testing.assert_allclose(parameters, np.tile(list(TRUE.values()), (len(parameters), 1)))


def test_iterations_restart_from_estimate(tmp_path):
    """The second pass starts from the first pass's estimate, with the initial uncertainty."""
    study = str(_study(tmp_path, iterations=2, times=TimeRange(start=0.5, stop=5.0, step=0.5)))
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
    options = _options(tmp_path, times=[0.5, 1.0])
    assert options.measurements is not None
    w = replace(options.measurements["w"], times=[0.5])
    study = tmp_path / "study.yaml"
    dump_study(replace(options, measurements={**options.measurements, "w": w}), study)
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

    operator = ObservationOperator(
        ["v"], ArraySampler(["v"], solver.state_spec), DifferenceModel()
    )
    measurement = Measurement("v", operator, times, np.array(values), stddev=0.1)
    parameters = ParametersOptions(
        reparameterization="log", select={"c": ParameterPrior(relative_stddev=0.5)}
    )
    parameterization = build_parameterization(parameters, solver.nominal_parameters())

    estimation = Estimation(
        solver,
        parameterization,
        [measurement],
        EstimationOptions(sigma_points="unique"),
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


def test_log_output(tmp_path, kipe_caplog):
    """Setup summary, one table row per assimilation step, estimate with its 1σ range."""
    study = str(_study(tmp_path, times=[0.5, 1.0]))
    assert main(["synthesis", study]) == 0
    kipe_caplog.clear()
    with kipe_caplog.at_level(logging.INFO, logger="kipe"):
        assert main(["estimation", study]) == 0

    messages = [record.getMessage() for record in kipe_caplog.records]
    assert messages[0].split() == ["study", str(Path(study).resolve())]
    assert messages[1].startswith("kipe ")  # versions, host, ranks (kipe.provenance)
    assert messages[2] == "ROUKF estimation"
    assert any(m.endswith("simplex (4 sigma points)") for m in messages)
    header = next(m for m in messages if "innovation" in m)
    assert header.split() == ["step", "time", "a", "b", "c", "innovation"]
    rows = [m.split() for m in messages if m.split()[:1] in (["1"], ["2"])]
    assert [row[:2] for row in rows] == [["1", "0.5"], ["2", "1"]]
    estimate = messages.index(next(m for m in messages if m.startswith("  estimate after")))
    assert messages[estimate + 1].split() == ["parameter", "estimate", "1σ", "range"]
    assert messages[estimate + 2].split()[0] == "a"


def test_log_level_option_overrides_study_file(tmp_path, kipe_caplog):
    """``--log-level debug`` shows the per-sigma-point details and where they were logged."""
    study = str(_study(tmp_path, times=[0.5]))
    assert main(["synthesis", study]) == 0
    kipe_caplog.clear()
    assert main(["estimation", study, "--log-level", "debug"]) == 0
    record = next(r for r in kipe_caplog.records if "sigma point 1/4" in r.getMessage())

    # at the debug level, the message shows where it was logged
    handlers = logging.getLogger("kipe").handlers
    handler = next(h for h in handlers if h.name == "kipe.console")
    assert handler.format(record).startswith("kipe: estimation:_propagate_and_observe:")


def test_log_stays_out_of_the_root_logger(tmp_path):
    """kipe's messages don't reach the root logger's handlers.

    An imported library may give the root logger a handler on stderr (UFL does, with
    ``logging.debug``); through it, kipe's messages would bypass ``--log-level``.
    """
    study = str(_study(tmp_path, times=[0.5]))
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)  # as logging.basicConfig() adds, on stderr
    logging.getLogger().addHandler(handler)
    try:
        assert main(["synthesis", study]) == 0
        assert main(["estimation", study, "--log-level", "warning"]) == 0
    finally:
        logging.getLogger().removeHandler(handler)
    assert stream.getvalue() == ""


def test_solver_failure_reports_the_sigma_point(tmp_path, kipe_caplog, monkeypatch):
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
    kipe_caplog.clear()
    with pytest.raises(RuntimeError, match="Newton did not converge"):
        main(["estimation", study])

    error = next(r for r in kipe_caplog.records if r.levelno == logging.ERROR).getMessage()
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


def test_realizations(tmp_path, capsys):
    """Without --realization, estimation runs each realization, into its own directory; with
    it, only that one. The plot needs it."""
    study = str(_study(tmp_path, times=[0.5, 1.0, 1.5, 2.0], realizations=3))
    assert main(["synthesis", study, "--log-level", "warning"]) == 0
    assert main(["estimation", study, "--log-level", "warning"]) == 0

    output = tmp_path / "results" / "estimation"
    estimates = [np.load(output / f"r{r}" / "history.npz")["parameters"][-1] for r in range(3)]
    assert not np.allclose(estimates[0], estimates[1])  # different noise
    assert not (output / "history.npz").exists()

    shutil.rmtree(output / "r1")
    assert main(["estimation", study, "--realization", "1", "--log-level", "warning"]) == 0
    rerun = np.load(output / "r1" / "history.npz")["parameters"][-1]
    np.testing.assert_array_equal(rerun, estimates[1])  # the filter is deterministic
    assert "realization: 1" in (output / "r1" / "run.yaml").read_text()

    assert main(["estimation", study, "--realization", "3"]) == 1
    assert "realization 3: the study has realizations 0 to 2" in capsys.readouterr().err

    assert main(["plot", study, "--terminal"]) == 1
    assert "choose one with --realization" in capsys.readouterr().err
    assert main(["plot", study, "--terminal", "--realization", "2"]) == 0


SHORT = [0.5, 1.0, 1.5, 2.0]  # 4 steps of 4 sigma points (simplex, 3 parameters)


def _interrupt(monkeypatch: pytest.MonkeyPatch, after: int) -> None:
    """Make the solver fail after the given number of propagations, as a killed job would."""
    propagate = Solver.propagate
    calls = 0

    def failing(self, *args):
        nonlocal calls
        calls += 1
        if calls > after:
            raise RuntimeError("interrupted")
        return propagate(self, *args)

    monkeypatch.setattr(Solver, "propagate", failing)


def _estimation(path: Path) -> Path:
    return path / "results" / "estimation"


def _assert_same_history(a: Path, b: Path) -> None:
    """Bit for bit: the same filter, from the same estimate, on the same data."""
    expected, actual = np.load(a / "history.npz"), np.load(b / "history.npz")
    assert actual.files == expected.files
    for key in expected.files:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert (b / "history.csv").read_text() == (a / "history.csv").read_text()


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    """An estimation with 2 iterations, without interruption; --resume without a history
    starts from the beginning."""
    path = tmp_path / "reference"
    path.mkdir()
    study = str(_study(path, times=SHORT, iterations=2))
    assert main(["synthesis", study, "--log-level", "warning"]) == 0
    assert main(["estimation", study, "--resume", "--log-level", "warning"]) == 0
    return _estimation(path)


def test_resume_an_interrupted_estimation(tmp_path, monkeypatch, reference):
    path = tmp_path / "interrupted"
    path.mkdir()
    study = str(_study(path, times=SHORT, iterations=2))
    assert main(["synthesis", study, "--log-level", "warning"]) == 0
    with monkeypatch.context() as patch:
        _interrupt(patch, after=16 + 5)  # in the second iteration
        with pytest.raises(RuntimeError, match="interrupted"):
            main(["estimation", study, "--log-level", "warning"])
    output = _estimation(path)
    assert set(np.load(output / "history.npz")["iteration"]) == {0}  # the completed one

    assert main(["estimation", study, "--resume", "--log-level", "warning"]) == 0
    _assert_same_history(reference, output)
    assert "resumed_after_iteration: 1" in (output / "run.yaml").read_text()
    log = (output / "kipe.log").read_text()
    assert "forward solver failed" in log  # the interrupted run's log is kept
    assert "resumed after iteration 1/2" in log


def test_resume_with_more_iterations(tmp_path, reference):
    """A complete estimation continues when the study asks for more iterations."""
    study = str(_study(tmp_path, times=SHORT, iterations=1))
    assert main(["synthesis", study, "--log-level", "warning"]) == 0
    assert main(["estimation", study, "--log-level", "warning"]) == 0

    study = str(_study(tmp_path, times=SHORT, iterations=2))
    assert main(["estimation", study, "--resume", "--log-level", "warning"]) == 0
    _assert_same_history(reference, _estimation(tmp_path))


def test_resume_a_complete_estimation(tmp_path, reference):
    """Nothing runs, nothing is written but the log."""
    before = {f.name: f.read_bytes() for f in reference.iterdir()}
    study = str(_study(tmp_path / "reference", times=SHORT, iterations=2))

    assert main(["estimation", study, "--resume"]) == 0
    after = {f.name: f.read_bytes() for f in reference.iterdir()}
    assert b"complete after 2 iterations, nothing to resume" in after.pop("kipe.log")
    before.pop("kipe.log")
    assert after == before


def test_resume_a_changed_study(tmp_path, capsys, reference):
    study = _study(tmp_path / "reference", times=SHORT, iterations=3, relative_stddev=0.3)

    assert main(["estimation", str(study), "--resume"]) == 1
    assert "the study has changed since (parameters)" in capsys.readouterr().err


def test_run_records_the_data_files(reference):
    data = YAML(typ="safe").load(reference / "run.yaml")["data"]
    assert list(data) == ["v", "w"]
    for file in data.values():
        assert file["sha256"] == hashlib.sha256(Path(file["path"]).read_bytes()).hexdigest()


def test_resume_with_changed_data(tmp_path, capsys):
    study = str(_study(tmp_path, times=SHORT, iterations=1))
    assert main(["synthesis", study, "--log-level", "warning"]) == 0
    assert main(["estimation", study, "--log-level", "warning"]) == 0
    times, values = read_numpy(tmp_path / "w.npz")
    write_numpy(tmp_path / "w.npz", times, values + 1e-3)

    study = str(_study(tmp_path, times=SHORT, iterations=2))
    assert main(["estimation", study, "--resume"]) == 1
    message = f"the data of w ({tmp_path / 'w.npz'}) have changed since the earlier run"
    assert message in capsys.readouterr().err


def test_resume_without_recorded_data(tmp_path, kipe_caplog, reference):
    """An earlier run of an older kipe: resumed, with a warning that the data are unchecked."""
    run = YAML(typ="safe").load(reference / "run.yaml")
    del run["data"]
    write_yaml(reference / "run.yaml", run)

    study = str(_study(tmp_path / "reference", times=SHORT, iterations=3))
    assert main(["estimation", study, "--resume", "--log-level", "warning"]) == 0
    assert "cannot compare the data with the earlier run" in kipe_caplog.text
