"""SciPy Acceleration integration, evaluation budgets and ROUKF restart behavior."""

import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI

import numpy as np
import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from kipe._acceleration import InvalidAccelerationPoint, run_acceleration
from kipe.estimation import Estimation, estimate
from kipe.measurements.data import write_numpy
from kipe.options import (
    AccelerationOptions,
    EstimationOptions,
    ForwardSolverOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    OutputOptions,
    ParameterPrior,
    ParametersOptions,
    StudyFileError,
    StudyOptions,
)


def test_scalar_linear_map_and_evaluation_budget():
    """Accelerate a linear map and perform exactly the requested number of evaluations."""
    inputs = []

    def evaluate(x):
        inputs.append(x.copy())
        return 0.9 * x + 0.1

    run_acceleration(
        evaluate, np.array([0.0]), AccelerationOptions(memory=1, regularization=0), 3
    )
    np.testing.assert_allclose(np.asarray(inputs).ravel(), [0, 0.1, 1], atol=1e-13)


@pytest.mark.parametrize("evaluations", [1, 5])
def test_exact_fixed_point_keeps_requested_pass_count(evaluations):
    """Honor the pass count even when SciPy detects an exact fixed point immediately."""
    inputs = []

    def evaluate(x):
        inputs.append(x.copy())
        return x.copy()

    run_acceleration(evaluate, np.array([2.0]), AccelerationOptions(), evaluations)
    np.testing.assert_array_equal(inputs, np.full((evaluations, 1), 2.0))


def test_invalid_extrapolation_restarts_at_last_plain_output():
    """Discard an invalid extrapolation and restart SciPy without evaluating the map there."""
    inputs = []
    rejected = []

    def evaluate(x):
        if x[0] > 0.5:
            rejected.append(x.copy())
            raise InvalidAccelerationPoint
        inputs.append(x.copy())
        return 0.9 * x + 0.1

    run_acceleration(
        evaluate, np.array([0.0]), AccelerationOptions(memory=1, regularization=0), 4
    )
    assert len(rejected) == 1
    np.testing.assert_allclose(np.asarray(inputs).ravel(), [0, 0.1, 0.19, 0.271])


def test_invalid_fallback_is_not_retried_indefinitely():
    """Stop if the ordinary map output cannot be used as the restart input either."""
    inputs = []

    def evaluate(x):
        inputs.append(x.copy())
        if x[0] >= 1:
            raise InvalidAccelerationPoint
        return np.ones(1)

    with pytest.raises(InvalidAccelerationPoint):
        run_acceleration(evaluate, np.zeros(1), AccelerationOptions(), 3)
    np.testing.assert_array_equal(np.asarray(inputs).ravel(), [0, 1, 1])


def test_constant_residual_preserves_plain_updates():
    """Handle dependent residual differences without a singular-system failure."""
    inputs = []

    def evaluate(x):
        inputs.append(x.copy())
        return x + 1

    run_acceleration(evaluate, np.zeros(1), AccelerationOptions(regularization=0), 4)
    np.testing.assert_allclose(np.asarray(inputs).ravel(), [0, 1, 2, 3])


@pytest.mark.parametrize("initial,output", [(np.inf, 1.0), (0.0, np.nan)])
def test_nonfinite_vectors_are_rejected(initial, output):
    """Reject invalid initial inputs and nonfinite map outputs explicitly."""
    expected = InvalidAccelerationPoint if np.isinf(initial) else ValueError
    with pytest.raises(expected):
        run_acceleration(
            lambda x: np.array([output]), np.array([initial]), AccelerationOptions(), 2
        )


def test_map_failure_is_not_hidden():
    """Propagate map errors instead of treating them as invalid extrapolations."""

    def evaluate(x):
        raise ValueError("forward solver failed")

    with pytest.raises(ValueError, match="forward solver failed"):
        run_acceleration(evaluate, np.zeros(1), AccelerationOptions(), 2)


@pytest.mark.parametrize("error", [ValueError("underflow"), OverflowError("overflow")])
def test_parameter_validation_errors_are_invalid_acceleration_points(monkeypatch, error):
    """Translate invalid physical parameters so acceleration can restart."""
    estimation = object.__new__(Estimation)
    estimation._log_setup = lambda: None
    estimation._history = SimpleNamespace(steps=[])
    estimation._completed_iterations = 0
    estimation._iterations = 2
    estimation._acceleration = AccelerationOptions()

    def validate_physical(theta):
        raise error

    estimation._parameterization = SimpleNamespace(
        anchor_theta=lambda: np.zeros(1), validate_physical=validate_physical
    )
    monkeypatch.setattr(
        "kipe.estimation.run_acceleration",
        lambda evaluate, *_: evaluate(np.zeros(1)),
    )

    with pytest.raises(InvalidAccelerationPoint) as exc_info:
        estimation.run()
    assert exc_info.value.__cause__ is error


def test_linear_least_squares_matches_scipy():
    """Match SciPy's input sequence and improve the linear least-squares estimate."""
    from scipy.optimize import NoConvergence, anderson

    H = np.vander(np.linspace(0, 1, 80), 3, increasing=True)
    y = H @ np.ones(3) + 0.5 * np.random.default_rng(4).standard_normal(80)
    fim = H.T @ H / 0.5**2
    rhs = H.T @ y / 0.5**2

    def G(x):
        return np.linalg.solve(fim + 10 * np.eye(3), rhs + 10 * x)

    inputs = []
    outputs = []

    def evaluate(x):
        inputs.append(x.copy())
        outputs.append(G(x))
        return outputs[-1]

    run_acceleration(evaluate, np.full(3, 2.0), AccelerationOptions(), 9)
    reference = []
    try:
        anderson(
            lambda x: G(x) - x,
            np.full(3, 2.0),
            alpha=1,
            M=5,
            w0=0.01,
            maxiter=8,
            f_tol=1e-16,
            line_search=None,
            callback=lambda x, f: reference.append(x.copy()),
        )
    except NoConvergence:
        pass
    np.testing.assert_allclose(inputs[1:], reference, atol=1e-8, rtol=1e-8)
    plain = np.full(3, 2.0)
    for _ in range(9):
        plain = G(plain)
    mle = np.linalg.solve(fim, rhs)
    assert np.linalg.norm(outputs[-1] - mle) < np.linalg.norm(plain - mle) * 0.1


@pytest.mark.parametrize(
    "options", [{"memory": 0}, {"regularization": -1}, {"regularization": np.inf}]
)
def test_invalid_options(options):
    """Reject unsupported memory and regularization values."""
    with pytest.raises(ValidationError):
        AccelerationOptions(**options)


class _LinearSolver:
    H = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    def __init__(self):
        self.state_spec = {"y": SimpleNamespace(local_size=3, kind="replicated")}

    def nominal_parameters(self):
        return {"a": 0.6, "b": -0.4}

    def initial_state(self):
        return 0.0, {"y": np.zeros(3)}

    def propagate(self, t0, t1, state, parameters):
        return {"y": self.H @ np.array([parameters["a"], parameters["b"]])}


@pytest.mark.parametrize("reparameterization", ["additive", "log", "multiplicative"])
@pytest.mark.parametrize(
    "completed,interrupted,legacy_key",
    [
        (1, False, None),
        (2, False, None),
        (4, False, None),
        (2, True, "anderson"),
        (2, True, "aceleration"),
    ],
)
def test_roukf_acceleration_and_resume(
    tmp_path, monkeypatch, reparameterization, completed, interrupted, legacy_key
):
    """Resume the same estimates and covariances without rerunning completed filter passes."""
    comm = MPI.COMM_WORLD
    tmp_path = Path(comm.bcast(str(tmp_path), root=0))
    module = types.ModuleType("aa_solver")
    module.Solver = _LinearSolver
    monkeypatch.setitem(sys.modules, "aa_solver", module)
    if comm.rank == 0:
        write_numpy(
            tmp_path / "y.npz", np.array([1.0]), np.array([_LinearSolver.H @ [0.6, -0.4]])
        )
    comm.barrier()
    study = StudyOptions(
        output=OutputOptions(path=str(tmp_path / "accelerated")),
        forward_solver=ForwardSolverOptions(factory="aa_solver:Solver"),
        parameters=ParametersOptions(
            reparameterization=reparameterization,
            priors={
                name: ParameterPrior(
                    mean=-2 if reparameterization == "log" and name == "b" else 2,
                    **(
                        {"stddev": 0.1}
                        if reparameterization == "additive"
                        else {"relative_stddev": 0.1}
                    ),
                )
                for name in ["a", "b"]
            },
        ),
        measurements={
            "y": MeasurementOptions(
                fields=["y"],
                times=[1.0],
                data=NumpyDataOptions(type="numpy", path=str(tmp_path / "y.npz")),
                noise=NoiseOptions(stddev=0.5),
            )
        },
        estimation=EstimationOptions(
            iterations=5, acceleration=AccelerationOptions(memory=2, regularization=0)
        ),
    )
    result = estimate(study)
    if reparameterization in {"additive", "multiplicative"}:
        np.testing.assert_allclose(list(result.values()), [0.6, -0.4], atol=1e-8)
    if reparameterization == "log":
        assert result["a"] > 0 and result["b"] < 0
    plain = replace(
        study,
        output=OutputOptions(path=str(tmp_path / "plain")),
        estimation=EstimationOptions(iterations=5),
    )
    plain_result = estimate(plain)
    if reparameterization in {"additive", "multiplicative"}:
        # With fixed anchors, each linear pass has the same prior precision.
        sigma = 0.1 if reparameterization == "additive" else 2 * 0.1
        precision = np.eye(2) / sigma**2
        information = _LinearSolver.H.T @ _LinearSolver.H / 0.5**2
        rhs = information @ np.array([0.6, -0.4])
        expected = np.full(2, 2.0)
        for _ in range(5):
            expected = np.linalg.solve(information + precision, rhs + precision @ expected)
        np.testing.assert_allclose(list(plain_result.values()), expected, atol=1e-12)
    assert np.linalg.norm(np.array(list(result.values())) - [0.6, -0.4]) < np.linalg.norm(
        np.array(list(plain_result.values())) - [0.6, -0.4]
    )
    resumed = replace(study, output=OutputOptions(path=str(tmp_path / "resumed")))
    propagate = _LinearSolver.propagate
    if interrupted:
        completed_calls = 0

        def interrupted_propagate(self, *args, **kwargs):
            nonlocal completed_calls
            if completed_calls == 3 * completed:
                raise RuntimeError("interrupted forward solve")
            completed_calls += 1
            return propagate(self, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(_LinearSolver, "propagate", interrupted_propagate)
            with pytest.raises(RuntimeError, match="interrupted forward solve"):
                estimate(resumed)
    else:
        estimate(replace(resumed, estimation=replace(study.estimation, iterations=completed)))
    comm.barrier()
    if interrupted:
        # Existing archived studies and run metadata must remain usable after the rename.
        if comm.rank == 0:
            yaml = YAML()
            saved_study = tmp_path / "resumed/estimation/study.yaml"
            saved_run = tmp_path / "resumed/estimation/run.yaml"
            for path in [saved_study, saved_run]:
                saved = yaml.load(path)
                settings = saved["estimation"] if path == saved_study else saved
                settings[legacy_key] = settings.pop("acceleration")
                with path.open("w") as stream:
                    yaml.dump(saved, stream)
        comm.barrier()
    calls = []

    def counted_propagate(self, *args, **kwargs):
        calls.append(1)
        return propagate(self, *args, **kwargs)

    monkeypatch.setattr(_LinearSolver, "propagate", counted_propagate)
    continued = estimate(resumed, resume=True)
    # The default simplex stencil has three sigma points for these two parameters.
    assert len(calls) == 3 * (5 - completed)
    np.testing.assert_allclose(list(continued.values()), list(result.values()), atol=1e-12)
    comm.barrier()
    for name in ["theta", "parameters", "P_theta"]:
        with (
            np.load(tmp_path / "accelerated/estimation/history.npz") as full,
            np.load(tmp_path / "resumed/estimation/history.npz") as part,
        ):
            np.testing.assert_allclose(full[name], part[name], atol=1e-12)
            if name == "theta":
                physical = full["parameters"]
                if reparameterization == "additive":
                    expected_theta = physical - 2
                elif reparameterization == "multiplicative":
                    expected_theta = physical / 2
                else:
                    expected_theta = np.log(physical / np.array([2, -2]))
                np.testing.assert_allclose(full["theta"], expected_theta, atol=1e-12)
    with pytest.raises(StudyFileError, match=r"estimation\.acceleration"):
        estimate(
            replace(
                resumed, estimation=replace(study.estimation, iterations=6, acceleration=None)
            ),
            resume=True,
        )
    comm.barrier()

    if comm.rank == 0:
        yaml = YAML()
        path = tmp_path / "resumed/estimation/run.yaml"
        run = yaml.load(path)
        run["acceleration"]["scipy_version"] = "different-version"
        with path.open("w") as stream:
            yaml.dump(run, stream)
    comm.barrier()
    with pytest.raises(StudyFileError, match="acceleration backend or SciPy version"):
        estimate(
            replace(resumed, estimation=replace(study.estimation, iterations=6)), resume=True
        )
    if comm.rank == 0:
        del run["acceleration"]
        with path.open("w") as stream:
            yaml.dump(run, stream)
    comm.barrier()
    with pytest.raises(StudyFileError, match="acceleration backend or SciPy version"):
        estimate(
            replace(resumed, estimation=replace(study.estimation, iterations=6)), resume=True
        )
    comm.barrier()
