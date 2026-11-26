"""SciPy Anderson integration, evaluation budgets and ROUKF restart behavior."""

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

from kipe._anderson import InvalidAndersonPoint, run_anderson
from kipe.estimation import estimate
from kipe.measurements.data import write_numpy
from kipe.options import (
    AndersonOptions,
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
from kipe.parameters import build_parameterization


def test_scalar_linear_map_and_evaluation_budget():
    """Accelerate a linear map and perform exactly the requested number of evaluations."""
    inputs = []

    def evaluate(x):
        inputs.append(x.copy())
        return 0.9 * x + 0.1

    run_anderson(evaluate, np.array([0.0]), AndersonOptions(memory=1, regularization=0), 3)
    np.testing.assert_allclose(np.asarray(inputs).ravel(), [0, 0.1, 1], atol=1e-13)


@pytest.mark.parametrize("evaluations", [1, 5])
def test_exact_fixed_point_keeps_requested_pass_count(evaluations):
    """Honor the pass count even when SciPy detects an exact fixed point immediately."""
    inputs = []

    def evaluate(x):
        inputs.append(x.copy())
        return x.copy()

    run_anderson(evaluate, np.array([2.0]), AndersonOptions(), evaluations)
    np.testing.assert_array_equal(inputs, np.full((evaluations, 1), 2.0))


def test_invalid_extrapolation_restarts_at_last_plain_output():
    """Discard an invalid extrapolation and restart SciPy without evaluating the map there."""
    inputs = []
    rejected = []

    def evaluate(x):
        if x[0] > 0.5:
            rejected.append(x.copy())
            raise InvalidAndersonPoint
        inputs.append(x.copy())
        return 0.9 * x + 0.1

    run_anderson(evaluate, np.array([0.0]), AndersonOptions(memory=1, regularization=0), 4)
    assert len(rejected) == 1
    np.testing.assert_allclose(np.asarray(inputs).ravel(), [0, 0.1, 0.19, 0.271])


def test_constant_residual_preserves_plain_updates():
    """Handle dependent residual differences without a singular-system failure."""
    inputs = []

    def evaluate(x):
        inputs.append(x.copy())
        return x + 1

    run_anderson(evaluate, np.zeros(1), AndersonOptions(regularization=0), 4)
    np.testing.assert_allclose(np.asarray(inputs).ravel(), [0, 1, 2, 3])


@pytest.mark.parametrize("initial,output", [(np.inf, 1.0), (0.0, np.nan)])
def test_nonfinite_vectors_are_rejected(initial, output):
    """Reject invalid initial inputs and nonfinite map outputs explicitly."""
    expected = InvalidAndersonPoint if np.isinf(initial) else ValueError
    with pytest.raises(expected):
        run_anderson(lambda x: np.array([output]), np.array([initial]), AndersonOptions(), 2)


def test_map_failure_is_not_hidden():
    """Propagate map errors instead of treating them as invalid extrapolations."""

    def evaluate(x):
        raise ValueError("forward solver failed")

    with pytest.raises(ValueError, match="forward solver failed"):
        run_anderson(evaluate, np.zeros(1), AndersonOptions(), 2)


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

    run_anderson(evaluate, np.full(3, 2.0), AndersonOptions(), 9)
    reference = []
    try:
        anderson(
            lambda x: G(x) - x,
            np.full(3, 2.0),
            alpha=1,
            M=3,
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
    "kind,mean", [("additive", 2.0), ("log", 2.0), ("log", -2.0), ("multiplicative", 2.0)]
)
def test_fixed_parameter_coordinates_round_trip(kind, mean):
    """Invert each parameter transformation, including negative log anchors."""
    prior = ParameterPrior(
        mean=mean, **({"stddev": 0.1} if kind == "additive" else {"relative_stddev": 0.1})
    )
    parameters = build_parameterization(
        ParametersOptions(reparameterization=kind, priors={"a": prior}), ["a"]
    )
    values = {"a": mean * 1.2}
    assert parameters.to_physical(parameters.to_theta(values)) == pytest.approx(values)


@pytest.mark.parametrize(
    "options", [{"memory": 0}, {"regularization": -1}, {"regularization": np.inf}]
)
def test_invalid_options(options):
    """Reject unsupported memory and regularization values."""
    with pytest.raises(ValidationError):
        AndersonOptions(**options)


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
    "completed,interrupted", [(1, False), (2, False), (4, False), (2, True)]
)
def test_roukf_acceleration_and_resume(
    tmp_path, monkeypatch, reparameterization, completed, interrupted
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
            iterations=5, anderson=AndersonOptions(memory=2, regularization=0)
        ),
    )
    result = estimate(study)
    if reparameterization == "additive":
        np.testing.assert_allclose(list(result.values()), [0.6, -0.4], atol=1e-8)
    if reparameterization == "log":
        assert result["a"] > 0 and result["b"] < 0
    plain = replace(
        study,
        output=OutputOptions(path=str(tmp_path / "plain")),
        estimation=EstimationOptions(iterations=5),
    )
    plain_result = estimate(plain)
    assert np.linalg.norm(np.array(list(plain_result.values())) - [0.6, -0.4]) > 0.5
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
    with pytest.raises(StudyFileError, match=r"estimation\.anderson"):
        estimate(
            replace(resumed, estimation=replace(study.estimation, iterations=6, anderson=None)),
            resume=True,
        )
    comm.barrier()

    if comm.rank == 0:
        yaml = YAML()
        path = tmp_path / "resumed/estimation/run.yaml"
        run = yaml.load(path)
        run["anderson"]["scipy_version"] = "different-version"
        with path.open("w") as stream:
            yaml.dump(run, stream)
    comm.barrier()
    with pytest.raises(StudyFileError, match="Anderson backend or SciPy version"):
        estimate(
            replace(resumed, estimation=replace(study.estimation, iterations=6)), resume=True
        )
    if comm.rank == 0:
        del run["anderson"]
        with path.open("w") as stream:
            yaml.dump(run, stream)
    comm.barrier()
    with pytest.raises(StudyFileError, match="Anderson backend or SciPy version"):
        estimate(
            replace(resumed, estimation=replace(study.estimation, iterations=6)), resume=True
        )
    comm.barrier()
