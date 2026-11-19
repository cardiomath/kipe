"""Anderson mixing, the notebook's linear map, and accelerated ROUKF restart behavior."""

import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI

import numpy as np
import pytest
from pydantic import ValidationError

from kipe._anderson import Anderson
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


def test_scalar_linear_map_and_constant_residual():
    aa = Anderson(AndersonOptions(memory=1, regularization=0))
    x = np.array([0.0])
    x = aa.update(x, 0.9 * x + 0.1)
    np.testing.assert_allclose(x, [0.1])
    x = aa.update(x, 0.9 * x + 0.1)
    np.testing.assert_allclose(x, [1.0], atol=1e-13)
    aa.reset()
    # Dependent differences must not introduce NaNs or a singular-system exception.
    for _ in range(4):
        g = x + 1
        x = aa.update(x, g)
        np.testing.assert_array_equal(x, g)


def test_matches_reference_notebook_scipy_anderson():
    scipy = pytest.importorskip("scipy.optimize")
    H = np.vander(np.linspace(0, 1, 80), 3, increasing=True)
    y = H @ np.ones(3) + 0.5 * np.random.default_rng(4).standard_normal(80)
    fim = H.T @ H / 0.5**2
    rhs = H.T @ y / 0.5**2

    def G(x):
        return np.linalg.solve(fim + 10 * np.eye(3), rhs + 10 * x)

    x = np.full(3, 2.0)
    aa = Anderson(AndersonOptions(memory=3, regularization=0.01))
    ours = []
    plain = x.copy()
    for _ in range(9):
        x = aa.update(x, G(x))
        ours.append(x.copy())
        plain = G(plain)
    reference = []
    try:
        scipy.anderson(
            lambda x: G(x) - x,
            np.full(3, 2.0),
            alpha=1,
            M=3,
            w0=0.01,
            maxiter=9,
            f_tol=1e-16,
            line_search=None,
            callback=lambda x, f: reference.append(x.copy()),
        )
    except scipy.NoConvergence:
        pass
    np.testing.assert_allclose(ours[:5], reference[:5], atol=1e-8, rtol=1e-8)
    mle = np.linalg.solve(fim, rhs)
    assert np.linalg.norm(x - mle) < np.linalg.norm(plain - mle) * 0.1


@pytest.mark.parametrize(
    "kind,mean", [("additive", 2.0), ("log", 2.0), ("log", -2.0), ("multiplicative", 2.0)]
)
def test_fixed_parameter_coordinates_round_trip(kind, mean):
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
def test_roukf_acceleration_and_resume(tmp_path, monkeypatch, reparameterization):
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
    estimate(replace(resumed, estimation=replace(study.estimation, iterations=2)))
    comm.barrier()
    continued = estimate(resumed, resume=True)
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
