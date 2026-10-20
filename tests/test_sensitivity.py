"""``kipe sensitivity``: the sensitivities and the Fisher information, from forward runs."""

import shutil
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from kipe import sensitivity
from kipe.cli import main
from kipe.estimation import estimate
from kipe.measurements import write_numpy
from kipe.options import (
    EstimationOptions,
    ForwardSolverOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    OutputOptions,
    ParameterPrior,
    ParametersOptions,
    Reparameterization,
    SensitivityOptions,
    StudyFileError,
    StudyOptions,
    dump_study,
)

EXAMPLE = Path(__file__).parents[1] / "examples" / "fitzhugh_nagumo" / "study.yaml"
TIMES = [1.0, 2.0, 3.0]
NOISE = 0.5
STDDEV = {"a": 0.2, "b": 0.1}  # additive: in the unit of the parameters
RELATIVE_STDDEV = {"a": 0.15, "b": 0.3}  # multiplicative


@dataclass(frozen=True)
class _FieldSpec:
    local_size: int
    kind: str = "replicated"


class _LinearSolver:
    """y(t) = (a t + b, 2 a - b t): linear in a and b, so centered differences are exact."""

    @property
    def state_spec(self) -> dict[str, _FieldSpec]:
        return {"y": _FieldSpec(2)}

    def nominal_parameters(self) -> dict[str, float]:
        return {"a": 1.5, "b": -0.5}

    def initial_state(self) -> tuple[float, dict[str, np.ndarray]]:
        return 0.0, {"y": np.zeros(2)}

    def propagate(self, t0, t1, state, parameters):
        p = self.nominal_parameters() | parameters
        return {"y": np.array([p["a"] * t1 + p["b"], 2 * p["a"] - p["b"] * t1])}


@pytest.fixture
def linear(monkeypatch):
    module = types.ModuleType("linear_solver")
    module.Solver = _LinearSolver
    monkeypatch.setitem(sys.modules, "linear_solver", module)  # removed after the test


def _study(
    tmp_path: Path,
    step: float = 0.1,
    reparameterization: Reparameterization = "additive",
    estimation: EstimationOptions | None = None,
) -> StudyOptions:
    if reparameterization == "additive":
        priors = {name: ParameterPrior(stddev=s) for name, s in STDDEV.items()}
    else:
        priors = {name: ParameterPrior(relative_stddev=s) for name, s in RELATIVE_STDDEV.items()}
    return StudyOptions(
        output=OutputOptions(path=str(tmp_path / "results")),
        forward_solver=ForwardSolverOptions("linear_solver:Solver"),
        parameters=ParametersOptions(reparameterization=reparameterization, select=priors),
        measurements={
            "y": MeasurementOptions(
                fields=["y"],
                times=TIMES,
                data=NumpyDataOptions(type="numpy", path=str(tmp_path / "y.npz")),
                noise=NoiseOptions(stddev=NOISE),
            )
        },
        sensitivity=SensitivityOptions(step=step),
        estimation=estimation or EstimationOptions(),
    )


def _analyze(study: StudyOptions) -> sensitivity.Sensitivity:
    for i in range(sensitivity.run_count(study)):
        sensitivity.run(study, i)
    return sensitivity.assemble(study)


def test_linear_model(tmp_path, linear):
    """The exact sensitivities, Fisher information and Cramér-Rao bound."""
    result = _analyze(_study(tmp_path))

    t = np.array(TIMES)
    J = np.zeros((2 * len(t), 2))  # the two entries of y at each time, one after another

    J[0::2] = np.column_stack([t, np.ones_like(t)])  # ∂(a t + b)/∂(a, b)
    J[1::2] = np.column_stack([2 * np.ones_like(t), -t])  # ∂(2 a - b t)/∂(a, b)
    np.testing.assert_allclose(result.jacobians["y"], J, rtol=1e-9, atol=1e-12)

    M = J.T @ J / NOISE**2
    np.testing.assert_allclose(result.fim, M, rtol=1e-9)
    np.testing.assert_allclose(result.covariance, np.linalg.inv(M), rtol=1e-9)

    _, s, Vt = np.linalg.svd(J / NOISE, full_matrices=False)
    np.testing.assert_allclose(result.singular_values, s)
    assert result.condition_number == pytest.approx(s[0] / s[-1], rel=1e-9)
    np.testing.assert_allclose(np.abs(result.right_singular_vectors), np.abs(Vt.T), rtol=1e-9)

    saved = np.load(tmp_path / "results" / "sensitivity" / "sensitivity.npz")
    np.testing.assert_array_equal(saved["fim"], result.fim)
    np.testing.assert_array_equal(saved["J_y"], result.jacobians["y"])
    assert saved["names"].tolist() == ["a", "b"]


def test_runs_one_by_one(tmp_path, linear):
    """The tasks of a job array, then the assembly: the result of a single command."""
    study = _study(tmp_path)
    path = tmp_path / "study.yaml"
    dump_study(study, path)

    for i in range(4):
        assert main(["sensitivity", str(path), "--run", str(i), "--log-level", "warning"]) == 0
    assert main(["sensitivity", str(path), "--assemble", "--log-level", "warning"]) == 0

    one_by_one = dict(np.load(tmp_path / "results" / "sensitivity" / "sensitivity.npz"))

    assert main(["sensitivity", str(path), "--log-level", "warning"]) == 0
    at_once = np.load(tmp_path / "results" / "sensitivity" / "sensitivity.npz")
    for key in at_once.files:
        np.testing.assert_array_equal(one_by_one[key], at_once[key])


def test_runs_are_perturbations(tmp_path, linear):
    """Run i: parameter i // 2, + for even i, by step σ_θ."""
    study = _study(tmp_path, step=0.5)
    sensitivity.run(study, 3)  # -b

    times, values = np.load(sensitivity.run_directory(study, 3) / "y.npz").values()
    a, b = 1.5, -0.5 - 0.5 * STDDEV["b"]
    np.testing.assert_allclose(values, [[a * t + b, 2 * a - b * t] for t in TIMES])
    np.testing.assert_array_equal(times, TIMES)


def test_missing_runs(tmp_path, linear):
    study = _study(tmp_path)
    sensitivity.run(study, 0)
    sensitivity.run(study, 2)
    with pytest.raises(StudyFileError, match="sensitivity runs 1, 3 are missing or unfinished"):
        sensitivity.assemble(study)


def test_runs_of_another_step(tmp_path, linear):
    study = _study(tmp_path)
    _analyze(study)
    with pytest.raises(StudyFileError, match="runs 0, 1, 2, 3 have other parameters"):
        sensitivity.assemble(replace(study, sensitivity=SensitivityOptions(step=0.2)))


def test_no_such_run(tmp_path, linear, capsys):
    path = tmp_path / "study.yaml"
    dump_study(_study(tmp_path), path)
    assert main(["sensitivity", str(path), "--run", "4"]) == 1
    assert "sensitivity run 4: the study has runs 0 to 3" in capsys.readouterr().err


def test_needs_times(tmp_path, linear):
    study = _study(tmp_path)
    assert study.measurements is not None
    measurement = replace(study.measurements["y"], times=None)
    with pytest.raises(StudyFileError, match=r"measurements\.y: the sensitivity analysis needs"):
        sensitivity.run(replace(study, measurements={"y": measurement}), 0)


@pytest.mark.parametrize("reparameterization", ["additive", "multiplicative"])
@pytest.mark.parametrize("sigma_points", ["simplex", "canonical"])
@pytest.mark.parametrize("iterations", [1, 3])
def test_fisher_information_is_what_the_filter_learns(
    tmp_path, linear, reparameterization, sigma_points, iterations
):
    """For observations linear in theta, the ROUKF is the Kalman filter: after each pass over
    the data, its covariance is the posterior (M + P_0^-1)^-1 of the Fisher information M and
    the prior, the Bayesian Cramer-Rao bound.

    The data are the model's own, without noise: the estimate stays at the nominal values, so
    for multiplicative, theta keeps its scale over the passes (each recenters it at the
    previous estimate), and needs no rescaling to compare with M.
    """
    options = EstimationOptions(sigma_points=sigma_points, iterations=iterations)
    study = _study(tmp_path, reparameterization=reparameterization, estimation=options)
    result = _analyze(study)

    a, b = _LinearSolver().nominal_parameters().values()
    data = [[a * t + b, 2 * a - b * t] for t in TIMES]
    write_numpy(tmp_path / "y.npz", np.array(TIMES), np.array(data))
    estimate(study)

    P = np.load(tmp_path / "results" / "estimation" / "history.npz")["P_theta"][-1]
    P_0 = np.diag(result.stddev_theta**2)
    np.testing.assert_allclose(P, np.linalg.inv(result.fim + np.linalg.inv(P_0)), rtol=1e-9)
    np.testing.assert_allclose(P, result.posterior_covariance, rtol=1e-9)


def test_rescaled_filter_covariance_for_multiplicative(tmp_path, linear):
    """With noisy data, the estimate leaves the nominal values, and each pass recenters theta
    at the previous one: for multiplicative, the last pass's theta and prior are relative to
    its center phi^0. With D = diag(phi^0 / phi_true), the filter's covariance D P D in the
    theta of the sensitivity analysis is (M + (D P_0 D)^-1)^-1."""
    options = EstimationOptions(iterations=3)
    study = _study(tmp_path, reparameterization="multiplicative", estimation=options)
    result = _analyze(study)

    a, b = _LinearSolver().nominal_parameters().values()
    data = np.array([[a * t + b, 2 * a - b * t] for t in TIMES])
    noise = np.random.default_rng(1).normal(0.0, NOISE, data.shape)
    write_numpy(tmp_path / "y.npz", np.array(TIMES), data + noise)
    estimate(study)

    history = np.load(tmp_path / "results" / "estimation" / "history.npz")
    center = history["parameters"][history["iteration"] == 2][0]  # first row of the last pass
    D = np.diag(center / np.array([a, b]))
    assert np.abs(np.diag(D) - 1).max() > 0.05  # not the trivial case D = 1

    P, P_0 = history["P_theta"][-1], np.diag(result.stddev_theta**2)
    expected = np.linalg.inv(result.fim + np.linalg.inv(D @ P_0 @ D))
    np.testing.assert_allclose(D @ P @ D, expected, rtol=1e-9)


def test_example_study(tmp_path, monkeypatch):
    """The FitzHugh-Nagumo example: after its iterations, the filter's standard deviations are
    within 10 % of the Bayesian Cramer-Rao bound, its correlations within 0.15: the filter is
    consistent on this model, whose data dominate the prior."""
    shutil.copy(EXAMPLE, tmp_path / "study.yaml")
    monkeypatch.chdir(tmp_path)  # paths in the study file are relative to the working directory
    for command in ["synthesis", "estimation", "sensitivity"]:
        assert main([command, "study.yaml", "--log-level", "warning"]) == 0

    P = np.load("results/estimation/history.npz")["P_theta"][-1]
    bound = np.load("results/sensitivity/sensitivity.npz")["posterior_covariance"]
    stddev, stddev_bound = np.sqrt(np.diag(P)), np.sqrt(np.diag(bound))
    np.testing.assert_allclose(stddev, stddev_bound, rtol=0.1)

    correlation = P / np.outer(stddev, stddev)
    bound_correlation = bound / np.outer(stddev_bound, stddev_bound)
    # 0.08 apart (2026-10): the filter linearizes around its estimates, over its sigma points
    np.testing.assert_allclose(correlation, bound_correlation, atol=0.15)
