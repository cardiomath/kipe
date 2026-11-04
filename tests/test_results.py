"""Reading the results of a study: the final estimates of its noise realizations."""

from pathlib import Path

import numpy as np
import pytest

from kipe.estimation import estimate
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
    StudyFileError,
    StudyOptions,
    SynthesisOptions,
    TimeRange,
)
from kipe.results import read_realizations
from kipe.synthesis import synthesize

REALIZATIONS = 3
ITERATIONS = 2


def _study(tmp_path: Path, reparameterization: Reparameterization = "log") -> StudyOptions:
    """A FitzHugh-Nagumo study estimating a and c from v, with three noise realizations."""
    spread = {"stddev": 0.05} if reparameterization == "additive" else {"relative_stddev": 0.2}
    return StudyOptions(
        output=OutputOptions(path=str(tmp_path / "results")),
        forward_solver=ForwardSolverOptions(
            factory="kipe.examples.fitzhugh_nagumo:Solver", arguments={"dt": 0.05}
        ),
        parameters=ParametersOptions(
            reparameterization=reparameterization,
            priors={
                "a": ParameterPrior(mean=0.25, **spread),
                "c": ParameterPrior(mean=2.5, **spread),
            },
        ),
        measurements={
            "v": MeasurementOptions(
                fields=["v"],
                times=TimeRange(start=0.5, stop=10.0, step=0.5),
                data=NumpyDataOptions(
                    type="numpy", path=str(tmp_path / "data" / "r{realization}" / "v.npz")
                ),
                noise=NoiseOptions(stddev=0.05, seed=0),
            )
        },
        synthesis=SynthesisOptions(realizations=REALIZATIONS),
        estimation=EstimationOptions(iterations=ITERATIONS),
    )


def _run(study: StudyOptions) -> None:
    synthesize(study)
    for r in range(REALIZATIONS):
        estimate(study, realization=r)


def _history(tmp_path: Path, realization: int) -> dict[str, np.ndarray]:
    path = tmp_path / "results" / "estimation" / f"r{realization}" / "history.npz"
    with np.load(path) as history:
        return dict(history)


@pytest.mark.parametrize("reparameterization", ["log", "multiplicative", "additive"])
def test_read_realizations(tmp_path, reparameterization):
    """The estimate at the end of each outer iteration, and the filter's covariance of the
    final one in physical units: J P_theta J, with dphi/dtheta at the last iteration's
    anchor, the estimate of the iteration before."""
    study = _study(tmp_path, reparameterization)
    _run(study)
    result = read_realizations(study)

    assert result.names == ["a", "c"]
    np.testing.assert_array_equal(result.realizations, [0, 1, 2])
    assert result.parameters.shape == (REALIZATIONS, 2)
    assert result.covariance.shape == (REALIZATIONS, 2, 2)
    assert result.iterations.shape == (REALIZATIONS, ITERATIONS, 2)

    for r in range(REALIZATIONS):
        history = _history(tmp_path, r)
        first = history["parameters"][history["iteration"] == 0][-1]
        last = history["parameters"][-1]
        np.testing.assert_array_equal(result.iterations[r], [first, last])
        np.testing.assert_array_equal(result.parameters[r], last)

        J = {"log": last, "multiplicative": first, "additive": np.ones(2)}[reparameterization]
        expected = np.outer(J, J) * history["P_theta"][-1]
        np.testing.assert_allclose(result.covariance[r], expected, rtol=1e-14)


def test_incomplete_realizations_are_skipped(tmp_path, kipe_caplog):
    """E.g., while a job array runs: a realization without all outer iterations is left out."""
    study = _study(tmp_path)
    _run(study)
    (tmp_path / "results" / "estimation" / "r1" / "history.npz").unlink()

    result = read_realizations(study)
    np.testing.assert_array_equal(result.realizations, [0, 2])
    assert "r1: 0 of 2 outer iterations, skipped" in kipe_caplog.text


def test_no_completed_realization(tmp_path):
    study = _study(tmp_path)
    with pytest.raises(StudyFileError, match=r"no realization .* has completed its 2 outer"):
        read_realizations(study)
