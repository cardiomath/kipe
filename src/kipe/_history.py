"""The history of an estimation: written as the filter runs, read back to resume it.

:class:`History` writes ``history.csv``, one row per assimilation step as the filter runs,
and ``history.npz``, all steps of the completed outer iterations with the full parameter
covariance. :func:`read_completed_steps` reads them back, for an estimation that continues an
interrupted one (``kipe estimation --resume``).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import numpy as np

from kipe import roukf
from kipe._types import NDArray_f64
from kipe.options import StudyFileError, StudyOptions, load_study
from kipe.parameters import Parameterization


@dataclass(frozen=True)
class Step:
    """The estimate after one assimilation step, as recorded in the history."""

    iteration: int
    time: float
    parameters: NDArray_f64  # physical values
    lower: NDArray_f64  # physical 1σ range
    upper: NDArray_f64
    theta: NDArray_f64
    P_theta: NDArray_f64


class History:
    """The estimation history: one entry per assimilation step.

    Rows are appended to ``history.csv`` as the filter runs; :meth:`write_npz` writes all
    steps, including the full parameter covariance. Only rank 0 writes.

    Per parameter, e.g., ``c``, the CSV has the columns ``c`` (physical value), ``lower_c`` and
    ``upper_c`` (its physical 1σ range), ``theta_c`` and ``stddev_theta_c``.

    Args:
        path: output directory
        names: names of the estimated parameters
        comm: the communicator
        steps: the steps of an earlier run that this one continues (:func:`read_steps`); the
            CSV starts with them
    """

    def __init__(
        self, path: Path, names: list[str], comm: MPI.Comm, steps: Sequence[Step] = ()
    ) -> None:
        self._path = path
        self._names = names
        self._writes = comm.rank == 0
        self._steps: list[Step] = list(steps)
        if self._writes:
            path.mkdir(parents=True, exist_ok=True)
            header = ["iteration", "time"]
            for name in names:
                header += [name, f"lower_{name}", f"upper_{name}"]
                header += [f"theta_{name}", f"stddev_theta_{name}"]
            lines = [",".join(header)] + [_csv_row(step) for step in self._steps]
            (path / "history.csv").write_text("\n".join(lines) + "\n")

    @property
    def steps(self) -> list[Step]:
        """The recorded steps, those of an earlier run included."""
        return self._steps

    def record(
        self,
        iteration: int,
        t: float,
        filter_state: roukf.FilterState,
        parameterization: Parameterization,
    ) -> None:
        r"""Record the estimate after an assimilation step.

        Args:
            iteration: index of the pass
            t: time of the step
            filter_state: the filter state, with the estimate :math:`\theta`
            parameterization: maps :math:`\theta` to physical values
        """
        theta, P_theta = filter_state.theta, filter_state.covariance()
        stddev = np.sqrt(np.diag(P_theta))
        ranges = np.array(parameterization.physical_range(theta, stddev))
        step = Step(
            iteration=iteration,
            time=t,
            parameters=np.array(list(parameterization.to_physical(theta).values())),
            lower=ranges[:, 0],
            upper=ranges[:, 1],
            theta=theta.copy(),
            P_theta=P_theta.copy(),
        )
        self._steps.append(step)

        if self._writes:
            with (self._path / "history.csv").open("a") as f:
                f.write(_csv_row(step) + "\n")

    def write_npz(self) -> None:
        """Write all steps to ``history.npz``, e.g., after each outer iteration."""
        if not self._writes:
            return

        np.savez(
            self._path / "history.npz",
            names=np.array(self._names),
            iteration=np.array([step.iteration for step in self._steps]),
            time=np.array([step.time for step in self._steps]),
            parameters=np.array([step.parameters for step in self._steps]),
            lower=np.array([step.lower for step in self._steps]),
            upper=np.array([step.upper for step in self._steps]),
            theta=np.array([step.theta for step in self._steps]),
            P_theta=np.array([step.P_theta for step in self._steps]),
        )


def _csv_row(step: Step) -> str:
    """Return the row of ``history.csv`` of a step, with the values at full precision.

    Args:
        step: the step

    Returns:
        the row, without line break
    """
    row = [str(step.iteration), repr(step.time)]
    stddev = np.sqrt(np.diag(step.P_theta))
    columns = zip(step.parameters, step.lower, step.upper, step.theta, stddev, strict=True)
    for values in columns:
        row += [repr(float(value)) for value in values]

    return ",".join(row)


def read_steps(path: Path, names: list[str]) -> list[Step]:
    """Read the steps of an estimation from its ``history.npz``.

    Args:
        path: output directory of the estimation
        names: names of the estimated parameters, which the history must have

    Returns:
        the steps; none if there is no ``history.npz``

    Raises:
        StudyFileError: if the history has other parameters
    """
    if not (path / "history.npz").exists():
        return []

    with np.load(path / "history.npz") as history:
        if history["names"].tolist() != names:
            raise StudyFileError(
                f"{path / 'history.npz'} estimates {', '.join(history['names'])}, the study "
                f"{', '.join(names)}"
            )
        return [
            Step(
                iteration=int(history["iteration"][i]),
                time=float(history["time"][i]),
                parameters=history["parameters"][i],
                lower=history["lower"][i],
                upper=history["upper"][i],
                theta=history["theta"][i],
                P_theta=history["P_theta"][i],
            )
            for i in range(len(history["time"]))
        ]


def read_completed_steps(output: Path, study: StudyOptions) -> list[Step]:
    """Return the steps of the completed outer iterations of an earlier run of the study.

    Checks that the study has not changed since: the saved ``study.yaml`` of the run must have
    the same forward solver, parameters, measurements and sigma points. Every rank reads the
    files, and so gets the same steps, or the same error.

    Args:
        output: output directory of the estimation
        study: the study options, with parameters

    Returns:
        the steps; none if there is no earlier run to continue

    Raises:
        StudyFileError: if the study has changed, or the run's study file is missing
    """
    assert study.parameters is not None
    steps = read_steps(output, list(study.parameters.select))
    if steps:
        _check_unchanged(output, study)

    return steps


def _check_unchanged(output: Path, study: StudyOptions) -> None:
    """Check that a study is the one of an earlier estimation, but for more iterations.

    Args:
        output: output directory of the earlier estimation, with its ``study.yaml``
        study: the study options

    Raises:
        StudyFileError: if the study differs, or the earlier run has no ``study.yaml``
    """
    saved = output / "study.yaml"
    if not saved.exists():
        raise StudyFileError(f"cannot resume: {saved} is missing, to compare the study with")
    earlier = load_study(saved)

    changed = [
        name
        for name in ("forward_solver", "parameters", "measurements")
        if getattr(earlier, name) != getattr(study, name)
    ]
    if earlier.estimation.sigma_points != study.estimation.sigma_points:
        changed.append("estimation.sigma_points")
    if changed:
        raise StudyFileError(
            f"cannot resume the estimation in {output}: the study has changed since "
            f"({', '.join(changed)}); only estimation.iterations may differ"
        )


def count_iterations(steps: Sequence[Step]) -> int:
    """Return the number of outer iterations of recorded steps.

    Args:
        steps: the steps, of complete iterations

    Returns:
        the number of iterations
    """
    return steps[-1].iteration + 1 if steps else 0


def final_thetas(steps: Sequence[Step]) -> list[NDArray_f64]:
    r"""Return the estimate :math:`\theta` at the end of each outer iteration of recorded steps.

    Args:
        steps: the steps, of complete iterations, in order

    Returns:
        the estimate of the last step of each iteration
    """
    last: dict[int, NDArray_f64] = {}
    for step in steps:
        last[step.iteration] = step.theta

    return list(last.values())
