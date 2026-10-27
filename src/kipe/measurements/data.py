"""Measurements and their data at the measurement times, and how the data are stored.

A :class:`Measurement` holds the data of one measurement at all its times, with its
observation operator (:mod:`kipe.measurements.operator`) and its noise. The data are stored in
the ``numpy`` format (:func:`read_numpy`, :func:`write_numpy`), in the order of the
measurement's positions, so that they do not depend on the number of ranks.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import numpy as np

from kipe._types import NDArray_f64
from kipe.measurements.operator import ObservationOperator
from kipe.options import StudyFileError, TimeRange


@dataclass(frozen=True)
class Measurement:
    """A measurement: its data at all measurement times, and its observation operator.

    Synthesis produces measurements, estimation consumes them.
    """

    name: str
    """Name of the measurement, for logs and error messages."""

    operator: ObservationOperator
    r"""The observation operator :math:`\mathcal{H}`, from a model state to predicted data."""

    times: NDArray_f64
    """The measurement times, shape ``(n,)``."""

    values: NDArray_f64
    """This rank's entries of the measured data at each of the n times, at the positions
    ``operator.global_indices``, shape ``(n, number of this rank's entries)``."""

    stddev: float
    """Standard deviation of the measurement noise, in the unit of the data."""


def measurement_times(times: list[float] | TimeRange) -> NDArray_f64:
    """Return the measurement times as an array.

    Args:
        times: the ``times`` of a measurement, a list or a range

    Returns:
        the times; for a range, from ``start`` up to and including ``stop`` (up to round-off)
    """
    if isinstance(times, TimeRange):
        n = math.floor((times.stop - times.start) / times.step + 1e-9)
        return times.start + times.step * np.arange(n + 1, dtype=np.float64)
    return np.array(times, dtype=np.float64)


def read_numpy(path: str | Path) -> tuple[NDArray_f64, NDArray_f64]:
    """Read measurement data from a ``.npz`` file with the arrays ``times`` and ``values``.

    Args:
        path: path to the ``.npz`` file

    Returns:
        - times, shape ``(n,)``
        - values, shape ``(n, m)``: m values at each of the n times

    Raises:
        StudyFileError: if the file cannot be read or does not hold the expected arrays
    """
    try:
        with np.load(path) as data:
            missing = sorted({"times", "values"} - set(data.files))
            if missing:
                missing_names = ", ".join(missing)
                raise StudyFileError(f"measurement data {path} has no array {missing_names}")
            times, values = data["times"], data["values"]
    except OSError as err:
        raise StudyFileError(f"cannot read measurement data {path}: {err}") from err

    if times.ndim != 1 or values.ndim != 2 or len(times) != len(values):
        raise StudyFileError(
            f"measurement data {path}: expected times (n,) and values (n, m), got "
            f"{times.shape} and {values.shape}"
        )
    return times.astype(np.float64), values.astype(np.float64)


def write_numpy(path: str | Path, times: NDArray_f64, values: NDArray_f64) -> None:
    """Write measurement data to a ``.npz`` file, creating its directory if needed.

    Args:
        path: path to the ``.npz`` file
        times: the n measurement times, shape ``(n,)``
        values: m values at each of the n times, shape ``(n, m)``
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, times=times, values=values)


def write_data(
    path: str | Path,
    operator: ObservationOperator,
    times: NDArray_f64,
    values: NDArray_f64,
    comm: MPI.Comm,
) -> None:
    """Write a measurement's data, gathered from all ranks, on rank 0 (:func:`write_numpy`).

    In the format of the data files: what synthesis writes, and the predicted data of the runs
    of the sensitivity analysis.

    Args:
        path: path to the ``.npz`` file
        operator: the measurement's observation operator, which says where each rank's
            entries go (``global_indices``)
        times: the measurement times
        values: this rank's entries of the data, one row per time
        comm: communicator over which the entries are distributed
    """
    parts = comm.gather((operator.global_indices, values), root=0)
    if comm.rank != 0:
        return

    assert parts is not None
    data = np.empty((len(times), operator.size))

    for global_indices, local_values in parts:
        data[:, global_indices] = local_values

    write_numpy(path, times, data)
