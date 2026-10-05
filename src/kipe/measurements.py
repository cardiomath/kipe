r"""Measurements, and how predicted data are compared with them.

The observation operator compares a model state with a measurement in two steps: a spatial
sampler (:mod:`kipe.sampling`) evaluates the observed fields at the measurement locations,
giving :math:`y`; a :class:`MeasurementModel` predicts the data from it,
:math:`\hat{z} = M(y)`, and compares them with the measured data :math:`z`, giving the
innovation :math:`\Gamma(z, \hat{z})`. Models act on arrays only, so they are the same for
every backend.

At each assimilation time, a measurement is passed around as a :class:`MeasurementSnapshot`.

To add a model: subclass :class:`MeasurementModel`, implement both methods, add its options
to :mod:`kipe.options` and a ``case`` to :func:`build_model`.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kipe._types import NDArray_f64
from kipe.options import DifferenceModelOptions, StudyFileError, TimeRange


@dataclass(frozen=True)
class MeasurementSnapshot:
    """A measurement at one assimilation time."""

    # NOTE: this class is going to be extended by auxiliary/additional data, like pcmri
    # (background phase), or geometry data

    z: NDArray_f64
    """Measured data, in the measurement space (e.g., values at the measurement locations)."""


class MeasurementModel(ABC):
    """Turns sampled model fields into predicted data and compares them with the measurement."""

    @abstractmethod
    def predict(self, y: NDArray_f64, m: MeasurementSnapshot) -> NDArray_f64:
        r"""Predict the measured data from the sampled state, :math:`\hat{z} = M(y)`.

        Args:
            y: sampled state :math:`y`, as returned by
                :meth:`kipe.sampling.SpatialSampler.sample`
            m: the measurement at the current assimilation time

        Returns:
            predicted data :math:`\hat{z}`, in the measurement space
        """

    @abstractmethod
    def innovation(self, z_hat: NDArray_f64, m: MeasurementSnapshot) -> NDArray_f64:
        r"""Compare predicted and measured data, :math:`\Gamma(z, \hat{z})`.

        Args:
            z_hat: predicted data :math:`\hat{z}`, as returned by :meth:`predict`
            m: the measurement at the current assimilation time

        Returns:
            innovation, before weighting with the measurement noise
        """


class DifferenceModel(MeasurementModel):
    r"""The measured data are the sampled fields: :math:`\hat{z} = y`, and the innovation is
    :math:`\Gamma = z - \hat{z}`.
    """

    def predict(self, y: NDArray_f64, m: MeasurementSnapshot) -> NDArray_f64:
        """Return the sampled fields as predicted data.

        Args:
            y: sampled state
            m: the measurement at the current assimilation time (unused)

        Returns:
            ``y``
        """
        return y

    def innovation(self, z_hat: NDArray_f64, m: MeasurementSnapshot) -> NDArray_f64:
        """Return measured minus predicted data.

        Args:
            z_hat: predicted data
            m: the measurement at the current assimilation time

        Returns:
            ``m.z - z_hat``
        """
        return m.z - z_hat


def build_model(options: DifferenceModelOptions) -> MeasurementModel:
    """Construct the measurement model given by the ``model`` options of a measurement.

    Args:
        options: ``model`` options of the measurement

    Returns:
        the measurement model
    """
    match options:
        case DifferenceModelOptions():
            return DifferenceModel()


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
        - times, shape (n,)
        - values, shape (n, m): the data at each time

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
        times: measurement times, shape (n,)
        values: the data at each time, shape (n, m)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, times=times, values=values)
