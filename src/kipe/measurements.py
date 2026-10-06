r"""Measurements, and how predicted data are compared with them.

The observation operator compares a model state with a measurement in two steps: a spatial
sampler (:mod:`kipe.sampling`) evaluates the observed fields at the measurement locations,
giving :math:`y`; a :class:`MeasurementModel` predicts the data from it,
:math:`\hat{z} = M(y)`, and compares them with the measured data :math:`z`, giving the
innovation :math:`\Gamma(z, \hat{z})`. Models act on arrays only, so they are the same for
every backend. The :class:`ObservationOperator` :math:`\mathcal{H} = M \circ S` combines the
two steps: it takes a model state to predicted data.

To add a model: subclass :class:`MeasurementModel`, implement both methods, add its options
to :mod:`kipe.options` and a ``case`` to :func:`build_model`.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import numpy as np

from kipe._types import NDArray_f64
from kipe.forward_solver import FieldSpec, State
from kipe.options import DifferenceModelOptions, MeasurementOptions, StudyFileError, TimeRange
from kipe.sampling import SpatialSampler, build_sampler


class MeasurementModel(ABC):
    """Turns sampled model fields into predicted data and compares them with the measurement."""

    # NOTE: a measurement context (time-dependent geometry, auxiliary data such as the magnitude
    # or background phase of PC-MRI) is expected soon. It becomes an additional argument of
    # both methods, e.g., `context: MeasurementContext` (see PLAN.md, "Measurement context").

    @abstractmethod
    def predict(self, y: NDArray_f64) -> NDArray_f64:
        r"""Predict the measured data from the sampled state, :math:`\hat{z} = M(y)`.

        Args:
            y: sampled state :math:`y`, as returned by
                :meth:`kipe.sampling.SpatialSampler.sample`

        Returns:
            predicted data :math:`\hat{z}`, in the measurement space
        """

    @abstractmethod
    def innovation(self, z_hat: NDArray_f64, z: NDArray_f64) -> NDArray_f64:
        r"""Compare predicted and measured data, :math:`\Gamma(z, \hat{z})`.

        Args:
            z_hat: predicted data :math:`\hat{z}`, as returned by :meth:`predict`
            z: measured data at the current assimilation time, in the measurement space

        Returns:
            innovation, before weighting with the measurement noise
        """


class DifferenceModel(MeasurementModel):
    r"""The measured data are the sampled fields: :math:`\hat{z} = y`, and the innovation is
    :math:`\Gamma = z - \hat{z}`.
    """

    def predict(self, y: NDArray_f64) -> NDArray_f64:
        """Return the sampled fields as predicted data.

        Args:
            y: sampled state

        Returns:
            ``y``
        """
        return y

    def innovation(self, z_hat: NDArray_f64, z: NDArray_f64) -> NDArray_f64:
        """Return measured minus predicted data.

        Args:
            z_hat: predicted data
            z: measured data

        Returns:
            ``z - z_hat``
        """
        return z - z_hat


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
        case _:
            assert_never(options)


@dataclass(frozen=True)
class ObservationOperator:
    r"""The observation operator :math:`\mathcal{H} = M \circ S` of a measurement.

    Takes a model state :math:`\chi` to predicted data :math:`\hat{z} = \mathcal{H}(\chi)`:
    the spatial sampler :math:`S` evaluates the observed fields at the measurement locations,
    the measurement model :math:`M` predicts the data from them.
    """

    fields: list[str]
    """Names of the observed state fields."""

    sampler: SpatialSampler
    """The spatial sampler :math:`S`."""

    model: MeasurementModel
    """The measurement model :math:`M`."""

    def __call__(self, state: State) -> NDArray_f64:
        r"""Return the predicted data :math:`\hat{z} = \mathcal{H}(\chi)` of a model state.

        Args:
            state: the model state :math:`\chi`

        Returns:
            predicted data, in the measurement space
        """
        y = self.sampler.sample({name: state[name] for name in self.fields})

        return self.model.predict(y)


def build_observation_operator(
    name: str, options: MeasurementOptions, state_spec: Mapping[str, FieldSpec]
) -> ObservationOperator:
    """Construct the observation operator of a measurement.

    Args:
        name: name of the measurement, for error messages
        options: the measurement options
        state_spec: the forward solver's state fields

    Returns:
        the observation operator

    Raises:
        StudyFileError: if the measurement observes a field the solver does not have
    """
    unknown = [f for f in options.fields if f not in state_spec]
    if unknown:
        raise StudyFileError(
            f"measurements.{name}: the forward solver has no field {', '.join(unknown)}. "
            f"Available: {', '.join(state_spec)}"
        )

    return ObservationOperator(
        fields=options.fields,
        sampler=build_sampler(options.spatial_sampling, options.fields),
        model=build_model(options.model),
    )


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
    """The measured data: m values at each of the n times, shape ``(n, m)``."""

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
