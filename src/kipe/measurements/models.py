r"""Measurement models: from the sampled fields to predicted data, and their comparison.

A :class:`MeasurementModel` predicts the measured data from the sampled state,
:math:`\hat{z} = M(y)`, and compares them with the measured data, giving the innovation
:math:`\Gamma(z, \hat{z})`. Models act on arrays only, so they are the same for every backend.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, assert_never

from kipe._types import NDArray_f64, NDArray_i64
from kipe.options import (
    DifferenceModelOptions,
    MeasurementModelOptions,
    PlaneModelOptions,
)
from kipe.sampling import SpatialSampler


class MeasurementModel(ABC):
    """Turns sampled model fields into predicted data and compares them with the measurement."""

    gather_input: ClassVar[bool] = False
    """Request a complete globally ordered input and prediction on root only.

    False selects rank-local input and prediction.
    """

    def clear_prediction(self) -> None:
        """Leave prediction state unchanged.

        This default implementation does nothing. Overrides must clear only transient
        prediction results, retain persistent history and perform no MPI collectives.
        """
        return None

    # NOTE: a measurement context (time-dependent geometry, auxiliary data such as the magnitude
    # or background phase of PC-MRI, connectivity of STL vertices) is expected soon.
    # It becomes an additional argument of # both methods, e.g., `context: MeasurementContext`.

    @abstractmethod
    def predict(self, y: NDArray_f64) -> NDArray_f64:
        r"""Predict the measured data from the sampled state, :math:`\hat{z} = M(y)`.

        Args:
            y: sampled state :math:`y`

        Returns:
            predicted entries :math:`\hat{z}`, at the positions :meth:`layout` gives;
            globally gathered models return the complete output on root
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

    def layout(self, sampler: SpatialSampler) -> tuple[int, NDArray_i64]:
        """Return the size of the predicted data, and the positions of this rank's entries.

        Return the sampler's global size and this rank's sampled-entry indices unchanged.

        Args:
            sampler: the measurement's spatial sampler

        Returns:
            - the number of entries of the predicted data, over all ranks
            - the positions of this rank's entries, in the order :meth:`predict` returns them
        """
        return sampler.size, sampler.global_indices


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


def build_model(
    options: MeasurementModelOptions,
) -> MeasurementModel:
    """Construct the measurement model given by the ``model`` options of a measurement.

    Args:
        options: ``model`` options of the measurement

    Returns:
        the measurement model
    """
    match options:
        case DifferenceModelOptions():
            return DifferenceModel()
        case PlaneModelOptions():
            from kipe.measurements.plane import PlaneModel, PlaneOrientation

            model = PlaneModel()
            model.orientation = PlaneOrientation(options.normal_toward)
            return model
        case _:
            assert_never(options)
