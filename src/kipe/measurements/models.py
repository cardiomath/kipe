r"""Measurement models: from the sampled fields to predicted data, and their comparison.

A :class:`MeasurementModel` predicts the measured data from the sampled state,
:math:`\hat{z} = M(y)`, and compares them with the measured data, giving the innovation
:math:`\Gamma(z, \hat{z})`. Models act on arrays only, so they are the same for every backend.

To add a model: subclass :class:`MeasurementModel`, implement both methods, add its options
to :mod:`kipe.options` and a ``case`` to :func:`build_model`.
"""

from abc import ABC, abstractmethod
from typing import assert_never

from kipe._types import NDArray_f64
from kipe.options import DifferenceModelOptions


class MeasurementModel(ABC):
    """Turns sampled model fields into predicted data and compares them with the measurement."""

    # NOTE: a measurement context (time-dependent geometry, auxiliary data such as the magnitude
    # or background phase of PC-MRI, connectivity of STL vertices) is expected soon.
    # It becomes an additional argument of # both methods, e.g., `context: MeasurementContext`.

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
