r"""Measurement models: from the sampled fields to predicted data, and their comparison.

A :class:`MeasurementModel` predicts the measured data from the sampled state,
:math:`\hat{z} = M(y)`, and compares them with the measured data, giving the innovation
:math:`\Gamma(z, \hat{z})`. Models act on arrays only, so they are the same for every backend.

To add a model: subclass :class:`MeasurementModel`, implement both methods (and
:meth:`~MeasurementModel.layout` if the model combines entries), add its options to
:mod:`kipe.options` and a ``case`` to :func:`build_model`.
"""

from abc import ABC, abstractmethod
from typing import assert_never

from mpi4py import MPI

from kipe._types import NDArray_f64, NDArray_i64
from kipe.options import (
    DifferenceModelOptions,
    MeasurementModelOptions,
    PlaneModelOptions,
    StudyFileError,
)
from kipe.sampling import SpatialSampler


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
            this rank's entries of the predicted data :math:`\hat{z}`, at the positions
            :meth:`layout` gives
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

        By default those of the sampler: a model that acts entry by entry, like
        :class:`DifferenceModel`, predicts one value per sampled entry, on the rank that holds
        it. A model that combines entries has its own, e.g., a plane fitted to sampled
        vertices: three coefficients, held by rank 0 after gathering the coordinates. As for the
        sampler, each entry is held by exactly one rank, and the positions do not depend on
        the number of ranks.

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
    sampler: SpatialSampler | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> MeasurementModel:
    """Construct the measurement model given by the ``model`` options of a measurement.

    Args:
        options: ``model`` options of the measurement
        sampler: spatial sampler, required by models that combine its entries
        comm: communicator of the run

    Returns:
        the measurement model
    """
    match options:
        case DifferenceModelOptions():
            return DifferenceModel()
        case PlaneModelOptions():
            from kipe.measurements.plane import PlaneModel

            if sampler is None:
                raise StudyFileError("plane model needs a spatial sampler")
            return PlaneModel(sampler, comm)
        case _:
            assert_never(options)
