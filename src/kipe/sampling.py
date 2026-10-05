r"""Spatial sampling: bringing the model fields to the measurement locations.

A :class:`SpatialSampler` evaluates the fields a measurement observes where the measurement
is taken, giving the sampled state :math:`y` that the measurement model (see
:mod:`kipe.measurements`) compares with the data. The samplers here need only the field
arrays and their own configuration; samplers that need the discretization (mesh, degrees of
freedom) live with their backend, e.g., in :mod:`kipe.fenicsx.sampling`.

To add a sampler: subclass :class:`SpatialSampler`, implement :meth:`~SpatialSampler.sample`,
add its options to :mod:`kipe.options` and a ``case`` to :func:`build_sampler`.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

from kipe._types import NDArray_f64
from kipe.options import ArraySamplingOptions


class SpatialSampler(ABC):
    """Evaluates the model fields at the measurement locations."""

    # NOTE: a measurement context (time-dependent geometry, auxiliary data such as the magnitude
    # or background phase of PC-MRI) is expected soon. It becomes an additional argument of
    # this method, e.g., `context: MeasurementContext` (see PLAN.md, "Measurement context").

    @abstractmethod
    def sample(self, fields: Mapping[str, NDArray_f64]) -> NDArray_f64:
        r"""Evaluate the model fields at the measurement locations.

        Args:
            fields: the state fields the measurement observes (its ``fields``), rank-local

        Returns:
            sampled state :math:`y`, rank-local; a new array, not a view of ``fields``
            or of internal buffers
        """


class ArraySampler(SpatialSampler):
    """Samples all entries of the observed fields, concatenated in the order of ``fields``.

    Args:
        fields: names of the observed fields, in the order of concatenation
    """

    def __init__(self, fields: list[str]) -> None:
        self._fields = fields

    def sample(self, fields: Mapping[str, NDArray_f64]) -> NDArray_f64:
        """Concatenate the observed fields.

        Args:
            fields: the observed state fields

        Returns:
            all entries of the fields, concatenated (a new array)
        """
        return np.concatenate([fields[name] for name in self._fields])


def build_sampler(options: ArraySamplingOptions, fields: list[str]) -> SpatialSampler:
    """Construct the sampler given by the ``spatial_sampling`` options of a measurement.

    Args:
        options: ``spatial_sampling`` options of the measurement
        fields: names of the fields the measurement observes

    Returns:
        the spatial sampler
    """
    match options:
        case ArraySamplingOptions():
            return ArraySampler(fields)
