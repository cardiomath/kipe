r"""Spatial sampling: bringing the model fields to the measurement locations.

A :class:`SpatialSampler` evaluates the fields a measurement observes where the measurement
is taken, giving the sampled state :math:`y` that the measurement model (see
:mod:`kipe.measurements`) compares with the data. The samplers here need only the field
arrays and their own configuration; samplers that need the discretization (mesh, degrees of
freedom) live with their backend, e.g., in :mod:`kipe.fenicsx.sampling`.

In parallel, each entry of the sampled vector is held by exactly one rank: a sampler knows
the vector's ``size`` over all ranks and, per rank, the ``global_indices`` of the entries it
returns. The measurement's data are read and written in the order of these positions, so they
do not depend on the number of ranks.

To add a sampler: subclass :class:`SpatialSampler`, set ``size`` and ``global_indices``,
implement :meth:`~SpatialSampler.sample`, add its options to :mod:`kipe.options` and a
``case`` to :func:`build_sampler`.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import assert_never

from mpi4py import MPI

import numpy as np

from kipe._types import NDArray_f64, NDArray_i64
from kipe.forward_solver import FieldSpec, ForwardSolver
from kipe.options import ArraySamplingOptions, MeshSamplingOptions, StudyFileError


class SpatialSampler(ABC):
    """Evaluates the model fields at the measurement locations."""

    size: int
    """Number of entries of the sampled vector, over all ranks."""

    global_indices: NDArray_i64
    """Positions of this rank's entries in the sampled vector, in the order :meth:`sample`
    returns them; each entry is held by exactly one rank."""

    # NOTE: a measurement context (time-dependent geometry, auxiliary data such as the magnitude
    # or background phase of PC-MRI) is expected soon. It becomes an additional argument of
    # this method, e.g., `context: MeasurementContext` (see PLAN.md, "Measurement context").

    @abstractmethod
    def sample(self, fields: Mapping[str, NDArray_f64]) -> NDArray_f64:
        r"""Evaluate the model fields at the measurement locations.

        Args:
            fields: the state fields the measurement observes (its ``fields``), rank-local

        Returns:
            this rank's entries of the sampled state :math:`y`, at the positions
            ``global_indices``; a new array, not a view of ``fields`` or of internal buffers
        """


class ArraySampler(SpatialSampler):
    """Samples all entries of the observed fields, concatenated in the order of ``fields``.

    The fields must be replicated, i.e., held in full by every rank; rank 0 holds all entries
    of the sampled vector, the other ranks none.

    Args:
        fields: names of the observed fields, in the order of concatenation
        state_spec: the forward solver's state fields
        comm: communicator of the run

    Raises:
        StudyFileError: if a field is distributed over the ranks; its entries would depend on
            the partition, which needs a sampler of the backend, e.g., ``fenicsx.mesh``
    """

    def __init__(
        self,
        fields: list[str],
        state_spec: Mapping[str, FieldSpec],
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        distributed = [name for name in fields if state_spec[name].kind == "distributed"]
        if distributed:
            raise StudyFileError(
                f"field {', '.join(distributed)} is distributed over the MPI ranks; array "
                "sampling needs replicated fields, use a sampler of the backend instead"
            )

        self._fields = fields
        self._holds = comm.rank == 0
        self.size = sum(state_spec[name].local_size for name in fields)
        self.global_indices = np.arange(self.size if self._holds else 0, dtype=np.int64)

    def sample(self, fields: Mapping[str, NDArray_f64]) -> NDArray_f64:
        """Concatenate the observed fields, on rank 0.

        Args:
            fields: the observed state fields

        Returns:
            all entries of the fields, concatenated (a new array); empty on the other ranks
        """
        if not self._holds:
            return np.empty(0)

        return np.concatenate([fields[name] for name in self._fields])


def build_sampler(
    options: ArraySamplingOptions | MeshSamplingOptions,
    fields: list[str],
    solver: ForwardSolver,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> SpatialSampler:
    """Construct the sampler given by the ``spatial_sampling`` options of a measurement.

    Args:
        options: ``spatial_sampling`` options of the measurement
        fields: names of the fields the measurement observes
        solver: the forward solver; backend samplers take what they need from it, e.g., the
            function spaces
        comm: communicator of the run

    Returns:
        the spatial sampler

    Raises:
        StudyFileError: if a backend sampler is used but its backend is not installed
    """
    match options:
        case ArraySamplingOptions():
            return ArraySampler(fields, solver.state_spec, comm)
        case MeshSamplingOptions():
            try:  # dolfinx only when a study uses it
                from kipe.fenicsx.sampling import build_mesh_sampler
            except ImportError as err:
                raise StudyFileError(f"fenicsx.mesh needs dolfinx: {err}") from err
            return build_mesh_sampler(options, fields, solver, comm)
        case _:
            assert_never(options)
