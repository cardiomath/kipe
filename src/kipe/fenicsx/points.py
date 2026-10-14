"""Spatial sampler at points: ``fenicsx.points``.

:class:`PointSampler` evaluates a field of the solver at given points. Its entries are in the
order of the points; for a vector field, the components of each point one after another. This
order is fixed by the study file, so the data do not depend on the number of ranks (see
:mod:`kipe.sampling`).

A small, complete sampler: the how-to "Write a spatial sampler" in the documentation walks
through it.
"""

from collections.abc import Mapping

from mpi4py import MPI

import numpy as np
from dolfinx import fem, geometry

from kipe._types import NDArray_f64
from kipe.fenicsx.sampling import observed_function_space
from kipe.forward_solver import ForwardSolver
from kipe.options import PointSamplingOptions, StudyFileError
from kipe.sampling import SpatialSampler


class PointSampler(SpatialSampler):
    """Evaluates a field at given points.

    Each point is evaluated by one rank: the lowest rank that owns a cell containing it. A point
    on the boundary between the cells of two ranks lies in cells of both.

    Args:
        field: name of the observed field
        source: the field's function space, on the solver's mesh
        points: coordinates of the points, one row per point
        comm: communicator of the run

    Raises:
        StudyFileError: if the points do not have the coordinates of the mesh's dimension, or a
            point lies outside the mesh
    """

    def __init__(
        self,
        field: str,
        source: fem.FunctionSpace,
        points: list[list[float]],
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        mesh = source.mesh
        gdim = mesh.geometry.dim
        if any(len(point) != gdim for point in points):
            raise StudyFileError(f"fenicsx.points: each point needs {gdim} coordinates")
        x = np.zeros((len(points), 3))  # dolfinx takes points in 3D
        x[:, :gdim] = points

        # the cell of this rank that contains each point, -1 if none; owned cells only, so
        # that a cell is not found by its owner and by the ranks that have it as a ghost
        tdim = mesh.topology.dim
        n_owned = mesh.topology.index_map(tdim).size_local
        tree = geometry.bb_tree(mesh, tdim, np.arange(n_owned, dtype=np.int32))
        candidates = geometry.compute_collisions_points(tree, x)
        colliding = geometry.compute_colliding_cells(mesh, candidates, x)
        cells = np.full(len(x), -1, dtype=np.int32)
        for i in range(len(x)):
            if len(colliding.links(i)):
                cells[i] = colliding.links(i)[0]

        # each point to the lowest rank that found it; comm.size: no rank did
        holder = np.where(cells >= 0, comm.rank, comm.size).astype(np.int32)
        comm.Allreduce(MPI.IN_PLACE, holder, op=MPI.MIN)
        outside = np.flatnonzero(holder == comm.size)
        if len(outside):
            raise StudyFileError(
                f"fenicsx.points: {len(outside)} of {len(x)} points lie outside the solver's "
                f"mesh, e.g., {list(points[outside[0]])}"
            )
        mine = np.flatnonzero(holder == comm.rank)

        self._field = field
        self._function = fem.Function(source)
        self._n_owned = source.dofmap.index_map.size_local * source.dofmap.index_map_bs
        self._points, self._cells = x[mine], cells[mine]

        # the sampled vector: the values of the points, components of a point one after another
        value_size = int(np.prod(source.value_shape))
        self.size = len(x) * value_size
        self.global_indices = (mine[:, np.newaxis] * value_size + np.arange(value_size)).ravel()

    def sample(self, fields: Mapping[str, NDArray_f64]) -> NDArray_f64:
        """Evaluate the field at this rank's points.

        Args:
            fields: the observed state field, this rank's owned degrees of freedom

        Returns:
            the values at this rank's points (a new array), at the positions
            ``global_indices``
        """
        self._function.x.array[: self._n_owned] = fields[self._field]
        # the values of the ghost degrees of freedom, which cells at the partition boundary use
        self._function.x.scatter_forward()

        values: NDArray_f64 = self._function.eval(self._points, self._cells)
        return values.ravel()


def build_point_sampler(
    options: PointSamplingOptions,
    fields: list[str],
    solver: ForwardSolver,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> PointSampler:
    """Construct a :class:`PointSampler` from the ``spatial_sampling`` options of a measurement.

    Args:
        options: ``spatial_sampling`` options of the measurement
        fields: names of the fields the measurement observes; exactly one
        solver: the forward solver, which provides the field's function space
        comm: communicator of the run

    Returns:
        the sampler
    """
    field, source = observed_function_space("fenicsx.points", fields, solver)
    return PointSampler(field, source, options.points, comm)
