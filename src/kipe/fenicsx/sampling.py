"""Spatial samplers on meshes: ``fenicsx.mesh``.

:class:`MeshSampler` interpolates a field of the solver onto a measurement mesh, which need not
match the solver's mesh. Its entries are in the order of the measurement mesh file: one value
per vertex for ``P1`` (by the vertex's row in the file), one per cell for ``DG0`` (by the
cell's row). dolfinx keeps these rows however it distributes and renumbers the mesh, so the
data do not depend on the number of ranks (see :mod:`kipe.sampling`).
"""

from collections.abc import Mapping
from typing import Literal

from mpi4py import MPI

import numpy as np
from dolfinx import fem, io

from kipe._types import NDArray_f64, NDArray_i64
from kipe.fenicsx.forward_solver import SupportsFunctionSpaces
from kipe.forward_solver import ForwardSolver
from kipe.options import MeshSamplingOptions, StudyFileError
from kipe.sampling import SpatialSampler

type Element = Literal["P1", "DG0"]


class MeshSampler(SpatialSampler):
    """Interpolates a scalar field onto a measurement mesh.

    Args:
        field: name of the observed field
        source: the field's function space, on the solver's mesh
        mesh: path to the measurement mesh, an XDMF file with a mesh named ``mesh``
        element: finite element on the measurement mesh
        comm: communicator of the run

    Raises:
        StudyFileError: if the measurement mesh has curved cells (``P1`` needs a linear
            geometry)
    """

    def __init__(
        self,
        field: str,
        source: fem.FunctionSpace,
        mesh: str,
        element: Element,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        with io.XDMFFile(comm, mesh, "r") as f:
            measurement_mesh = f.read_mesh(name="mesh")

        if element == "P1" and measurement_mesh.geometry.cmap.degree != 1:
            raise StudyFileError(
                f"the measurement mesh {mesh} has curved cells; P1 needs linear ones"
            )

        family = ("Lagrange", 1) if element == "P1" else ("DG", 0)
        target = fem.functionspace(measurement_mesh, family)

        tdim = measurement_mesh.topology.dim
        cells = measurement_mesh.topology.index_map(tdim)
        self._cells = np.arange(cells.size_local + cells.num_ghosts, dtype=np.int32)
        self._interpolation = fem.create_interpolation_data(target, source, self._cells)

        self._field = field
        self._source = fem.Function(source)
        self._target = fem.Function(target)
        self._n_source = source.dofmap.index_map.size_local
        self._n_owned = target.dofmap.index_map.size_local

        self.size = target.dofmap.index_map.size_global
        self.global_indices = _file_index(target, element)[: self._n_owned]

    def sample(self, fields: Mapping[str, NDArray_f64]) -> NDArray_f64:
        """Interpolate the field onto the measurement mesh.

        Args:
            fields: the observed state field, this rank's owned degrees of freedom

        Returns:
            the values of this rank's owned degrees of freedom on the measurement mesh (a new
            array), at the positions ``global_indices``
        """
        self._source.x.array[: self._n_source] = fields[self._field]
        self._source.x.scatter_forward()
        self._target.interpolate_nonmatching(self._source, self._cells, self._interpolation)

        return self._target.x.array[: self._n_owned].copy()


def build_mesh_sampler(
    options: MeshSamplingOptions,
    fields: list[str],
    solver: ForwardSolver,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> MeshSampler:
    """Construct a :class:`MeshSampler` from the ``spatial_sampling`` options of a measurement.

    Args:
        options: ``spatial_sampling`` options of the measurement
        fields: names of the fields the measurement observes; exactly one
        solver: the forward solver, which provides the field's function space
        comm: communicator of the run

    Returns:
        the sampler

    Raises:
        StudyFileError: if the measurement does not observe exactly one distributed, scalar
            field whose function space the solver provides
    """
    if len(fields) != 1:
        raise StudyFileError(f"fenicsx.mesh samples one field, got {', '.join(fields)}")
    (field,) = fields

    if solver.state_spec[field].kind != "distributed":
        raise StudyFileError(f"fenicsx.mesh needs a distributed field, {field} is replicated")
    if not isinstance(solver, SupportsFunctionSpaces):
        raise StudyFileError(
            f"fenicsx.mesh needs the function space of {field}, but the forward solver has no "
            "function_space method (see kipe.fenicsx.forward_solver)"
        )

    source = solver.function_space(field)
    if source.dofmap.index_map_bs != 1:
        raise StudyFileError(f"fenicsx.mesh supports scalar fields only, {field} is not scalar")

    return MeshSampler(field, source, options.mesh, options.element, comm)


def _file_index(V: fem.FunctionSpace, element: Element) -> NDArray_i64:
    """Return, for each local degree of freedom of V, its row in the mesh file.

    P1 on a linear mesh: per cell, the degrees of freedom and the geometry nodes are listed in
    the same order, and each node knows its row (``input_global_indices``). DG0: one degree of
    freedom per cell, and each cell knows its row (``original_cell_index``).

    Args:
        V: the function space on the measurement mesh
        element: its finite element

    Returns:
        the rows, for owned and ghost degrees of freedom
    """
    index_map = V.dofmap.index_map
    file_index = np.empty(index_map.size_local + index_map.num_ghosts, dtype=np.int64)

    mesh = V.mesh
    if element == "P1":
        file_index[V.dofmap.list] = mesh.geometry.input_global_indices[mesh.geometry.dofmap]
    else:
        n_cells = len(V.dofmap.list)
        file_index[V.dofmap.list[:, 0]] = mesh.topology.original_cell_index[:n_cells]

    return file_index
