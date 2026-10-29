"""Spatial samplers on meshes: ``fenicsx.mesh``.

:class:`MeshSampler` interpolates a field of the solver onto a measurement mesh, which need not
match the solver's mesh, optionally projected onto a direction or the surface normal. Its
entries are in the order of the measurement mesh file: one value per vertex for ``P1`` (by the
vertex's row in the file), one per cell for ``DG0`` (by the cell's row); for a vector field
without projection, the components of each vertex or cell one after another. dolfinx keeps
these rows however it distributes and renumbers the mesh, so the data do not depend on the
number of ranks (see :mod:`kipe.sampling`).
"""

from collections.abc import Mapping
from typing import Literal

from mpi4py import MPI

import numpy as np
import ufl
from dolfinx import fem, geometry, io
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr

from kipe._types import NDArray_f64, NDArray_i64
from kipe.fenicsx.sampling.common import observed_function_space
from kipe.forward_solver import ForwardSolver
from kipe.options import MeshSamplingOptions, StudyFileError
from kipe.sampling import SpatialSampler

type Element = Literal["P1", "DG0"]
type Projection = list[float] | Literal["surface_normal"] | None


class MeshSampler(SpatialSampler):
    """Interpolates a field onto a measurement mesh, optionally projected.

    Args:
        field: name of the observed field
        source: the field's function space, on the solver's mesh
        mesh: path to the measurement mesh, an XDMF file
        element: finite element on the measurement mesh
        projection: a direction (one entry per component of the field) or
            ``"surface_normal"`` (the measurement mesh must be a surface); all components if
            None
        padding: distance by which a measurement point may lie outside the solver's mesh
        mesh_name: name of the mesh in the XDMF file
        comm: communicator of the run

    Raises:
        StudyFileError: if the file has no mesh named ``mesh_name``, the measurement mesh has
            curved cells (``P1`` needs a linear geometry), the projection does not fit the field
            or the mesh, or points of the measurement mesh lie outside the solver's mesh,
            beyond the padding
    """

    def __init__(
        self,
        field: str,
        source: fem.FunctionSpace,
        mesh: str,
        element: Element,
        projection: Projection = None,
        padding: float = 0.0,
        mesh_name: str = "mesh",
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        with io.XDMFFile(comm, mesh, "r") as f:
            try:
                measurement_mesh = f.read_mesh(name=mesh_name)
            except RuntimeError as err:
                raise StudyFileError(
                    f"cannot read the mesh {mesh_name!r} of {mesh} (spatial_sampling.mesh_name, "
                    f"e.g., Grid for meshio's files): {err}"
                ) from err

        if element == "P1" and measurement_mesh.geometry.cmap.degree != 1:
            raise StudyFileError(
                f"the measurement mesh {mesh} has curved cells; P1 needs linear ones"
            )
        shape = source.value_shape
        if len(shape) > 1:
            raise StudyFileError(
                f"fenicsx.mesh supports scalar and vector fields, {field} is not"
            )

        family = ("Lagrange", 1) if element == "P1" else ("DG", 0)
        target = fem.functionspace(measurement_mesh, (*family, shape))

        tdim = measurement_mesh.topology.dim
        cells = measurement_mesh.topology.index_map(tdim)
        self._cells = np.arange(cells.size_local + cells.num_ghosts, dtype=np.int32)
        # dolfinx's default padding at the least, for round-off
        self._interpolation = fem.create_interpolation_data(
            target, source, self._cells, padding=max(padding, 1e-14)
        )
        _check_inside(
            self._interpolation,
            len(self._cells),
            cells.size_local,
            cells.size_global,
            padding,
            comm,
        )

        self._field = field
        self._source = fem.Function(source)
        self._target = fem.Function(target)
        self._n_source = source.dofmap.index_map.size_local * source.dofmap.index_map_bs

        # the projection: u . d interpolated into a scalar space on the measurement mesh; exact
        # for P1 and DG0, which evaluate at the vertices and midpoints, where u's values are
        self._projected: fem.Function | None = None
        output = target
        if projection is not None:
            vector = _projection_vector(projection, field, shape, measurement_mesh, mesh)
            output = fem.functionspace(measurement_mesh, family)
            self._projected = fem.Function(output)
            points = output.element.interpolation_points  # a method in 0.9, a property from 0.10
            self._expression = fem.Expression(
                ufl.dot(self._target, vector), points() if callable(points) else points
            )

        bs = output.dofmap.index_map_bs
        n_owned = output.dofmap.index_map.size_local
        self._n_values = n_owned * bs
        self.size = output.dofmap.index_map.size_global * bs
        rows = _file_index(output, element)[:n_owned]
        self.global_indices = (rows[:, np.newaxis] * bs + np.arange(bs)).ravel()

    def sample(self, fields: Mapping[str, NDArray_f64]) -> NDArray_f64:
        """Interpolate the field onto the measurement mesh, and project it.

        Args:
            fields: the observed state field, this rank's owned degrees of freedom

        Returns:
            the values of this rank's owned degrees of freedom on the measurement mesh (a new
            array), at the positions ``global_indices``
        """
        self._source.x.array[: self._n_source] = fields[self._field]
        self._source.x.scatter_forward()
        self._target.interpolate_nonmatching(self._source, self._cells, self._interpolation)

        output = self._target
        if self._projected is not None:
            self._projected.interpolate(self._expression)
            output = self._projected

        return output.x.array[: self._n_values].copy()


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
        StudyFileError: if the measurement does not observe exactly one distributed field whose
            function space the solver provides
    """
    field, source = observed_function_space("fenicsx.mesh", fields, solver)
    return MeshSampler(
        field,
        source,
        options.mesh,
        options.element,
        options.projection,
        options.padding,
        options.mesh_name,
        comm,
    )


def _check_inside(
    interpolation: geometry.PointOwnershipData,
    n_cells: int,
    n_owned: int,
    n_global: int,
    padding: float,
    comm: MPI.Comm,
) -> None:
    """Check that all points of the measurement mesh lie in the solver's mesh, within padding.

    dolfinx gives a point it does not find the value 0, without a warning.

    Args:
        interpolation: the interpolation data: the owner of each point (-1: not found), the
            points of each cell one after another
        n_cells: number of cells interpolated on this rank, owned and ghost
        n_owned: number of owned cells among them, the first ones
        n_global: number of cells of the measurement mesh
        padding: the padding the points were searched with
        comm: communicator of the run

    Raises:
        StudyFileError: if a cell has a point outside the solver's mesh
    """
    outside = 0
    if n_cells:
        src_owner = interpolation.src_owner  # a method in dolfinx 0.9, a property from 0.10
        owners = np.asarray(src_owner() if callable(src_owner) else src_owner)
        owners = owners.reshape(n_cells, -1)
        outside = int(np.any(owners[:n_owned] < 0, axis=1).sum())
    outside = comm.allreduce(outside)
    if outside:
        beyond = f"farther than padding {padding:g} " if padding else ""
        raise StudyFileError(
            f"{outside} of {n_global} cells of the measurement mesh lie (partly) {beyond}"
            "outside the solver's mesh; spatial_sampling.padding lets points within a distance "
            "take the value of the nearest cell"
        )


def _projection_vector(
    projection: list[float] | Literal["surface_normal"],
    field: str,
    shape: tuple[int, ...],
    mesh: Mesh,
    path: str,
) -> Expr:
    """Return the vector a field is projected onto: the normalized direction, or the cell normal.

    Args:
        projection: a direction or ``"surface_normal"``
        field: name of the field, for errors
        shape: the field's value shape
        mesh: the measurement mesh
        path: its file, for errors

    Returns:
        the UFL vector

    Raises:
        StudyFileError: if the field is not a vector field, the direction does not have one
            entry per component, or the surface normal is asked for on a mesh that is not a
            surface with one dimension less than its space
    """
    if not shape:
        raise StudyFileError(f"projection needs a vector field, {field} is scalar")

    if projection == "surface_normal":
        tdim, gdim = mesh.topology.dim, mesh.geometry.dim
        if tdim != gdim - 1 or shape != (gdim,):
            raise StudyFileError(
                f"projection surface_normal needs a surface mesh ({path} has cells of dimension "
                f"{tdim} in {gdim}D) and a field with {gdim} components ({field}: {shape[0]})"
            )
        return ufl.CellNormal(mesh)

    if len(projection) != shape[0]:
        raise StudyFileError(
            f"projection: the direction has {len(projection)} entries, {field} has "
            f"{shape[0]} components"
        )
    direction = np.asarray(projection, dtype=np.float64)
    return ufl.as_vector((direction / np.linalg.norm(direction)).tolist())


def _file_index(V: fem.FunctionSpace, element: Element) -> NDArray_i64:
    """Return, for each local degree of freedom (block) of V, its row in the mesh file.

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
