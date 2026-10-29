"""Displaced coordinates of tagged mesh vertices: ``fenicsx.displaced_vertices``.

Selection and ordering are fixed at construction. Tags are combined globally before
intersecting their vertex sets, including when different ranks hold the adjacent surfaces.
The original geometry input indices define a partition-independent order. Like PointSampler,
this implementation replicates the selected reference points on all ranks at construction;
each sample returns only this rank's entries, with components of each vertex together.
"""

from collections.abc import Mapping
from typing import Literal

from mpi4py import MPI

import numpy as np
from dolfinx import fem, io
from dolfinx.mesh import entities_to_geometry

from kipe._types import NDArray_f64
from kipe.fenicsx.sampling.common import observed_function_space
from kipe.fenicsx.sampling.points import PointSampler
from kipe.forward_solver import ForwardSolver
from kipe.options import DisplacedVertexSamplingOptions, StudyFileError


class DisplacedVertexSampler(PointSampler):
    """Evaluate ``X + u(X)`` at vertices selected by a union or intersection of tags.

    Only topological vertices are selected, including on higher-order meshes. The displacement
    is evaluated through its function space; geometry indices are never treated as field DOFs.
    The reference mesh is not modified.

    Args:
        field: name of the displacement field
        source: displacement function space on the reference mesh
        mesh: XDMF file containing the selection mesh and its facet tags
        tags: nonempty list of tag IDs
        operation: union or intersection of the tags' vertex sets; intersection by default
        comm: communicator of the mesh and run
        mesh_name: name of the mesh in the XDMF file
        tags_name: name of the facet tags in the XDMF file

    Raises:
        StudyFileError: if the field is not a displacement vector, a tag is absent globally,
            or the selection is empty or invalid
    """

    def __init__(
        self,
        field: str,
        source: fem.FunctionSpace,
        mesh: str,
        tags: list[int],
        operation: Literal["union", "intersection"] = "intersection",
        comm: MPI.Comm = MPI.COMM_WORLD,
        mesh_name: str = "mesh",
        tags_name: str = "Facet tags",
    ) -> None:
        gdim = source.mesh.geometry.dim
        if source.value_shape != (gdim,):
            raise StudyFileError(
                f"fenicsx.displaced_vertices needs a displacement vector with {gdim} components"
            )
        if not tags or operation not in ("union", "intersection"):
            raise StudyFileError(
                "fenicsx.displaced_vertices needs tags and operation union or intersection"
            )
        try:
            with io.XDMFFile(comm, mesh, "r") as file:
                selection_mesh = file.read_mesh(name=mesh_name)
                tdim = selection_mesh.topology.dim
                selection_mesh.topology.create_connectivity(tdim - 1, tdim)
                mesh_tags = file.read_meshtags(selection_mesh, name=tags_name)
        except (RuntimeError, OSError) as err:
            raise StudyFileError(
                f"fenicsx.displaced_vertices: cannot read mesh {mesh_name!r} and facet tags "
                f"{tags_name!r} from {mesh}: {err}"
            ) from err
        if selection_mesh.geometry.dim != gdim:
            raise StudyFileError(
                "fenicsx.displaced_vertices: selection mesh and displacement must have "
                "the same geometric dimension"
            )
        dim = mesh_tags.dim
        if dim != tdim - 1:
            raise StudyFileError("fenicsx.displaced_vertices needs facet tags")
        selection_mesh.topology.create_connectivity(0, tdim)
        if dim:
            selection_mesh.topology.create_connectivity(dim, 0)
            connectivity = selection_mesh.topology.connectivity(dim, 0)

        vertex_sets = []
        coordinates: dict[int, list[float]] = {}
        for tag in dict.fromkeys(tags):
            entities = mesh_tags.find(tag)
            vertices = (
                np.unique(np.concatenate([connectivity.links(e) for e in entities]))
                if dim and len(entities)
                else entities
            )
            geometry_nodes = (
                entities_to_geometry(selection_mesh, 0, vertices.astype(np.int32), False).ravel()
                if len(vertices)
                else np.empty(0, dtype=np.int32)
            )
            ids = selection_mesh.geometry.input_global_indices[geometry_nodes]
            x = selection_mesh.geometry.x[geometry_nodes, :gdim]
            parts = comm.allgather((ids, x))
            global_ids = np.unique(np.concatenate([part[0] for part in parts]))
            if not len(global_ids):
                raise StudyFileError(f"fenicsx.displaced_vertices: tag {tag} was not found")
            vertex_sets.append(global_ids)
            for part_ids, part_x in parts:
                coordinates.update(zip(part_ids.tolist(), part_x.tolist(), strict=True))

        selected = vertex_sets[0]
        combine = np.union1d if operation == "union" else np.intersect1d
        for vertices in vertex_sets[1:]:
            selected = combine(selected, vertices)
        if not len(selected):
            raise StudyFileError("fenicsx.displaced_vertices: the selected vertex set is empty")

        points = [coordinates[int(node)] for node in selected]
        super().__init__(field, source, points, comm)
        self._reference_coordinates = self._points[:, :gdim].ravel().copy()

    def sample(self, fields: Mapping[str, NDArray_f64]) -> NDArray_f64:
        """Return this rank's displaced coordinates as a new flat array.

        Args:
            fields: displacement field, this rank's owned degrees of freedom

        Returns:
            reference coordinates plus evaluated displacement, at ``global_indices``
        """
        return self._reference_coordinates + super().sample(fields)


def build_displaced_vertex_sampler(
    options: DisplacedVertexSamplingOptions,
    fields: list[str],
    solver: ForwardSolver,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> DisplacedVertexSampler:
    """Construct the coordinate sampler from study options and solver capabilities.

    Args:
        options: spatial sampling options
        fields: names of the observed fields; exactly one displacement field
        solver: solver providing ``function_space(field)``
        comm: communicator of the run

    Returns:
        the coordinate sampler

    Raises:
        StudyFileError: if the solver does not provide the required capabilities
    """
    kind = "fenicsx.displaced_vertices"
    field, source = observed_function_space(kind, fields, solver)
    return DisplacedVertexSampler(
        field,
        source,
        options.mesh,
        options.tags,
        options.operation,
        comm,
        options.mesh_name,
        options.tags_name,
    )
