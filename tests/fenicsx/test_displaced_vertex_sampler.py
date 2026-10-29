"""Coordinates of tagged vertices, including tag intersections across MPI ranks."""

import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

import numpy as np
from dolfinx import fem, io, mesh

from kipe.fenicsx.sampling.displaced_vertices import DisplacedVertexSampler
from kipe.measurements.operator import build_observation_operator
from kipe.options import (
    DisplacedVertexSamplingOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    StudyFileError,
)

COMM = MPI.COMM_WORLD


@pytest.fixture
def domain(shared_tmp_path):
    """Read the same file on any number of ranks, with known geometry row order."""
    path = shared_tmp_path / "coordinates.xdmf"
    reference = None
    if COMM.rank == 0:
        serial = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
        reference = serial.geometry.x[:, :2].copy()
        with io.XDMFFile(MPI.COMM_SELF, path, "w") as f:
            f.write_mesh(serial)
    reference = COMM.bcast(reference, root=0)
    with io.XDMFFile(COMM, path, "r") as f:
        distributed = f.read_mesh()
    return distributed, reference


def _tags(domain):
    fdim = domain.topology.dim - 1
    indices, values = [], []
    for tag, marker in [
        (10, lambda x: np.isclose(x[0], 0)),
        (20, lambda x: np.isclose(x[1], 0)),
        (30, lambda x: np.isclose(x[0], 1)),
    ]:
        facets = mesh.locate_entities_boundary(domain, fdim, marker)
        indices.extend(facets)
        values.extend([tag] * len(facets))
    order = np.argsort(indices)
    return mesh.meshtags(
        domain,
        fdim,
        np.asarray(indices, dtype=np.int32)[order],
        np.asarray(values, dtype=np.int32)[order],
    )


def _displacement(x):
    return np.vstack([0.1 + x[0] ** 2, 0.2 + x[1] ** 2])


def _owned(V, function):
    u = fem.Function(V)
    u.interpolate(function)
    n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    return u.x.array[:n].copy()


def _gathered(sampler, local):
    parts = COMM.allgather((sampler.global_indices, local))
    indices = np.concatenate([part[0] for part in parts])
    np.testing.assert_array_equal(np.sort(indices), np.arange(sampler.size))
    full = np.full(sampler.size, np.nan)
    for indices, values in parts:
        full[indices] = values
    return full.reshape(-1, 2)


@pytest.mark.parametrize("operation", ["union", "intersection"])
def test_displaced_coordinates_in_file_order(domain, operation):
    distributed, reference = domain
    V = fem.functionspace(distributed, ("Lagrange", 2, (2,)))
    sampler = DisplacedVertexSampler("u", V, _tags(distributed), [10, 20], operation)
    original_mesh = distributed.geometry.x.copy()
    owned = _owned(V, _displacement)
    original_field = owned.copy()
    actual = _gathered(sampler, sampler.sample({"u": owned}))
    left, bottom = np.isclose(reference[:, 0], 0), np.isclose(reference[:, 1], 0)
    selected = reference[left | bottom if operation == "union" else left & bottom]
    np.testing.assert_allclose(actual, selected + _displacement(selected.T).T, atol=1e-12)
    np.testing.assert_array_equal(distributed.geometry.x, original_mesh)
    np.testing.assert_array_equal(owned, original_field)

    # A later call must use the current field and leave the previous result valid.
    zero = np.zeros_like(owned)
    np.testing.assert_allclose(_gathered(sampler, sampler.sample({"u": zero})), selected)
    np.testing.assert_allclose(actual, selected + _displacement(selected.T).T, atol=1e-12)


def test_duplicate_and_single_tags(domain):
    distributed, reference = domain
    V = fem.functionspace(distributed, ("Lagrange", 1, (2,)))
    sampler = DisplacedVertexSampler("u", V, _tags(distributed), [10, 10])
    actual = _gathered(
        sampler, sampler.sample({"u": np.zeros(V.dofmap.index_map.size_local * 2)})
    )
    np.testing.assert_allclose(actual, reference[np.isclose(reference[:, 0], 0)])


def test_cell_tags_are_rejected(domain):
    distributed, _ = domain
    V = fem.functionspace(distributed, ("Lagrange", 1, (2,)))
    tdim = distributed.topology.dim
    cells = np.arange(distributed.topology.index_map(tdim).size_local, dtype=np.int32)
    tags = mesh.meshtags(distributed, tdim, cells, np.full(len(cells), 10, dtype=np.int32))
    with pytest.raises(StudyFileError, match="needs facet tags"):
        DisplacedVertexSampler("u", V, tags, [10])


@pytest.mark.parametrize(
    ("tags", "operation", "message"),
    [
        ([999], "union", "tag 999 was not found"),
        ([10, 30], "intersection", "vertex set is empty"),
        ([], "union", "needs tags"),
        ([10], "invalid", "operation union or intersection"),
    ],
)
def test_invalid_selection(domain, tags, operation, message):
    distributed, _ = domain
    V = fem.functionspace(distributed, ("Lagrange", 1, (2,)))
    with pytest.raises(StudyFileError, match=message):
        DisplacedVertexSampler("u", V, _tags(distributed), tags, operation)


@pytest.mark.parametrize("shape", [(), (3,)])
def test_requires_displacement_vector(domain, shape):
    distributed, _ = domain
    V = fem.functionspace(distributed, ("Lagrange", 1, shape))
    with pytest.raises(StudyFileError, match="displacement vector with 2 components"):
        DisplacedVertexSampler("u", V, _tags(distributed), [10])


def test_builder_uses_solver_facet_tags(domain):
    distributed, reference = domain
    V = fem.functionspace(distributed, ("Lagrange", 1, (2,)))

    class Spec:
        kind = "distributed"
        local_size = V.dofmap.index_map.size_local * 2

    class Solver:
        def __init__(self):
            self.state_spec = {"u": Spec()}

        def function_space(self, field):
            assert field == "u"
            return V

        def facet_tags(self, field):
            assert field == "u"
            return _tags(distributed)

    options = MeasurementOptions(
        fields=["u"],
        data=NumpyDataOptions(type="numpy", path="unused.npz"),
        noise=NoiseOptions(stddev=0.1),
        spatial_sampling=DisplacedVertexSamplingOptions(
            type="fenicsx.displaced_vertices", tags=[10, 20]
        ),
    )
    operator = build_observation_operator("coordinates", options, Solver())
    actual = _gathered(operator.sampler, operator({"u": np.zeros(Spec.local_size)}))
    expected = reference[np.isclose(reference[:, 0], 0) & np.isclose(reference[:, 1], 0)]
    np.testing.assert_allclose(actual, expected)

    class WithoutFacetTags:
        def __init__(self):
            self.state_spec = {"u": Spec()}

        function_space = Solver.function_space

    with pytest.raises(StudyFileError, match="needs a facet_tags method"):
        build_observation_operator("coordinates", options, WithoutFacetTags())


def test_three_dimensional_surface_intersection():
    domain = mesh.create_unit_cube(COMM, 2, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    sampler = DisplacedVertexSampler("u", V, _tags(domain), [10, 20])
    translation = np.array([0.1, 0.2, 0.3])
    values = sampler.sample({
        "u": _owned(V, lambda x: np.tile(translation[:, None], (1, x.shape[1])))
    })
    parts = COMM.allgather((sampler.global_indices, values))
    full = np.empty(sampler.size)
    for indices, local in parts:
        full[indices] = local
    points = full.reshape(-1, 3) - translation
    assert len(points) == 3
    np.testing.assert_allclose(points[:, :2], 0, atol=1e-12)
    np.testing.assert_allclose(np.sort(points[:, 2]), [0, 0.5, 1], atol=1e-12)
