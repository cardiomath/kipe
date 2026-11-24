"""Displaced facet vertices fitted to a plane, in serial and under MPI."""

import pytest

pytest.importorskip("dolfinx")

from types import SimpleNamespace

from mpi4py import MPI

import numpy as np
from dolfinx import fem, mesh

from kipe.measurements.operator import build_observation_operator
from kipe.options import (
    DisplacedVertexSamplingOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    PlaneModelOptions,
)


def test_displaced_vertices_to_plane_coefficients():
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 2, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 1.0))
    tags = mesh.meshtags(domain, 2, facets, np.full(len(facets), 30, dtype=np.int32))
    owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    class Solver:
        def __init__(self):
            self.state_spec = {"u": SimpleNamespace(local_size=owned_size, kind="distributed")}

        def function_space(self, field):
            assert field == "u"
            return V

        def facet_tags(self, field):
            assert field == "u"
            return tags

    options = MeasurementOptions(
        fields=["u"],
        data=NumpyDataOptions(type="numpy", path="unused.npz"),
        noise=NoiseOptions(stddev=0.01),
        spatial_sampling=DisplacedVertexSamplingOptions(
            type="fenicsx.displaced_vertices", tags=[30], combine="union"
        ),
        model=PlaneModelOptions(),
    )
    operator = build_observation_operator("plane", options, Solver(), comm)
    assert operator.sampler.size == 27  # nine 3D vertices on the top surface
    assert operator.size == 3
    np.testing.assert_array_equal(
        np.concatenate(comm.allgather(operator.global_indices)), [0, 1, 2]
    )
    indices = operator.global_indices
    np.testing.assert_allclose(
        operator({"u": np.zeros(owned_size)}), np.array([0.0, 0.0, 1.0])[indices], atol=1e-12
    )
    u = fem.Function(V)
    u.interpolate(
        lambda x: np.vstack((
            np.zeros_like(x[0]),
            np.zeros_like(x[0]),
            0.2 * x[0] - 0.3 * x[1] + 0.1,
        ))
    )
    original = domain.geometry.x.copy()
    np.testing.assert_allclose(
        operator({"u": u.x.array[:owned_size].copy()}),
        np.array([0.2, -0.3, 1.1])[indices],
        atol=1e-12,
    )
    np.testing.assert_array_equal(domain.geometry.x, original)
