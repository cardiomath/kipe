"""The fenicsx.points sampler: values at points, the order of the data, errors."""

import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

import numpy as np
from dolfinx import fem, mesh

from kipe._types import NDArray_f64
from kipe.fenicsx.examples.heat import Solver as Heat
from kipe.fenicsx.sampling.points import PointSampler
from kipe.measurements.operator import build_observation_operator
from kipe.options import (
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    PointSamplingOptions,
    StudyFileError,
)
from kipe.sampling import SpatialSampler

COMM = MPI.COMM_WORLD

# inside cells, on cell edges and vertices of the 12 x 12 mesh, and on the boundary: points on
# the partition boundary lie in cells of two ranks, but must be evaluated once
POINTS = [[0.31, 0.47], [0.5, 0.5], [0.25, 0.75], [0.0, 0.0], [1.0, 0.6], [0.123, 1.0]]


def _linear(x: NDArray_f64) -> NDArray_f64:
    """A linear function: P1 represents it exactly, so its point values are known."""
    return x[0] + 10 * x[1]


def _vector(x: NDArray_f64) -> NDArray_f64:
    """A linear vector field, (x + 10 y, 2 x - y)."""
    return np.vstack([x[0] + 10 * x[1], 2 * x[0] - x[1]])


def _owned(V: fem.FunctionSpace, f) -> NDArray_f64:
    """The owned degrees of freedom of f, interpolated into V: a state field."""
    u = fem.Function(V)
    u.interpolate(f)
    return u.x.array[: V.dofmap.index_map.size_local * V.dofmap.index_map_bs].copy()


def _gathered(sampler: SpatialSampler, local: NDArray_f64) -> NDArray_f64:
    """The sampled vector of all ranks, assembled by global_indices, on every rank."""
    full = np.full(sampler.size, np.nan)
    for indices, values in COMM.allgather((sampler.global_indices, local)):
        full[indices] = values
    return full


def test_values_at_the_points():
    V = Heat(n=12).function_space("u")
    sampler = PointSampler("u", V, POINTS)

    values = _gathered(sampler, sampler.sample({"u": _owned(V, _linear)}))
    np.testing.assert_allclose(values, _linear(np.array(POINTS).T), rtol=0, atol=1e-12)


def test_each_point_on_exactly_one_rank():
    sampler = PointSampler("u", Heat(n=12).function_space("u"), POINTS)

    indices = np.concatenate(COMM.allgather(sampler.global_indices))
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(POINTS)))


def test_vector_field_components_one_point_after_another():
    V = fem.functionspace(mesh.create_unit_square(COMM, 12, 12), ("Lagrange", 1, (2,)))
    sampler = PointSampler("u", V, POINTS)

    values = _gathered(sampler, sampler.sample({"u": _owned(V, _vector)}))
    expected = _vector(np.array(POINTS).T).T.ravel()  # (x0, y0, x1, y1, ...)
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)


def test_from_the_study_options():
    """Through the measurement's options: the case in kipe.sampling.build_sampler."""
    heat = Heat(n=12)
    options = MeasurementOptions(
        fields=["u"],
        data=NumpyDataOptions(type="numpy", path="unused.npz"),
        noise=NoiseOptions(stddev=0.1),
        spatial_sampling=PointSamplingOptions(type="fenicsx.points", points=POINTS),
    )
    operator = build_observation_operator("u", options, heat)

    state = {"u": _owned(heat.function_space("u"), _linear)}
    values = _gathered(operator.sampler, operator(state))
    np.testing.assert_allclose(values, _linear(np.array(POINTS).T), rtol=0, atol=1e-12)


def test_point_outside_the_mesh():
    V = Heat(n=4).function_space("u")
    with pytest.raises(StudyFileError, match=r"1 of 2 points lie outside .* \[1\.5, 0\.5\]"):
        PointSampler("u", V, [[0.5, 0.5], [1.5, 0.5]])


def test_points_need_the_mesh_dimension():
    V = Heat(n=4).function_space("u")
    with pytest.raises(StudyFileError, match="each point needs 2 coordinates"):
        PointSampler("u", V, [[0.5, 0.5, 0.0]])
