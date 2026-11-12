"""Plane fitting and the observation operator, including distributed vertex coordinates."""

import sys
import types
from types import SimpleNamespace

from mpi4py import MPI

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from kipe import sensitivity
from kipe.estimation import estimate
from kipe.measurements.data import read_numpy
from kipe.measurements.models import build_model
from kipe.measurements.operator import build_observation_operator
from kipe.measurements.plane import PlaneModel, fit_plane
from kipe.options import (
    EstimationOptions,
    ForwardSolverOptions,
    MeasurementModelOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    OutputOptions,
    ParameterPrior,
    ParametersOptions,
    PlaneModelOptions,
    StudyFileError,
    StudyOptions,
    SynthesisNoiseOptions,
    SynthesisOptions,
)
from kipe.sampling import SpatialSampler
from kipe.synthesis import synthesize


def _vertices(a=0.2, b=-0.3, c=1.1):
    xy = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    return np.column_stack((xy, a * xy[:, 0] + b * xy[:, 1] + c))


def test_fit_plane_exact_and_translated_vertices():
    vertices = _vertices()
    original = vertices.copy()
    np.testing.assert_allclose(fit_plane(vertices), [0.2, -0.3, 1.1], atol=1e-14)
    translation = np.array([0.4, -0.2, 0.7])
    expected_shift = translation[2] - 0.2 * translation[0] + 0.3 * translation[1]
    np.testing.assert_allclose(
        fit_plane(vertices + translation), [0.2, -0.3, 1.1 + expected_shift], atol=1e-14
    )
    np.testing.assert_array_equal(vertices, original)
    # Exactly three noncollinear vertices also define a plane.
    np.testing.assert_allclose(fit_plane(vertices[:3]), [0.2, -0.3, 1.1], atol=1e-14)


def test_fit_plane_orthogonal_distances_and_vertex_order():
    """Symmetric perturbations along the normal preserve the exact best-fit plane."""
    vertices = _vertices()
    normal = np.array([-0.2, 0.3, 1.0])
    normal /= np.linalg.norm(normal)
    perturbed = np.vstack((vertices + 0.01 * normal, vertices - 0.01 * normal))
    np.testing.assert_allclose(fit_plane(perturbed), [0.2, -0.3, 1.1], atol=1e-14)
    np.testing.assert_allclose(fit_plane(perturbed[::-1]), fit_plane(perturbed), atol=1e-14)


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        (np.empty((0, 3)), "at least three 3D vertices"),
        (np.zeros((2, 3)), "at least three 3D vertices"),
        (np.zeros((4, 2)), "at least three 3D vertices"),
        (np.zeros(12), "at least three 3D vertices"),
        (np.full((4, 3), np.nan), "finite"),
        (np.full((4, 3), np.inf), "finite"),
        (np.ones((4, 3)), "noncollinear"),
        (np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), "noncollinear"),
        (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]), "vertical"),
        (np.array([[0, 0, 0], [1e-10, 1, 0], [1e-10, 0, 1]]), "vertical"),
        (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]), "unique"),
    ],
)
def test_fit_plane_rejects_invalid_vertices(vertices, message):
    with pytest.raises(ValueError, match=message):
        fit_plane(vertices)


class _VertexSampler(SpatialSampler):
    """Split coordinate components across ranks, in reverse local order."""

    def __init__(self, size, comm):
        self.size = size
        self.global_indices = np.arange(size, dtype=np.int64)[comm.rank :: comm.size][::-1]

    def sample(self, fields):
        return fields["vertices"][self.global_indices].copy()


@pytest.mark.parametrize("size", [0, 6, 10])
def test_plane_model_rejects_invalid_sample_size(size):
    with pytest.raises(StudyFileError, match="three 3D vertices"):
        PlaneModel(_VertexSampler(size, MPI.COMM_SELF), MPI.COMM_SELF)


def test_plane_model_rejects_repeated_indices():
    sampler = _VertexSampler(12, MPI.COMM_SELF)
    sampler.global_indices[-1] = sampler.global_indices[0]
    with pytest.raises(StudyFileError, match="exactly one rank"):
        PlaneModel(sampler, MPI.COMM_SELF)


def test_plane_model_options_and_builder():
    options = TypeAdapter(MeasurementModelOptions).validate_python({"type": "plane"})
    assert options == PlaneModelOptions()
    with pytest.raises(ValidationError):
        TypeAdapter(MeasurementModelOptions).validate_python({"type": "plane", "unknown": 1})
    with pytest.raises(StudyFileError, match="needs a spatial sampler"):
        build_model(options)
    model = build_model(options, _VertexSampler(12, MPI.COMM_SELF), MPI.COMM_SELF)
    assert isinstance(model, PlaneModel)


def test_distributed_vertices_and_repeated_predictions():
    """Gather split coordinate components in global order and own the coefficients once."""
    comm = MPI.COMM_WORLD
    sampler = _VertexSampler(12, comm)
    model = PlaneModel(sampler, comm)
    size, indices = model.layout(sampler)
    assert size == 3
    np.testing.assert_array_equal(
        np.concatenate(comm.allgather(indices)), np.arange(3, dtype=np.int64)
    )
    first = model.predict(sampler.sample({"vertices": _vertices().ravel()}))
    np.testing.assert_allclose(first, np.array([0.2, -0.3, 1.1])[indices], atol=1e-14)
    second = model.predict(sampler.sample({"vertices": _vertices(-0.4, 0.5, 2.0).ravel()}))
    np.testing.assert_allclose(second, np.array([-0.4, 0.5, 2.0])[indices], atol=1e-14)
    np.testing.assert_allclose(first, np.array([0.2, -0.3, 1.1])[indices], atol=1e-14)
    measured = np.array([0.3, -0.1, 1.5])[indices]
    np.testing.assert_allclose(model.innovation(first, measured), measured - first)


def test_distributed_vertices_with_empty_ranks():
    comm = MPI.COMM_WORLD
    sampler = _VertexSampler(12, MPI.COMM_SELF)
    if comm.rank:
        sampler.global_indices = np.empty(0, dtype=np.int64)
    model = PlaneModel(sampler, comm)
    result = model.predict(sampler.sample({"vertices": _vertices().ravel()}))
    expected = np.array([0.2, -0.3, 1.1]) if comm.rank == 0 else np.empty(0)
    np.testing.assert_allclose(result, expected, atol=1e-14)


@pytest.mark.parametrize("invalid", ["nonfinite", "length", "vertical"])
def test_collective_fit_errors(invalid):
    """An invalid fit must raise on every rank instead of leaving peers in a collective."""
    comm = MPI.COMM_WORLD
    sampler = _VertexSampler(12, comm)
    model = PlaneModel(sampler, comm)
    vertices = _vertices()
    if invalid == "vertical":
        vertices[:, [0, 2]] = vertices[:, [2, 0]]
        vertices[:, 0] = 0.0
    values = sampler.sample({"vertices": vertices.ravel()})
    if comm.rank == 0:
        if invalid == "nonfinite":
            values[0] = np.nan
        elif invalid == "length":
            values = values[:-1]
    with pytest.raises(ValueError):
        model.predict(values)


class _AnalyticSolver:
    """Vertices on z = 0.2*x - 0.3*y + height*t, with no PDE."""

    def __init__(self):
        self.state_spec = {"vertices": SimpleNamespace(local_size=12, kind="replicated")}

    def nominal_parameters(self):
        return {"height": 1.5}

    def initial_state(self):
        return 0.0, {"vertices": _vertices(c=0.0).ravel()}

    def propagate(self, t0, t1, state, parameters):
        height = (self.nominal_parameters() | parameters)["height"]
        return {"vertices": _vertices(c=height * t1).ravel()}


def test_plane_synthesis_estimation_and_sensitivity(tmp_path, monkeypatch):
    """The real operator reduces 12 coordinates to three data and estimates plane motion."""
    module = types.ModuleType("plane_solver")
    module.Solver = _AnalyticSolver
    monkeypatch.setitem(sys.modules, "plane_solver", module)
    study = StudyOptions(
        output=OutputOptions(path=str(tmp_path / "results")),
        forward_solver=ForwardSolverOptions(factory="plane_solver:Solver"),
        parameters=ParametersOptions(
            reparameterization="additive",
            priors={"height": ParameterPrior(mean=1.0, stddev=0.5)},
        ),
        measurements={
            "plane": MeasurementOptions(
                fields=["vertices"],
                times=[0.5, 1.0],
                data=NumpyDataOptions(type="numpy", path=str(tmp_path / "plane.npz")),
                noise=NoiseOptions(stddev=1e-4),
                model=PlaneModelOptions(),
            )
        },
        synthesis=SynthesisOptions(noise={"plane": SynthesisNoiseOptions(stddev=0.0)}),
        estimation=EstimationOptions(iterations=1),
    )
    operator = build_observation_operator(
        "plane", study.measurements["plane"], _AnalyticSolver()
    )
    assert operator.sampler.size == 12
    assert operator.size == 3
    synthesize(study)
    times, values = read_numpy(tmp_path / "plane.npz")
    np.testing.assert_allclose(times, [0.5, 1.0])
    np.testing.assert_allclose(values, [[0.2, -0.3, 0.75], [0.2, -0.3, 1.5]], atol=1e-14)
    assert estimate(study)["height"] == pytest.approx(1.5, abs=1e-6)
    for i in range(sensitivity.run_count(study)):
        sensitivity.run(study, i)
    result = sensitivity.assemble(study)
    np.testing.assert_allclose(
        result.jacobians["plane"].ravel(), [0.0, 0.0, 0.5, 0.0, 0.0, 1.0], atol=1e-10
    )
