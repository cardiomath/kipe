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
from kipe.measurements.operator import ObservationOperator, build_observation_operator
from kipe.measurements.plane import PlaneModel, fit_from_coordinates, orient_normal
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
    StudyOptions,
    SynthesisNoiseOptions,
    SynthesisOptions,
)
from kipe.sampling import SpatialSampler
from kipe.synthesis import synthesize


def _vertices(a=0.2, b=-0.3, c=1.1):
    """Return four corners of a square lifted onto the plane ``z = a*x + b*y + c``."""
    xy = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    return np.column_stack((xy, a * xy[:, 0] + b * xy[:, 1] + c))


def test_fit_from_coordinates_exact_and_translated_vertices():
    """Recover known coefficients, account for translations, and preserve input coordinates.

    Translating all points by (dx, dy, dz) must preserve both slopes and change the
    intercept by dz - a*dx - b*dy. The same fit must work with exactly three noncollinear
    vertices, and fitting must not mutate the supplied array.
    """
    vertices = _vertices()
    original = vertices.copy()
    np.testing.assert_allclose(
        fit_from_coordinates(vertices).coefficients, [0.2, -0.3, 1.1], atol=1e-14
    )
    translation = np.array([0.4, -0.2, 0.7])
    expected_shift = translation[2] - 0.2 * translation[0] + 0.3 * translation[1]
    np.testing.assert_allclose(
        fit_from_coordinates(vertices + translation).coefficients,
        [0.2, -0.3, 1.1 + expected_shift],
        atol=1e-14,
    )
    np.testing.assert_array_equal(vertices, original)
    # Exactly three noncollinear vertices also define a plane.
    np.testing.assert_allclose(
        fit_from_coordinates(vertices[:3]).coefficients, [0.2, -0.3, 1.1], atol=1e-14
    )


def test_fit_from_coordinates_orthogonal_distances_and_vertex_order():
    """Recover the orthogonal best-fit plane despite symmetric off-plane perturbations.

    Equal displacements to either side along the normal preserve the expected plane.
    Reversing the vertices must not change the fitted coefficients.
    """
    vertices = _vertices()
    normal = np.array([-0.2, 0.3, 1.0])
    normal /= np.linalg.norm(normal)
    perturbed = np.vstack((vertices + 0.01 * normal, vertices - 0.01 * normal))
    np.testing.assert_allclose(
        fit_from_coordinates(perturbed).coefficients, [0.2, -0.3, 1.1], atol=1e-14
    )
    np.testing.assert_allclose(
        fit_from_coordinates(perturbed[::-1]).coefficients,
        fit_from_coordinates(perturbed).coefficients,
        atol=1e-14,
    )


def test_normal_toward_opposite_sides_and_translated_mesh():
    """Opposite target points reverse the normal but preserve the plane coefficients.

    Translating both vertices and target must preserve the selected unit normal; the
    orientation must also be independent of vertex ordering.
    """
    vertices = _vertices()
    above = fit_from_coordinates(vertices, normal_toward=(0, 0, 5))
    below = fit_from_coordinates(vertices[::-1], normal_toward=(0, 0, -5))
    np.testing.assert_allclose(above.normal, -below.normal)
    np.testing.assert_allclose(above.coefficients, below.coefficients)
    assert np.dot(above.normal, np.array([0, 0, 5]) - above.centroid) > 0
    assert np.dot(below.normal, np.array([0, 0, -5]) - below.centroid) > 0
    assert np.linalg.norm(above.normal) == pytest.approx(1)
    shift = np.array([10, -20, 30])
    moved = fit_from_coordinates(
        vertices + shift, normal_toward=tuple(shift + np.array([0, 0, -5]))
    )
    np.testing.assert_allclose(moved.normal, below.normal, atol=1e-14)


@pytest.mark.parametrize("target", [None, (0, 0, -5)])
def test_normals_consistent_across_times_and_evaluation_order(target):
    """Align moving planes with one fixed first normal, with or without a target point.

    After establishing the reference, reordered time evaluations and reversed vertex
    arrays must produce the same normals and leave the coefficients unchanged.
    """
    frames = [_vertices(a=0.1 + 0.02 * t, b=-0.2, c=1 + 0.1 * t) for t in range(5)]
    first = fit_from_coordinates(frames[0], normal_toward=target).normal
    expected = [fit_from_coordinates(points, reference_normal=first) for points in frames]
    for i in [4, 1, 3, 0, 2, 0]:
        actual = fit_from_coordinates(
            frames[i][::-1], normal_toward=target, reference_normal=first
        )
        np.testing.assert_allclose(actual.normal, expected[i].normal, atol=1e-14)
        assert np.dot(actual.normal, expected[0].normal) > 0
        np.testing.assert_allclose(
            actual.coefficients, fit_from_coordinates(frames[i]).coefficients, atol=1e-14
        )


def test_orientation_helper_uses_first_normal_not_previous_or_target():
    """Retain the first normal's hemisphere when the point or the previous fit disagrees.

    Without any reference, preserve the supplied sign. Use a point only for the first
    orientation; subsequent orientations must follow the first normal even when two
    successive normals have a negative dot product with each other.
    """
    first = np.array([0.0, 0.0, -1.0])
    np.testing.assert_array_equal(orient_normal(first, np.zeros(3)), first)
    point = (0.0, 0.0, -5.0)
    np.testing.assert_array_equal(orient_normal(-first, np.zeros(3), normal_toward=point), first)
    # The target is above this later plane, but the original normal remains the reference.
    np.testing.assert_array_equal(
        orient_normal(-first, np.array([0, 0, -6]), normal_toward=point, reference_normal=first),
        first,
    )
    left = np.array([-1.0, 0.0, -0.1])
    right = np.array([1.0, 0.0, -0.1])
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    for normal in [left, right, -left, -right]:
        result = orient_normal(normal, np.zeros(3), reference_normal=first)
        assert np.dot(result, first) > 0
    assert np.dot(left, right) < 0  # aligning to the previous normal would give another sign


@pytest.mark.parametrize("reference", [np.zeros(3), np.array([np.nan, 0, 1]), np.ones(2)])
def test_orientation_helper_rejects_invalid_reference(reference):
    """A reference must be finite, three-dimensional and nonzero."""
    with pytest.raises(ValueError, match="reference_normal"):
        orient_normal(np.array([0.0, 0.0, 1.0]), np.zeros(3), reference_normal=reference)


def test_orientation_helper_rejects_perpendicular_reference():
    """Perpendicular normals do not determine a consistent side."""
    with pytest.raises(ValueError, match="perpendicular"):
        orient_normal(
            np.array([0.0, 0.0, 1.0]), np.zeros(3), reference_normal=np.array([1.0, 0, 0])
        )


@pytest.mark.parametrize("target", [(0, 0, 1.1), (0, 0), (0, 0, np.nan), (0, 0, np.inf)])
def test_normal_toward_rejects_undefined_orientation(target):
    """Reject a target on the first plane, a malformed target, and nonfinite coordinates."""
    with pytest.raises(ValueError, match="normal_toward"):
        fit_from_coordinates(_vertices(), normal_toward=target)


@pytest.mark.parametrize("target", [[1, 2], [1, 2, 3, 4], [1, 2, np.inf]])
def test_normal_toward_configuration_validation(target):
    """Require exactly three finite target coordinates when parsing plane-model options."""
    with pytest.raises(ValidationError):
        TypeAdapter(MeasurementModelOptions).validate_python({
            "type": "plane",
            "normal_toward": target,
        })


@pytest.mark.parametrize("target", [None, (0, 0, -5)])
def test_oriented_model_distributed_and_collective_failure(target):
    """Keep the first orientation reference across MPI predictions and failed fits.

    Only root exposes the normal, and callers receive a copy. Once the reference exists,
    a later plane passing through the target is valid. A nonfinite fit must fail on every
    rank and clear the latest normal without discarding the original reference.
    """
    comm = MPI.COMM_WORLD
    options = MeasurementOptions(
        fields=["vertices"],
        data=NumpyDataOptions(type="numpy", path="unused.npz"),
        noise=NoiseOptions(stddev=0.1),
        model=PlaneModelOptions(normal_toward=target),
    )
    operator = build_observation_operator("plane", options, _AnalyticSolver(), comm)
    model = operator.model
    assert isinstance(model, PlaneModel)
    assert model.normal is None
    predicted = operator({"vertices": _vertices().ravel()})
    expected = np.array([0.2, -0.3, 1.1])[operator.global_indices]
    np.testing.assert_allclose(predicted, expected, atol=1e-14)
    if comm.rank == 0:
        normal = model.normal
        assert normal is not None
        if target is not None:
            assert normal[2] < 0
        first = normal.copy()
        normal[:] = 0  # callers cannot mutate the stored reference through this property
        assert np.linalg.norm(model.normal) == pytest.approx(1)
    else:
        assert model.normal is None
    # Later planes can pass through the target: only the first fit uses it.
    operator({"vertices": _vertices(c=-5).ravel()})
    if comm.rank == 0:
        np.testing.assert_allclose(model.normal, first)
    invalid = np.full(12, np.nan)
    with pytest.raises(ValueError, match="finite"):
        operator({"vertices": invalid})
    assert model.normal is None
    operator({"vertices": _vertices(c=-6).ravel()})
    if comm.rank == 0:
        np.testing.assert_allclose(model.normal, first)
    # An input-layout error happens before model.predict; the generic lifecycle hook must
    # still clear the latest normal, while preserving the reference used after recovery.
    local = operator.sampler.sample({"vertices": _vertices(c=-6).ravel()})
    if comm.rank == 0:
        local = local[:-1]
    with pytest.raises(ValueError, match="sampler layout"):
        operator.predict(local)
    assert model.normal is None
    operator({"vertices": _vertices(c=-7).ravel()})
    if comm.rank == 0:
        np.testing.assert_allclose(model.normal, first)


def test_plane_factory_keeps_orientation_independent_between_models():
    """Every model gets independent orientation history, including when no point is supplied."""
    first = build_model(PlaneModelOptions(normal_toward=(0, 0, -5)))
    second = build_model(PlaneModelOptions(normal_toward=(0, 0, 5)))
    without_point = build_model(PlaneModelOptions())
    assert isinstance(first, PlaneModel)
    assert isinstance(second, PlaneModel)
    assert isinstance(without_point, PlaneModel)
    assert first.orientation is not second.orientation
    assert first.orientation is not without_point.orientation
    first.predict(_vertices().ravel())
    original = first.normal
    assert original is not None and original[2] < 0
    assert second.normal is None
    assert without_point.normal is None
    second.predict(_vertices().ravel())
    np.testing.assert_allclose(second.normal, -original)
    np.testing.assert_allclose(first.normal, original)
    first.clear_prediction()
    assert first.normal is None
    assert second.normal is not None
    first.predict(_vertices(c=-6).ravel())
    np.testing.assert_allclose(first.normal, original)


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
def test_fit_from_coordinates_rejects_invalid_vertices(vertices, message):
    """Reject invalid shapes, nonfinite points, degenerate fits and vertical planes."""
    with pytest.raises(ValueError, match=message):
        fit_from_coordinates(vertices)


class _VertexSampler(SpatialSampler):
    """Split coordinate components across ranks, in reverse local order."""

    def __init__(self, size, comm):
        """Assign alternating global coordinate entries to ranks and reverse local order."""
        self.size = size
        self.global_indices = np.arange(size, dtype=np.int64)[comm.rank :: comm.size][::-1]

    def sample(self, fields):
        """Copy this rank's coordinate entries from the replicated test field."""
        return fields["vertices"][self.global_indices].copy()


@pytest.mark.parametrize("size", [0, 6, 10])
def test_plane_model_rejects_invalid_sample_size(size):
    """Reject fewer than three vertices or flattened sizes that are not multiples of three."""
    with pytest.raises(ValueError):
        PlaneModel().predict(np.zeros(size))


def test_plane_model_options_and_builder():
    """Build a constructor-free array model without a sampler or MPI communicator."""
    options = TypeAdapter(MeasurementModelOptions).validate_python({"type": "plane"})
    assert options == PlaneModelOptions()
    with pytest.raises(ValidationError):
        TypeAdapter(MeasurementModelOptions).validate_python({"type": "plane", "unknown": 1})
    model = build_model(options)
    assert isinstance(model, PlaneModel)
    np.testing.assert_allclose(model.predict(_vertices().ravel()), [0.2, -0.3, 1.1], atol=1e-14)


def test_distributed_vertices_and_repeated_predictions():
    """Reconstruct globally ordered vertices and assign all three coefficients only to root.

    Repeated predictions must reflect the new plane without changing earlier returned
    arrays. Innovation must equal measured minus predicted coefficients on each rank.
    """
    comm = MPI.COMM_WORLD
    sampler = _VertexSampler(12, comm)
    model = PlaneModel()
    operator = ObservationOperator(["vertices"], sampler, model, comm)
    size, indices = operator.size, operator.global_indices
    assert size == 3
    np.testing.assert_array_equal(
        np.concatenate(comm.allgather(indices)), np.arange(3, dtype=np.int64)
    )
    first = operator({"vertices": _vertices().ravel()})
    np.testing.assert_allclose(first, np.array([0.2, -0.3, 1.1])[indices], atol=1e-14)
    second = operator({"vertices": _vertices(-0.4, 0.5, 2.0).ravel()})
    np.testing.assert_allclose(second, np.array([-0.4, 0.5, 2.0])[indices], atol=1e-14)
    np.testing.assert_allclose(first, np.array([0.2, -0.3, 1.1])[indices], atol=1e-14)
    measured = np.array([0.3, -0.1, 1.5])[indices]
    np.testing.assert_allclose(model.innovation(first, measured), measured - first)


def test_distributed_vertices_with_empty_ranks():
    """Fit correctly when root owns every coordinate and all other ranks own none."""
    comm = MPI.COMM_WORLD
    sampler = _VertexSampler(12, MPI.COMM_SELF)
    if comm.rank:
        sampler.global_indices = np.empty(0, dtype=np.int64)
    operator = ObservationOperator(["vertices"], sampler, PlaneModel(), comm)
    result = operator({"vertices": _vertices().ravel()})
    expected = np.array([0.2, -0.3, 1.1]) if comm.rank == 0 else np.empty(0)
    np.testing.assert_allclose(result, expected, atol=1e-14)


@pytest.mark.parametrize("invalid", ["nonfinite", "length", "length_on_peer", "vertical"])
def test_collective_fit_errors(invalid):
    """Raise on every rank for nonfinite coordinates, wrong local lengths or vertical planes.

    Failure on root must be communicated before other ranks continue, avoiding an unmatched
    collective operation after an invalid prediction.
    """
    comm = MPI.COMM_WORLD
    sampler = _VertexSampler(12, comm)
    operator = ObservationOperator(["vertices"], sampler, PlaneModel(), comm)
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
    if invalid == "length_on_peer" and comm.rank == min(1, comm.size - 1):
        values = values[:-1]
    with pytest.raises(ValueError):
        operator.predict(values)


class _AnalyticSolver:
    """Vertices on z = 0.2*x - 0.3*y + height*t, with no PDE."""

    def __init__(self):
        """Expose four replicated 3D vertices as a twelve-entry solver field."""
        self.state_spec = {"vertices": SimpleNamespace(local_size=12, kind="replicated")}

    def nominal_parameters(self):
        """Provide the true vertical motion rate used to generate synthetic measurements."""
        return {"height": 1.5}

    def initial_state(self):
        """Start at time zero with the plane intercept equal to zero."""
        return 0.0, {"vertices": _vertices(c=0.0).ravel()}

    def propagate(self, t0, t1, state, parameters):
        """Set the plane intercept to height*t1 while keeping both slopes fixed."""
        height = (self.nominal_parameters() | parameters)["height"]
        return {"vertices": _vertices(c=height * t1).ravel()}


@pytest.mark.parametrize("stddev", [1e-4, [1e-4, 2e-4, 3e-4]])
def test_plane_synthesis_estimation_and_sensitivity(tmp_path, monkeypatch, stddev):
    """Exercise the plane model through synthesis, estimation and sensitivity analysis.

    Synthetic data must contain three coefficients per time, not twelve coordinates.
    Estimation must recover the known vertical motion rate. Finite-difference sensitivities
    must match the analytic derivative, and the Fisher information must use the intercept's
    noise level correctly for both scalar and component-wise standard deviations.
    """
    # Synthesis writes on rank 0; every rank must read the same data and provenance.
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path if MPI.COMM_WORLD.rank == 0 else None, root=0)
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
                noise=NoiseOptions(stddev=stddev),
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
    MPI.COMM_WORLD.barrier()
    times, values = read_numpy(tmp_path / "plane.npz")
    np.testing.assert_allclose(times, [0.5, 1.0])
    np.testing.assert_allclose(values, [[0.2, -0.3, 0.75], [0.2, -0.3, 1.5]], atol=1e-14)
    assert estimate(study)["height"] == pytest.approx(1.5, abs=1e-6)
    for i in range(sensitivity.run_count(study)):
        sensitivity.run(study, i)
    MPI.COMM_WORLD.barrier()  # rank 0 must finish writing run metadata before assembly
    result = sensitivity.assemble(study)
    np.testing.assert_allclose(
        result.jacobians["plane"].ravel(), [0.0, 0.0, 0.5, 0.0, 0.0, 1.0], atol=1e-10
    )
    shift_stddev = stddev[2] if isinstance(stddev, list) else stddev
    np.testing.assert_allclose(result.fim, [[1.25 / shift_stddev**2]], rtol=1e-8)
    MPI.COMM_WORLD.barrier()
