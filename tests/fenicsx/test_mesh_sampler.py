"""The fenicsx.mesh sampler: order of the mesh file, a twin experiment, errors."""

from pathlib import Path

import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, io, mesh

from kipe._types import NDArray_f64
from kipe.cli import main
from kipe.examples.fitzhugh_nagumo import Solver as FitzHughNagumo
from kipe.fenicsx.examples.heat import FieldSpec as HeatFieldSpec
from kipe.fenicsx.examples.heat import Solver as Heat
from kipe.fenicsx.sampling import MeshSampler, build_mesh_sampler
from kipe.options import (
    EstimationOptions,
    ForwardSolverOptions,
    MeasurementOptions,
    MeshSamplingOptions,
    NoiseOptions,
    NumpyDataOptions,
    OutputOptions,
    ParameterPrior,
    ParametersOptions,
    StudyFileError,
    StudyOptions,
    TimeRange,
    dump_study,
)

COMM = MPI.COMM_WORLD


def _write_mesh(path: Path, n: int) -> Path:
    """Write a unit square mesh with n cells per side to an XDMF file; return its path.

    Written by rank 0 alone: dolfinx writes the vertices in an order that depends on the number
    of ranks, so this gives the same file, and the same noise per vertex, on any number.
    """
    if COMM.rank == 0:
        with io.XDMFFile(MPI.COMM_SELF, path, "w") as f:
            f.write_mesh(mesh.create_unit_square(MPI.COMM_SELF, n, n))
    COMM.barrier()

    return path


def _reference_geometry(n: int) -> tuple[NDArray_f64, NDArray_f64]:
    """Vertex coordinates and cells (vertex rows) in the order of the file _write_mesh writes.

    In serial, dolfinx writes the geometry in its local order, so the serial mesh in memory is
    the reference for the file's rows (checked by test_reference_is_the_file_content).
    """
    reference = None
    if COMM.rank == 0:
        written = mesh.create_unit_square(MPI.COMM_SELF, n, n)
        reference = (written.geometry.x[:, :2].copy(), written.geometry.dofmap.copy())

    return COMM.bcast(reference, root=0)


def _file_geometry(path: Path) -> tuple[NDArray_f64, NDArray_f64]:
    """Vertex coordinates and cells as stored in the file, read without dolfinx (needs h5py)."""
    import h5py

    with h5py.File(path.with_suffix(".h5"), "r") as f:
        return f["Mesh/mesh/geometry"][:, :2], f["Mesh/mesh/topology"][:]


def _gathered(sampler: MeshSampler, local: NDArray_f64) -> NDArray_f64:
    """The sampled vector of all ranks, assembled by global_indices, on every rank."""
    full = np.full(sampler.size, np.nan)

    for indices, values in COMM.allgather((sampler.global_indices, local)):
        full[indices] = values

    return full


def _linear(x: NDArray_f64) -> NDArray_f64:
    """A linear function: P1 interpolates it exactly, also between non-matching meshes."""
    return x[0] + 10 * x[1]


def _sample_linear(
    path: Path, element: str, padding: float = 0.0
) -> tuple[MeshSampler, NDArray_f64]:
    """Sample the linear function, given on the heat solver's mesh, onto a measurement mesh."""
    heat = Heat(n=12)
    u = fem.Function(heat.function_space("u"))
    u.interpolate(_linear)
    owned = u.x.array[: heat.state_spec["u"].local_size]

    options = MeshSamplingOptions(
        type="fenicsx.mesh", mesh=str(path), element=element, padding=padding
    )
    sampler = build_mesh_sampler(options, ["u"], heat)

    return sampler, _gathered(sampler, sampler.sample({"u": owned}))


def test_reference_is_the_file_content(shared_tmp_path):
    """The assumption behind _reference_geometry, checked against the file itself."""
    pytest.importorskip("h5py")
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    coordinates, cells = _reference_geometry(5)

    file_coordinates, file_cells = _file_geometry(path)
    np.testing.assert_array_equal(file_coordinates, coordinates)
    np.testing.assert_array_equal(file_cells, cells)


def test_p1_values_are_in_the_order_of_the_mesh_file(shared_tmp_path):
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    coordinates, _ = _reference_geometry(5)

    sampler, values = _sample_linear(path, "P1")
    assert sampler.size == len(coordinates)
    np.testing.assert_allclose(values, _linear(coordinates.T), rtol=0, atol=1e-12)


def test_dg0_values_are_in_the_order_of_the_mesh_file(shared_tmp_path):
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    coordinates, cells = _reference_geometry(5)
    midpoints = coordinates[cells].mean(axis=1)

    sampler, values = _sample_linear(path, "DG0")
    assert sampler.size == len(cells)
    np.testing.assert_allclose(values, _linear(midpoints.T), rtol=0, atol=1e-12)


def test_each_entry_on_exactly_one_rank(shared_tmp_path):
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    sampler, _ = _sample_linear(path, "P1")
    indices = np.concatenate(COMM.allgather(sampler.global_indices))
    np.testing.assert_array_equal(np.sort(indices), np.arange(sampler.size))


def _study(directory: Path, measurement_mesh: Path) -> Path:
    """Write a heat equation twin experiment: kappa and a from a coarser measurement mesh."""
    study = StudyOptions(
        output=OutputOptions(path=str(directory / "results")),
        forward_solver=ForwardSolverOptions(
            factory="kipe.fenicsx.examples.heat:Solver", arguments={"n": 16, "dt": 0.01}
        ),
        parameters=ParametersOptions(
            reparameterization="log",
            select={
                "kappa": ParameterPrior(initial=0.12, relative_stddev=0.1),
                "a": ParameterPrior(initial=0.8, relative_stddev=0.1),
            },
        ),
        measurements={
            "u": MeasurementOptions(
                fields=["u"],
                times=TimeRange(start=0.05, stop=0.5, step=0.05),
                data=NumpyDataOptions(type="numpy", path=str(directory / "u.npz")),
                noise=NoiseOptions(stddev=1e-3, seed=0),
                spatial_sampling=MeshSamplingOptions(
                    type="fenicsx.mesh", mesh=str(measurement_mesh)
                ),
            )
        },
        estimation=EstimationOptions(iterations=3),
    )
    path = directory / "study.yaml"

    if COMM.rank == 0:
        dump_study(study, path)
    COMM.barrier()

    return path


def test_twin_experiment_recovers_kappa_and_a(shared_tmp_path):
    measurement_mesh = _write_mesh(shared_tmp_path / "measurement.xdmf", 6)
    study = str(_study(shared_tmp_path, measurement_mesh))

    assert main(["synthesis", study, "--log-level", "warning"]) == 0
    assert main(["estimation", study, "--log-level", "warning"]) == 0

    COMM.barrier()  # rank 0 writes the history at the end
    history = np.load(shared_tmp_path / "results" / "estimation" / "history.npz")
    np.testing.assert_allclose(history["parameters"][-1], [0.1, 1.0], rtol=0.02)


def test_one_field_only():
    options = MeshSamplingOptions(type="fenicsx.mesh", mesh="unused.xdmf")
    with pytest.raises(StudyFileError, match="samples one field"):
        build_mesh_sampler(options, ["u", "v"], Heat(n=4))


def test_replicated_field_is_rejected():
    options = MeshSamplingOptions(type="fenicsx.mesh", mesh="unused.xdmf")
    with pytest.raises(StudyFileError, match="needs a distributed field"):
        build_mesh_sampler(options, ["v"], FitzHughNagumo())


def test_solver_without_function_spaces_is_rejected():
    class WithoutSpaces:
        state_spec = Heat(n=4).state_spec

    options = MeshSamplingOptions(type="fenicsx.mesh", mesh="unused.xdmf")
    with pytest.raises(StudyFileError, match="no function_space method"):
        build_mesh_sampler(options, ["u"], WithoutSpaces())  # type: ignore[arg-type]


# vector fields and projections


class _VectorField:
    """Stand-in for a solver with one vector field on the unit square: what samplers use."""

    def __init__(self, n: int = 12) -> None:
        self._V = fem.functionspace(mesh.create_unit_square(COMM, n, n), ("Lagrange", 1, (2,)))
        self._n_owned = self._V.dofmap.index_map.size_local * 2
        self.state_spec = {"u": HeatFieldSpec(self._n_owned)}

    def function_space(self, field: str) -> fem.FunctionSpace:
        return self._V

    def owned(self) -> NDArray_f64:
        """The owned entries of the linear vector field _vector."""
        u = fem.Function(self._V)
        u.interpolate(_vector)
        return u.x.array[: self._n_owned].copy()


def _vector(x: NDArray_f64) -> NDArray_f64:
    """A linear vector field, (x + 10 y, 2 x - y): P1 interpolates it exactly."""
    return np.vstack([x[0] + 10 * x[1], 2 * x[0] - x[1]])


def _sample_vector(path: Path, element: str = "P1", projection=None) -> NDArray_f64:
    """Sample the linear vector field onto a measurement mesh, gathered on every rank."""
    solver = _VectorField()
    options = MeshSamplingOptions(
        type="fenicsx.mesh", mesh=str(path), element=element, projection=projection
    )
    sampler = build_mesh_sampler(options, ["u"], solver)
    return _gathered(sampler, sampler.sample({"u": solver.owned()}))


def _write_line_mesh(path: Path, k: int, x0: float = 1.0) -> tuple[NDArray_f64, NDArray_f64]:
    """Write the vertical line x = x0, by default the right edge of the unit square, as a line
    mesh in 2D with k cells.

    Returns:
        its vertex coordinates and cells, in the order of the file (see _reference_geometry)
    """
    reference = None
    if COMM.rank == 0:
        x = np.column_stack([np.full(k + 1, x0), np.linspace(0.0, 1.0, k + 1)])
        cells = np.column_stack([np.arange(k), np.arange(1, k + 1)]).astype(np.int64)
        element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(2,)))
        written = mesh.create_mesh(MPI.COMM_SELF, cells, x, element)
        with io.XDMFFile(MPI.COMM_SELF, path, "w") as f:
            f.write_mesh(written)
        reference = (written.geometry.x[:, :2].copy(), written.geometry.dofmap.copy())
    COMM.barrier()

    return COMM.bcast(reference, root=0)


def test_vector_field_without_projection(shared_tmp_path):
    """All components, those of each vertex one after another."""
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    coordinates, _ = _reference_geometry(5)

    values = _sample_vector(path)
    expected = _vector(coordinates.T).T.ravel()
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)


def test_projection_onto_a_direction(shared_tmp_path):
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    coordinates, _ = _reference_geometry(5)

    values = _sample_vector(path, projection=[1.0, 1.0])
    expected = _vector(coordinates.T).sum(axis=0) / np.sqrt(2)  # normalized direction
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)

    values = _sample_vector(path, projection=[0.0, 2.0])  # the y component
    np.testing.assert_allclose(values, _vector(coordinates.T)[1], rtol=0, atol=1e-12)


def test_projection_onto_the_surface_normal(shared_tmp_path):
    """On x = 1, u . n = ±(1 + 10 y), with one sign for all cells, as they are oriented alike."""
    path = shared_tmp_path / "edge.xdmf"
    coordinates, cells = _write_line_mesh(path, 4)
    midpoints = coordinates[cells].mean(axis=1)

    values = _sample_vector(path, element="DG0", projection="surface_normal")
    normal_component = _vector(midpoints.T)[0]
    sign = np.sign(values[0])
    np.testing.assert_allclose(values, sign * normal_component, rtol=0, atol=1e-12)


def test_projection_needs_a_vector_field(shared_tmp_path):
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 3)
    options = MeshSamplingOptions(type="fenicsx.mesh", mesh=str(path), projection=[1.0])
    with pytest.raises(StudyFileError, match="needs a vector field"):
        build_mesh_sampler(options, ["u"], Heat(n=4))


def test_direction_needs_one_entry_per_component(shared_tmp_path):
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 3)
    with pytest.raises(StudyFileError, match="has 3 entries"):
        _sample_vector(path, projection=[1.0, 0.0, 0.0])


def test_surface_normal_needs_a_surface_mesh(shared_tmp_path):
    path = _write_mesh(shared_tmp_path / "measurement.xdmf", 3)
    with pytest.raises(StudyFileError, match="needs a surface mesh"):
        _sample_vector(path, element="DG0", projection="surface_normal")


# --- measurement points outside the solver's mesh ----------------------------------------------


@pytest.mark.parametrize("element", ["P1", "DG0"])
def test_points_outside_the_mesh_are_an_error(shared_tmp_path, element):
    """A line 2e-3 to the right of the solver's unit square: by default, and with too small a
    padding, an error (dolfinx would give these points the value 0)."""
    path = shared_tmp_path / "outside.xdmf"
    _write_line_mesh(path, 4, x0=1.002)

    with pytest.raises(StudyFileError, match="4 of 4 cells of the measurement mesh lie"):
        _sample_linear(path, element)
    with pytest.raises(StudyFileError, match=r"farther than padding 0\.001 outside"):
        _sample_linear(path, element, padding=1e-3)


@pytest.mark.parametrize("element", ["P1", "DG0"])
def test_padding_takes_the_value_of_the_nearest_cell(shared_tmp_path, element):
    """Within the padding, a point takes the value of the field in the nearest cell,
    extrapolated: for the linear function, its exact value."""
    path = shared_tmp_path / "outside.xdmf"
    coordinates, cells = _write_line_mesh(path, 4, x0=1.002)
    points = coordinates if element == "P1" else coordinates[cells].mean(axis=1)

    _, values = _sample_linear(path, element, padding=1e-2)
    np.testing.assert_allclose(values, _linear(points.T), rtol=0, atol=1e-12)
