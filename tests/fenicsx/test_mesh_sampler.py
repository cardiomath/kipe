"""The fenicsx.mesh sampler: order of the mesh file, a twin experiment, errors."""

from pathlib import Path

import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

import numpy as np
from dolfinx import fem, io, mesh

from kipe._types import NDArray_f64
from kipe.cli import main
from kipe.examples.fitzhugh_nagumo import Solver as FitzHughNagumo
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


def _sample_linear(path: Path, element: str) -> tuple[MeshSampler, NDArray_f64]:
    """Sample the linear function, given on the heat solver's mesh, onto a measurement mesh."""
    heat = Heat(n=12)
    u = fem.Function(heat.function_space("u"))
    u.interpolate(_linear)
    owned = u.x.array[: heat.state_spec["u"].local_size]

    options = MeshSamplingOptions(type="fenicsx.mesh", mesh=str(path), element=element)
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
