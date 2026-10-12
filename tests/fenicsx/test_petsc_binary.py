"""fenicsx.petscbinary data: written as chimera-synthmeas does, read on the same ranks."""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

from petsc4py import PETSc

import numpy as np
from dolfinx import fem, io, mesh

from kipe._types import NDArray_f64
from kipe.fenicsx.data import _read_vector
from kipe.fenicsx.examples.heat import Solver as Heat
from kipe.measurements import ObservationOperator, build_observation_operator, read_data
from kipe.options import (
    MeasurementOptions,
    MeshSamplingOptions,
    NoiseOptions,
    PetscBinaryDataOptions,
    StudyFileError,
)

COMM = MPI.COMM_WORLD
INDICES = [0, 10, 20]  # chimera-synthmeas writes the indices of the solver's checkpoints
TIMES = [0.1, 0.2, 0.3]
PATTERN = "measurements/{i}/u_np{mpi_size}-{mpi_rank}.dat"


def _write_mesh(path: Path, n: int) -> Path:
    """Write a unit square mesh to an XDMF file, from rank 0 alone; return its path."""
    if COMM.rank == 0:
        with io.XDMFFile(MPI.COMM_SELF, path, "w") as f:
            f.write_mesh(mesh.create_unit_square(MPI.COMM_SELF, n, n))
    COMM.barrier()

    return path


def _field(t: float) -> Callable[[NDArray_f64], NDArray_f64]:
    """A linear function of space at time t: P1 and DG0 interpolate it exactly."""
    return lambda x: (1 + t) * (x[0] + 10 * x[1])


def _write_vector(path: Path, values: NDArray_f64) -> None:
    """Write one rank's values to a PETSc binary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    viewer = PETSc.Viewer().createBinary(str(path), "w", comm=MPI.COMM_SELF)
    PETSc.Vec().createWithArray(values, comm=MPI.COMM_SELF).view(viewer)
    viewer.destroy()


def _write_data(
    directory: Path, mesh_path: Path, element: str, mpi_size: int = COMM.size, extra: int = 0
) -> PetscBinaryDataOptions:
    """Write data as chimera-synthmeas does: the measurement function's owned values per rank.

    Args:
        directory: where the data go
        mesh_path: the measurement mesh, read on all ranks as chimera-synthmeas does
        element: the measurement space's element
        mpi_size: the number of ranks in the file names
        extra: number of values appended on rank 0, to make its files the wrong size
    """
    with io.XDMFFile(COMM, mesh_path, "r") as f:
        measurement_mesh = f.read_mesh(name="mesh")
    family = ("Lagrange", 1) if element == "P1" else ("DG", 0)
    u = fem.Function(fem.functionspace(measurement_mesh, family))

    pattern = str(directory / PATTERN)
    for i, t in zip(INDICES, TIMES, strict=True):
        u.interpolate(_field(t))
        values = u.x.petsc_vec.array_r
        if COMM.rank == 0:
            values = np.append(values, np.zeros(extra))
        path = Path(pattern.format(i=i, mpi_size=mpi_size, mpi_rank=COMM.rank))
        _write_vector(path, values)

    if COMM.rank == 0:
        np.savetxt(
            directory / "measurements" / "timestamps.csv", np.c_[INDICES, TIMES], delimiter=","
        )
    COMM.barrier()

    return PetscBinaryDataOptions(type="fenicsx.petscbinary", path=pattern)


def _operator(
    mesh_path: Path, element: str, data: PetscBinaryDataOptions
) -> ObservationOperator:
    """The observation operator of the heat solver's u, sampled onto the measurement mesh."""
    options = MeasurementOptions(
        fields=["u"],
        data=data,
        noise=NoiseOptions(stddev=1.0),
        spatial_sampling=MeshSamplingOptions(
            type="fenicsx.mesh", mesh=str(mesh_path), element=element
        ),
    )
    return build_observation_operator("u", options, Heat(n=12))


@pytest.mark.parametrize("element", ["P1", "DG0"])
def test_data_match_sampled_field(shared_tmp_path, element):
    """The files hold what kipe's sampler returns for the same field, entry by entry."""
    mesh_path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    data = _write_data(shared_tmp_path, mesh_path, element)
    operator = _operator(mesh_path, element, data)

    times, values = read_data("u", data, operator)
    np.testing.assert_array_equal(times, TIMES)

    heat = Heat(n=12)
    u = fem.Function(heat.function_space("u"))
    for t, measured in zip(TIMES, values, strict=True):
        u.interpolate(_field(t))
        predicted = operator({"u": u.x.array[: heat.state_spec["u"].local_size]})
        np.testing.assert_allclose(measured, predicted, rtol=0, atol=1e-12)


def test_other_number_of_ranks(shared_tmp_path):
    mesh_path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    data = _write_data(shared_tmp_path, mesh_path, "P1", mpi_size=COMM.size + 1)
    operator = _operator(mesh_path, "P1", data)

    with pytest.raises(
        StudyFileError, match=f"measurements.u: no measurement data .* {COMM.size}"
    ):
        read_data("u", data, operator)


def test_wrong_size_on_one_rank_fails_on_all(shared_tmp_path):
    mesh_path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    data = _write_data(shared_tmp_path, mesh_path, "P1", extra=1)
    operator = _operator(mesh_path, "P1", data)

    with pytest.raises(StudyFileError, match="values, this rank's part of the measurement has"):
        read_data("u", data, operator)


def test_missing_timestamps(shared_tmp_path):
    mesh_path = _write_mesh(shared_tmp_path / "measurement.xdmf", 5)
    data = PetscBinaryDataOptions(
        type="fenicsx.petscbinary", path=str(shared_tmp_path / PATTERN)
    )
    operator = _operator(mesh_path, "P1", data)

    with pytest.raises(StudyFileError, match="cannot read the measurement times"):
        read_data("u", data, operator)


def test_file_written_with_petsc_binary_io(tmp_path, monkeypatch):
    """chimera-synthmeas writes with PETSc's PetscBinaryIO script; kipe reads with petsc4py."""
    import petsc4py

    scripts = Path(petsc4py.get_config()["PETSC_DIR"]) / "lib" / "petsc" / "bin"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    PetscBinaryIO = pytest.importorskip("PetscBinaryIO")

    values = np.linspace(0.0, 1.0, 7)
    path = tmp_path / "u.dat"
    binary_io = PetscBinaryIO.PetscBinaryIO()
    binary_io.writeBinaryFile(str(path), [values.view(PetscBinaryIO.Vec)])

    np.testing.assert_array_equal(_read_vector(str(path), 7, 1), values)
