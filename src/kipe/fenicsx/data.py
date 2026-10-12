"""Measurement data in PETSc binary files: ``fenicsx.petscbinary``.

The format chimera-synthmeas writes for chimera-roukf: one file per time and rank, each with
the values of the degrees of freedom that rank owns on the measurement mesh, in dolfinx's local
order, and a ``timestamps.csv`` with the index and the time of each file. These are the entries
kipe's :class:`kipe.fenicsx.sampling.MeshSampler` returns, for the same mesh, element and
projection on the same number of ranks, so they are used as they are, without reordering.
"""

from pathlib import Path

from mpi4py import MPI

from petsc4py import PETSc

import numpy as np

from kipe._types import NDArray_f64
from kipe.options import PetscBinaryDataOptions, StudyFileError


def read_petsc_binary(
    options: PetscBinaryDataOptions, n_values: int, comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[NDArray_f64, NDArray_f64]:
    """Read this rank's values of measurement data at all times in ``timestamps.csv``.

    Collective: an error on one rank is raised on all.

    Args:
        options: the ``data`` options of the measurement
        n_values: number of values this rank holds, as the measurement's sampler returns
        comm: communicator of the run

    Returns:
        - times, shape ``(n,)``
        - this rank's values, shape ``(n, n_values)``

    Raises:
        StudyFileError: if ``timestamps.csv`` cannot be read, there are no files for this
            number of ranks, or a file does not have ``n_values`` values
    """
    error = None
    try:
        indices, times = _read_timestamps(options.path)
        values = np.empty((len(times), n_values))
        for k, i in enumerate(indices):
            path = options.path.format(i=i, mpi_size=comm.size, mpi_rank=comm.rank)
            values[k] = _read_vector(path, n_values, comm.size)
    except StudyFileError as err:
        error = str(err)

    errors = [e for e in comm.allgather(error) if e is not None]
    if errors:
        raise StudyFileError(errors[0])

    return times, values


def _read_timestamps(pattern: str) -> tuple[list[int], NDArray_f64]:
    """Read the indices and times from the ``timestamps.csv`` of a path pattern.

    The file is in the directory that contains the first path component with ``{i}``.

    Args:
        pattern: path pattern of the data files

    Returns:
        - the indices
        - the times

    Raises:
        StudyFileError: if the file cannot be read or does not have two columns
    """
    directory = Path(pattern)
    while "{i" not in directory.name:
        directory = directory.parent
    path = directory.parent / "timestamps.csv"

    try:
        table = np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as err:
        raise StudyFileError(f"cannot read the measurement times {path}: {err}") from err

    if table.shape[1] != 2:
        raise StudyFileError(
            f"measurement times {path}: expected two columns (index, time), got {table.shape[1]}"
        )

    return [int(i) for i in table[:, 0]], table[:, 1].astype(np.float64)


def _read_vector(path: str, n_values: int, n_ranks: int) -> NDArray_f64:
    """Read one rank's values from a PETSc binary file.

    Args:
        path: the file
        n_values: number of values the file must have
        n_ranks: number of ranks of the run, for the error message

    Returns:
        the values

    Raises:
        StudyFileError: if the file does not exist, cannot be read, or does not have
            ``n_values`` values
    """
    if not Path(path).is_file():
        raise StudyFileError(
            f"no measurement data {path}: PETSc binary data are read on the number of ranks "
            f"they were written on, is it {n_ranks}?"
        )

    viewer = PETSc.Viewer().createBinary(path, "r", comm=MPI.COMM_SELF)
    vector = PETSc.Vec().create(comm=MPI.COMM_SELF)
    try:
        vector.load(viewer)
        values = vector.getArray(readonly=True).astype(np.float64)
    except PETSc.Error as err:
        raise StudyFileError(f"cannot read measurement data {path}: {err}") from err
    finally:
        vector.destroy()
        viewer.destroy()

    if len(values) != n_values:
        raise StudyFileError(
            f"measurement data {path} has {len(values)} values, this rank's part of the "
            f"measurement has {n_values}: a different measurement mesh, element or projection?"
        )
    return values
