"""Write the measurement mesh of the heat example: the unit square with 6 x 6 cells.

Coarser than the solver's mesh (16 x 16) and not matching it. Run from this directory:

    python measurement_mesh.py

The file's vertex order defines the order of the measurement data, so synthesis and estimation
must use the same file (on any number of ranks). It is written in serial, by one rank, so that
the script gives the same file however it is run; written in parallel, the order would depend
on the number of ranks.
"""

from pathlib import Path

from mpi4py import MPI

from dolfinx import io, mesh


def write_measurement_mesh(path: Path = Path("data/measurement_mesh.xdmf"), n: int = 6) -> None:
    """Write the unit square with n x n cells to an XDMF file.

    Args:
        path: the XDMF file; its directory is created if needed
        n: number of cells per side
    """
    if MPI.COMM_WORLD.rank != 0:  # one rank writes, also if run with mpirun
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with io.XDMFFile(MPI.COMM_SELF, path, "w") as f:
        f.write_mesh(mesh.create_unit_square(MPI.COMM_SELF, n, n))


if __name__ == "__main__":
    write_measurement_mesh()
