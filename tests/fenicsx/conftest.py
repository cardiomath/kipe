"""Fixtures for the FEniCSx tests, which also run on several MPI ranks."""

from pathlib import Path

from mpi4py import MPI

import pytest


@pytest.fixture
def shared_tmp_path(tmp_path: Path) -> Path:
    """A temporary directory shared by all ranks: rank 0's."""
    return Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
