"""The heat equation example (examples/heat) runs end to end and recovers the parameters."""

import runpy
import shutil
from pathlib import Path

import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

import numpy as np

from kipe.cli import main

EXAMPLE = Path(__file__).parents[2] / "examples" / "heat"


def test_heat_example(shared_tmp_path, monkeypatch):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        shutil.copy(EXAMPLE / "study.yaml", shared_tmp_path / "study.yaml")
        script = runpy.run_path(str(EXAMPLE / "measurement_mesh.py"))
        script["write_measurement_mesh"](shared_tmp_path / "data" / "measurement_mesh.xdmf")
    comm.barrier()

    # paths in the study file are relative to the working directory
    monkeypatch.chdir(shared_tmp_path)
    assert main(["synthesis", "study.yaml", "--log-level", "warning"]) == 0
    assert main(["estimation", "study.yaml", "--log-level", "warning"]) == 0

    comm.barrier()  # rank 0 writes the history at the end
    estimate = np.load("results/estimation/history.npz")["parameters"][-1]
    np.testing.assert_allclose(estimate, [0.1, 1.0], rtol=0.02)
