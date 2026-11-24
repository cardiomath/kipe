"""The sampler layer validates ownership independently of the measurement model."""

from types import SimpleNamespace

from mpi4py import MPI

import numpy as np
import pytest

from kipe.measurements.operator import build_observation_operator
from kipe.options import MeasurementOptions, NoiseOptions, NumpyDataOptions, StudyFileError
from kipe.sampling import SpatialSampler


class _IndexSampler(SpatialSampler):
    """Expose a configurable layout independently of meshes and forward solvers."""

    def __init__(self, size, indices):
        self.size = size
        self.global_indices = indices

    def sample(self, fields):
        """Return the declared positions of a replicated test field."""
        return fields["u"][self.global_indices].copy()


def test_sampler_layout_accepts_permuted_indices_and_empty_ranks():
    """Allow arbitrary local order, distributed ownership and empty rank-local selections."""
    comm = MPI.COMM_WORLD
    for size in [0, 1, 12]:
        indices = np.arange(size, dtype=np.int64)[comm.rank :: comm.size][::-1].copy()
        original = indices.copy()
        sampler = _IndexSampler(size, indices)
        sampler.validate_layout(comm)
        np.testing.assert_array_equal(sampler.global_indices, original)


@pytest.mark.parametrize("invalid", ["duplicate", "missing", "out_of_range", "negative"])
def test_sampler_layout_rejects_invalid_ownership_on_every_rank(invalid):
    """Detect invalid global coverage even when only one rank supplies incorrect indices."""
    comm = MPI.COMM_WORLD
    indices = np.arange(12, dtype=np.int64)[comm.rank :: comm.size][::-1].copy()
    if comm.rank == 0:
        if invalid == "duplicate":
            indices[-1] = indices[0]
        elif invalid == "missing":
            indices = indices[:-1]
        else:
            indices[-1] = 12 if invalid == "out_of_range" else -1
    with pytest.raises(StudyFileError, match="exactly one rank"):
        _IndexSampler(12, indices).validate_layout(comm)


@pytest.mark.parametrize(
    "invalid", ["negative_size", "noninteger_size", "float_indices", "matrix_indices"]
)
def test_sampler_layout_rejects_malformed_metadata(invalid):
    """Report invalid sizes and index types collectively rather than failing on root alone."""
    comm = MPI.COMM_WORLD
    size = 12
    indices = np.arange(size, dtype=np.int64)[comm.rank :: comm.size].copy()
    if comm.rank == min(1, comm.size - 1):
        if invalid == "negative_size":
            size = -1
        elif invalid == "noninteger_size":
            size = 12.5
        elif invalid == "float_indices":
            indices = indices.astype(float)
        else:
            indices = indices.reshape(1, -1)
    with pytest.raises(StudyFileError, match="sampler"):
        _IndexSampler(size, indices).validate_layout(comm)


def test_sampler_layout_rejects_inconsistent_sizes():
    """Require the same global size on every process, including empty processes."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs multiple ranks with different global sizes")
    indices = np.arange(12, dtype=np.int64)[comm.rank :: comm.size]
    with pytest.raises(StudyFileError, match="agree"):
        _IndexSampler(12 + comm.rank, indices).validate_layout(comm)


def test_factory_validates_sampler_for_a_local_difference_model(monkeypatch):
    """Invalid ownership must fail even when the model never gathers its input."""
    comm = MPI.COMM_WORLD
    indices = np.arange(12, dtype=np.int64)[comm.rank :: comm.size].copy()
    if comm.rank == 0:
        indices[-1] = indices[0]
    sampler = _IndexSampler(12, indices)
    monkeypatch.setattr("kipe.sampling.ArraySampler", lambda *args: sampler)
    options = MeasurementOptions(
        fields=["u"],
        data=NumpyDataOptions(type="numpy", path="unused.npz"),
        noise=NoiseOptions(stddev=0.1),
    )
    with pytest.raises(StudyFileError, match=r"measurements\.invalid.*exactly one rank"):
        build_observation_operator(
            "invalid", options, SimpleNamespace(state_spec={"u": None}), comm
        )
