"""Independent noise uses global data order, including distributed and empty ranks."""

from types import SimpleNamespace

from mpi4py import MPI

import numpy as np
import pytest
from pydantic import ValidationError

from kipe.measurements.data import noise_stddev
from kipe.options import NoiseOptions, StudyFileError, SynthesisNoiseOptions
from kipe.synthesis import _add_noise


def test_component_noise_global_order_and_realizations():
    comm = MPI.COMM_WORLD
    indices = np.arange(3, dtype=np.int64)[comm.rank :: comm.size][::-1]
    operator = SimpleNamespace(size=3, global_indices=indices)
    stddev = [0.01, 0.1, 1.0]
    values = np.zeros((2, len(indices)))
    for realization in [0, 2]:
        noisy = _add_noise(values, operator, stddev, 7, realization)
        expected = np.array([
            np.random.default_rng([7, k, realization]).normal(0, stddev, 3)[indices]
            for k in range(2)
        ])
        np.testing.assert_array_equal(noisy, expected)
    np.testing.assert_array_equal(noise_stddev(stddev, 3, indices), np.array(stddev)[indices])
    zero = _add_noise(values, operator, [0, 0, 0], None, 0)
    np.testing.assert_array_equal(zero, values)
    with pytest.raises(StudyFileError, match="expected 3"):
        _add_noise(values, operator, [0, 0], None, 0)


@pytest.mark.parametrize("stddev", [[], [0.1, -1], [0.1, np.nan], [np.inf], np.inf])
def test_invalid_component_noise_configuration(stddev):
    for options in [NoiseOptions, SynthesisNoiseOptions]:
        with pytest.raises(ValidationError):
            options(stddev=stddev)


def test_zero_only_allowed_for_synthesis():
    assert SynthesisNoiseOptions(stddev=[0, 0.1, 0]).stddev == [0, 0.1, 0]
    with pytest.raises(ValidationError):
        NoiseOptions(stddev=[0, 0.1, 0])
    with pytest.raises(StudyFileError, match="expected 3"):
        noise_stddev([0.1, 0.2], 3)
