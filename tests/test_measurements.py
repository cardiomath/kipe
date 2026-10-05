"""Measurement models, numpy data, measurement times and spatial samplers."""

import numpy as np
import pytest

from kipe.measurements import (
    DifferenceModel,
    MeasurementModel,
    MeasurementSnapshot,
    build_model,
    measurement_times,
    read_numpy,
    write_numpy,
)
from kipe.options import ArraySamplingOptions, DifferenceModelOptions, StudyFileError, TimeRange
from kipe.sampling import ArraySampler, SpatialSampler, build_sampler

SNAPSHOT = MeasurementSnapshot(z=np.array([1.0, 2.0, 3.0]))


def test_difference_model():
    model = build_model(DifferenceModelOptions())
    assert isinstance(model, DifferenceModel)
    z_hat = model.predict(np.array([0.5, 2.0, 4.0]), SNAPSHOT)
    np.testing.assert_array_equal(z_hat, [0.5, 2.0, 4.0])
    np.testing.assert_array_equal(model.innovation(z_hat, SNAPSHOT), [0.5, 0.0, -1.0])


def test_incomplete_model_cannot_be_instantiated():
    """A model missing a method fails on instantiation, not deep inside the filter."""

    class InnovationOnly(MeasurementModel):
        def innovation(self, y, m):
            return y

    with pytest.raises(TypeError, match="predict"):
        InnovationOnly()


def test_array_sampler_concatenates_in_order_of_fields():
    sampler = build_sampler(ArraySamplingOptions(), ["w", "v"])
    assert isinstance(sampler, ArraySampler)
    fields = {"v": np.array([1.0]), "w": np.array([2.0, 3.0])}
    np.testing.assert_array_equal(sampler.sample(fields, SNAPSHOT), [2.0, 3.0, 1.0])


def test_array_sampler_returns_a_new_array():
    """The filter keeps sampled states while the fields' arrays may be reused."""
    v = np.array([1.0, 2.0])
    y = ArraySampler(["v"]).sample({"v": v}, SNAPSHOT)
    v[:] = 0.0
    np.testing.assert_array_equal(y, [1.0, 2.0])


def test_incomplete_sampler_cannot_be_instantiated():
    class NoSample(SpatialSampler):
        pass

    with pytest.raises(TypeError, match="sample"):
        NoSample()


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        ([0.5, 1.0, 2.0], [0.5, 1.0, 2.0]),
        (TimeRange(0.0, 0.3, 0.1), [0.0, 0.1, 0.2, 0.3]),  # stop included despite round-off
        (TimeRange(0.0, 1.0, 0.3), [0.0, 0.3, 0.6, 0.9]),  # stop not on the grid
        (TimeRange(0.5, 0.5, 0.1), [0.5]),
    ],
)
def test_measurement_times(times, expected):
    np.testing.assert_allclose(measurement_times(times), expected)


def test_numpy_round_trip(tmp_path):
    path = tmp_path / "data" / "v.npz"  # directory is created
    times = np.array([0.5, 1.0])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    write_numpy(path, times, values)
    read_times, read_values = read_numpy(path)
    np.testing.assert_array_equal(read_times, times)
    np.testing.assert_array_equal(read_values, values)


def test_read_numpy_errors(tmp_path):
    with pytest.raises(StudyFileError, match="cannot read"):
        read_numpy(tmp_path / "missing.npz")

    np.savez(tmp_path / "no_values.npz", times=np.zeros(2))
    with pytest.raises(StudyFileError, match="no array values"):
        read_numpy(tmp_path / "no_values.npz")

    np.savez(tmp_path / "mismatch.npz", times=np.zeros(2), values=np.zeros((3, 1)))
    with pytest.raises(StudyFileError, match="expected times"):
        read_numpy(tmp_path / "mismatch.npz")
