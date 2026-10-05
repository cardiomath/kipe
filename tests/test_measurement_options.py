"""The ``measurements`` section of the study file."""

import pytest
from pydantic import TypeAdapter, ValidationError

from kipe.options import (
    ArraySamplingOptions,
    DifferenceModelOptions,
    MeasurementOptions,
    StudyOptions,
    TimeRange,
)

BASE = {
    "output": {"path": "results"},
    "forward_solver": {"factory": "kipe.examples.fitzhugh_nagumo:Solver"},
}

MEASUREMENT = {
    "fields": ["v"],
    "data": {"type": "numpy", "path": "data/v.npz"},
    "noise": {"stddev": 0.1},
}


def _study(measurements: dict, **sections) -> StudyOptions:
    data = BASE | {"measurements": measurements} | sections
    return TypeAdapter(StudyOptions).validate_python(data)


def test_defaults():
    study = _study({"v": MEASUREMENT})
    measurement = study.measurements["v"]
    assert measurement.times is None
    assert measurement.spatial_sampling == ArraySamplingOptions()
    assert measurement.model == DifferenceModelOptions()


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        ([0.5, 1.0, 2.0], [0.5, 1.0, 2.0]),
        ({"start": 0.5, "stop": 20.0, "step": 0.5}, TimeRange(0.5, 20.0, 0.5)),
    ],
)
def test_times(times, expected):
    measurement = TypeAdapter(MeasurementOptions).validate_python(MEASUREMENT | {"times": times})
    assert measurement.times == expected


@pytest.mark.parametrize(
    "change",
    [
        {"fields": []},
        {"times": {"start": 1.0, "stop": 0.5, "step": 0.1}},  # stop before start
        {"times": {"start": 0.0, "stop": 1.0, "step": 0.0}},
        {"noise": {"stddev": 0.0}},
        {"spatial_sampling": {"type": "fenicsx.voxel"}},  # not available yet
        {"model": {"type": "pcmri_phase"}},  # not available yet
        {"data": {"type": "csv", "path": "data/v.csv"}},
    ],
)
def test_invalid_measurement(change):
    with pytest.raises(ValidationError):
        TypeAdapter(MeasurementOptions).validate_python(MEASUREMENT | change)
