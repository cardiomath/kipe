"""The ``measurements`` and ``synthesis`` sections of the study file."""

import pytest
from pydantic import TypeAdapter, ValidationError

from kipe.options import (
    ArraySamplingOptions,
    CheckpointsTruthOptions,
    DifferenceModelOptions,
    MeasurementOptions,
    MeshSamplingOptions,
    RunTruthOptions,
    StudyOptions,
    SynthesisNoiseOptions,
    TimeRange,
)

BASE = {
    "output": {"path": "results"},
    "forward_solver": {"factory": "kipe.examples.fitzhugh_nagumo:Solver"},
}

MEASUREMENT = {
    "fields": ["v"],
    "data": {"type": "numpy", "path": "data/v.npz"},
    "noise": {"stddev": 0.1, "seed": 0},
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
    assert study.synthesis.truth == RunTruthOptions()


def test_synthesis_truth_checkpoints():
    truth = {"type": "checkpoints", "path": "results/reference"}
    study = _study({"v": MEASUREMENT}, synthesis={"truth": truth})
    assert study.synthesis.truth == CheckpointsTruthOptions(**truth)
    with pytest.raises(ValidationError, match="path"):
        _study({"v": MEASUREMENT}, synthesis={"truth": {"type": "checkpoints"}})


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
        {"noise": {"stddev": 0.1, "seed": -1}},
        {"spatial_sampling": {"type": "fenicsx.voxel"}},  # not available yet
        {"model": {"type": "pcmri_phase"}},  # not available yet
        {"data": {"type": "csv", "path": "data/v.csv"}},
    ],
)
def test_invalid_measurement(change):
    with pytest.raises(ValidationError):
        TypeAdapter(MeasurementOptions).validate_python(MEASUREMENT | change)


def test_synthesis_noise():
    noise = {"v": {"stddev": 0}}
    study = _study({"v": MEASUREMENT}, synthesis={"noise": noise})
    assert study.synthesis.noise["v"] == SynthesisNoiseOptions(stddev=0.0)


@pytest.mark.parametrize(
    ("noise", "message"),
    [
        ({"w": {"stddev": 0.1}}, "synthesis.noise: no measurement w"),
        ({"v": {"stddev": -0.1}}, "greater than or equal to 0"),
    ],
)
def test_invalid_synthesis_noise(noise, message):
    with pytest.raises(ValidationError, match=message):
        _study({"v": MEASUREMENT}, synthesis={"noise": noise})


def test_measurements_must_not_share_noise_seeds():
    w = MEASUREMENT | {"fields": ["w"]}  # same seed 0 as v
    with pytest.raises(ValidationError, match="v and w share the noise seed 0"):
        _study({"v": MEASUREMENT, "w": w})


def test_measurements_without_seeds():
    """Seeds are only needed by synthesis; estimation runs without them."""
    noise = {"stddev": 0.1}
    study = _study({
        "v": MEASUREMENT | {"noise": noise},
        "w": MEASUREMENT | {"fields": ["w"], "noise": noise},
    })
    assert all(m.noise.seed is None for m in study.measurements.values())


def _mesh_sampling(**options) -> MeshSamplingOptions:
    """The spatial sampling of a study with one fenicsx.mesh measurement."""
    sampling = {"type": "fenicsx.mesh", "mesh": "data/mesh.xdmf"} | options
    study = _study({"u": MEASUREMENT | {"fields": ["u"], "spatial_sampling": sampling}})
    assert study.measurements is not None
    spatial_sampling = study.measurements["u"].spatial_sampling
    assert isinstance(spatial_sampling, MeshSamplingOptions)
    return spatial_sampling


def test_mesh_sampling():
    assert _mesh_sampling().element == "P1"
    assert _mesh_sampling().projection is None
    assert _mesh_sampling(projection=[0, 0, 1]).projection == [0.0, 0.0, 1.0]
    assert _mesh_sampling(element="DG0", projection="surface_normal").projection == (
        "surface_normal"
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"projection": "surface_normal"}, "needs element DG0"),  # P1, the default
        ({"projection": [0, 0]}, "must not be zero"),
        ({"projection": "normal"}, "valid list"),
        ({"element": "P2"}, "element"),
    ],
)
def test_invalid_mesh_sampling(options, message):
    with pytest.raises(ValidationError, match=message):
        _mesh_sampling(**options)
