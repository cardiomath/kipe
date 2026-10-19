"""``kipe synthesis``: synthetic data from a forward run."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kipe.cli import main
from kipe.examples import fitzhugh_nagumo
from kipe.examples.fitzhugh_nagumo import Solver
from kipe.measurements import read_numpy
from kipe.options import (
    CheckpointsTruthOptions,
    ForwardSolverOptions,
    MeasurementOptions,
    NoiseOptions,
    NumpyDataOptions,
    OutputOptions,
    RunTruthOptions,
    StudyFileError,
    StudyOptions,
    SynthesisOptions,
    TimeRange,
    load_study,
)
from kipe.synthesis import synthesize


def _study(tmp_path: Path, measurements: str) -> Path:
    path = tmp_path / "study.yaml"
    path.write_text(f"""\
output: {{path: {tmp_path / "results"}}}
forward_solver:
  factory: "kipe.examples.fitzhugh_nagumo:Solver"
  arguments: {{dt: 0.05}}
measurements:
{measurements}""")
    return path


def _exact(times: list[float], field: str) -> np.ndarray:
    """Noise-free values of a field at the given times, computed directly with the solver."""
    solver = Solver(dt=0.05)
    t, state = solver.initial_state()
    values = []
    for t_k in times:
        if t_k > t:
            state = solver.propagate(t, t_k, state, {})
            t = t_k
        values.append(state[field].copy())
    return np.array(values)


def test_synthesis(tmp_path):
    """Data = exact values + noise drawn from default_rng([seed, k]) at the k-th time."""
    study = _study(
        tmp_path,
        f"""\
  v:
    fields: [v]
    times: [0.0, 0.5, 1.0]                # includes the start time
    data: {{type: numpy, path: {tmp_path / "v.npz"}}}
    noise: {{stddev: 0.1, seed: 3}}
  w:
    fields: [w]
    times: {{start: 0.25, stop: 1.0, step: 0.25}}   # interleaved with the times of v
    data: {{type: numpy, path: {tmp_path / "w.npz"}}}
    noise: {{stddev: 0.2, seed: 4}}
""",
    )
    assert main(["synthesis", str(study)]) == 0

    for name, times, stddev, seed in [
        ("v", [0.0, 0.5, 1.0], 0.1, 3),
        ("w", [0.25, 0.5, 0.75, 1.0], 0.2, 4),
    ]:
        read_times, values = read_numpy(tmp_path / f"{name}.npz")
        np.testing.assert_allclose(read_times, times)
        noise = np.array([
            np.random.default_rng([seed, k]).normal(0.0, stddev, 1) for k in range(len(times))
        ])
        np.testing.assert_allclose(values, _exact(times, name) + noise, atol=1e-12)


def test_synthesis_noise(tmp_path, kipe_caplog):
    """synthesis.noise: none for v (no seed needed), 5 times the assumed noise for w."""
    study = _study(
        tmp_path,
        f"""\
  v:
    fields: [v]
    times: [0.5, 1.0]
    data: {{type: numpy, path: {tmp_path / "v.npz"}}}
    noise: {{stddev: 0.1}}
  w:
    fields: [w]
    times: [0.5, 1.0]
    data: {{type: numpy, path: {tmp_path / "w.npz"}}}
    noise: {{stddev: 0.1, seed: 4}}
synthesis:
  noise:
    v: {{stddev: 0}}
    w: {{stddev: 0.5}}
""",
    )
    assert main(["synthesis", str(study)]) == 0

    times = [0.5, 1.0]
    _, v = read_numpy(tmp_path / "v.npz")
    np.testing.assert_array_equal(v, _exact(times, "v"))

    _, w = read_numpy(tmp_path / "w.npz")
    noise = np.array([np.random.default_rng([4, k]).normal(0.0, 0.5, 1) for k in range(2)])
    np.testing.assert_allclose(w, _exact(times, "w") + noise, atol=1e-12)

    assert "0 (assumes 0.1)" in kipe_caplog.text
    assert "0.5 (assumes 0.1)" in kipe_caplog.text


def test_synthesize_returns_the_written_measurements(tmp_path):
    study = _study(
        tmp_path,
        f"""\
  v:
    fields: [v]
    times: [0.5, 1.0]
    data: {{type: numpy, path: {tmp_path / "v.npz"}}}
    noise: {{stddev: 0.1, seed: 0}}
""",
    )
    [(measurement,)] = synthesize(load_study(study))
    times, values = read_numpy(tmp_path / "v.npz")
    assert measurement.name == "v"
    assert measurement.stddev == pytest.approx(0.1)
    np.testing.assert_array_equal(measurement.times, times)
    np.testing.assert_array_equal(measurement.values, values)


def test_synthesize_alias(tmp_path):
    study = _study(
        tmp_path,
        f"""\
  v:
    fields: [v]
    times: [1.0]
    data: {{type: numpy, path: {tmp_path / "v.npz"}}}
    noise: {{stddev: 0.1, seed: 0}}
""",
    )
    assert main(["synthesize", str(study)]) == 0
    assert (tmp_path / "v.npz").exists()


@pytest.mark.parametrize(
    ("measurement", "message"),
    [
        ("fields: [v]\n    noise: {stddev: 0.1, seed: 0}", "synthesis needs 'times'"),
        ("fields: [v]\n    times: [1.0]\n    noise: {stddev: 0.1}", "needs a noise 'seed'"),
        (
            "fields: [u]\n    times: [1.0]\n    noise: {stddev: 0.1, seed: 0}",
            "has no field u. Available: v, w",
        ),
        (
            "fields: [v]\n    times: [-1.0]\n    noise: {stddev: 0.1, seed: 0}",
            "before the start",
        ),
    ],
)
def test_synthesis_errors(tmp_path, capsys, measurement, message):
    study = _study(
        tmp_path,
        f"""\
  v:
    {measurement}
    data: {{type: numpy, path: {tmp_path / "v.npz"}}}
""",
    )
    assert main(["synthesis", str(study)]) == 1
    assert message in capsys.readouterr().err


# --- truth from checkpoints --------------------------------------------------------------------


def _options(
    tmp_path: Path,
    truth: RunTruthOptions | CheckpointsTruthOptions,
    folder: str,
    solver: str = "Checkpointed",
    realizations: int = 1,
) -> StudyOptions:
    """A study measuring v and w at interleaved times, with noise, writing to ``folder``.

    ``solver`` is a factory in :mod:`kipe.examples.fitzhugh_nagumo`; ``Checkpointed`` is made
    by :func:`_checkpoints`. With several realizations, each writes to ``folder/r<r>``.
    """
    data = tmp_path / folder / ("r{realization}" if realizations > 1 else "")
    measurements = {
        "v": MeasurementOptions(
            fields=["v"],
            times=[0.0, 0.5, 1.0],
            data=NumpyDataOptions(type="numpy", path=str(data / "v.npz")),
            noise=NoiseOptions(stddev=0.1, seed=3),
        ),
        "w": MeasurementOptions(
            fields=["w"],
            times=TimeRange(start=0.25, stop=1.0, step=0.25),
            data=NumpyDataOptions(type="numpy", path=str(data / "w.npz")),
            noise=NoiseOptions(stddev=0.2, seed=4),
        ),
    }
    return StudyOptions(
        output=OutputOptions(path=str(tmp_path / folder)),
        forward_solver=ForwardSolverOptions(
            factory=f"kipe.examples.fitzhugh_nagumo:{solver}", arguments={"dt": 0.05}
        ),
        measurements=measurements,
        synthesis=SynthesisOptions(truth=truth, realizations=realizations),
    )


CHECKPOINTS = CheckpointsTruthOptions(type="checkpoints", path="reference")


def _checkpoints(
    monkeypatch: pytest.MonkeyPatch, fields: list[str], times: list[float]
) -> list[tuple[str, float]]:
    """Make the factory ``kipe.examples.fitzhugh_nagumo:Checkpointed``: the FitzHugh-Nagumo
    solver, reading checkpoints of the given fields at the given times from a run with its
    nominal parameters. Return the paths and times it reads.

    A new class each time: ``isinstance`` checks against protocols are cached per class.
    """
    read = []

    class Checkpointed(Solver):
        def read_checkpoint(self, path: str, t: float) -> dict[str, np.ndarray]:
            if not np.isclose(times, t).any():
                raise ValueError(f"no checkpoint at t = {t:g}")
            read.append((path, t))
            return {field: _exact([t], field)[0] for field in fields}

    monkeypatch.setattr(fitzhugh_nagumo, "Checkpointed", Checkpointed, raising=False)
    return read


def test_synthesis_from_checkpoints(tmp_path, monkeypatch):
    """The same data as from a run, reading each time once, also when shared by v and w."""
    [run] = synthesize(_options(tmp_path, RunTruthOptions(), "run", solver="Solver"))
    read = _checkpoints(monkeypatch, ["v", "w"], [0.0, 0.25, 0.5, 0.75, 1.0])
    [checkpoints] = synthesize(_options(tmp_path, CHECKPOINTS, "checkpoints"))

    assert read == [("reference", t) for t in [0.0, 0.25, 0.5, 0.75, 1.0]]
    for a, b in zip(checkpoints, run, strict=True):
        np.testing.assert_array_equal(a.times, b.times)
        np.testing.assert_allclose(a.values, b.values, rtol=1e-14)
    _, v = read_numpy(tmp_path / "checkpoints" / "v.npz")
    np.testing.assert_array_equal(v, checkpoints[0].values)


def test_synthesis_from_partial_checkpoints(tmp_path, monkeypatch):
    """Checkpoints need only the observed fields."""
    _checkpoints(monkeypatch, ["v", "w"], [0.0, 0.25, 0.5, 0.75, 1.0])
    [expected] = synthesize(_options(tmp_path, CHECKPOINTS, "a"))

    _checkpoints(monkeypatch, ["v"], [0.0, 0.5, 1.0])
    options = _options(tmp_path, CHECKPOINTS, "b")
    assert options.measurements is not None
    options.measurements.pop("w")
    [(v,)] = synthesize(options)
    np.testing.assert_array_equal(v.values, expected[0].values)


@pytest.mark.parametrize(
    ("fields", "times", "message"),
    [
        (None, None, "the forward solver cannot read checkpoints"),
        (
            ["v"],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            "measurements.w: the checkpoint at t = 0.25 has no field w",
        ),
        (["v", "w"], [0.0, 0.25, 0.5, 1.0], "measurements.w: no checkpoint at t = 0.75"),
    ],
)
def test_synthesis_from_checkpoints_errors(tmp_path, monkeypatch, fields, times, message):
    solver = "Solver"  # without read_checkpoint
    if fields is not None:
        _checkpoints(monkeypatch, fields, times)
        solver = "Checkpointed"
    options = _options(tmp_path, CHECKPOINTS, "c", solver)
    with pytest.raises(StudyFileError, match=message):
        synthesize(options)


def test_realizations(tmp_path, monkeypatch):
    """Realization 0 has the data of a study without realizations; realization r the noise
    drawn from default_rng([seed, k, r]) at the k-th time, on the same truth, read once."""
    [single] = synthesize(_options(tmp_path, RunTruthOptions(), "single", solver="Solver"))
    read = _checkpoints(monkeypatch, ["v", "w"], [0.0, 0.25, 0.5, 0.75, 1.0])
    realizations = synthesize(_options(tmp_path, CHECKPOINTS, "multi", realizations=3))

    assert len(read) == 5  # each checkpoint once, not once per realization
    assert len(realizations) == 3
    for a, b in zip(realizations[0], single, strict=True):
        np.testing.assert_allclose(a.values, b.values, rtol=1e-14)

    v = realizations[2][0]
    times = [0.0, 0.5, 1.0]
    noise = [np.random.default_rng([3, k, 2]).normal(0.0, 0.1, 1) for k in range(3)]
    np.testing.assert_allclose(v.values, _exact(times, "v") + noise, rtol=1e-12)
    _, written = read_numpy(tmp_path / "multi" / "r2" / "v.npz")
    np.testing.assert_array_equal(written, v.values)


def test_synthesis_writes_checksums_of_the_data(tmp_path):
    """In the format of sha256sum, all realizations."""
    realizations = synthesize(_options(tmp_path, RunTruthOptions(), "multi", "Solver", 2))

    lines = (tmp_path / "multi" / "synthesis" / "data.sha256").read_text().splitlines()
    paths = [
        tmp_path / "multi" / f"r{r}" / f"{measurement.name}.npz"
        for r, measurements in enumerate(realizations)
        for measurement in measurements
    ]
    assert lines == [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}" for p in paths]
