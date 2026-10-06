"""``kipe synthesis``: synthetic data from a forward run, and the example study file."""

from pathlib import Path

import numpy as np
import pytest

from kipe.cli import main
from kipe.examples.fitzhugh_nagumo import Solver
from kipe.measurements import read_numpy
from kipe.options import load_study
from kipe.synthesis import synthesize

EXAMPLE = Path(__file__).parents[1] / "examples" / "fitzhugh_nagumo" / "study.yaml"


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


def test_example_study_file_is_valid():
    study = load_study(EXAMPLE)
    assert set(study.measurements) == {"v", "w"}


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
    (measurement,) = synthesize(load_study(study))
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
