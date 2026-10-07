"""Generation of synthetic measurement data from a forward run: ``kipe synthesis``.

The forward solver runs with its nominal parameters through all measurement times. At each
time, the observation operator gives the predicted data and noise is added. The resulting
measurements are written to their ``data`` files, and returned.
"""

import logging
import time
from collections.abc import Mapping
from typing import assert_never

import numpy as np

from kipe._formatting import format_table
from kipe._types import NDArray_f64
from kipe.forward_solver import ForwardSolver, build_forward_solver
from kipe.measurements import (
    Measurement,
    ObservationOperator,
    build_observation_operator,
    measurement_times,
    write_numpy,
)
from kipe.options import MeasurementOptions, RunTruthOptions, StudyFileError, StudyOptions

logger = logging.getLogger(__name__)


def synthesize(study: StudyOptions) -> list[Measurement]:
    """Generate the data of all measurements of the study and write them.

    Args:
        study: the study options

    Returns:
        the generated measurements

    Raises:
        StudyFileError: if a measurement lacks what synthesis needs, or observes a field the
            solver does not have
    """
    if not study.measurements:
        raise StudyFileError("synthesis needs a measurements section")

    started = time.perf_counter()
    solver = build_forward_solver(study.forward_solver)
    operators = {
        name: build_observation_operator(name, options, solver.state_spec)
        for name, options in study.measurements.items()
    }

    logger.info("synthesis")
    logger.info("  %-16s%s", "forward solver", study.forward_solver.factory)
    logger.info("  %-16s%s", "truth", study.synthesis.truth.type)
    logger.info("")
    rows = [("measurement", "fields", "σ", "seed")]
    for name, options in study.measurements.items():
        seed = "" if options.noise.seed is None else str(options.noise.seed)
        rows.append((name, ", ".join(options.fields), f"{options.noise.stddev:g}", seed))
    for line in format_table(rows, "<<>>"):
        logger.info("  %s", line)

    match study.synthesis.truth:
        case RunTruthOptions():
            measurements = _generate_from_run(solver, study.measurements, operators)
        case _:
            assert_never(study.synthesis.truth)

    for measurement in measurements:
        path = study.measurements[measurement.name].data.path
        write_numpy(path, measurement.times, measurement.values)
        logger.info("wrote %s: %d times to %s", measurement.name, len(measurement.times), path)
    logger.info("%-11s%.3g s", "run time", time.perf_counter() - started)

    return measurements


def _generate_from_run(
    solver: ForwardSolver,
    options: Mapping[str, MeasurementOptions],
    operators: Mapping[str, ObservationOperator],
) -> list[Measurement]:
    """Generate the data by running the solver with its nominal parameters.

    The noise of the k-th time of a measurement is drawn from
    ``numpy.random.default_rng([seed, k])``.

    Args:
        solver: the forward solver
        options: the measurement options: times, noise
        operators: the observation operators of the measurements

    Returns:
        the generated measurements

    Raises:
        StudyFileError: if a measurement lacks times or a noise seed, or a measurement time is
            before the solver's start time
    """
    times = {name: _times(name, o) for name, o in options.items()}
    seeds = {name: _seed(name, o) for name, o in options.items()}
    values: dict[str, list[NDArray_f64]] = {name: [] for name in options}

    # all (time, measurement, index) events, in the order they occur
    events = sorted(
        ((float(t), name, k) for name in options for k, t in enumerate(times[name])),
        key=lambda event: event[0],
    )

    t, state = solver.initial_state()
    for t_k, name, k in events:
        if t_k < t - _tol(t):
            raise StudyFileError(f"measurements.{name}: time {t_k} is before the start {t}")
        elif t_k > t + _tol(t):
            state = solver.propagate(t, t_k, state, {})
            t = t_k
        else:
            # same time (up to round-off): sample the current state, no solver call
            pass

        z = operators[name](state)
        noise = np.random.default_rng([seeds[name], k]).normal(
            0.0, options[name].noise.stddev, z.shape
        )
        values[name].append(z + noise)

    return [
        Measurement(name, operators[name], times[name], np.array(values[name]), o.noise.stddev)
        for name, o in options.items()
    ]


def _times(name: str, options: MeasurementOptions) -> NDArray_f64:
    """Return the times of a measurement, which synthesis requires.

    Args:
        name: name of the measurement
        options: its options

    Returns:
        the measurement times

    Raises:
        StudyFileError: if the measurement has no times
    """
    if options.times is None:
        raise StudyFileError(f"measurements.{name}: synthesis needs 'times'")

    return measurement_times(options.times)


def _seed(name: str, options: MeasurementOptions) -> int:
    """Return the noise seed of a measurement, which synthesis requires.

    Args:
        name: name of the measurement
        options: its options

    Returns:
        the noise seed

    Raises:
        StudyFileError: if the measurement has no noise seed
    """
    if options.noise.seed is None:
        raise StudyFileError(f"measurements.{name}: synthesis needs a noise 'seed'")

    return options.noise.seed


def _tol(t: float) -> float:
    """Return the tolerance below which two times are the same.

    Args:
        t: time

    Returns:
        tolerance
    """
    return 1e-12 * max(1.0, abs(t))
