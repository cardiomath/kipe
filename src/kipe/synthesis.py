"""Generation of synthetic measurement data from a forward run: ``kipe synthesis``.

The true state at the measurement times comes from the forward solver (``synthesis.truth``):
it runs with its nominal parameters through all measurement times (``run``), or reads the
states from the checkpoints of an earlier run (``checkpoints``). At each time, the observation
operator gives the predicted data and noise is added: the measurement's
``noise.stddev``, or ``synthesis.noise`` where given (e.g., none, for noise-free data). The
resulting measurements are written to their ``data`` files, and returned. The provenance of the
run is written to ``<output.path>/synthesis/`` (:mod:`kipe.provenance`).

In parallel, each rank generates the entries of the data it holds (``global_indices``); the
noise of an entry depends only on its position, so the data do not depend on the number of
ranks. Rank 0 writes the files.
"""

import logging
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import assert_never

from mpi4py import MPI

import numpy as np

from kipe._formatting import format_table
from kipe._types import NDArray_f64
from kipe.forward_solver import (
    ForwardSolver,
    State,
    SupportsCheckpoints,
    build_forward_solver,
)
from kipe.measurements import (
    Measurement,
    ObservationOperator,
    build_observation_operator,
    measurement_times,
    write_numpy,
)
from kipe.options import (
    CheckpointsTruthOptions,
    MeasurementOptions,
    RunTruthOptions,
    StudyFileError,
    StudyOptions,
    SynthesisOptions,
)
from kipe.provenance import recorded_run

logger = logging.getLogger(__name__)


def synthesize(study: StudyOptions, comm: MPI.Comm = MPI.COMM_WORLD) -> list[Measurement]:
    """Generate the data of all measurements of the study; write them and the provenance.

    Args:
        study: the study options
        comm: communicator over which the model state is distributed

    Returns:
        the generated measurements, with this rank's entries of their data

    Raises:
        StudyFileError: if a measurement lacks what synthesis needs, or observes a field the
            solver does not have
    """
    if not study.measurements:
        raise StudyFileError("synthesis needs a measurements section")

    started = time.perf_counter()
    solver = build_forward_solver(study.forward_solver)
    operators = {
        name: build_observation_operator(name, options, solver, comm)
        for name, options in study.measurements.items()
    }

    output = Path(study.output.path) / "synthesis"
    with recorded_run(output, study, comm=comm):
        logger.info("synthesis")
        logger.info("  %-16s%s", "forward solver", study.forward_solver.factory)
        truth = study.synthesis.truth
        if isinstance(truth, CheckpointsTruthOptions):
            logger.info("  %-16s%s at %s", "truth", truth.type, truth.path)
        else:
            logger.info("  %-16s%s", "truth", truth.type)
        logger.info("")
        stddevs = {
            name: _added_stddev(name, options, study.synthesis)
            for name, options in study.measurements.items()
        }
        rows = [("measurement", "fields", "σ", "seed")]
        for name, options in study.measurements.items():
            seed = "" if options.noise.seed is None else str(options.noise.seed)
            sigma = f"{stddevs[name]:g}"
            if stddevs[name] != options.noise.stddev:
                sigma += f" (assumes {options.noise.stddev:g})"
            rows.append((name, ", ".join(options.fields), sigma, seed))
        for line in format_table(rows, "<<>>"):
            logger.info("  %s", line)

        match study.synthesis.truth:
            case RunTruthOptions():
                measurements = _generate_from_run(solver, study.measurements, operators, stddevs)
            case CheckpointsTruthOptions(path=path):
                measurements = _generate_from_checkpoints(
                    solver, path, study.measurements, operators, stddevs
                )
            case _:
                assert_never(study.synthesis.truth)

        for measurement in measurements:
            path = study.measurements[measurement.name].data.path
            _write(path, measurement, comm)
            logger.info(
                "wrote %s: %d times to %s", measurement.name, len(measurement.times), path
            )
        logger.info("%-11s%.3g s", "run time", time.perf_counter() - started)

    return measurements


def _generate_from_run(
    solver: ForwardSolver,
    options: Mapping[str, MeasurementOptions],
    operators: Mapping[str, ObservationOperator],
    stddevs: Mapping[str, float],
) -> list[Measurement]:
    """Generate the data by running the solver with its nominal parameters.

    Args:
        solver: the forward solver
        options: the measurement options: times, noise seed, assumed noise
        operators: the observation operators of the measurements
        stddevs: standard deviations of the noise added to each measurement's data

    Returns:
        the generated measurements

    Raises:
        StudyFileError: if a measurement lacks times or a noise seed, or a measurement time is
            before the solver's start time
    """
    times = {name: _times(name, o) for name, o in options.items()}
    seeds = {name: _seed(name, o) for name, o in options.items() if stddevs[name] > 0}
    values: dict[str, list[NDArray_f64]] = {name: [] for name in options}

    t, state = solver.initial_state()
    for t_k, name, k in _events(times):
        if t_k < t - _tol(t):
            raise StudyFileError(f"measurements.{name}: time {t_k} is before the start {t}")
        elif t_k > t + _tol(t):
            state = solver.propagate(t, t_k, state, {})
            t = t_k
        else:
            # same time (up to round-off): sample the current state, no solver call
            pass

        values[name].append(_observe(operators[name], state, stddevs[name], seeds.get(name), k))

    return _measurements(options, operators, times, values)


def _generate_from_checkpoints(
    solver: ForwardSolver,
    path: str,
    options: Mapping[str, MeasurementOptions],
    operators: Mapping[str, ObservationOperator],
    stddevs: Mapping[str, float],
) -> list[Measurement]:
    """Generate the data from the checkpoints of an earlier run, read at the measurement times.

    Args:
        solver: the forward solver, which must read checkpoints
        path: location of the checkpoints
        options: the measurement options: times, noise seed, assumed noise
        operators: the observation operators of the measurements
        stddevs: standard deviations of the noise added to each measurement's data

    Returns:
        the generated measurements

    Raises:
        StudyFileError: if the solver cannot read checkpoints, a measurement lacks times or a
            noise seed, or a checkpoint is missing or lacks a field a measurement observes
    """
    if not isinstance(solver, SupportsCheckpoints):
        raise StudyFileError(
            "synthesis.truth checkpoints: the forward solver cannot read checkpoints "
            "(it has no read_checkpoint)"
        )

    times = {name: _times(name, o) for name, o in options.items()}
    seeds = {name: _seed(name, o) for name, o in options.items() if stddevs[name] > 0}
    values: dict[str, list[NDArray_f64]] = {name: [] for name in options}

    t: float | None = None
    state: State = {}
    for t_k, name, k in _events(times):
        if t is None or abs(t_k - t) > _tol(t):  # a new time: read its checkpoint
            try:
                state = solver.read_checkpoint(path, t_k)
            except ValueError as err:
                raise StudyFileError(f"measurements.{name}: {err}") from err
            t = t_k

        missing = [f for f in operators[name].fields if f not in state]
        if missing:
            raise StudyFileError(
                f"measurements.{name}: the checkpoint at t = {t_k:g} has no field "
                f"{', '.join(missing)}"
            )
        values[name].append(_observe(operators[name], state, stddevs[name], seeds.get(name), k))

    return _measurements(options, operators, times, values)


def _events(times: Mapping[str, NDArray_f64]) -> Iterator[tuple[float, str, int]]:
    """Return the measurement events in the order they occur.

    Args:
        times: the times of each measurement

    Returns:
        the events (time, measurement name, index of the time in the measurement's times),
        sorted by time
    """
    events = [(float(t), name, k) for name, ts in times.items() for k, t in enumerate(ts)]
    return iter(sorted(events, key=lambda event: event[0]))


def _observe(
    operator: ObservationOperator, state: State, stddev: float, seed: int | None, k: int
) -> NDArray_f64:
    """Return this rank's entries of the data of a measurement at its k-th time.

    The noise of the k-th time is drawn from ``numpy.random.default_rng([seed, k])``; none is
    drawn without a seed (standard deviation 0).

    Args:
        operator: the observation operator of the measurement
        state: the true state at the k-th time
        stddev: standard deviation of the added noise
        seed: noise seed of the measurement, ``None`` for no noise
        k: index of the time in the measurement's times

    Returns:
        the predicted data plus noise, at the positions ``operator.global_indices``
    """
    z = operator(state)
    if seed is not None:
        # the noise of all entries, the same on every rank; each keeps those it holds
        noise = np.random.default_rng([seed, k]).normal(0.0, stddev, operator.size)
        z = z + noise[operator.global_indices]
    return z


def _measurements(
    options: Mapping[str, MeasurementOptions],
    operators: Mapping[str, ObservationOperator],
    times: Mapping[str, NDArray_f64],
    values: Mapping[str, list[NDArray_f64]],
) -> list[Measurement]:
    """Return the generated measurements.

    Args:
        options: the measurement options, for the assumed noise
        operators: the observation operators of the measurements
        times: the times of each measurement
        values: this rank's entries of the data of each measurement, one array per time

    Returns:
        the measurements, in the order of the options
    """
    return [
        Measurement(name, operators[name], times[name], np.array(values[name]), o.noise.stddev)
        for name, o in options.items()
    ]


def _write(path: str, measurement: Measurement, comm: MPI.Comm) -> None:
    """Write a measurement's data, gathered from all ranks, on rank 0.

    Args:
        path: path to the ``.npz`` file
        measurement: the measurement, with this rank's entries of its data
        comm: communicator over which the entries are distributed
    """
    parts = comm.gather((measurement.operator.global_indices, measurement.values), root=0)
    if comm.rank != 0:
        return

    assert parts is not None
    values = np.empty((len(measurement.times), measurement.operator.size))
    for global_indices, local_values in parts:
        values[:, global_indices] = local_values
    write_numpy(path, measurement.times, values)


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


def _added_stddev(name: str, options: MeasurementOptions, synthesis: SynthesisOptions) -> float:
    """Return the standard deviation of the noise synthesis adds to a measurement's data.

    Args:
        name: name of the measurement
        options: its options
        synthesis: the synthesis options

    Returns:
        the standard deviation from ``synthesis.noise`` if it lists the measurement, else the
        measurement's ``noise.stddev``
    """
    if name in synthesis.noise:
        return synthesis.noise[name].stddev

    return options.noise.stddev


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
