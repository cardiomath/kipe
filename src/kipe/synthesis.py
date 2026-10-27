"""Generation of synthetic measurement data from a forward run: ``kipe synthesis``.

The true state at the measurement times comes from the forward solver (``synthesis.truth``):
it runs with its nominal parameters through all measurement times (``run``), or reads the
states from the checkpoints of an earlier run (``checkpoints``). At each time, the observation
operator gives the predicted data and noise is added: the measurement's
``noise.stddev``, or ``synthesis.noise`` where given (e.g., none, for noise-free data). The
resulting measurements are written to their ``data`` files, and returned. With several noise
realizations (``synthesis.realizations``), the true state is computed once, and each
realization gets its own noise and data files. The provenance of the run is written to
``<output.path>/synthesis/`` (:mod:`kipe.provenance`).

The noise-free data come from :func:`observe_run` or :func:`observe_checkpoints`. They are
public: the sensitivity analysis (:mod:`kipe.sensitivity`) uses :func:`observe_run` for its
perturbed runs.

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
    Parameters,
    State,
    SupportsCheckpoints,
    build_forward_solver,
)
from kipe.measurements.data import Measurement, measurement_times, write_data
from kipe.measurements.operator import ObservationOperator, build_observation_operator
from kipe.options import (
    CheckpointsTruthOptions,
    MeasurementOptions,
    RunTruthOptions,
    StudyFileError,
    StudyOptions,
    SynthesisOptions,
    TruthOptions,
    data_path,
)
from kipe.provenance import recorded_run, sha256

logger = logging.getLogger(__name__)


def synthesize(study: StudyOptions, comm: MPI.Comm = MPI.COMM_WORLD) -> list[list[Measurement]]:
    """Generate the data of all measurements of the study; write them and the provenance.

    Args:
        study: the study options
        comm: communicator over which the model state is distributed

    Returns:
        the generated measurements of each noise realization, with this rank's entries of
        their data: ``synthesize(study)[r][i]`` is measurement i of realization r

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

    stddevs = {
        name: _added_stddev(name, options, study.synthesis)
        for name, options in study.measurements.items()
    }

    output = Path(study.output.path) / "synthesis"
    with recorded_run(output, study, comm=comm):
        _log_plan(study, stddevs)
        times = {name: _times(name, o) for name, o in study.measurements.items()}
        seeds = {
            name: _seed(name, o) for name, o in study.measurements.items() if stddevs[name] > 0
        }
        noise_free = _noise_free(study.synthesis.truth, solver, times, operators)

        realizations = []
        written = []
        for r in range(study.synthesis.realizations):
            measurements = _realization(study, r, operators, times, noise_free, stddevs, seeds)
            for measurement in measurements:
                path = data_path(study.measurements[measurement.name], r)
                write_data(
                    path, measurement.operator, measurement.times, measurement.values, comm
                )
                written.append(path)
            realizations.append(measurements)

        if comm.rank == 0:  # rank 0 wrote the files
            lines = [f"{sha256(path)}  {path}\n" for path in written]
            (output / "data.sha256").write_text("".join(lines))
        _log_written(study, times)
        logger.info("%-11s%.3g s", "run time", time.perf_counter() - started)

    return realizations


def _log_plan(study: StudyOptions, stddevs: Mapping[str, float]) -> None:
    """Log what synthesis does: the solver, the truth, the realizations and the measurements.

    Args:
        study: the study options, with measurements
        stddevs: standard deviations of the noise added to each measurement's data
    """
    assert study.measurements is not None
    logger.info("synthesis")
    logger.info("  %-16s%s", "forward solver", study.forward_solver.factory)
    truth = study.synthesis.truth
    if isinstance(truth, CheckpointsTruthOptions):
        logger.info("  %-16s%s at %s", "truth", truth.type, truth.path)
    else:
        logger.info("  %-16s%s", "truth", truth.type)
    if study.synthesis.realizations > 1:
        logger.info("  %-16s%d", "realizations", study.synthesis.realizations)
    logger.info("")

    rows = [("measurement", "fields", "σ", "seed")]
    for name, options in study.measurements.items():
        seed = "" if options.noise.seed is None else str(options.noise.seed)
        sigma = f"{stddevs[name]:g}"
        if stddevs[name] != options.noise.stddev:
            sigma += f" (assumes {options.noise.stddev:g})"
        rows.append((name, ", ".join(options.fields), sigma, seed))
    for line in format_table(rows, "<<>>"):
        logger.info("  %s", line)


def _log_written(study: StudyOptions, times: Mapping[str, NDArray_f64]) -> None:
    """Log where the data of each measurement went.

    Args:
        study: the study options, with measurements
        times: the times of each measurement
    """
    assert study.measurements is not None
    n = study.synthesis.realizations
    each = f"{n} realizations of " if n > 1 else ""
    for name, options in study.measurements.items():
        logger.info(
            "wrote %s: %s%d times to %s", name, each, len(times[name]), options.data.path
        )


def _noise_free(
    truth: TruthOptions,
    solver: ForwardSolver,
    times: Mapping[str, NDArray_f64],
    operators: Mapping[str, ObservationOperator],
) -> dict[str, NDArray_f64]:
    """Return the noise-free data of the measurements, from the source of the truth.

    Args:
        truth: the source of the true state, ``synthesis.truth``
        solver: the forward solver
        times: the times of each measurement
        operators: the observation operators of the measurements

    Returns:
        this rank's entries of each measurement's noise-free data, one row per time
    """
    match truth:
        case RunTruthOptions():
            return observe_run(solver, times, operators)
        case CheckpointsTruthOptions(path=path):
            return observe_checkpoints(solver, path, times, operators)
        case _:
            assert_never(truth)


def _realization(
    study: StudyOptions,
    realization: int,
    operators: Mapping[str, ObservationOperator],
    times: Mapping[str, NDArray_f64],
    noise_free: Mapping[str, NDArray_f64],
    stddevs: Mapping[str, float],
    seeds: Mapping[str, int],
) -> list[Measurement]:
    """Return the measurements of one noise realization.

    Args:
        study: the study options, with measurements
        realization: number of the noise realization
        operators: the observation operators of the measurements
        times: the times of each measurement
        noise_free: this rank's entries of each measurement's noise-free data
        stddevs: standard deviations of the noise added to each measurement's data
        seeds: noise seeds of the measurements with noise

    Returns:
        the measurements, in the order of the study, with this rank's entries of their data
    """
    assert study.measurements is not None
    return [
        Measurement(
            name,
            operators[name],
            times[name],
            _add_noise(
                noise_free[name], operators[name], stddevs[name], seeds.get(name), realization
            ),
            options.noise.stddev,
        )
        for name, options in study.measurements.items()
    ]


def observe_run(
    solver: ForwardSolver,
    times: Mapping[str, NDArray_f64],
    operators: Mapping[str, ObservationOperator],
    parameters: Parameters | None = None,
) -> dict[str, NDArray_f64]:
    """Return the noise-free data of a run of the solver: its predicted measurements.

    The truth of synthesis with the nominal parameters, the perturbed runs of the sensitivity
    analysis (:mod:`kipe.sensitivity`) with others.

    Args:
        solver: the forward solver
        times: the times of each measurement
        operators: the observation operators of the measurements
        parameters: physical values of solver parameters; the nominal ones for all others

    Returns:
        this rank's entries of each measurement's noise-free data, one row per time

    Raises:
        StudyFileError: if a measurement time is before the solver's start time
    """
    values: dict[str, list[NDArray_f64]] = {name: [] for name in times}

    t, state = solver.initial_state()
    for t_k, name, _ in _events(times):
        if t_k < t - _tol(t):
            raise StudyFileError(f"measurements.{name}: time {t_k} is before the start {t}")
        elif t_k > t + _tol(t):
            state = solver.propagate(t, t_k, state, dict(parameters or {}))
            t = t_k
        else:
            # same time (up to round-off): sample the current state, no solver call
            pass

        values[name].append(operators[name](state))

    return {name: np.array(v) for name, v in values.items()}


def observe_checkpoints(
    solver: ForwardSolver,
    path: str,
    times: Mapping[str, NDArray_f64],
    operators: Mapping[str, ObservationOperator],
) -> dict[str, NDArray_f64]:
    """Return the noise-free data of the checkpoints of an earlier run: its predicted
    measurements, from the states read at the measurement times.

    The truth of synthesis from checkpoints (``synthesis.truth: {type: checkpoints}``).

    Args:
        solver: the forward solver, which must read checkpoints
        path: location of the checkpoints
        times: the times of each measurement
        operators: the observation operators of the measurements

    Returns:
        this rank's entries of each measurement's noise-free data, one row per time

    Raises:
        StudyFileError: if the solver cannot read checkpoints, or a checkpoint is missing or
            lacks a field a measurement observes
    """
    if not isinstance(solver, SupportsCheckpoints):
        raise StudyFileError(
            "synthesis.truth checkpoints: the forward solver cannot read checkpoints "
            "(it has no read_checkpoint)"
        )

    values: dict[str, list[NDArray_f64]] = {name: [] for name in times}

    t: float | None = None
    state: State = {}
    for t_k, name, _ in _events(times):
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
        values[name].append(operators[name](state))

    return {name: np.array(v) for name, v in values.items()}


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


def _add_noise(
    values: NDArray_f64,
    operator: ObservationOperator,
    stddev: float,
    seed: int | None,
    realization: int,
) -> NDArray_f64:
    """Return a measurement's data of one noise realization: the noise-free data plus noise.

    The noise at the k-th time of realization r is drawn from
    ``numpy.random.default_rng([seed, k, r])``, which is ``default_rng([seed, k])`` for r = 0:
    NumPy's seed sequence ignores trailing zeros. None is drawn without a seed (standard
    deviation 0).

    Args:
        values: this rank's entries of the noise-free data, one row per time
        operator: the observation operator of the measurement
        stddev: standard deviation of the added noise
        seed: noise seed of the measurement, ``None`` for no noise
        realization: number of the noise realization

    Returns:
        the data, at the positions ``operator.global_indices``
    """
    if seed is None:
        return values.copy()

    noisy = np.empty_like(values)
    for k, z in enumerate(values):
        # the noise of all entries, the same on every rank; each keeps those it holds
        noise = np.random.default_rng([seed, k, realization]).normal(0.0, stddev, operator.size)
        noisy[k] = z + noise[operator.global_indices]
    return noisy


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
