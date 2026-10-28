"""Generation of synthetic measurement data from a forward run: ``kipe synthesis``.

The true data at the measurement times come from the source of the truth
(``synthesis.truth``): the forward solver runs with its nominal parameters through all
measurement times (``run``), or reads the states from the checkpoints of an earlier run
(``checkpoints``), and the observation operators give the true data; or they are read from
files (``data``), e.g., the noise-free data of an earlier synthesis. Noise is then added: the
measurement's ``noise.stddev``, or ``synthesis.noise`` where given (e.g., none, for
noise-free data). The resulting measurements are written to their ``data`` files, and
returned. With several noise realizations (``synthesis.realizations``), the true data are
computed once, and each realization gets its own noise and data files. From a run or
checkpoints, the noise-free data are written too, to ``<output.path>/synthesis/noise_free/``.
The provenance of the run is written to ``<output.path>/synthesis/``
(:mod:`kipe.provenance`).

The noise-free data come from :func:`observe_run` or :func:`observe_checkpoints`. They are
public: the sensitivity analysis (:mod:`kipe.sensitivity`) uses :func:`observe_run` for its
perturbed runs.

In parallel, each rank generates the entries of the data it holds (``global_indices``); the
noise of an entry depends only on its position, so the data do not depend on the number of
ranks. Rank 0 writes the files.
"""

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import assert_never

from mpi4py import MPI

import numpy as np

from kipe._formatting import format_table
from kipe._types import NDArray_f64
from kipe.forward_solver import (
    ForwardSolver,
    Parameters,
    SupportsCheckpoints,
    build_forward_solver,
)
from kipe.measurements.data import (
    Measurement,
    measurement_schedule,
    measurement_times,
    read_data,
    time_tolerance,
    write_data,
)
from kipe.measurements.operator import ObservationOperator, build_observation_operator
from kipe.options import (
    CheckpointsTruthOptions,
    DataTruthOptions,
    MeasurementOptions,
    RunTruthOptions,
    StudyFileError,
    StudyOptions,
    data_path,
    truth_path,
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
        their data: ``synthesize(study)[r][i]`` is measurement i of realization r; their
        files are written when it returns, on every rank

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
    with recorded_run(output, study, _run_info(study), comm):
        noise = _noise(study)
        _log_plan(study, noise)
        times, true_data = _truth(study, solver, operators)

        written = []
        if not isinstance(study.synthesis.truth, DataTruthOptions):  # need not be noise-free
            written += _write_noise_free(output, operators, times, true_data, comm)

        realizations = []
        for r in range(study.synthesis.realizations):
            measurements = _realization(study, r, operators, times, true_data, noise)
            written += [_write(study, measurement, r, comm) for measurement in measurements]
            realizations.append(measurements)

        _write_checksums(output, written, comm)
        _log_written(study, times)
        logger.info("%-11s%.3g s", "run time", time.perf_counter() - started)

    comm.barrier()  # rank 0 has written the data, e.g., for an estimation that follows
    return realizations


def _log_plan(study: StudyOptions, noise: Mapping[str, tuple[float, int | None]]) -> None:
    """Log what synthesis does: the solver, the truth, the realizations and the measurements.

    Args:
        study: the study options, with measurements
        noise: standard deviation and seed of the noise added to each measurement's data
    """
    assert study.measurements is not None
    logger.info("synthesis")
    logger.info("  %-16s%s", "forward solver", study.forward_solver.factory)
    truth = study.synthesis.truth
    if isinstance(truth, CheckpointsTruthOptions | DataTruthOptions):
        logger.info("  %-16s%s at %s", "truth", truth.type, truth.path)
    else:
        logger.info("  %-16s%s", "truth", truth.type)
    if study.synthesis.realizations > 1:
        logger.info("  %-16s%d", "realizations", study.synthesis.realizations)
    logger.info("")

    rows = [("measurement", "fields", "σ", "seed")]
    for name, options in study.measurements.items():
        stddev, _ = noise[name]
        seed = "" if options.noise.seed is None else str(options.noise.seed)
        sigma = f"{stddev:g}"
        if stddev != options.noise.stddev:
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
    if not isinstance(study.synthesis.truth, DataTruthOptions):
        output = Path(study.output.path) / "synthesis" / "noise_free"
        logger.info("wrote the noise-free data to %s", output)


def _truth(
    study: StudyOptions,
    solver: ForwardSolver,
    operators: Mapping[str, ObservationOperator],
) -> tuple[dict[str, NDArray_f64], dict[str, NDArray_f64]]:
    """Return the times and the true data of the measurements, from the source of the truth.

    Args:
        study: the study options, with measurements
        solver: the forward solver
        operators: the observation operators of the measurements

    Returns:
        - the times of each measurement
        - this rank's entries of each measurement's true data, one row per time

    Raises:
        StudyFileError: if a measurement lacks what its source of the truth needs, e.g., its
            times for a run, or its true data do not fit it
    """
    assert study.measurements is not None
    truth = study.synthesis.truth
    if isinstance(truth, DataTruthOptions):
        times, values = {}, {}
        for name, options in study.measurements.items():
            selected = None if options.times is None else measurement_times(options.times)
            path = truth_path(truth, name)
            times[name], values[name] = read_data(name, path, operators[name], selected)
        return times, values

    times = {name: _times(name, options) for name, options in study.measurements.items()}
    match truth:
        case RunTruthOptions():
            return times, observe_run(solver, times, operators)
        case CheckpointsTruthOptions(path=path):
            return times, observe_checkpoints(solver, path, times, operators)
        case _:
            assert_never(truth)


def _run_info(study: StudyOptions) -> dict[str, dict[str, dict[str, str]]]:
    """Return what ``run.yaml`` records beyond the study: the files of given true data.

    Args:
        study: the study options

    Returns:
        with the truth source ``data``: ``{"truth": measurement name -> path and SHA-256 hash
        of its true data}``; otherwise nothing

    Raises:
        StudyFileError: if a file of the true data cannot be read
    """
    truth = study.synthesis.truth
    if not isinstance(truth, DataTruthOptions):
        return {}

    files = {}
    for name in study.measurements or {}:
        path = truth_path(truth, name)
        try:
            files[name] = {"path": path, "sha256": sha256(path)}
        except OSError as err:
            raise StudyFileError(f"cannot read the true data {path}: {err}") from err
    return {"truth": files}


def _write_noise_free(
    output: Path,
    operators: Mapping[str, ObservationOperator],
    times: Mapping[str, NDArray_f64],
    true_data: Mapping[str, NDArray_f64],
    comm: MPI.Comm,
) -> list[str]:
    """Write the noise-free data of the measurements, to ``noise_free/`` in the output.

    Args:
        output: output directory of the synthesis
        operators: the observation operators of the measurements
        times: the times of each measurement
        true_data: this rank's entries of each measurement's true data
        comm: communicator of the run; rank 0 writes

    Returns:
        the paths written
    """
    written = []
    for name, operator in operators.items():
        path = str(output / "noise_free" / f"{name}.npz")
        write_data(path, operator, times[name], true_data[name], comm)
        written.append(path)
    return written


def _write(
    study: StudyOptions, measurement: Measurement, realization: int, comm: MPI.Comm
) -> str:
    """Write the data of a measurement of one noise realization, to its ``data.path``.

    Args:
        study: the study options, with measurements
        measurement: the measurement, with this rank's entries of its data
        realization: number of the noise realization
        comm: communicator of the run; rank 0 writes

    Returns:
        the path written
    """
    assert study.measurements is not None
    path = data_path(study.measurements[measurement.name], realization)
    write_data(path, measurement.operator, measurement.times, measurement.values, comm)
    return path


def _write_checksums(output: Path, written: list[str], comm: MPI.Comm) -> None:
    """Write ``data.sha256``: the hash and path of each file written, as ``sha256sum`` does.

    Args:
        output: output directory of the synthesis
        written: the paths of the data files written
        comm: communicator of the run; rank 0, which wrote the files, writes
    """
    if comm.rank == 0:
        lines = [f"{sha256(path)}  {path}\n" for path in written]
        (output / "data.sha256").write_text("".join(lines))


def _realization(
    study: StudyOptions,
    realization: int,
    operators: Mapping[str, ObservationOperator],
    times: Mapping[str, NDArray_f64],
    true_data: Mapping[str, NDArray_f64],
    noise: Mapping[str, tuple[float, int | None]],
) -> list[Measurement]:
    """Return the measurements of one noise realization.

    Args:
        study: the study options, with measurements
        realization: number of the noise realization
        operators: the observation operators of the measurements
        times: the times of each measurement
        true_data: this rank's entries of each measurement's true data
        noise: standard deviation and seed of the noise added to each measurement's data

    Returns:
        the measurements, in the order of the study, with this rank's entries of their data
    """
    assert study.measurements is not None
    return [
        Measurement(
            name,
            operators[name],
            times[name],
            _add_noise(true_data[name], operators[name], *noise[name], realization),
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
    for t_k, measured in measurement_schedule(times):
        if t_k < t - time_tolerance(t):
            name = measured[0][0]
            raise StudyFileError(f"measurements.{name}: time {t_k} is before the start {t}")
        elif t_k > t + time_tolerance(t):
            state = solver.propagate(t, t_k, state, dict(parameters or {}))
            t = t_k
        else:
            # the start time (up to round-off): sample the initial state, no solver call
            pass

        for name, _ in measured:
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

    for t, measured in measurement_schedule(times):
        try:
            state = solver.read_checkpoint(path, t)
        except ValueError as err:
            raise StudyFileError(f"measurements.{measured[0][0]}: {err}") from err

        for name, _ in measured:
            missing = [f for f in operators[name].fields if f not in state]
            if missing:
                raise StudyFileError(
                    f"measurements.{name}: the checkpoint at t = {t:g} has no field "
                    f"{', '.join(missing)}"
                )
            values[name].append(operators[name](state))

    return {name: np.array(v) for name, v in values.items()}


def _add_noise(
    values: NDArray_f64,
    operator: ObservationOperator,
    stddev: float,
    seed: int | None,
    realization: int,
) -> NDArray_f64:
    """Return a measurement's data of one noise realization: the true data plus noise.

    The noise at the k-th time of realization r is drawn from
    ``numpy.random.default_rng([seed, k, r])``, which is ``default_rng([seed, k])`` for r = 0:
    NumPy's seed sequence ignores trailing zeros. None is drawn without a seed (standard
    deviation 0).

    Args:
        values: this rank's entries of the true data, one row per time
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


def _noise(study: StudyOptions) -> dict[str, tuple[float, int | None]]:
    """Return the noise synthesis adds to each measurement's data: its standard deviation and
    seed.

    The standard deviation comes from ``synthesis.noise`` if it lists the measurement, else
    from the measurement's ``noise.stddev``. A measurement with noise needs a seed.

    Args:
        study: the study options, with measurements

    Returns:
        measurement name -> standard deviation and seed (None without noise)

    Raises:
        StudyFileError: if a measurement with noise has no noise seed
    """
    assert study.measurements is not None
    noise = {}
    for name, options in study.measurements.items():
        synthesis = study.synthesis.noise.get(name)
        stddev = options.noise.stddev if synthesis is None else synthesis.stddev
        if stddev > 0 and options.noise.seed is None:
            raise StudyFileError(f"measurements.{name}: synthesis needs a noise 'seed'")
        noise[name] = (stddev, options.noise.seed if stddev > 0 else None)

    return noise
