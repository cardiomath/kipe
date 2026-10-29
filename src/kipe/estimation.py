"""Parameter estimation with the ROUKF: ``kipe estimation``.

:class:`Estimation` runs the reduced-order unscented Kalman filter (:mod:`kipe.roukf`) over a
set of measurements: at each measurement time, the sigma points are propagated with the
forward solver, compared with the data of the measurements at that time through their
observation operators, and the estimate is corrected. The measurements may have different
times; the filter steps through all of them in order. Outer iterations repeat the pass,
starting from the previous estimate.
:func:`estimate` builds all of this from a study file.

The estimation history is written to ``<output.path>/estimation/``: ``history.csv`` (one row
per assimilation step, written as the filter runs) and ``history.npz`` (all steps of the
completed outer iterations, including the full parameter covariance, written after each). An
interrupted estimation can continue from it (``kipe estimation --resume``). :func:`estimate`
also writes the provenance of the run there (:mod:`kipe.provenance`). With several noise
realizations (``synthesis.realizations``), each estimation uses the data of one realization
and writes to its subdirectory, e.g., ``estimation/r07/``.
"""

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mpi4py import MPI

import numpy as np

from kipe import roukf
from kipe._formatting import format_table
from kipe._history import (
    History,
    Step,
    count_iterations,
    final_thetas,
    read_completed_steps,
)
from kipe._types import NDArray_f64
from kipe.forward_solver import (
    FieldSpec,
    ForwardSolver,
    Parameters,
    State,
    build_forward_solver,
)
from kipe.measurements.data import (
    Measurement,
    measurement_schedule,
    measurement_times,
    read_data,
    time_tolerance,
)
from kipe.measurements.operator import build_observation_operator
from kipe.options import (
    EstimationOptions,
    StudyFileError,
    StudyOptions,
    data_path,
    estimation_directory,
)
from kipe.parameters import Parameterization, build_parameterization
from kipe.provenance import data_files, recorded_run

logger = logging.getLogger(__name__)

_WIDTH = 78  # width of the separator lines in the log


class Estimation:
    """Parameter estimation with the ROUKF over a set of measurements.

    Args:
        solver: the forward solver
        parameterization: the estimated parameters, with their initial estimate and uncertainty
        measurements: the measurements, with distinct names; each at its own times, those at
            the same time (up to round-off) are assimilated together
        options: number of outer iterations and sigma-point stencil
        output: directory for the estimation history
        comm: communicator over which the model state is distributed
        completed_steps: the steps of the completed outer iterations of an earlier,
            interrupted run of the same estimation, which this one continues

    Raises:
        ValueError: if two measurements have the same name
        StudyFileError: if a measurement has two times that are the same up to round-off
    """

    def __init__(
        self,
        solver: ForwardSolver,
        parameterization: Parameterization,
        measurements: list[Measurement],
        options: EstimationOptions,
        output: Path,
        comm: MPI.Comm = MPI.COMM_WORLD,
        completed_steps: Sequence[Step] = (),
    ) -> None:
        self._solver = solver
        self._parameterization = parameterization
        self._measurements = measurements
        self._iterations = options.iterations
        self._sigma_points = options.sigma_points
        self._output = output
        self._comm = comm

        self._measurements_by_name = {m.name: m for m in measurements}
        if len(self._measurements_by_name) < len(measurements):
            raise ValueError("the measurements must have distinct names")
        self._schedule = measurement_schedule({m.name: m.times for m in measurements})
        self._layout = _StateLayout(solver.state_spec)
        self._stencil = roukf.sigma_point_stencil(
            options.sigma_points, len(parameterization.names)
        )
        self._history = History(output, parameterization.names, comm, completed_steps)
        self._completed_iterations = count_iterations(completed_steps)

    @property
    def times(self) -> list[float]:
        """The assimilation times: the times of all measurements, in order."""
        return [t for t, _ in self._schedule]

    def run(self) -> Parameters:
        """Run the outer iterations, each a pass of the filter over all measurements.

        Each pass after the first starts from the previous estimate, with the initial
        uncertainty. After each pass, the history is written to ``history.npz``. A run that
        continues an earlier one starts after its completed iterations, from their estimate.

        Returns:
            the estimated parameters, as physical values
        """
        self._log_setup()
        started = time.perf_counter()

        # the estimate after the completed iterations, anchored as they were
        parameterization = self._parameterization
        for theta in final_thetas(self._history.steps):
            estimate = parameterization.to_physical(theta)
            parameterization = parameterization.with_anchor(estimate)
        if self._completed_iterations:
            logger.info(
                "resumed after iteration %d/%d", self._completed_iterations, self._iterations
            )

        for iteration in range(self._completed_iterations, self._iterations):
            title = f" iteration {iteration + 1}/{self._iterations} "
            logger.info("%s", title.center(_WIDTH, "━"))
            iteration_started = time.perf_counter()

            filter_state = self._assimilate(parameterization, iteration)

            elapsed = time.perf_counter() - iteration_started
            logger.info("")
            logger.info(
                "  estimate after iteration %d/%d (%.3g s)",
                iteration + 1,
                self._iterations,
                elapsed,
            )
            for line in format_table(_estimate_rows(filter_state, parameterization), "<><"):
                logger.info("    %s", line)

            estimate = parameterization.to_physical(filter_state.theta)
            # get parameterization of next iteration, anchored at the current estimate
            parameterization = parameterization.with_anchor(estimate)
            self._history.write_npz()

        logger.info("%s", "━" * _WIDTH)
        logger.info("%-11s%s", "history", self._output / "history.{csv,npz}")
        logger.info("%-11s%.3g s", "run time", time.perf_counter() - started)

        return estimate

    def _log_setup(self) -> None:
        """Log what is estimated, from which measurements, and how."""
        solver = type(self._solver)
        logger.info("ROUKF estimation")
        logger.info("  %-16s%s:%s", "forward solver", solver.__module__, solver.__qualname__)
        logger.info("  %-16s%s", "output", self._output)
        logger.info("")
        logger.info("  filter")
        n_points = self._stencil.points.shape[1]
        logger.info("    %-14s%s (%d sigma points)", "stencil", self._sigma_points, n_points)
        logger.info("    %-14s%d", "iterations", self._iterations)
        logger.info("    %-14s%d", "steps", len(self._schedule))
        logger.info("    %-14s%d", "MPI ranks", self._comm.size)
        logger.info("")

        parameters = [("parameter", "reparameterization", "initial", "1σ range")]
        ranges = self._parameterization.one_sigma_range()
        for p, (lower, upper) in zip(self._parameterization.parameters, ranges, strict=True):
            parameters.append((
                p.name,
                p.reparameterization,
                f"{p.anchor:.4g}",
                f"[{lower:.4g}, {upper:.4g}]",
            ))
        for line in format_table(parameters, "<<><"):
            logger.info("  %s", line)
        logger.info("")

        measurements = [("measurement", "fields", "times", "σ")]
        for m in self._measurements:
            times = f"{len(m.times)} ({m.times[0]:g} … {m.times[-1]:g})"
            measurements.append((m.name, ", ".join(m.operator.fields), times, f"{m.stddev:g}"))
        for line in format_table(measurements, "<<<>"):
            logger.info("  %s", line)

    def _assimilate(
        self, parameterization: Parameterization, iteration: int
    ) -> roukf.FilterState:
        r"""Run one pass of the filter over all measurement times.

        Args:
            parameterization: the estimated parameters, relative to this pass's initial
                estimate
            iteration: index of this pass

        Returns:
            the filter state at the end of the pass

        Raises:
            StudyFileError: if the first measurement time is not after the solver's start time
        """
        t, solver_state = self._solver.initial_state()
        if self.times[0] <= t + time_tolerance(t):
            raise StudyFileError(
                f"measurement time {self.times[0]} is not after the solver's start time {t}"
            )

        filter_state = roukf.initial_state(
            self._layout.to_vector(solver_state),
            parameterization.anchor_theta(),
            parameterization.stddev_theta(),
            self._stencil,
        )
        self._history.record(iteration, t, filter_state, parameterization)
        logger.info("%s", _table_header(parameterization.names))

        for k, t_next in enumerate(self.times):
            # 1. sampling
            x_sigma, theta_sigma = roukf.sample(filter_state, self._stencil)

            # 2. propagation and 3. innovations
            x_sigma, Gamma = self._propagate_and_observe(
                t, float(t_next), x_sigma, theta_sigma, parameterization, k
            )

            # 4. correction
            filter_state = roukf.update(x_sigma, theta_sigma, Gamma, self._stencil, self._comm)
            t = float(t_next)
            self._history.record(iteration, t, filter_state, parameterization)
            physical = parameterization.to_physical(filter_state.theta)
            logger.info("%s", _table_row(k + 1, t, physical, self._innovation_rms(Gamma)))

        return filter_state

    def _propagate_and_observe(
        self,
        t: float,
        t_next: float,
        x_sigma: NDArray_f64,
        theta_sigma: NDArray_f64,
        parameterization: Parameterization,
        k: int,
    ) -> tuple[NDArray_f64, NDArray_f64]:
        """Propagate the sigma points to the next measurement time and compute their innovations.

        Propagation and innovation are interleaved, sigma point by sigma point, because the
        state returned by the solver is only valid until its next call.

        Args:
            t: current time
            t_next: the next measurement time
            x_sigma: the sigma-point states at ``t``, one per column, shape ``(n, r)``
            theta_sigma: the sigma-point parameters, one per column, shape ``(p, r)``
            parameterization: maps the parameters to physical values for the solver
            k: index of the next assimilation step

        Returns:
            - the propagated sigma-point states, one per column, shape ``(n, r)``
            - their innovations, weighted with the noise, shape ``(m, r)``
        """
        x_propagated = np.empty_like(x_sigma)
        Gamma = np.empty((self._innovation_size(k), x_sigma.shape[1]))

        r = x_sigma.shape[1]
        for i in range(r):
            phi = parameterization.to_physical(theta_sigma[:, i])
            values = ", ".join(f"{name} = {value:.4g}" for name, value in phi.items())
            logger.debug("step %d, sigma point %d/%d: %s", k + 1, i + 1, r, values)
            started = time.perf_counter()

            try:
                solver_state = self._solver.propagate(
                    t, t_next, self._layout.to_state(x_sigma[:, i]), phi
                )
            except Exception:
                # which parameters made the solver fail; the exception itself follows
                logger.error(
                    "forward solver failed in step %d (t = %g → %g), sigma point %d/%d: %s",
                    k + 1, t, t_next, i + 1, r, values,
                )  # fmt: skip
                raise
            elapsed = time.perf_counter() - started
            logger.debug("  └─ propagated %g → %g (%.2g s)", t, t_next, elapsed)

            x_propagated[:, i] = self._layout.to_vector(solver_state)
            Gamma[:, i] = self._innovation(solver_state, k)

        return x_propagated, Gamma

    def _innovation(self, state: State, k: int) -> NDArray_f64:
        """Return the innovations of the measurements of the k-th step, weighted with the noise.

        Args:
            state: the propagated state of one sigma point
            k: index of the assimilation step

        Returns:
            the innovations, concatenated in the order of the measurements
        """
        blocks = []
        for name, row in self._schedule[k][1]:
            measurement = self._measurements_by_name[name]
            z_hat = measurement.operator(state)
            z = measurement.values[row]
            innovation = measurement.operator.model.innovation(z_hat, z)
            blocks.append(innovation / measurement.stddev)

        return np.concatenate(blocks)

    def _innovation_rms(self, Gamma: NDArray_f64) -> float:
        """Return the RMS of the mean innovation, over all ranks.

        The innovations are weighted with the noise, so the RMS approaches 1 if the model and
        the assumed noise fit the data.

        Args:
            Gamma: the innovations of the sigma points, shape ``(m, r)``

        Returns:
            the RMS of the mean innovation
        """
        mean = roukf.mean(Gamma, self._stencil)
        local = np.array([mean @ mean, mean.size])  # each entry is held by exactly one rank
        self._comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)

        return float(np.sqrt(local[0] / local[1]))

    def _innovation_size(self, k: int) -> int:
        """Return the number of innovation entries of the measurements of the k-th step.

        Args:
            k: index of the assimilation step

        Returns:
            the number of this rank's entries
        """
        measurements = self._measurements_by_name
        return sum(measurements[name].values.shape[1] for name, _ in self._schedule[k][1])


def estimate(
    study: StudyOptions,
    comm: MPI.Comm = MPI.COMM_WORLD,
    realization: int = 0,
    resume: bool = False,
) -> Parameters:
    """Estimate the parameters of a study; write the estimation history and the provenance.

    Builds the forward solver, the estimated parameters and the measurements (with their data)
    from the study file, then runs the :class:`Estimation`.

    With ``resume``, an interrupted estimation continues after its completed outer iterations,
    as recorded in its ``history.npz``, with the same result as without the interruption. The
    study may have more ``estimation.iterations`` than before, but must otherwise be the same.
    Without a history, the estimation starts from the beginning; if it is complete, nothing
    runs.

    Args:
        study: the study options
        comm: communicator over which the model state is distributed
        realization: number of the noise realization whose data are used
            (``synthesis.realizations``)
        resume: continue an interrupted estimation

    Returns:
        the estimated parameters, as physical values

    Raises:
        StudyFileError: if the study lacks what estimation needs, has no such realization, or
            the data do not fit; with ``resume``, if the study has changed since the earlier run
    """
    if study.parameters is None:
        raise StudyFileError("estimation needs a parameters section")

    output = estimation_directory(study, realization)
    data = data_files(study, realization)
    completed_steps: list[Step] = []
    if resume:
        completed_steps = read_completed_steps(output, study, data, comm)
        comm.barrier()  # every rank has read the earlier run's files before rank 0 rewrites them
    completed_iterations = count_iterations(completed_steps)
    if completed_iterations >= study.estimation.iterations:
        logger.info(
            "%s: complete after %d iterations, nothing to resume", output, completed_iterations
        )
        return dict(
            zip(study.parameters.priors, completed_steps[-1].parameters.tolist(), strict=True)
        )

    solver = build_forward_solver(study.forward_solver)
    parameterization = build_parameterization(
        study.parameters, solver.nominal_parameters().keys()
    )
    measurements = _build_measurements(study, solver, comm, realization)
    estimation = Estimation(
        solver, parameterization, measurements, study.estimation, output, comm, completed_steps
    )

    run: dict[str, Any] = {"assimilation_times": estimation.times}
    if study.synthesis.realizations > 1:
        run["realization"] = realization
    run["data"] = data
    if completed_iterations:
        run["resumed_after_iteration"] = completed_iterations

    with recorded_run(output, study, run, comm):
        return estimation.run()


def _table_header(names: list[str]) -> str:
    """Return the header of the per-step table in the log.

    Args:
        names: names of the estimated parameters

    Returns:
        the header line
    """
    return f"{'step':>6}{'time':>9}" + "".join(f"{n:>11}" for n in names) + f"{'innovation':>12}"


def _table_row(step: int, t: float, physical: Parameters, rms: float) -> str:
    """Return one row of the per-step table in the log.

    Args:
        step: number of the assimilation step
        t: its time
        physical: the estimate, as physical values
        rms: RMS of the mean innovation

    Returns:
        the table row
    """
    values = "".join(f"{v:>11.4g}" for v in physical.values())
    return f"{step:>6}{t:>9.4g}{values}{rms:>12.3g}"


def _estimate_rows(
    filter_state: roukf.FilterState, parameterization: Parameterization
) -> list[tuple[str, str, str]]:
    """Return the table of the estimate with its physical 1σ range, with a header.

    Args:
        filter_state: the filter state
        parameterization: maps the estimate to physical values

    Returns:
        the header and one row per parameter: name, estimate, 1σ range
    """
    theta = filter_state.theta
    stddev = np.sqrt(np.diag(filter_state.covariance()))
    physical = parameterization.to_physical(theta)
    ranges = parameterization.physical_range(theta, stddev)

    rows = [("parameter", "estimate", "1σ range")]
    for (name, value), (lower, upper) in zip(physical.items(), ranges, strict=True):
        rows.append((name, f"{value:.4g}", f"[{lower:.4g}, {upper:.4g}]"))

    return rows


def _build_measurements(
    study: StudyOptions, solver: ForwardSolver, comm: MPI.Comm, realization: int
) -> list[Measurement]:
    """Read the data of the study's measurements and build their observation operators.

    Every rank reads the whole data and keeps the entries it holds (``global_indices``), see
    :func:`~kipe.measurements.data.read_data`.

    Args:
        study: the study options
        solver: the forward solver
        comm: communicator of the run
        realization: number of the noise realization whose data are read

    Returns:
        the measurements, with this rank's entries of their data at the selected times

    Raises:
        StudyFileError: if there are no measurements, a measurement observes a field the solver
            does not have, its data do not have its number of entries, or a selected time is
            not in the data
    """
    if not study.measurements:
        raise StudyFileError("estimation needs a measurements section")

    measurements = []
    for name, options in study.measurements.items():
        operator = build_observation_operator(name, options, solver, comm)
        selected = None if options.times is None else measurement_times(options.times)
        times, values = read_data(name, data_path(options, realization), operator, selected)
        measurements.append(Measurement(name, operator, times, values, options.noise.stddev))

    return measurements


class _StateLayout:
    """Conversion between the solver's state (named fields) and one rank-local vector.

    The fields are concatenated in the order of the solver's ``state_spec``.

    Args:
        state_spec: the solver's state fields
    """

    def __init__(self, state_spec: Mapping[str, FieldSpec]) -> None:
        self._slices: dict[str, slice] = {}
        offset = 0
        for name, spec in state_spec.items():
            self._slices[name] = slice(offset, offset + spec.local_size)
            offset += spec.local_size

    def to_vector(self, state: State) -> NDArray_f64:
        """Concatenate the fields of a state into a new vector.

        Args:
            state: the state

        Returns:
            the state as one vector
        """
        return np.concatenate([state[name] for name in self._slices])

    def to_state(self, vector: NDArray_f64) -> State:
        """Split a vector into the fields of a state (new, contiguous arrays).

        Args:
            vector: the state as one vector

        Returns:
            the state
        """
        return {name: vector[s].copy() for name, s in self._slices.items()}
