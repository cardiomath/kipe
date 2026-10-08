"""Parameter estimation with the ROUKF: ``kipe estimation``.

:class:`Estimation` runs the reduced-order unscented Kalman filter (:mod:`kipe.roukf`) over a
set of measurements: at each measurement time, the sigma points are propagated with the
forward solver, compared with the data through the observation operators, and the estimate is
corrected. Outer iterations repeat the pass, starting from the previous estimate.
:func:`estimate` builds all of this from a study file.

The estimation history is written to ``<output.path>/estimation/``: ``history.csv`` (one row
per assimilation step, written as the filter runs) and ``history.npz`` (all steps, including
the full parameter covariance). :func:`estimate` also writes the provenance of the run there
(:mod:`kipe.provenance`).
"""

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import numpy as np

from kipe import roukf
from kipe._formatting import format_table
from kipe._types import NDArray_f64
from kipe.forward_solver import (
    FieldSpec,
    ForwardSolver,
    Parameters,
    State,
    build_forward_solver,
)
from kipe.measurements import (
    Measurement,
    build_observation_operator,
    measurement_times,
    read_numpy,
)
from kipe.options import EstimationOptions, StudyFileError, StudyOptions
from kipe.parameters import Parameterization, build_parameterization
from kipe.provenance import recorded_run

logger = logging.getLogger(__name__)

_WIDTH = 78  # width of the separator lines in the log


class Estimation:
    """Parameter estimation with the ROUKF over a set of measurements.

    Args:
        solver: the forward solver
        parameterization: the estimated parameters, with their initial estimate and uncertainty
        measurements: the measurements; they must share their times
        options: number of outer iterations and sigma-point stencil
        output: directory for the estimation history
        comm: communicator over which the model state is distributed

    Raises:
        StudyFileError: if the measurements have different times, or one observes a
            distributed field (not supported yet)
    """

    def __init__(
        self,
        solver: ForwardSolver,
        parameterization: Parameterization,
        measurements: list[Measurement],
        options: EstimationOptions,
        output: Path,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        self._solver = solver
        self._parameterization = parameterization
        self._measurements = measurements
        self._iterations = options.iterations
        self._particles = options.particles
        self._output = output
        self._comm = comm

        self._times = _common_times(measurements)
        self._rank_contributes = _rank_contributes(measurements, solver.state_spec, comm)
        self._layout = _StateLayout(solver.state_spec)
        self._stencil = roukf.sigma_point_stencil(options.particles, len(parameterization.names))
        self._history = _History(output, parameterization.names, comm)

    def run(self) -> Parameters:
        """Run the outer iterations, each a pass of the filter over all measurements.

        Each pass after the first starts from the previous estimate, with the initial
        uncertainty.

        Returns:
            the estimated parameters, as physical values
        """
        self._log_setup()
        started = time.perf_counter()

        parameterization = self._parameterization
        for iteration in range(self._iterations):
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
            parameterization = parameterization.recenter(filter_state.theta)

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
        sigma_points = self._stencil.points.shape[1]
        logger.info("    %-14s%s (%d sigma points)", "stencil", self._particles, sigma_points)
        logger.info("    %-14s%d", "iterations", self._iterations)
        logger.info("    %-14s%d", "MPI ranks", self._comm.size)
        logger.info("")

        parameters = [("parameter", "reparameterization", "initial", "1σ range")]
        ranges = self._parameterization.one_sigma_range()
        for p, (lower, upper) in zip(self._parameterization.parameters, ranges, strict=True):
            parameters.append((
                p.name,
                p.reparameterization,
                f"{p.initial:.4g}",
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
        if self._times[0] <= t + 1e-12 * max(1.0, abs(t)):
            raise StudyFileError(
                f"measurement time {self._times[0]} is not after the solver's start time {t}"
            )

        filter_state = roukf.initial_state(
            self._layout.to_vector(solver_state),
            parameterization.initial_theta(),
            parameterization.stddev_theta(),
            self._stencil,
        )
        self._history.record(iteration, t, filter_state, parameterization)
        logger.info("%s", _table_header(parameterization.names))

        for k, t_next in enumerate(self._times):
            # 1. sampling
            x_sigma, theta_sigma = roukf.sample(filter_state, self._stencil)

            # 2. propagation and 3. innovations
            x_sigma, Gamma = self._propagate_and_observe(
                t, float(t_next), x_sigma, theta_sigma, parameterization, k
            )

            # 4. correction
            filter_state = roukf.update(
                x_sigma,
                theta_sigma,
                Gamma,
                self._stencil,
                self._comm,
                contributes=self._rank_contributes,
            )
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
            k: index of the next measurement time

        Returns:
            - the propagated sigma-point states, one per column, shape ``(n, r)``
            - their innovations, weighted with the noise, shape ``(m, r)``
        """
        x_propagated = np.empty_like(x_sigma)
        Gamma = np.empty((self._innovation_size(), x_sigma.shape[1]))

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
        """Return the innovations of all measurements at the k-th time, weighted with the noise.

        Args:
            state: the propagated state of one sigma point
            k: index of the assimilation time

        Returns:
            the innovations, concatenated in the order of the measurements
        """
        blocks = []
        for measurement in self._measurements:
            z_hat = measurement.operator(state)
            z = measurement.values[k]
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
        local = np.array([mean @ mean, mean.size] if self._rank_contributes else [0.0, 0.0])
        self._comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)

        return float(np.sqrt(local[0] / local[1]))

    def _innovation_size(self) -> int:
        """Return the number of innovation entries of all measurements at one time.

        Returns:
            the number of entries
        """
        return sum(measurement.values.shape[1] for measurement in self._measurements)


def estimate(study: StudyOptions, comm: MPI.Comm = MPI.COMM_WORLD) -> Parameters:
    """Estimate the parameters of a study; write the estimation history and the provenance.

    Builds the forward solver, the estimated parameters and the measurements (with their data)
    from the study file, then runs the :class:`Estimation`.

    Args:
        study: the study options
        comm: communicator over which the model state is distributed

    Returns:
        the estimated parameters, as physical values

    Raises:
        StudyFileError: if the study lacks what estimation needs, or the data do not fit
    """
    if study.parameters is None:
        raise StudyFileError("estimation needs a parameters section")

    solver = build_forward_solver(study.forward_solver)
    parameterization = build_parameterization(study.parameters, solver.nominal_parameters())
    measurements = _build_measurements(study, solver.state_spec)
    output = Path(study.output.path) / "estimation"
    estimation = Estimation(
        solver, parameterization, measurements, study.estimation, output, comm
    )

    times = {"assimilation_times": _common_times(measurements).tolist()}  # maybe from the data
    with recorded_run(output, study, times, comm):
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
    study: StudyOptions, state_spec: Mapping[str, FieldSpec]
) -> list[Measurement]:
    """Read the data of the study's measurements and build their observation operators.

    Args:
        study: the study options
        state_spec: the solver's state fields

    Returns:
        the measurements, with their data at the selected times

    Raises:
        StudyFileError: if there are no measurements, a measurement observes a field the solver
            does not have, or a selected time is not in the data
    """
    if not study.measurements:
        raise StudyFileError("estimation needs a measurements section")

    measurements = []
    for name, options in study.measurements.items():
        operator = build_observation_operator(name, options, state_spec)
        times, values = read_numpy(options.data.path)
        if options.times is not None:
            indices = [_index_of(t, times, name) for t in measurement_times(options.times)]
            times, values = times[indices], values[indices]

        measurements.append(Measurement(name, operator, times, values, options.noise.stddev))

    return measurements


def _index_of(t: float, times: NDArray_f64, name: str) -> int:
    """Return the index of time ``t`` in the data's times.

    Args:
        t: selected time
        times: the times in the data
        name: name of the measurement, for the error message

    Returns:
        the index

    Raises:
        StudyFileError: if ``t`` is not in the data
    """
    matches = np.flatnonzero(np.isclose(times, t, rtol=0, atol=1e-12 * max(1.0, abs(t))))
    if len(matches) == 0:
        raise StudyFileError(f"measurements.{name}: time {t} is not in the data")

    return int(matches[0])


def _common_times(measurements: list[Measurement]) -> NDArray_f64:
    """Return the assimilation times, which all measurements must share.

    Args:
        measurements: the measurements

    Returns:
        the times

    Raises:
        StudyFileError: if the measurements have different times
    """
    times = measurements[0].times
    for measurement in measurements[1:]:
        if len(measurement.times) != len(times) or not np.allclose(
            measurement.times, times, rtol=0, atol=1e-12 * max(1.0, np.abs(times).max())
        ):
            raise StudyFileError(
                f"measurements {measurements[0].name} and {measurement.name} have different "
                "times; for now, all measurements must share their times"
            )

    return times


def _rank_contributes(
    measurements: list[Measurement], state_spec: Mapping[str, FieldSpec], comm: MPI.Comm
) -> bool:
    """Return whether this rank's innovation entries count in the gain.

    Measurements of replicated fields are held in full by every rank, so only rank 0
    contributes.

    Args:
        measurements: the measurements
        state_spec: the solver's state fields
        comm: the communicator

    Returns:
        whether this rank contributes

    Raises:
        StudyFileError: if a measurement observes a distributed field (not supported yet)
    """
    for measurement in measurements:
        for name in measurement.operator.fields:
            if state_spec[name].kind == "distributed":
                raise StudyFileError(
                    f"measurements.{measurement.name}: field {name} is distributed over the "
                    "MPI ranks; measurements of distributed fields are not supported yet"
                )

    return comm.rank == 0


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


@dataclass(frozen=True)
class _Step:
    """The estimate after one assimilation step, as recorded in the history."""

    iteration: int
    time: float
    parameters: NDArray_f64  # physical values
    lower: NDArray_f64  # physical 1σ range
    upper: NDArray_f64
    theta: NDArray_f64
    P_theta: NDArray_f64


class _History:
    """The estimation history: one entry per assimilation step.

    Rows are appended to ``history.csv`` as the filter runs; :meth:`write_npz` writes all
    steps, including the full parameter covariance. Only rank 0 writes.

    Per parameter, e.g., ``c``, the CSV has the columns ``c`` (physical value), ``lower_c`` and
    ``upper_c`` (its physical 1σ range), ``theta_c`` and ``stddev_theta_c``.

    Args:
        path: output directory
        names: names of the estimated parameters
        comm: the communicator
    """

    def __init__(self, path: Path, names: list[str], comm: MPI.Comm) -> None:
        self._path = path
        self._names = names
        self._writes = comm.rank == 0
        self._steps: list[_Step] = []
        if self._writes:
            path.mkdir(parents=True, exist_ok=True)
            header = ["iteration", "time"]
            for name in names:
                header += [name, f"lower_{name}", f"upper_{name}"]
                header += [f"theta_{name}", f"stddev_theta_{name}"]
            (path / "history.csv").write_text(",".join(header) + "\n")

    def record(
        self,
        iteration: int,
        t: float,
        filter_state: roukf.FilterState,
        parameterization: Parameterization,
    ) -> None:
        r"""Record the estimate after an assimilation step.

        Args:
            iteration: index of the pass
            t: time of the step
            filter_state: the filter state, with the estimate :math:`\theta`
            parameterization: maps :math:`\theta` to physical values
        """
        theta, P_theta = filter_state.theta, filter_state.covariance()
        stddev = np.sqrt(np.diag(P_theta))
        ranges = np.array(parameterization.physical_range(theta, stddev))
        step = _Step(
            iteration=iteration,
            time=t,
            parameters=np.array(list(parameterization.to_physical(theta).values())),
            lower=ranges[:, 0],
            upper=ranges[:, 1],
            theta=theta.copy(),
            P_theta=P_theta.copy(),
        )
        self._steps.append(step)

        if self._writes:
            row = [str(iteration), repr(t)]
            columns = zip(step.parameters, step.lower, step.upper, theta, stddev, strict=True)
            for values in columns:
                row += [repr(float(value)) for value in values]
            with (self._path / "history.csv").open("a") as f:
                f.write(",".join(row) + "\n")

    def write_npz(self) -> None:
        """Write all steps to ``history.npz``."""
        if not self._writes:
            return

        np.savez(
            self._path / "history.npz",
            names=np.array(self._names),
            iteration=np.array([step.iteration for step in self._steps]),
            time=np.array([step.time for step in self._steps]),
            parameters=np.array([step.parameters for step in self._steps]),
            lower=np.array([step.lower for step in self._steps]),
            upper=np.array([step.upper for step in self._steps]),
            theta=np.array([step.theta for step in self._steps]),
            P_theta=np.array([step.P_theta for step in self._steps]),
        )
