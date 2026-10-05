"""Generation of synthetic measurement data from a forward run: ``kipe synthesis``.

The forward solver runs with its nominal parameters through all measurement times. At each
time, the observed fields are sampled, the data predicted and noise added; each measurement
is written to its ``data`` file.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import assert_never

import numpy as np

from kipe._types import NDArray_f64
from kipe.forward_solver import FieldSpec, ForwardSolver, build_forward_solver
from kipe.measurements import (
    MeasurementModel,
    build_model,
    measurement_times,
    write_numpy,
)
from kipe.options import MeasurementOptions, RunTruthOptions, StudyFileError, StudyOptions
from kipe.sampling import SpatialSampler, build_sampler

logger = logging.getLogger(__name__)


@dataclass
class _Measurement:
    """One measurement to generate: its options, times, components and generated values."""

    name: str
    options: MeasurementOptions
    seed: int
    times: NDArray_f64
    sampler: SpatialSampler
    model: MeasurementModel
    values: list[NDArray_f64] = field(default_factory=list)


def synthesize(study: StudyOptions) -> None:
    """Generate and write the data of all measurements of the study.

    Args:
        study: the study options

    Raises:
        StudyFileError: if a measurement lacks what synthesis needs, or observes a field the
            solver does not have
    """
    solver = build_forward_solver(study.forward_solver)
    measurements = _build_measurements(study, solver.state_spec)

    match study.synthesis.truth:
        case RunTruthOptions():
            _generate_from_run(solver, measurements)
        case _:
            assert_never(study.synthesis.truth)

    for measurement in measurements:
        write_numpy(
            measurement.options.data.path, measurement.times, np.array(measurement.values)
        )
        logger.info(
            "wrote %s: %d times to %s",
            measurement.name,
            len(measurement.times),
            measurement.options.data.path,
        )


def _generate_from_run(solver: ForwardSolver, measurements: list[_Measurement]) -> None:
    """Generate the data by running the solver with its nominal parameters.

    Fills the ``values`` of each measurement.

    Args:
        solver: the forward solver
        measurements: the measurements to generate

    Raises:
        StudyFileError: if a measurement time is before the solver's start time
    """
    # all (time, measurements, index) events, in the order they occur
    events = sorted(
        ((float(t), m, k) for m in measurements for k, t in enumerate(m.times)),
        key=lambda event: event[0],
    )

    t, state = solver.initial_state()
    for t_k, measurement, k in events:
        if t_k < t - _tol(t):
            raise StudyFileError(
                f"measurements.{measurement.name}: time {t_k} is before the start {t}"
            )
        elif t_k > t + _tol(t):
            state = solver.propagate(t, t_k, state, {})
            t = t_k
        else:
            # same time (up to round-off): sample the current state, no solver call
            pass

        fields = {name: state[name] for name in measurement.options.fields}
        y = measurement.sampler.sample(fields)
        z = measurement.model.predict(y)
        noise = np.random.default_rng([measurement.seed, k]).normal(
            0.0, measurement.options.noise.stddev, z.shape
        )
        measurement.values.append(z + noise)


def _build_measurements(
    study: StudyOptions, state_spec: Mapping[str, FieldSpec]
) -> list[_Measurement]:
    """Check the measurements for synthesis and build their components.

    Args:
        study: the study options
        state_spec: the solver's state fields

    Returns:
        time series for each configured measurement

    Raises:
        StudyFileError: if there are no measurements, one lacks ``times`` or a noise ``seed``,
            or observes a field the solver does not have
    """
    if not study.measurements:
        raise StudyFileError("synthesis needs a measurements section")

    measurements = []
    for name, options in study.measurements.items():
        if options.times is None:
            raise StudyFileError(f"measurements.{name}: synthesis needs 'times'")
        if options.noise.seed is None:
            raise StudyFileError(f"measurements.{name}: synthesis needs a noise 'seed'")
        unknown = [f for f in options.fields if f not in state_spec]
        if unknown:
            raise StudyFileError(
                f"measurements.{name}: the forward solver has no field {', '.join(unknown)}. "
                f"Available: {', '.join(state_spec)}"
            )

        measurements.append(
            _Measurement(
                name=name,
                options=options,
                seed=options.noise.seed,
                times=measurement_times(options.times),
                sampler=build_sampler(options.spatial_sampling, options.fields),
                model=build_model(options.model),
            )
        )

    return measurements


def _tol(t: float) -> float:
    """Return the tolerance below which two times are the same.

    Args:
        t: time

    Returns:
        tolerance
    """
    return 1e-12 * max(1.0, abs(t))
