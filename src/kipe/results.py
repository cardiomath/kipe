r"""Results of a study, read back for analysis: the estimates of its noise realizations.

The estimation of each noise realization writes its history (:mod:`kipe.estimation`).
:func:`read_realizations` collects the final estimate of each realization, the filter's
covariance of it, and the estimate at the end of each outer iteration: for the statistics of
the estimates over the noise, e.g., their spread compared with the filter's uncertainty.

The covariances are in physical units. The filter's :math:`\bs{\theta}` of the last outer
iteration is relative to its anchor, the estimate of the iteration before, which differs from
realization to realization; for ``multiplicative`` parameters, even the scale of
:math:`\bs{\theta}` does. The physical covariance, :math:`\bs{J} \bs{P}_\theta \bs{J}` with
:math:`\bs{J} = \operatorname{diag}(\mathrm{d}\phi_i / \mathrm{d}\theta_i)` at the final estimate
(:meth:`kipe.parameters.Parameterization.derivative`), is comparable across realizations, to
first order.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kipe._history import Step, count_iterations, read_steps
from kipe._types import NDArray_f64, NDArray_i64
from kipe.options import StudyFileError, StudyOptions, estimation_directory
from kipe.parameters import build_parameterization

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Realizations:
    """The final estimates of a study's noise realizations, from :func:`read_realizations`."""

    names: list[str]
    """Names of the estimated parameters."""

    realizations: NDArray_i64
    """Numbers of the realizations read, shape ``(R,)``."""

    parameters: NDArray_f64
    """The final estimate of each realization, as physical values, shape ``(R, p)``."""

    covariance: NDArray_f64
    """The filter's covariance of each final estimate, in physical units, shape
    ``(R, p, p)``."""

    iterations: NDArray_f64
    """The estimate at the end of each outer iteration, as physical values, shape
    ``(R, iterations, p)``."""


def read_realizations(study: StudyOptions) -> Realizations:
    """Read the final estimates of a study's noise realizations.

    A realization whose estimation has not completed its ``estimation.iterations`` outer
    iterations, or has not run, is skipped with a warning, so that the results of a job array
    can be read while it runs.

    Args:
        study: the study options

    Returns:
        the estimates of the realizations read

    Raises:
        StudyFileError: if the study has no parameters section, a history estimates other
            parameters, or no realization has completed its estimation
    """
    if study.parameters is None:
        raise StudyFileError("reading the estimates needs a parameters section")

    names = list(study.parameters.priors)
    parameterization = build_parameterization(study.parameters, names)
    n_iterations = study.estimation.iterations

    read, parameters, covariance, iterations = [], [], [], []
    for r in range(study.synthesis.realizations):
        directory = estimation_directory(study, r)
        steps = read_steps(directory, names)
        completed = count_iterations(steps)
        if completed < n_iterations:
            logger.warning(
                "%s: %d of %d outer iterations, skipped", directory, completed, n_iterations
            )
            continue

        finals = _last_of_each_iteration(steps)[:n_iterations]
        last = finals[-1]
        # the last iteration's theta is relative to its anchor: the estimate before it
        anchors = parameterization
        if n_iterations > 1:
            anchors = parameterization.with_anchor(
                dict(zip(names, finals[-2].parameters.tolist(), strict=True))
            )
        J = anchors.derivative(last.theta)

        read.append(r)
        parameters.append(last.parameters)
        covariance.append(J[:, np.newaxis] * last.P_theta * J[np.newaxis, :])
        iterations.append([step.parameters for step in finals])

    if not read:
        raise StudyFileError(
            f"no realization of {study.output.path} has completed its {n_iterations} outer "
            "iterations"
        )

    return Realizations(
        names=names,
        realizations=np.array(read, dtype=np.int64),
        parameters=np.array(parameters),
        covariance=np.array(covariance),
        iterations=np.array(iterations),
    )


def _last_of_each_iteration(steps: Sequence[Step]) -> list[Step]:
    """Return the last step of each outer iteration.

    Args:
        steps: the steps, of complete iterations, in order

    Returns:
        the last step of each iteration, in order
    """
    last: dict[int, Step] = {}
    for step in steps:
        last[step.iteration] = step

    return list(last.values())
