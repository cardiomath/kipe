"""Run SciPy acceleration with a fixed budget of map evaluations."""

import logging
from collections.abc import Callable

import numpy as np
from scipy.optimize import anderson

from kipe._types import NDArray_f64
from kipe.options import AccelerationOptions

logger = logging.getLogger(__name__)


class InvalidAccelerationPoint(Exception):
    """The proposed input cannot be represented in the map's parameter domain."""


class _Completed(Exception):
    """The requested number of map evaluations has been completed."""


def run_acceleration(
    evaluate: Callable[[NDArray_f64], NDArray_f64],
    initial: NDArray_f64,
    options: AccelerationOptions,
    evaluations: int,
) -> None:
    """Evaluate a fixed-point map using SciPy to select successive inputs.

    Args:
        evaluate: map returning G(x); raises InvalidAccelerationPoint for inadmissible inputs
        initial: initial input vector
        options: acceleration memory and regularization
        evaluations: number of successful map evaluations, including the initial input

    Raises:
        InvalidAccelerationPoint: if the initial input or the unaccelerated fallback is
            inadmissible
        ValueError: if a map output is nonfinite
    """
    completed = 0
    last_output: NDArray_f64 | None = None

    def residual(x: NDArray_f64) -> NDArray_f64:
        nonlocal completed, last_output
        if not np.all(np.isfinite(x)):
            raise InvalidAccelerationPoint
        output = evaluate(x)
        if not np.all(np.isfinite(output)):
            raise ValueError("nonfinite acceleration map output")
        last_output = output.copy()
        completed += 1
        # Stop inside the evaluation to avoid an extra expensive map call or an unused step.
        if completed == evaluations:
            raise _Completed
        return output - x

    start = initial.copy()
    while completed < evaluations:
        before_restart = completed
        try:
            anderson(
                residual,
                start,
                alpha=1,
                M=options.memory,
                w0=options.regularization,
                iter=evaluations,
                maxiter=evaluations + 1,
                line_search=None,
            )
        except _Completed:
            return
        except InvalidAccelerationPoint:
            if last_output is None or completed == before_restart:
                raise
            logger.warning("invalid acceleration extrapolation; using the unaccelerated output")
        # On an invalid input, restart from the last ordinary output with fresh history.
        # At an exact root SciPy returns early; continue to honor the evaluation budget.
        assert last_output is not None
        start = last_output.copy()
