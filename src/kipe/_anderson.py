"""Limited-memory Anderson mixing for an externally evaluated fixed-point map."""

import numpy as np

from kipe._types import NDArray_f64
from kipe.options import AndersonOptions


class Anderson:
    """Mix previous inputs and corrections without additional forward evaluations.

    Args:
        options: history length and regularization of the residual least-squares problem
    """

    def __init__(self, options: AndersonOptions) -> None:
        self._options = options
        self._inputs: list[NDArray_f64] = []
        self._residuals: list[NDArray_f64] = []

    def reset(self) -> None:
        """Discard the history after an inadmissible extrapolation."""
        self._inputs.clear()
        self._residuals.clear()

    def update(self, x: NDArray_f64, g: NDArray_f64) -> NDArray_f64:
        """Return the next input from the current input and its unaccelerated output.

        Args:
            x: fixed-point input in fixed, scaled parameter coordinates
            g: output of one complete outer iteration, in the same coordinates

        Returns:
            the accelerated next input; the plain output on the first call or numerical failure
        """
        self._inputs.append(x.copy())
        self._residuals.append((g - x).copy())
        keep = self._options.memory + 1
        self._inputs = self._inputs[-keep:]
        self._residuals = self._residuals[-keep:]
        if len(self._inputs) == 1:
            return g.copy()

        dx = np.diff(np.asarray(self._inputs), axis=0).T
        df = np.diff(np.asarray(self._residuals), axis=0).T
        # Extended Anderson with alpha=1, as in the reference notebook. Solve the
        # augmented least-squares system instead of forming its normal equations.
        penalty = self._options.regularization * np.linalg.norm(df, axis=0)
        matrix = np.vstack((df, np.diag(penalty)))
        rhs = np.concatenate((self._residuals[-1], np.zeros(df.shape[1])))
        try:
            weights = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
            candidate = g - (dx + df) @ weights
        except np.linalg.LinAlgError:
            self.reset()
            return g.copy()
        if not np.all(np.isfinite(candidate)):
            self.reset()
            return g.copy()
        return candidate
