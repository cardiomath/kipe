r"""FitzHugh-Nagumo model, solved with RK4, as a minimal kipe forward solver.

.. math::

    \dot{v} &= c\left(v - \frac{v^3}{3} + w\right) \\
    \dot{w} &= -\frac{1}{c}(v - a + bw)

This is a common benchmark problem [Ram+07]_ with the following properties:

- Attracting limit cycle: small errors are attenuated (in contrast to chaotic Lorenz63 problem).
- Nonlinearity
- Signed states (possible problem with Lotka-Volterra)

The public members of :class:`Solver`, together with :class:`FieldSpec`, are the complete
kipe interface; the private methods are the model. This module is fully independent of kipe.

.. [Ram+07] Ramsay, J.O., Hooker, G., Campbell, D. and Cao, J. (2007), Parameter estimation
   for differential equations: a generalized smoothing approach. Journal of the Royal
   Statistical Society: Series B (Statistical Methodology), 69: 741-796.
   https://doi.org/10.1111/j.1467-9868.2007.00610.x
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

type NDArray_f64 = npt.NDArray[np.float64]
type State = dict[str, NDArray_f64]
type Parameters = dict[str, float]


@dataclass(frozen=True)
class FieldSpec:
    """Layout of one state field on the current MPI rank."""

    local_size: int
    """Number of entries of the field held by this rank: the owned entries for a distributed
    field, the full size for a replicated one."""

    kind: Literal["distributed", "replicated"] = "replicated"
    """``distributed``: each entry is owned by exactly one rank (e.g., the owned dofs of a finite
    element function).
    ``replicated``: every rank holds the full, identical field (e.g., the state of FHN or a small
    lumped-parameter model)."""


class Solver:
    """FitzHugh-Nagumo solver with a fixed RK4 time step.

    The defaults are the setup of [Ram+07]_.

    Args:
        dt: time step size
        v0: initial value of :math:`v`
        w0: initial value of :math:`w`
        a: nominal value of :math:`a`
        b: nominal value of :math:`b`
        c: nominal value of :math:`c`
    """

    def __init__(
        self,
        dt: float = 0.05,
        v0: float = -1.0,
        w0: float = 1.0,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 3.0,
    ) -> None:
        self._dt = dt
        self._x0 = (v0, w0)
        self._nominal = {"a": a, "b": b, "c": c}

    # kipe interface

    @property
    def state_spec(self) -> dict[str, FieldSpec]:
        """Names and layout of the state fields: two scalars, ``v`` and ``w``."""
        return {"v": FieldSpec(local_size=1), "w": FieldSpec(local_size=1)}

    def nominal_parameters(self) -> Parameters:
        """Return the estimable parameters and their nominal values.

        Returns:
            parameter name -> nominal value dict (a new dict)
        """
        return dict(self._nominal)

    def initial_state(self) -> tuple[float, State]:
        """Return the start time and the initial state.

        Returns:
            - start time ``t0 = 0``
            - initial state :math:`(v_0, w_0)`
        """
        v0, w0 = self._x0
        return 0.0, {"v": np.array([v0]), "w": np.array([w0])}

    def timestep(self, t0: float, state: State, parameters: Parameters) -> tuple[float, State]:
        """Perform one RK4 step from ``(t0, state)``.

        Args:
            t0: time of ``state``
            state: state at ``t0``
            parameters: parameter values for this step, overriding the nominal values

        Returns:
            - time ``t1 = t0 + dt``
            - state at ``t1``
        """
        x = np.array([state["v"][0], state["w"][0]])
        x = self._rk4_step(x, self._nominal | parameters)
        return t0 + self._dt, {"v": x[0:1], "w": x[1:2]}

    def propagate(self, t0: float, t1: float, state: State, parameters: Parameters) -> State:
        """Propagate the state from ``t0`` to ``t1`` with RK4 steps.

        Args:
            t0: time of ``state``
            t1: target time
            state: state at ``t0``
            parameters: parameter values for this call, overriding the nominal values

        Returns:
            state at ``t1``

        Raises:
            ValueError: if ``t1 - t0`` is not a positive multiple of ``dt``
        """
        n = round((t1 - t0) / self._dt)
        if n < 1 or not np.isclose(t0 + n * self._dt, t1, rtol=0, atol=1e-12 * max(1, abs(t1))):
            raise ValueError(f"cannot reach t1 = {t1} from t0 = {t0} with dt = {self._dt}")
        for _ in range(n):
            t0, state = self.timestep(t0, state, parameters)
        return state

    # model

    @staticmethod
    def _rhs(x: NDArray_f64, a: float, b: float, c: float) -> NDArray_f64:
        r"""Evaluate the right-hand side of the FitzHugh-Nagumo equations.

        Args:
            x: state vector :math:`(v, w)`
            a: parameter :math:`a`
            b: parameter :math:`b`
            c: parameter :math:`c`

        Returns:
            time derivative :math:`(\dot{v}, \dot{w})`
        """
        v, w = x
        return np.array([c * (v - v**3 / 3 + w), -(v - a + b * w) / c])

    def _rk4_step(self, x: NDArray_f64, parameters: Parameters) -> NDArray_f64:
        """Perform one classical RK4 step of size ``dt``.

        Args:
            x: state vector :math:`(v, w)`
            parameters: values of ``a``, ``b`` and ``c``

        Returns:
            new state vector (the input is not modified)
        """
        dt = self._dt
        k1 = self._rhs(x, **parameters)
        k2 = self._rhs(x + dt / 2 * k1, **parameters)
        k3 = self._rhs(x + dt / 2 * k2, **parameters)
        k4 = self._rhs(x + dt * k3, **parameters)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
