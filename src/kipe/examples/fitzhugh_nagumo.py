r"""FitzHugh-Nagumo example ODE, solved with RK4, with kipe interface.

.. math::

    \dot{v} &= c\left(v - \frac{v^3}{3} + w\right) \\
    \dot{w} &= -\frac{1}{c}(v - a + bw)

This is a common benchmark problem [Ram+07]_ with the following properties:

- Attracting limit cycle: small errors are attenuated (in contrast to chaotic Lorenz63 problem).
- Nonlinearity
- Signed states (possible problem with Lotka-Volterra)

This module is fully independent of kipe.

.. [Ram+07] Ramsay, J.O., Hooker, G., Campbell, D. and Cao, J. (2007), Parameter estimation
   for differential equations: a generalized smoothing approach. Journal of the Royal
   Statistical Society: Series B (Statistical Methodology), 69: 741-796.
   https://doi.org/10.1111/j.1467-9868.2007.00610.x
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

type NDArray64 = npt.NDArray[np.float64]
type State = dict[str, NDArray64]
type Parameters = dict[str, float]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldSpec:
    """Layout of one state field on the current MPI rank.

    Required by kipe interface.
    """

    local_size: int
    """Number of entries of the field held by this rank: the owned entries for a distributed
    field, the full size for a replicated one."""

    kind: Literal["distributed", "replicated"] = "replicated"
    """``distributed``: each entry is owned by exactly one rank (e.g. the owned dofs of a finite
    element function).
    ``replicated``: every rank holds the full, identical field (e.g. the state of FHN or a small
    lumped-parameter model)."""


class Solver:
    r"""Stateless FitzHugh-Nagumo solver with inverse interface.

    Args:
        parameters: nominal model parameters :math:`(a, b, c)`
        initial_state: initial state :math:`(v_0, w_0)`
        dt: time step size
    """

    def __init__(
        self,
        parameters: tuple[float, float, float],
        initial_state: tuple[float, float],
        dt: float,
    ) -> None:
        self._initial_state = np.array(initial_state)
        self._dt = dt

        self._nominal = dict(zip("abc", parameters, strict=True))

    @property
    def state_spec(self) -> dict[str, FieldSpec]:
        """Names and layout of the state fields.

        The state consists of two scalar fields, ``v`` and ``w``.

        Required by kipe interface.
        """
        return {
            "v": FieldSpec(local_size=1),
            "w": FieldSpec(local_size=1),
        }

    def nominal_parameters(self) -> Parameters:
        """Return the estimable parameters and their nominal values.

        Required by kipe interface.

        Returns:
            parameter name -> nominal physical value dict
        """
        return dict(self._nominal)

    def initial_state(self) -> tuple[float, State]:
        """Return the start time and the initial state.

        Returns:
            - start time ``t0``
            - initial state, valid until the next call
        """
        return (0.0, self._to_state(self._initial_state.copy()))

    @staticmethod
    def _rhs(x: NDArray64, parameters: Parameters) -> NDArray64:
        """Evaluate the right-hand side of the FHN equations.

        Args:
            x: state
            parameters: model parameters

        Returns:
            rhs
        """
        a, b, c = parameters["a"], parameters["b"], parameters["c"]
        v = x[0]
        w = x[1]
        return np.array((
            c * (v - v**3 / 3 + w),
            -1 / c * (v - a + b * w),
        ))

    @staticmethod
    def _to_state(x: NDArray64) -> State:
        """Convert the state vector to named mapping representation.

        Args:
            x: state vector (concatenation of field arrays)

        Returns:
            named state mapping
        """
        return {"v": x[0:1], "w": x[1:2]}

    @staticmethod
    def _to_vector(state: State) -> NDArray64:
        """Convert the name state mapping to vector (concatenation of field arrays).

        Args:
            state: named state mapping

        Returns:
            state vector (concatenation of field arrays)
        """
        return np.concatenate([state["v"], state["w"]])

    def _step(self, t0: float, x: NDArray64, parameters: Parameters) -> NDArray64:
        """Perform one RK4 step from ``t0`` with state array.

        Args:
            t0: time of the given state
            x: state array at time ``t0``
            parameters: model parameters used for this step

        Returns:
            - state array at ``t1``
        """
        dt = self._dt
        k1 = self._rhs(x, parameters)
        k2 = self._rhs(x + dt / 2 * k1, parameters)
        k3 = self._rhs(x + dt / 2 * k2, parameters)
        k4 = self._rhs(x + dt * k3, parameters)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def timestep(self, t0: float, state: State, parameters: Parameters) -> tuple[float, State]:
        """Perform one RK4 step from ``(t0, state)``.

        Required by kipe interface.

        Args:
            t0: time of the given state
            state: state at time ``t0``
            parameters: model parameters used for this step

        Returns:
            - time ``t1 = t0 + dt``
            - state at ``t1``
        """
        dt = self._dt

        parameters = self.nominal_parameters() | parameters
        x = self._step(t0, self._to_vector(state), parameters)

        t1 = t0 + dt
        logger.debug("t = %g: x = %s", t1, x)

        return t1, self._to_state(x)

    def propagate(self, t0: float, t1: float, state: State, parameters: Parameters) -> State:
        """Propagate state from ``(t0, state)`` to ``t1``.

        Required by kipe interface.

        Args:
            t0: time of the given state
            t1: target time
            state: state at time ``t0``
            parameters: model parameters for propagation

        Returns:
            state at time ``t1``

        Raises:
            ValueError: if ``t1`` cannot be reached exactly
        """
        dt = self._dt
        n = round((t1 - t0) / dt)
        if n < 1 or not np.isclose(t0 + n * dt, t1, rtol=0.0, atol=1e-12 * max(1.0, abs(t1))):
            raise ValueError(
                f"Target time {t1} unreachable from {t0} with dt = {self._dt}. "
                "t1 - t0 must be a multiple of dt."
            )

        logger.debug("Propagating model from %g -> %g, %d", t0, t1, n)

        ti = t0
        state_i = state

        for _ in range(n):
            ti, state_i = self.timestep(ti, state_i, parameters)

        logger.debug("\tx = %s", state_i)

        return state_i

    def solve(self, T: float) -> tuple[NDArray64, NDArray64]:
        """Forward solve of FHN with RK4, using the nominal parameters.

        Args:
            T: end time, must be a multiple of the time step

        Returns:
            - times at which the solution is computed, shape ``(n + 1,)``
            - solution array, shape ``(n + 1, 2)``

        Raises:
            ValueError: if ``T`` is not a multiple of the time step
        """
        dt = self._dt
        n = round(T / dt)
        if not np.isclose(n * dt, T, rtol=0.0, atol=1e-12 * max(1.0, abs(T))):
            raise ValueError(f"{T = } is not a multiple of the time step {dt = }")
        times = np.linspace(0, n * dt, n + 1)

        states = np.zeros((len(times), len(self._initial_state)))
        states[0] = self._initial_state[:]

        for i, t in enumerate(times[:-1]):
            states[i + 1] = self._step(t, states[i], self._nominal)

        return times, states


def main() -> None:
    """Perform one forward solve with setup from [Ram+07]_ and plot result."""
    p = (0.2, 0.2, 3.0)
    x0 = (-1.0, 1.0)
    dt = 0.05
    T = 20.0

    sol = Solver(p, x0, dt)
    t, x = sol.solve(T)

    try:
        import matplotlib.pyplot as plt

        plt.plot(t, x)
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel(r"$v$, $w$")
        plt.legend((r"$v$", r"$w$"))
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
