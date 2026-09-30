r"""FitzHugh-Nagumo example ODE, solved with RK4.

.. math::

    \dot{v} &= c\left(v - \frac{v^3}{3} + w\right) \\
    \dot{w} &= -\frac{1}{c}(v - a + bw)

This is a common benchmark problem [Ram+07]_ with the following properties:

- Attracting limit cycle: small errors are attenuated (in contrast to chaotic Lorenz63 problem).
- Nonlinearity
- Signed states (possible problem with Lotka-Volterra)

.. [Ram+07] Ramsay, J.O., Hooker, G., Campbell, D. and Cao, J. (2007), Parameter estimation
   for differential equations: a generalized smoothing approach. Journal of the Royal
   Statistical Society: Series B (Statistical Methodology), 69: 741-796.
   https://doi.org/10.1111/j.1467-9868.2007.00610.x
"""

import logging

import numpy as np
import numpy.typing as npt

type NDArray64 = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class Solver:
    r"""Stateless FitzHugh-Nagumo solver.

    Args:
        parameters: nominal model parameters :math:`\theta = (a, b, c)`
        initial_state: initial state :math:`(v_0, w_0)`
        dt: time step size
    """

    def __init__(
        self,
        parameters: tuple[float, float, float],
        initial_state: tuple[float, float],
        dt: float,
    ) -> None:
        self._parameters = parameters
        self._initial_state = np.array(initial_state)
        self._dt = dt

    @staticmethod
    def _rhs(x: NDArray64, parameters: tuple[float, float, float]) -> NDArray64:
        """Evaluate the right-hand side of the FHN equations.

        Args:
            x: state
            parameters: model parameters

        Returns:
            rhs
        """
        a, b, c = parameters
        v = x[0]
        w = x[1]
        return np.array((
            c * (v - v**3 / 3 + w),
            -1 / c * (v - a + b * w),
        ))

    def timestep(
        self, t: float, state: NDArray64, parameters: tuple[float, float, float]
    ) -> NDArray64:
        """Advance one RK4 step from ``(t, state)``.

        Does not modify the solver.

        Args:
            t: time of the given state
            state: state at time ``t``
            parameters: model parameters used for this step

        Returns:
            state at time ``t + dt``
        """
        dt = self._dt

        k1 = self._rhs(state, parameters)
        k2 = self._rhs(state + dt / 2 * k1, parameters)
        k3 = self._rhs(state + dt / 2 * k2, parameters)
        k4 = self._rhs(state + dt * k3, parameters)
        state_new = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        logger.debug("t = %g: x = %s", t + dt, state_new)

        return state_new

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
            raise ValueError(f"T = {T} is not a multiple of the time step dt = {dt}")
        times = np.linspace(0, n * dt, n + 1)

        states = np.zeros((len(times), len(self._initial_state)))
        states[0] = self._initial_state[:]

        for i, t in enumerate(times[:-1]):
            states[i + 1] = self.timestep(t, states[i], self._parameters)

        return times, states


if __name__ == "__main__":
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
