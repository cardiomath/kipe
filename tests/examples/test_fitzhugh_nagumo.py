"""Tests of the FitzHugh-Nagumo example solver against a tight scipy reference solution."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kipe.examples.fitzhugh_nagumo import Solver

PARAMETERS = (0.2, 0.2, 3.0)  # a, b, c (Ramsay et al. 2007)
INITIAL_STATE = (-1.0, 1.0)  # v, w
T = 20.0


def _rhs(t: float, x: np.ndarray, a: float, b: float, c: float) -> list[float]:
    """FitzHugh-Nagumo right-hand side, Ramsay et al. form (independent of the solver's)."""
    v, w = x
    return [c * (v - v**3 / 3 + w), -(v - a + b * w) / c]


@pytest.fixture(scope="module")
def reference():
    """Dense reference solution with tight tolerances."""
    return solve_ivp(
        _rhs,
        (0.0, T),
        INITIAL_STATE,
        args=PARAMETERS,
        rtol=1e-11,
        atol=1e-12,
        dense_output=True,
    ).sol


def _max_error(dt: float, reference) -> float:
    """Maximum error of the RK4 solution over all time steps."""
    times, states = Solver(PARAMETERS, INITIAL_STATE, dt).solve(T)
    assert times[-1] == pytest.approx(T)
    return float(np.abs(states - reference(times).T).max())


@pytest.mark.parametrize(("dt", "tol"), [(0.1, 5e-3), (0.05, 3e-4), (0.02, 1e-5), (0.01, 1e-6)])
def test_matches_reference(dt: float, tol: float, reference) -> None:
    """RK4 trajectory agrees with the scipy reference at every time step."""
    assert _max_error(dt, reference) < tol


def test_fourth_order_convergence(reference) -> None:
    """Halving dt reduces the error by about 2**4 = 16."""
    ratio = _max_error(0.1, reference) / _max_error(0.05, reference)
    assert 10 < ratio < 25
