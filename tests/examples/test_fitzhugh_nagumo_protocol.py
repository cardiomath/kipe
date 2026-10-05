"""Protocol contract tests of the FitzHugh-Nagumo reference implementation.

Written against the forward solver protocol (named-field states, ``parameters`` dicts). This file
is also type-checked by mypy, which verifies structural conformance (``_as_protocol``).
Generic protocol checks (state_spec vs. returned states, returned copies, ...) belong in a
reusable conformance helper, not here.
"""

import numpy as np
import pytest

from kipe.examples.fitzhugh_nagumo import Solver
from kipe.forward_solver import ForwardSolver

DT = 0.05


def _solver() -> Solver:
    return Solver(dt=DT)


def _as_protocol(solver: Solver) -> ForwardSolver:
    """Static conformance check: mypy rejects this if Solver doesn't match the protocol."""
    return solver


def _vector(state: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([state["v"], state["w"]])


@pytest.mark.parametrize("t1", [DT, 0.3, 1.0, 20.0])
def test_propagate_accepts_times_on_grid(t1: float) -> None:
    """Float round-off must not reject valid target times (e.g., 0.3 % 0.05 != 0 in floats)."""
    solver = _solver()
    t0, state = solver.initial_state()
    solver.propagate(t0, t1, state, {})


@pytest.mark.parametrize("t1", [0.33, 0.0, -1.0])
def test_propagate_rejects_unreachable_times(t1: float) -> None:
    solver = _solver()
    t0, state = solver.initial_state()
    with pytest.raises(ValueError):
        solver.propagate(t0, t1, state, {})


def test_propagate_is_restartable() -> None:
    """Propagating in pieces equals one propagation: the returned state is complete."""
    solver = _solver()
    t0, state = solver.initial_state()
    whole = solver.propagate(t0, 20.0, state, {})

    t0, state = solver.initial_state()
    half = solver.propagate(t0, 10.0, state, {})
    pieces = solver.propagate(10.0, 20.0, half, {})

    np.testing.assert_allclose(_vector(pieces), _vector(whole), rtol=0, atol=1e-12)


def test_parameters_overrides_nominal_values() -> None:
    """Empty parameters runs with nominal values; parameters entries override for the call."""
    solver = _solver()
    t0, state = solver.initial_state()
    nominal = solver.propagate(t0, 5.0, state, {})

    t0, state = solver.initial_state()
    perturbed = solver.propagate(t0, 5.0, state, {"c": 2.5})
    assert not np.allclose(_vector(perturbed), _vector(nominal))

    # a perturbed call must not change the solver's nominal values
    assert solver.nominal_parameters() == Solver().nominal_parameters()
