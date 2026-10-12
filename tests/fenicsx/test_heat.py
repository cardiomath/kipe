"""The dolfinx heat equation example: exact solution, statelessness, parameters, interface."""

import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

import numpy as np
from dolfinx import fem

from kipe.fenicsx.examples.heat import Solver
from kipe.forward_solver import ForwardSolver


def _max(values: np.ndarray) -> float:
    """Maximum over all ranks; -inf for a rank without entries."""
    local = float(np.max(values)) if values.size else -np.inf
    return MPI.COMM_WORLD.allreduce(local, op=MPI.MAX)


def _exact(solver: Solver, t: float, kappa: float, a: float) -> np.ndarray:
    """Owned entries of the exact solution at time t."""
    rate = 2 * np.pi**2 * kappa
    amplitude = a / rate * (1 - np.exp(-rate * t))
    u = fem.Function(solver.function_space("u"))
    u.interpolate(lambda x: amplitude * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    return u.x.array[: solver.state_spec["u"].local_size].copy()


def test_conforms_to_protocol():
    assert isinstance(Solver(n=4), ForwardSolver)


def test_matches_exact_solution():
    """Backward Euler and P1: first order in time, second order in space."""
    solver = Solver(n=32, dt=0.002)
    t0, state = solver.initial_state()
    state = solver.propagate(t0, 0.5, state, {})
    exact = _exact(solver, 0.5, kappa=0.1, a=1.0)

    error = _max(np.abs(state["u"] - exact)) / _max(np.abs(exact))
    assert error < 0.01


def test_calls_are_stateless():
    """Stateless: two calls give the same as one."""
    solver = Solver(n=8)
    t0, state = solver.initial_state()
    once = solver.propagate(t0, 0.2, state, {})["u"]
    half = solver.propagate(t0, 0.1, state, {})
    twice = solver.propagate(0.1, 0.2, half, {})["u"]
    assert _max(np.abs(once - twice)) < 1e-12 * max(_max(np.abs(once)), 1.0)


def test_parameters_override_nominal_values():
    """The solution is linear in the source amplitude a."""
    solver = Solver(n=8)
    t0, state = solver.initial_state()
    nominal = solver.propagate(t0, 0.1, state, {})["u"]
    doubled = solver.propagate(t0, 0.1, state, {"a": 2.0})["u"]
    assert _max(np.abs(doubled - 2 * nominal)) < 1e-12 * _max(np.abs(nominal))
    assert solver.nominal_parameters() == {"kappa": 0.1, "a": 1.0}


def test_rejects_unreachable_times():
    solver = Solver(n=4, dt=0.01)
    t0, state = solver.initial_state()
    with pytest.raises(ValueError, match="cannot reach"):
        solver.propagate(t0, 0.015, state, {})


def test_state_is_owned_dofs():
    solver = Solver(n=4)
    spec = solver.state_spec["u"]
    assert spec.kind == "distributed"
    assert spec.local_size == solver.function_space("u").dofmap.index_map.size_local
    with pytest.raises(KeyError):
        solver.function_space("v")
