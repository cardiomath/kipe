"""The ROUKF linear algebra, without a forward solver."""

from mpi4py import MPI

import numpy as np
import pytest

from kipe import roukf

KINDS = ["simplex", "canonical"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("p", [1, 2, 5])
def test_stencil_has_zero_mean(kind, p):
    stencil = roukf.sigma_point_stencil(kind, p)
    assert stencil.alpha * stencil.points.shape[1] == pytest.approx(1.0)
    np.testing.assert_allclose(roukf.mean(stencil.points, stencil), 0.0, atol=1e-14)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("p", [1, 2, 5])
def test_stencil_has_unit_covariance(kind, p):
    """P_alpha = I: the sigma points represent the covariance they are sampled with."""
    stencil = roukf.sigma_point_stencil(kind, p)
    np.testing.assert_allclose(stencil.p_alpha(), np.eye(p), atol=1e-14)


@pytest.mark.parametrize(("kind", "r"), [("simplex", 4), ("canonical", 6), ("unique", 1)])
def test_stencil_size(kind, r):
    assert roukf.sigma_point_stencil(kind, 3).points.shape == (3, r)


@pytest.mark.parametrize("kind", ["simplex", "canonical"])
def test_sampling_reproduces_estimate_and_sensitivity(kind):
    """Mean and sensitivity of freshly sampled sigma points are the estimate and its factor."""
    rng = np.random.default_rng(0)
    stencil = roukf.sigma_point_stencil(kind, 3)
    x, theta = rng.normal(size=5), rng.normal(size=3)
    L_x, L_theta = rng.normal(size=(5, 3)), rng.normal(size=(3, 3))
    U_inv = np.linalg.inv(stencil.p_alpha())  # = I for these stencils
    state = roukf.FilterState(x, theta, L_x, L_theta, U_inv)

    X, Theta = roukf.sample(state, stencil)
    np.testing.assert_allclose(roukf.mean(X, stencil), x, atol=1e-14)
    np.testing.assert_allclose(roukf.mean(Theta, stencil), theta, atol=1e-14)
    np.testing.assert_allclose(roukf.sensitivity(X, stencil), L_x, atol=1e-14)
    np.testing.assert_allclose(roukf.sensitivity(Theta, stencil), L_theta, atol=1e-14)


def test_initial_state():
    stddev = np.array([0.5, 1.0, 2.0])
    stencil = roukf.sigma_point_stencil("simplex", 3)
    state = roukf.initial_state(np.arange(4.0), np.ones(3), stddev, stencil)
    np.testing.assert_array_equal(state.L_x, np.zeros((4, 3)))  # reduced order
    np.testing.assert_allclose(state.covariance(), np.diag(stddev**2), atol=1e-14)


def test_unique_does_not_correct():
    """unique: one sigma point at the estimate, no spread, no correction."""
    stencil = roukf.sigma_point_stencil("unique", 2)
    x, theta = np.arange(3.0), np.array([1.0, 2.0])
    state = roukf.initial_state(x, theta, np.ones(2), stencil)
    X, Theta = roukf.sample(state, stencil)
    np.testing.assert_array_equal(X[:, 0], x)
    np.testing.assert_array_equal(Theta[:, 0], theta)
    _, M = roukf.gain(np.ones((4, 1)), stencil, MPI.COMM_WORLD)
    np.testing.assert_array_equal(M, 0.0)


def test_gain_without_contribution():
    """A rank that does not contribute adds nothing; on one rank: no information."""
    stencil = roukf.sigma_point_stencil("simplex", 2)
    U_inv, M = roukf.gain(np.ones((4, 3)), stencil, MPI.COMM_WORLD, contributes=False)
    np.testing.assert_allclose(U_inv, np.linalg.inv(stencil.p_alpha()))
    np.testing.assert_array_equal(M, 0.0)
