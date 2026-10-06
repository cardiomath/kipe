r"""One ROUKF step equals the Kalman update for a linear observation.

The unscented transform is exact for linear observations, so a ROUKF assimilation step must
reproduce the Kalman filter's posterior mean and covariance. This holds for every stencil with
an invertible :math:`P_\alpha` thanks to the consistent initialization :math:`U^0 =
P_\alpha`; with a state that depends linearly on the parameters, the corrected state follows
the corrected parameters exactly. The derivation uses the zero mean of the stencil, the
Cholesky factor :math:`C C^T = P_\alpha^{-1}` and the linearity of the observation.
"""

from mpi4py import MPI

import numpy as np
import pytest

from kipe import roukf

KINDS = ["simplex", "canonical", "star"]  # unique has no spread, hence no update


def _assert_close(actual, expected, tol=1e-11):
    """Equal up to round-off: entry errors relative to the largest entry of ``expected``.

    The errors are round-off, amplified by the conditioning of the problem; measured up to
    5e-13. A per-entry relative tolerance would be too strict for small entries.
    """
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol * np.abs(expected).max())


def _roukf_step(kind, x0, theta0, stddev, propagate, observe, z, noise):
    """One ROUKF assimilation step, stage by stage.

    Args:
        kind: kind of sigma-point stencil
        x0: initial state, shape ``(n,)``; its uncertainty is zero (reduced order)
        theta0: prior mean of the parameters, shape ``(p,)``
        stddev: prior standard deviations of the parameters, shape ``(p,)``
        propagate: the model ``propagate(x, theta) -> x``, on the sigma points (one per
            column)
        observe: the observation operator ``observe(x, theta) -> z_hat``, on the sigma points
        z: data, shape ``(m,)``
        noise: noise standard deviation

    Returns:
        - corrected state, shape ``(n,)``
        - corrected parameters, shape ``(p,)``
        - parameter covariance, shape ``(p, p)``
    """
    stencil = roukf.sigma_point_stencil(kind, len(theta0))
    state = roukf.initial_state(x0, theta0, stddev, stencil)

    # 1. sampling
    x_sigma, theta_sigma = roukf.sample(state, stencil)

    # 2. propagation
    x_sigma = propagate(x_sigma, theta_sigma)

    # 3. innovations, weighted with the noise
    Gamma = (z[:, np.newaxis] - observe(x_sigma, theta_sigma)) / noise

    # 4. correction
    state = roukf.update(x_sigma, theta_sigma, Gamma, stencil, MPI.COMM_WORLD)

    return state.x, state.theta, state.covariance()


def _kalman_update(theta0, P0, H, z, noise):
    """The Kalman update of a Gaussian prior for a linear observation, in the gain form.

    The unknown (the "state" of the Kalman filter) is theta, with trivial dynamics.

    Args:
        theta0: prior mean, shape ``(p,)``
        P0: prior covariance, shape ``(p, p)``
        H: observation operator, shape ``(m, p)``
        z: data, shape ``(m,)``
        noise: noise standard deviation

    Returns:
        - posterior mean, shape ``(p,)``
        - posterior covariance, shape ``(p, p)``
    """
    S = H @ P0 @ H.T + noise**2 * np.eye(len(z))  # innovation covariance
    K = P0 @ H.T @ np.linalg.inv(S)  # Kalman gain

    theta = theta0 + K @ (z - H @ theta0)
    P = (np.eye(len(theta0)) - K @ H) @ P0
    return theta, P


@pytest.mark.parametrize("kind", KINDS)
def test_one_step_equals_kalman_update_for_linear_observation(kind):
    """For z = H theta + noise, one ROUKF step is exactly the Kalman update.

    The unscented transform is exact for linear observations, so the ROUKF must reproduce
    the posterior mean and covariance of the Kalman filter. Parameters only, no state.
    """
    rng = np.random.default_rng(1)
    p, m, noise = 3, 4, 0.1  # number of parameters, measured values; noise stddev
    H = rng.normal(size=(m, p))  # observation operator
    theta0 = rng.normal(size=p)  # prior mean
    stddev = np.array([0.5, 1.0, 2.0])  # prior standard deviations
    z = H @ rng.normal(size=p)  # data

    _, theta, P = _roukf_step(
        kind,
        x0=np.zeros(0),
        theta0=theta0,
        stddev=stddev,
        propagate=lambda x, theta: x,
        observe=lambda x, theta: H @ theta,
        z=z,
        noise=noise,
    )
    theta_kalman, P_kalman = _kalman_update(theta0, np.diag(stddev**2), H, z, noise)

    _assert_close(theta, theta_kalman)
    _assert_close(P, P_kalman)


@pytest.mark.parametrize("kind", KINDS)
def test_state_follows_corrected_parameters(kind):
    """For a state x = A theta observed as z = H x + noise, the corrected state is A theta+.

    The state has no uncertainty of its own (reduced order); it is corrected through its
    sensitivity to the parameters. For a linear model this is exact: x+ = A theta+, and
    theta+ is the Kalman update for the observation operator H A.
    """
    rng = np.random.default_rng(2)
    n, p, m, noise = 5, 3, 4, 0.1  # state size, parameters, measured values; noise stddev
    A = rng.normal(size=(n, p))  # model: state from parameters
    H = rng.normal(size=(m, n))  # observation operator, acting on the state
    theta0 = rng.normal(size=p)
    stddev = np.array([0.5, 1.0, 2.0])
    z = H @ A @ rng.normal(size=p)

    x, theta, _ = _roukf_step(
        kind,
        x0=A @ theta0,
        theta0=theta0,
        stddev=stddev,
        propagate=lambda x, theta: A @ theta,
        observe=lambda x, theta: H @ x,
        z=z,
        noise=noise,
    )
    theta_kalman, _ = _kalman_update(theta0, np.diag(stddev**2), H @ A, z, noise)

    _assert_close(theta, theta_kalman)
    _assert_close(x, A @ theta)
