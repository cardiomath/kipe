r"""The reduced-order unscented Kalman filter (ROUKF): its linear algebra.

This module is the filter itself, on plain arrays: it knows no forward solver, no measurements
and no files, so it can be tested on its own. Running it over a study (propagating the sigma
points with the solver, computing the innovations, outer iterations, output) is done in
:mod:`kipe.estimation`, which calls :func:`initial_state`, :func:`sample` and :func:`update`.

The method of [MC11]_, in the formulation and notation of [NAB26]_. The filter estimates the
parameters :math:`\theta \in \mathbb{R}^p`; the model state :math:`\chi` (``x`` in the
code; :math:`n` entries on this MPI rank) is corrected along, but has no uncertainty of its
own at the start (reduced order: :math:`L_\chi^0 = 0`). The uncertainty is carried by the
factors :math:`L_\chi` (:math:`n \times p`), :math:`L_\theta` and :math:`U` (both
:math:`p \times p`), with the parameter covariance

.. math::

   P_\theta = L_\theta U^{-1} L_\theta^T.

The sigma points are the estimates perturbed along a fixed :class:`SigmaPointStencil`
:math:`I_{(*)}`, whose :math:`r` columns :math:`I_{(i)} \in \mathbb{R}^p` all have the weight
:math:`\alpha = 1/r`, with the weighted mean

.. math::

   E_\alpha(Y_{(*)}) = \alpha \sum_{i=1}^r Y_{(i)}

and :math:`Y_{(*)}` the matrix collecting the vectors :math:`Y_{(i)}` column-wise.

The estimate and its uncertainty factors form the :class:`FilterState`. One assimilation step
consists of four stages. Propagation and innovations involve the forward solver and the
measurements and are done in :mod:`kipe.estimation`; sampling and correction are done here.
[NAB26]_ counts the means of the propagated sigma points to the prediction; here they start
the correction, since they need no solver.

1. **Sampling** (:func:`sample`): with :math:`C` the lower Cholesky factor of
   :math:`U^{-1}`, the sigma points are

   .. math::

      \hat{\chi}_{(i)} = \hat{\chi}_+ + L_\chi C I_{(i)}, \qquad
      \hat{\theta}_{(i)} = \hat{\theta}_+ + L_\theta C I_{(i)}.

   Note that in many publications, the formula used :math:`C^T`. This is incorrect for a *lower*
   Cholesky factor :math:`C`.

2. **Propagation:** each sigma point is propagated with the forward model to the next
   measurement time (:mod:`kipe.estimation`), the parameters stay constant.

3. **Innovation:** per sigma point, with the spatial sampler and measurement model
   (:mod:`kipe.estimation`),

   .. math::

      \Gamma_{(i)} = Z - \mathcal{H}(\hat{\chi}_{(i)}).

4. **Correction** (:func:`update`): the a priori estimates (:func:`mean` of the propagated
   sigma points), their :func:`sensitivity`, the :func:`gain` and the corrected (a posteriori)
   estimates,

   .. math::

      \hat{\chi}_- = E_\alpha(\hat{\chi}_{(*)}), \qquad
      \hat{\theta}_- = E_\alpha(\hat{\theta}_{(*)}),

      L_\chi = \alpha \hat{\chi}_{(*)} I_{(*)}^T, \qquad
      L_\theta = \alpha \hat{\theta}_{(*)} I_{(*)}^T, \qquad
      L_\Gamma = \alpha \Gamma_{(*)} I_{(*)}^T,

      U = P_\alpha + L_\Gamma^T W^{-1} L_\Gamma, \qquad
      M = U^{-1} L_\Gamma^T W^{-1} E_\alpha(\Gamma_{(*)}),

      \hat{\chi}_+ = \hat{\chi}_- - L_\chi M, \qquad
      \hat{\theta}_+ = \hat{\theta}_- - L_\theta M.

Here the innovations are passed weighted with the noise, :math:`W^{-1/2} \Gamma_{(i)}`, so
:math:`W` does not appear in the code. States are rank-local vectors: the state fields,
concatenated. All functions are pure; only :func:`gain` communicates (two small sums over
the ranks).

.. [MC11] Moireau, P., & Chapelle, D. (2011). Reduced-order Unscented Kalman Filtering with
   application to parameter identification in large-dimensional systems. ESAIM: Control,
   Optimisation and Calculus of Variations, 17(2), 380-405.

.. [Ber12] Bertoglio, C. (2012). Forward and Inverse Problems in Fluid-Structure Interaction.
   Application to Hemodynamics (PhD thesis). Université Pierre et Marie Curie - Paris VI.

.. [NAB26] Nolte, D., Aróstica, R., & Bertoglio, C. Parameter estimation and identifiability
   in biventricular cardiac fluid-structure interaction from displacement and flow
   measurements. Submitted to Computer Methods in Applied Mechanics and Engineering.
"""

from dataclasses import dataclass
from typing import assert_never

from mpi4py import MPI

import numpy as np

from kipe._types import NDArray_f64
from kipe.options import Particles


@dataclass(frozen=True)
class SigmaPointStencil:
    r"""The stencil :math:`I_{(*)}` along which the sigma points are sampled, and its weight."""

    points: NDArray_f64
    r"""The stencil points :math:`I_{(i)}`, one per column, shape ``(p, r)``."""

    alpha: float
    r"""The weight :math:`\alpha = 1/r` of every stencil point."""

    def p_alpha(self) -> NDArray_f64:
        r"""Return :math:`P_\alpha = \alpha I_{(*)} I_{(*)}^T`.

        Returns:
            :math:`P_\alpha`, shape ``(p, p)``
        """
        return self.alpha * self.points @ self.points.T


@dataclass(frozen=True)
class FilterState:
    r"""The estimate and the factors that carry its uncertainty."""

    x: NDArray_f64
    r"""State estimate :math:`\hat{\chi}_+`, shape ``(n,)``."""

    theta: NDArray_f64
    r"""Parameter estimate :math:`\hat{\theta}_+`, shape ``(p,)``."""

    L_x: NDArray_f64
    r"""State sensitivity :math:`L_\chi`, shape ``(n, p)``."""

    L_theta: NDArray_f64
    r"""Parameter sensitivity :math:`L_\theta`, shape ``(p, p)``."""

    U_inv: NDArray_f64
    r""":math:`U^{-1}`, shape ``(p, p)``; zero for ``unique``."""

    def covariance(self) -> NDArray_f64:
        r"""Return the parameter covariance :math:`P_\theta = L_\theta U^{-1} L_\theta^T`.

        Returns:
            :math:`P_\theta`, shape ``(p, p)``
        """
        return self.L_theta @ self.U_inv @ self.L_theta.T


def sigma_point_stencil(particles: Particles, p: int) -> SigmaPointStencil:
    r"""Construct the sigma-point stencil of the given kind, after [MC11]_.

    - ``simplex``: :math:`p + 1` points on a regular simplex; :math:`P_\alpha = I`.
    - ``canonical``: :math:`2p` points :math:`\pm\sqrt{p}\,e_j`; :math:`P_\alpha = I`.
    - ``unique``: one point at the origin: no spread, hence no correction. A sanity check
      that runs the whole pipeline with fixed parameters.

    .. note::

       [MC11]_ also list ``star``, the canonical points and the origin. Its weights are not
       given; for :math:`P_\alpha = I` and weights summing to one, the origin must have weight
       zero, so it equals ``canonical``. Equal weights :math:`\alpha=1/(2p + 1)` would give
       :math:`P_\alpha = \frac{2p}{2p + 1} I`: the sigma points would represent too small a
       covariance, at every step.

    Args:
        particles: kind of stencil
        p: number of parameters

    Returns:
        the stencil
    """
    match particles:
        case "simplex":
            r = p + 1
            alpha = 1.0 / r
            points = np.zeros((p, r))
            points[0, 0:2] = (-1.0 / np.sqrt(2 * alpha), 1.0 / np.sqrt(2 * alpha))
            for i in range(1, p):
                d = i + 1
                points[i, 0 : i + 1] = 1.0 / np.sqrt(alpha * d * (d + 1))
                points[i, i + 1] = -d / np.sqrt(alpha * d * (d + 1))
        case "canonical":
            r = 2 * p
            points = np.zeros((p, r))
            for i in range(p):
                points[i, i] = np.sqrt(p)
                points[i, p + i] = -np.sqrt(p)
        case "unique":
            r = 1
            points = np.zeros((p, r))
        case _:
            assert_never(particles)

    alpha = 1.0 / r

    return SigmaPointStencil(points, alpha)


def initial_state(
    x0: NDArray_f64, theta0: NDArray_f64, stddev_theta: NDArray_f64, stencil: SigmaPointStencil
) -> FilterState:
    r"""Return the initial filter state.

    As in [NAB26]_, the initial factors are

    .. math::

       L_\chi^0 = 0, \qquad L_\theta^0 = \sqrt{P_\theta^0}, \qquad U^0 = P_\alpha,

    with :math:`\sqrt{P_\theta^0}` the Cholesky factor of the diagonal initial covariance, so
    that :math:`P_\theta^0 = L_\theta^0 (U^0)^{-1} (L_\theta^0)^T` for stencils with
    :math:`P_\alpha = I`. This differs from [MC11]_ and [Ber12]_, which use the inconsistent
    :math:`(U^0)^{-1} = (P_\theta^0)^{-1/2}`, :math:`L_\theta^0 = I`.

    Args:
        x0: initial state, shape ``(n,)``; known, without uncertainty of its own
        theta0: initial parameter estimate, shape ``(p,)``
        stddev_theta: initial standard deviations of :math:`\theta`, shape ``(p,)``
        stencil: the sigma-point stencil

    Returns:
        the initial filter state
    """
    p = len(theta0)
    P_alpha = stencil.p_alpha()

    L_x = np.zeros((len(x0), p))
    L_theta = np.diag(stddev_theta)
    U_inv = np.linalg.inv(P_alpha) if P_alpha.any() else np.zeros((p, p))

    return FilterState(x0.copy(), theta0.copy(), L_x, L_theta, U_inv)


def sample(state: FilterState, stencil: SigmaPointStencil) -> tuple[NDArray_f64, NDArray_f64]:
    r"""Sample the sigma points around the estimate.

    .. math::

       \hat{\chi}_{(i)} = \hat{\chi}_+ + L_\chi C I_{(i)}, \qquad
       \hat{\theta}_{(i)} = \hat{\theta}_+ + L_\theta C I_{(i)},

    with :math:`C` the lower Cholesky factor of :math:`U^{-1}`.

    Note that in many publications, the formula used :math:`C^T`. This is incorrect for a *lower*
    Cholesky factor :math:`C`.

    Args:
        state: the filter state, around whose estimate the sigma points are sampled
        stencil: the sigma-point stencil

    Returns:
        - sigma-point states, one per column, shape ``(n, r)``
        - sigma-point parameters, one per column, shape ``(p, r)``
    """
    U_inv = state.U_inv
    C = np.linalg.cholesky(U_inv) if U_inv.any() else np.zeros_like(U_inv)
    spread = C @ stencil.points  # C I_(i), one column per sigma point

    x_sigma = state.x[:, np.newaxis] + state.L_x @ spread
    theta_sigma = state.theta[:, np.newaxis] + state.L_theta @ spread

    return x_sigma, theta_sigma


def mean(ensemble: NDArray_f64, stencil: SigmaPointStencil) -> NDArray_f64:
    r"""Return the weighted mean :math:`E_\alpha(Y_{(*)}) = \alpha \sum_i Y_{(i)}`.

    Args:
        ensemble: one sigma point per column, each of size k (e.g., n for states, p for
            parameters), shape ``(k, r)``
        stencil: the sigma-point stencil

    Returns:
        the mean, shape ``(k,)``
    """
    return stencil.alpha * ensemble.sum(axis=1)


def sensitivity(ensemble: NDArray_f64, stencil: SigmaPointStencil) -> NDArray_f64:
    r"""Return the sensitivity :math:`\alpha Y_{(*)} I_{(*)}^T` of the sigma points.

    Args:
        ensemble: one sigma point per column, each of size k (e.g., n for states, p for
            parameters), shape ``(k, r)``
        stencil: the sigma-point stencil

    Returns:
        the sensitivity, shape ``(k, p)``
    """
    return stencil.alpha * ensemble @ stencil.points.T


def gain(
    Gamma: NDArray_f64, stencil: SigmaPointStencil, comm: MPI.Comm, *, contributes: bool = True
) -> tuple[NDArray_f64, NDArray_f64]:
    r"""Compute :math:`U^{-1}` and the correction factor :math:`M` from the innovations.

    .. math::

       L_\Gamma = \alpha \Gamma_{(*)} I_{(*)}^T, \qquad
       U = P_\alpha + L_\Gamma^T L_\Gamma, \qquad
       M = U^{-1} L_\Gamma^T E_\alpha(\Gamma_{(*)}),

    for innovations weighted with the noise (so :math:`W = I`). The products
    :math:`L_\Gamma^T L_\Gamma` and :math:`L_\Gamma^T E_\alpha(\Gamma_{(*)})` are sums over the
    innovation entries, summed over the ranks; each entry must be counted on exactly one rank
    (``contributes``).

    Args:
        Gamma: innovations, weighted with the noise, one sigma point per column; the m
            innovation entries held by this rank, shape ``(m, r)``
        stencil: the sigma-point stencil
        comm: communicator over which the innovation entries are distributed
        contributes: whether this rank's entries count; False on all but one rank for
            innovations that every rank holds in full

    Returns:
        - :math:`U^{-1}`, shape ``(p, p)``; zero for ``unique``
        - :math:`M`, shape ``(p,)``
    """
    p = stencil.points.shape[0]
    L_Gamma = sensitivity(Gamma, stencil)
    LtL = L_Gamma.T @ L_Gamma if contributes else np.zeros((p, p))
    LtG = L_Gamma.T @ mean(Gamma, stencil) if contributes else np.zeros(p)
    comm.Allreduce(MPI.IN_PLACE, LtL, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, LtG, op=MPI.SUM)

    P_alpha = stencil.p_alpha()
    if not P_alpha.any():  # unique: no spread, no correction
        return np.zeros((p, p)), np.zeros(p)

    U_inv = np.linalg.inv(P_alpha + LtL)
    M = U_inv @ LtG

    return U_inv, M


def update(
    x_sigma: NDArray_f64,
    theta_sigma: NDArray_f64,
    Gamma: NDArray_f64,
    stencil: SigmaPointStencil,
    comm: MPI.Comm,
    *,
    contributes: bool = True,
) -> FilterState:
    r"""Return the corrected filter state from the propagated sigma points and their innovations.

    The correction stage of the assimilation step: the a priori estimates (:func:`mean`), the
    sensitivities (:func:`sensitivity`), the :func:`gain` and the corrected estimates.

    Args:
        x_sigma: the propagated sigma-point states, one per column, shape ``(n, r)``
        theta_sigma: the sigma-point parameters, one per column, shape ``(p, r)``
        Gamma: their innovations, weighted with the noise, see :func:`gain`
        stencil: the sigma-point stencil
        comm: communicator over which the state and innovation entries are distributed
        contributes: whether this rank's innovation entries count, see :func:`gain`

    Returns:
        the corrected filter state
    """
    x_prior = mean(x_sigma, stencil)
    theta_prior = mean(theta_sigma, stencil)
    L_x = sensitivity(x_sigma, stencil)
    L_theta = sensitivity(theta_sigma, stencil)

    U_inv, M = gain(Gamma, stencil, comm, contributes=contributes)

    x_posterior = x_prior - L_x @ M
    theta_posterior = theta_prior - L_theta @ M

    return FilterState(x_posterior, theta_posterior, L_x, L_theta, U_inv)
