r"""The reduced-order unscented Kalman filter (ROUKF): its linear algebra.

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

One assimilation step consists of four stages [NAB26]_. This module provides the linear
algebra; propagation and innovations involve the forward solver and the measurements and are
done in :mod:`kipe.estimation`.

1. **Sampling** (:func:`sample`): with :math:`C` the lower Cholesky factor of
   :math:`U^{-1}`, the sigma points are

   .. math::

      \hat{\chi}_{(i)} = \hat{\chi}_+ + L_\chi C^T I_{(i)}, \qquad
      \hat{\theta}_{(i)} = \hat{\theta}_+ + L_\theta C^T I_{(i)}.

2. **Prediction:** each sigma point is propagated with the forward model to the next
   measurement time (:mod:`kipe.estimation`); their :func:`mean` gives the a priori
   estimates

   .. math::

      \hat{\chi}_- = E_\alpha(\hat{\chi}_{(*)}), \qquad
      \hat{\theta}_- = E_\alpha(\hat{\theta}_{(*)}).

3. **Innovation:** per sigma point, with the spatial sampler and measurement model
   (:mod:`kipe.estimation`),

   .. math::

      \Gamma_{(i)} = Z - \mathcal{H}(\hat{\chi}_{(i)}).

4. **Correction:** the :func:`sensitivity` of the sigma points, the :func:`gain` and the
   corrected (a posteriori) estimates (:func:`correct`),

   .. math::

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


def sigma_point_stencil(particles: Particles, p: int) -> SigmaPointStencil:
    r"""Construct the sigma-point stencil of the given kind, after [MC11]_.

    - ``simplex``: :math:`p + 1` points on a regular simplex; :math:`P_\alpha = I`.
    - ``canonical``: :math:`2p` points :math:`\pm\sqrt{p}\,e_j`; :math:`P_\alpha = I`.
    - ``star``: the canonical points and the origin, :math:`2p + 1` points.
    - ``unique``: one point at the origin: no spread, hence no correction. A sanity check
      that runs the whole pipeline with fixed parameters.

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
        case "star":
            r = 2 * p + 1
            points = np.zeros((p, r))
            for i in range(p):  # column 0 is the origin
                points[i, i + 1] = np.sqrt(p)
                points[i, p + i + 1] = -np.sqrt(p)
        case "unique":
            r = 1
            points = np.zeros((p, r))
        case _:
            assert_never(particles)

    return SigmaPointStencil(points, 1.0 / r)


def initial_factors(
    stddev_theta: NDArray_f64, n: int, stencil: SigmaPointStencil
) -> tuple[NDArray_f64, NDArray_f64, NDArray_f64]:
    r"""Return the initial factors :math:`L_\chi^0`, :math:`L_\theta^0` and :math:`(U^0)^{-1}`.

    As in [NAB26]_,

    .. math::

       L_\chi^0 = 0, \qquad L_\theta^0 = \sqrt{P_\theta^0}, \qquad U^0 = P_\alpha,

    with :math:`\sqrt{P_\theta^0}` the Cholesky factor of the diagonal initial covariance, so
    that :math:`P_\theta^0 = L_\theta^0 (U^0)^{-1} (L_\theta^0)^T` for stencils with
    :math:`P_\alpha = I`. This differs from [MC11]_ and [Ber12]_, which use the inconsistent
    :math:`(U^0)^{-1} = (P_\theta^0)^{-1/2}`, :math:`L_\theta^0 = I`.

    Args:
        stddev_theta: initial standard deviations of :math:`\theta`, shape ``(p,)``
        n: size of the rank-local state vector
        stencil: the sigma-point stencil

    Returns:
        - :math:`L_\chi^0`, shape ``(n, p)``
        - :math:`L_\theta^0`, shape ``(p, p)``
        - :math:`(U^0)^{-1}`, shape ``(p, p)``; zero for ``unique``
    """
    p = len(stddev_theta)
    P_alpha = stencil.p_alpha()

    L_x = np.zeros((n, p))
    L_theta = np.diag(stddev_theta)
    U_inv = np.linalg.inv(P_alpha) if P_alpha.any() else np.zeros((p, p))

    return L_x, L_theta, U_inv


def sample(
    x: NDArray_f64,
    theta: NDArray_f64,
    L_x: NDArray_f64,
    L_theta: NDArray_f64,
    U_inv: NDArray_f64,
    stencil: SigmaPointStencil,
) -> tuple[NDArray_f64, NDArray_f64]:
    r"""Sample the sigma points around the estimate.

    .. math::

       \hat{\chi}_{(i)} = \hat{\chi}_+ + L_\chi C^T I_{(i)}, \qquad
       \hat{\theta}_{(i)} = \hat{\theta}_+ + L_\theta C^T I_{(i)},

    with :math:`C` the lower Cholesky factor of :math:`U^{-1}`.

    Args:
        x: state estimate :math:`\hat{\chi}_+`, shape ``(n,)``
        theta: parameter estimate :math:`\hat{\theta}_+`, shape ``(p,)``
        L_x: state sensitivity :math:`L_\chi`, shape ``(n, p)``
        L_theta: parameter sensitivity :math:`L_\theta`, shape ``(p, p)``
        U_inv: :math:`U^{-1}`, shape ``(p, p)``
        stencil: the sigma-point stencil

    Returns:
        - sigma-point states, one per column, shape ``(n, r)``
        - sigma-point parameters, one per column, shape ``(p, r)``
    """
    C = np.linalg.cholesky(U_inv) if U_inv.any() else np.zeros_like(U_inv)
    spread = C.T @ stencil.points  # C^T I_(i), one column per sigma point

    x_sigma = x[:, np.newaxis] + L_x @ spread
    theta_sigma = theta[:, np.newaxis] + L_theta @ spread

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


def correct(estimate: NDArray_f64, L: NDArray_f64, M: NDArray_f64) -> NDArray_f64:
    r"""Return the corrected estimate,:math:`\hat{\psi}_+ = \hat{\psi}_- - L_\psi M`.

    The estimate :math:`\psi` can be the state :math:`\chi` or the parameters :math:`\theta`.

    Args:
        estimate: predicted estimate (state or parameters)
        L: its sensitivity
        M: correction factor from :func:`gain`

    Returns:
        the corrected estimate
    """
    return estimate - L @ M


def covariance(L_theta: NDArray_f64, U_inv: NDArray_f64) -> NDArray_f64:
    r"""Return the parameter covariance :math:`P_\theta = L_\theta U^{-1} L_\theta^T`.

    Args:
        L_theta: parameter sensitivity, shape ``(p, p)``
        U_inv: :math:`U^{-1}`, shape ``(p, p)``

    Returns:
        :math:`P_\theta`, shape ``(p, p)``
    """
    return L_theta @ U_inv @ L_theta.T
