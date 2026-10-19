Problem formulation
===================

Scientific references
---------------------

kipe estimates the parameters of a model from measurements with the reduced-order unscented
Kalman filter (ROUKF) of [MC11]_. This page follows the formulation and notation of
[NAB26]_; :mod:`kipe.roukf` implements the filter, :mod:`kipe.estimation` runs it over a study.

The estimation problem
----------------------

**The model.** The forward solver advances a state :math:`\bs{\chi}^k \in \mathbb{R}^r` from
one measurement time :math:`t^{k-1}` to the next, :math:`t^k`, taking as many time steps as it
needs,

.. math::

   \bs{\chi}^k = \mathcal{A}^k(\bs{\chi}^{k-1}, \bs{\phi}), \qquad k = 1, \dots, N_T,

with the physical parameters :math:`\bs{\phi} \in \mathbb{R}^p` and a known initial state
:math:`\bs{\chi}^0`. In kipe, :math:`\mathcal{A}^k` is the solver's
:meth:`~kipe.forward_solver.ForwardSolver.propagate`. The state is everything a time step
depends on, so that the model can be restarted from any state; the model carries no process
noise, so the parameters are the only unknowns.

**The parameters.** The filter estimates reparameterized parameters
:math:`\bs{\theta} \in \mathbb{R}^p`, relative to the initial estimate :math:`\bs{\phi}^0`
(``parameters.reparameterization``):

.. math::

   \text{log:} \quad \phi_i = \phi_i^0 \, 2^{\theta_i}, \qquad
   \text{multiplicative:} \quad \phi_i = \phi_i^0 \, \theta_i, \qquad
   \text{additive:} \quad \phi_i = \phi_i^0 + \theta_i,

so that the initial estimate is :math:`\theta^0_i = 0`, :math:`1` or :math:`0`. The parameters
are then of similar size, though their physical values may differ by orders of magnitude.
[NAB26]_ uses the multiplicative one. The prior covariance :math:`\bs{P}_\theta^0` is diagonal,
with the standard deviations of the study file, converted to :math:`\bs{\theta}`.

**The measurements.** :math:`\bs{Z}^k \in \mathbb{R}^m` collects all measurements at
:math:`t^k`, and the observation operator :math:`\mathcal{H}: \mathbb{R}^r \to \mathbb{R}^m`
predicts them from the state: in kipe, a spatial sampler and a measurement model per
measurement (:doc:`../user_guide/measurements`). The noise is Gaussian, zero-mean and independent, with the
covariance

.. math::

   \bs{W} = \operatorname{diag} \bs{\sigma}_Z^{\circ 2},

where :math:`\bs{\sigma}_Z` collects the standard deviations of the measurements
(``noise.stddev``).

**The problem.** The maximum a posteriori estimate of the parameters minimizes

.. math::

   J(\bs{\theta}) = \frac{1}{2} \big\| \bs{\theta} - \bs{\theta}^0
   \big\|^2_{(\bs{P}_\theta^0)^{-1}}
   + \frac{1}{2} \sum_{k=1}^{N_T} \big\| \bs{Z}^k - \mathcal{H}\big(\bs{\chi}^k(\bs{\theta})\big)
   \big\|^2_{\bs{W}^{-1}},

with :math:`\|\bs{x}\|^2_{\bs{A}} = \bs{x}^T \bs{A} \bs{x}` and :math:`\bs{\chi}^k(\bs{\theta})`
the state after :math:`k` steps with the parameters :math:`\bs{\phi}(\bs{\theta})`.
