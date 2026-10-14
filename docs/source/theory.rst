Theory
======

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
measurement (:doc:`measurements`). The noise is Gaussian, zero-mean and independent, with the
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

The ROUKF
---------

A Kalman filter propagates the state and its covariance with the model, and corrects both
whenever a measurement is available. For a linear model, it solves the problem above exactly.
The unscented Kalman filter extends it to nonlinear models: it represents the uncertainty by
:math:`N_\sigma` deterministically chosen *sigma points*, propagates them through the model,
and recovers the mean and covariance from them. For a joint estimate of state and parameters,
:math:`N_\sigma` grows with the dimension of both, which is infeasible for large models. The
ROUKF restricts the uncertainty to the :math:`p`-dimensional parameter space, so that
:math:`N_\sigma` grows with :math:`p` only.

We write :math:`\bs{\hat\chi}^k_\mp` and :math:`\bs{\hat\theta}^k_\mp` for the a priori
(predicted by the model) and a posteriori (corrected by the measurements) estimates at the
assimilation step :math:`k`, and :math:`\bs{Y}_{(*)}` for the matrix collecting the vectors
:math:`\bs{Y}_{(1)}, \dots, \bs{Y}_{(N_\sigma)}` column-wise.

Sigma points
^^^^^^^^^^^^

The sigma points perturb the estimates along the stencil
:math:`\bs{I}_{(1)}, \dots, \bs{I}_{(N_\sigma)} \in \mathbb{R}^p`, with equal weights
:math:`\alpha = 1/N_\sigma`, the weighted mean
:math:`E_\alpha(\bs{Y}_{(*)}) = \alpha \sum_i \bs{Y}_{(i)}` and

.. math::

   \bs{P}_\alpha = \alpha \, \bs{I}_{(*)} \bs{I}_{(*)}^T.

``estimation.sigma_points`` selects the stencil (:func:`kipe.roukf.sigma_point_stencil`):

- ``canonical``: :math:`N_\sigma = 2p`, a pair along each parameter,
  :math:`\bs{I}_{(i)} = \sqrt{p}\, \bs{e}_i` and :math:`\bs{I}_{(p+i)} = -\sqrt{p}\, \bs{e}_i`,
  as in [NAB26]_;
- ``simplex``: :math:`N_\sigma = p + 1`, the vertices of a regular simplex [MC11]_;
- ``unique``: :math:`N_\sigma = 1`, :math:`\bs{I}_{(1)} = \bs{0}`, without spread and so
  without correction: a check of the setup.

Both ``canonical`` and ``simplex`` have zero mean and :math:`\bs{P}_\alpha = \bs{1}`. A
symmetric stencil with :math:`2p + 1` points and equal weights would have
:math:`\bs{P}_\alpha = \frac{2p}{2p+1} \bs{1}`, and represent too small a spread.

The algorithm
^^^^^^^^^^^^^

The filter starts from :math:`\bs{\hat\chi}^0_+ = \bs{\chi}^0` and
:math:`\bs{\hat\theta}^0_+ = \bs{\theta}^0`, with

.. math::

   \bs{L}_\theta^0 = \sqrt{\bs{P}_\theta^0}, \qquad
   \bs{L}_\chi^0 = \bs{0} \in \mathbb{R}^{r \times p}, \qquad
   \bs{U}^0 = \bs{P}_\alpha,

:math:`\sqrt{\bs{P}_\theta^0}` the Cholesky factor of :math:`\bs{P}_\theta^0`. Each
assimilation step :math:`k = 1, \dots, N_T` consists of four stages.

**Sampling.** The sigma points perturb the current estimates along the stencil, for
:math:`i = 1, \dots, N_\sigma`,

.. math::

   \bs{\hat\chi}^{k-1}_{(i)} = \bs{\hat\chi}^{k-1}_+ + \bs{L}_\chi^{k-1} \bs{C}^{k-1} \bs{I}_{(i)},
   \qquad
   \bs{\hat\theta}^{k-1}_{(i)} = \bs{\hat\theta}^{k-1}_+
   + \bs{L}_\theta^{k-1} \bs{C}^{k-1} \bs{I}_{(i)},

with :math:`\bs{C}^{k-1}` the lower Cholesky factor of :math:`(\bs{U}^{k-1})^{-1}`, so that
the sigma points have the covariance :math:`\bs{L}_\theta \bs{U}^{-1} \bs{L}_\theta^T`
(:func:`kipe.roukf.sample`).

.. note::

   Some publications, e.g., [Ber12]_ and [NB22]_, write :math:`(\bs{C}^{k-1})^T` here, with
   :math:`\bs{C}^{k-1}` *a* Cholesky factor of :math:`(\bs{U}^{k-1})^{-1}`. That holds for the
   upper factor, :math:`\bs{C}^T \bs{C} = \bs{U}^{-1}`, as, e.g., MATLAB's ``chol`` returns
   it. With the lower factor, :math:`\bs{C} \bs{C}^T = \bs{U}^{-1}`, as numpy's ``cholesky``
   returns it and kipe uses, it is :math:`\bs{C}^{k-1}`: with its transpose, the sigma points
   have the intended covariance only while :math:`\bs{U}^{-1}` is diagonal, i.e., in the first
   step, since :math:`\bs{U}^0 = \bs{P}_\alpha = \bs{1}`.

**Prediction.** Each sigma point is propagated with the model, the parameters stay constant,

.. math::

   \bs{\hat\chi}^k_{(i)} = \mathcal{A}^k\big(\bs{\hat\chi}^{k-1}_{(i)},
   \bs{\phi}(\bs{\hat\theta}^{k-1}_{(i)})\big), \qquad
   \bs{\hat\theta}^k_{(i)} = \bs{\hat\theta}^{k-1}_{(i)},

and their means are the a priori estimates

.. math::

   \bs{\hat\chi}^k_- = E_\alpha\big(\bs{\hat\chi}^k_{(*)}\big), \qquad
   \bs{\hat\theta}^k_- = E_\alpha\big(\bs{\hat\theta}^k_{(*)}\big).

**Innovation.** For each sigma point, the discrepancy between the measurements and their
prediction,

.. math::

   \bs{\Gamma}^k_{(i)} = \bs{Z}^k - \mathcal{H}\big(\bs{\hat\chi}^k_{(i)}\big).

**Correction.** The sensitivities

.. math::

   \bs{L}_\chi^k = \alpha \, \bs{\hat\chi}^k_{(*)} \bs{I}_{(*)}^T, \qquad
   \bs{L}_\theta^k = \alpha \, \bs{\hat\theta}^k_{(*)} \bs{I}_{(*)}^T, \qquad
   \bs{L}_\Gamma^k = \alpha \, \bs{\Gamma}^k_{(*)} \bs{I}_{(*)}^T

and

.. math::

   \bs{U}^k = \bs{P}_\alpha + (\bs{L}_\Gamma^k)^T \bs{W}^{-1} \bs{L}_\Gamma^k

give the a posteriori estimates

.. math::

   \bs{\hat\chi}^k_+ = \bs{\hat\chi}^k_- - \bs{L}_\chi^k \bs{M}^k, \qquad
   \bs{\hat\theta}^k_+ = \bs{\hat\theta}^k_- - \bs{L}_\theta^k \bs{M}^k, \qquad
   \bs{M}^k = (\bs{U}^k)^{-1} (\bs{L}_\Gamma^k)^T \bs{W}^{-1}
   E_\alpha\big(\bs{\Gamma}^k_{(*)}\big),

and the covariance of the parameters,

.. math::

   \bs{P}_\theta^k = \bs{L}_\theta^k (\bs{U}^k)^{-1} (\bs{L}_\theta^k)^T.

After the last step, :math:`\bs{\hat\theta} = \bs{\hat\theta}^{N_T}_+` is the estimate and
:math:`\bs{P}_\theta = \bs{P}_\theta^{N_T}` its covariance: the posterior
:math:`\mathcal{N}(\bs{\hat\theta}, \bs{P}_\theta)`. kipe writes both after each step, to
the estimation history (:doc:`running`).

**In kipe**, sampling and correction are :func:`kipe.roukf.sample` and
:func:`kipe.roukf.update`; the propagation of the sigma points and their innovations, which
need the solver and the measurements, are done by :class:`kipe.estimation.Estimation`. The a
priori means, which [NAB26]_ counts to the prediction, start the correction in kipe, since they
need no solver. The innovations are weighted with the noise,
:math:`\bs{W}^{-1/2} \bs{\Gamma}^k_{(i)}`, so that :math:`\bs{W}` does not appear in the
filter's code.

Outer iterations
^^^^^^^^^^^^^^^^

The estimate of one pass over the measurements is biased towards the prior. Repeated passes
remove the bias (``estimation.iterations``): each starts from the previous estimate, with the
initial covariance and the initial state,

.. math::

   \bs{\phi}^{0,(j+1)} = \bs{\phi}\big(\bs{\hat\theta}^{(j)}\big), \qquad
   \bs{\theta}^{0,(j+1)} = \bs{\theta}^0, \qquad
   \bs{P}_\theta^{0,(j+1)} = \bs{P}_\theta^0, \qquad
   \bs{\hat\chi}^{0,(j+1)}_+ = \bs{\chi}^0,

with :math:`\bs{\phi}^{0,(0)} = \bs{\phi}^0`. For the multiplicative reparameterization,
:math:`\bs{\phi}(\bs{\hat\theta}^{(j)}) = \bs{\phi}^{0,(j)} \circ \bs{\hat\theta}^{(j)}`, as in
[NAB26]_. For a linear least-squares problem, the iterations converge linearly to the
minimizer of the data misfit (see [NAB26]_).

Verification: the Kalman filter for a linear observation
--------------------------------------------------------

For a linear observation, the unscented transform is exact, so each assimilation step of the
ROUKF gives the Kalman filter's posterior mean and covariance. This section derives it for one
step, for the parameters alone and with a state that depends linearly on them, with a stencil
with zero mean and :math:`\bs{P}_\alpha = \bs{1}`, as all of kipe's.
``tests/test_roukf_kalman.py`` checks it numerically, to round-off: one step, several steps,
and the state.

**Setting.** The prior is :math:`\bs{\theta} \sim \mathcal{N}(\bs{\theta}^0, \bs{P}_\theta^0)`,
the observation linear,

.. math::

   \bs{Z} = \bs{H} \bs{\theta} + \bs{\varepsilon}, \qquad
   \bs{\varepsilon} \sim \mathcal{N}(\bs{0}, \bs{W}).

**Sampling and prediction.** In the first step, :math:`\bs{U}^0 = \bs{P}_\alpha = \bs{1}`, so
:math:`\bs{C}^0 = \bs{1}`, and with :math:`\bs{S} = \bs{L}_\theta^0`,
:math:`\bs{S}\bs{S}^T = \bs{P}_\theta^0`,

.. math::

   \bs{\hat\theta}_{(i)} = \bs{\theta}^0 + \bs{S} \bs{I}_{(i)}, \qquad
   \bs{\hat\theta}_- = \bs{\theta}^0 + \bs{S} \, \alpha \sum_i \bs{I}_{(i)} = \bs{\theta}^0,

since the stencil has zero mean, and

.. math::

   \bs{L}_\theta = \alpha \, \bs{\hat\theta}_{(*)} \bs{I}_{(*)}^T
   = \bs{\theta}^0 \Big(\alpha \sum_i \bs{I}_{(i)}\Big)^T + \bs{S} \bs{P}_\alpha = \bs{S}.

The sigma points represent the prior exactly:
:math:`\bs{L}_\theta \bs{P}_\alpha^{-1} \bs{L}_\theta^T = \bs{S}\bs{S}^T = \bs{P}_\theta^0`.

**Innovation.** With the linear observation,

.. math::

   \bs{\Gamma}_{(i)} = \bs{Z} - \bs{H} \bs{\hat\theta}_{(i)}
   = \underbrace{\bs{Z} - \bs{H} \bs{\theta}^0}_{=: \bs{\bar\gamma}} - \bs{H} \bs{S} \bs{I}_{(i)},
   \qquad
   E_\alpha(\bs{\Gamma}_{(*)}) = \bs{\bar\gamma}, \qquad
   \bs{L}_\Gamma = -\bs{H} \bs{S}.

**Correction: covariance.**
:math:`\bs{U} = \bs{1} + \bs{S}^T \bs{H}^T \bs{W}^{-1} \bs{H} \bs{S}`, and since :math:`\bs{S}`
is invertible,

.. math::

   \bs{P}_\theta^+ = \bs{S} \bs{U}^{-1} \bs{S}^T
   = \big(\bs{S}^{-T} \bs{S}^{-1} + \bs{H}^T \bs{W}^{-1} \bs{H}\big)^{-1}
   = \big((\bs{P}_\theta^0)^{-1} + \bs{H}^T \bs{W}^{-1} \bs{H}\big)^{-1},

the Kalman posterior covariance, in its information form.

**Correction: mean.**
:math:`\bs{M} = \bs{U}^{-1} \bs{L}_\Gamma^T \bs{W}^{-1} \bs{\bar\gamma}
= -\bs{U}^{-1} \bs{S}^T \bs{H}^T \bs{W}^{-1} \bs{\bar\gamma}`, so

.. math::

   \bs{\hat\theta}_+ = \bs{\hat\theta}_- - \bs{L}_\theta \bs{M}
   = \bs{\theta}^0 + \bs{S} \bs{U}^{-1} \bs{S}^T \bs{H}^T \bs{W}^{-1} \bs{\bar\gamma}
   = \bs{\theta}^0 + \bs{P}_\theta^+ \bs{H}^T \bs{W}^{-1} \big(\bs{Z} - \bs{H}\bs{\theta}^0\big),

the Kalman update with the gain :math:`\bs{K} = \bs{P}_\theta^+ \bs{H}^T \bs{W}^{-1}`. The test
compares with the gain form

.. math::

   \bs{K} = \bs{P}_\theta^0 \bs{H}^T \big(\bs{H} \bs{P}_\theta^0 \bs{H}^T + \bs{W}\big)^{-1},
   \qquad
   \bs{P}_\theta^+ = (\bs{1} - \bs{K}\bs{H}) \bs{P}_\theta^0,

which is equal by the Woodbury identity.

**With a state.** For a model :math:`\bs{\hat\chi}_{(i)} = \bs{A} \bs{\hat\theta}_{(i)}` and
the observation :math:`\bs{Z} = \bs{H}\bs{\chi} + \bs{\varepsilon}`,
:math:`\bs{\hat\chi}_- = \bs{A}\bs{\hat\theta}_-` and
:math:`\bs{L}_\chi = \bs{A} \bs{L}_\theta`. The parameters see the observation operator
:math:`\bs{H}\bs{A}` (above, with :math:`\bs{H} \to \bs{H}\bs{A}`), and

.. math::

   \bs{\hat\chi}_+ = \bs{\hat\chi}_- - \bs{L}_\chi \bs{M}
   = \bs{A} \big(\bs{\hat\theta}_- - \bs{L}_\theta \bs{M}\big) = \bs{A} \bs{\hat\theta}_+.

**Several steps.** From the second step on, :math:`\bs{U}^{-1}` is no longer diagonal, and the
sampling with :math:`\bs{C}`, not :math:`\bs{C}^T`, keeps the sigma points' covariance at
:math:`\bs{P}_\theta^k`: only then do several steps equal the sequential Kalman filter. The test
checks this; it is not derived here, and it is the test that tells :math:`\bs{C}` from
:math:`\bs{C}^T`.
