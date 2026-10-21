The ROUKF
=========

Intro
-----

A Kalman filter is a recursive estimator: it corrects its estimate one measurement at a time.
For a dynamical model, it propagates the state and its covariance with the model, and
corrects both whenever a measurement is available.

For a linear model, it solves the estimation problem of :doc:`problem` exactly.
The unscented Kalman filter is a popular extension to nonlinear models: it represents the uncertainty by
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
------------

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
-------------

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
the estimation history (:doc:`../user_guide/estimation`).

**In kipe**, sampling and correction are :func:`kipe.roukf.sample` and
:func:`kipe.roukf.update`; the propagation of the sigma points and their innovations, which
need the solver and the measurements, are done by :class:`kipe.estimation.Estimation`. The a
priori means, which [NAB26]_ counts to the prediction, start the correction in kipe, since they
need no solver. The innovations are weighted with the noise,
:math:`\bs{W}^{-1/2} \bs{\Gamma}^k_{(i)}`, so that :math:`\bs{W}` does not appear in the
filter's code.

Outer iterations
----------------

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
