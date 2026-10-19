
Verification: the Kalman filter for a linear problem
=====================================================

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

**Several steps.** Step :math:`k` samples around :math:`\bs{\hat\theta}^{k-1}_+` with
:math:`\bs{S} = \bs{L}_\theta^{k-1} \bs{C}^{k-1}`, and

.. math::

   \bs{S}\bs{S}^T = \bs{L}_\theta^{k-1} (\bs{U}^{k-1})^{-1} (\bs{L}_\theta^{k-1})^T
   = \bs{P}_\theta^{k-1}:

the sigma points represent the previous posterior exactly, as those of the first step
represent the prior. The derivation above then holds for step :math:`k`, with this
:math:`\bs{S}` and the prior :math:`\mathcal{N}(\bs{\hat\theta}^{k-1}_+, \bs{P}_\theta^{k-1})`:
each step is a Kalman update, and several steps are the sequential Kalman filter.
