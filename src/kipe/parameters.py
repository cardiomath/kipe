r"""Estimated parameters: selection, initial estimate and reparameterization.

kipe estimates the reparameterized parameters :math:`\theta`, the forward solver receives the
physical values :math:`\phi`. :class:`Parameterization` maps between the two.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self, assert_never

import numpy as np

from kipe._types import NDArray_f64
from kipe.forward_solver import Parameters
from kipe.options import ParametersOptions, Reparameterization, StudyFileError


@dataclass(frozen=True)
class EstimatedParameter:
    r"""One estimated parameter: its reparameterization, the parameter value at which the
    reparameterization is anchored, and the standard deviation of :math:`\theta`.

    Raises:
        ValueError: if the anchor is not admissible for the reparameterization
    """

    name: str
    """Name of the parameter, as the forward solver knows it."""

    reparameterization: Reparameterization
    """See :attr:`kipe.options.ParametersOptions.reparameterization`."""

    anchor: float
    r"""The physical value :math:`\phi^0` that :math:`\theta` is relative to, e.g.,
    :math:`\phi = \phi^0 \, 2^\theta` for ``log``: in an estimation, first the study's initial
    estimate, then, in later outer iterations, the previous estimate
    (:meth:`Parameterization.recenter`); in the sensitivity analysis, the nominal value
    (:meth:`Parameterization.with_anchor`)."""

    stddev_theta: float
    r"""Initial standard deviation of :math:`\theta`."""

    def __post_init__(self) -> None:
        """Check that the anchor is admissible for the reparameterization."""
        if self.reparameterization == "log" and self.anchor == 0:
            raise ValueError(f"{self.name}: 'log' needs a nonzero initial estimate")

        if self.reparameterization == "multiplicative" and self.anchor == 0:
            raise ValueError(f"{self.name}: 'multiplicative' needs a nonzero initial estimate")

    def initial_theta(self) -> float:
        r"""Return :math:`\theta` at the anchor, where the filter starts: 0, or 1 for
        ``multiplicative``.

        Returns:
            initial :math:`\theta`
        """
        match self.reparameterization:
            case "log" | "additive":
                return 0.0
            case "multiplicative":
                return 1.0
            case _:
                assert_never(self.reparameterization)

    def to_physical(self, theta: float) -> float:
        r"""Map :math:`\theta` to the physical value.

        Args:
            theta: reparameterized parameter

        Returns:
            physical value
        """
        phi0 = self.anchor

        match self.reparameterization:
            case "log":
                return phi0 * 2.0**theta
            case "multiplicative":
                return phi0 * theta
            case "additive":
                return phi0 + theta
            case _:
                assert_never(self.reparameterization)

    def theta_step(self, step: float) -> float:
        r"""Return the step in :math:`\theta` corresponding to a step in the parameter.

        With ``log`` and ``multiplicative`` reparameterizations, ``step`` is the relative step in
        the parameter, at its anchor; with ``additive``, a fraction of the parameter's
        prior standard deviation, which gives the step its unit:

        - ``log``: :math:`h_\theta = h / \ln 2`, to first order;
        - ``multiplicative``: :math:`h_\theta = h`;
        - ``additive``: :math:`h_\theta = h \sigma_\theta`.

        Used by the sensitivity analysis (:mod:`kipe.sensitivity`), whose parameterization is
        anchored at the nominal values.

        Args:
            step: relative change of the parameter (``log``, ``multiplicative``), or fraction of
                its prior standard deviation (``additive``)

        Returns:
            step in :math:`\theta`
        """
        match self.reparameterization:
            case "log":
                return step / np.log(2.0)
            case "multiplicative":
                return step
            case "additive":
                return self.stddev_theta * step
            case _:
                assert_never(self.reparameterization)


@dataclass(frozen=True)
class Parameterization:
    r"""The estimated parameters as one vector :math:`\theta`, in the order of the study file.

    Converts :math:`\theta` to the physical values passed to the forward solver
    (:meth:`to_physical`), each entry by its parameter's reparameterization, relative to its
    anchor (:attr:`EstimatedParameter.anchor`). The anchors stay fixed during one pass of the
    filter over the measurements; outer iterations move them to the estimate (:meth:`recenter`).
    """

    parameters: list[EstimatedParameter]
    r"""The estimated parameters, in the order of the entries of :math:`\theta`."""

    @property
    def names(self) -> list[str]:
        """Names of the estimated parameters."""
        return [parameter.name for parameter in self.parameters]

    def stddev_theta(self) -> NDArray_f64:
        r"""Return the initial standard deviations of :math:`\theta`.

        Returns:
            initial standard deviations of :math:`\theta`
        """
        return np.array([parameter.stddev_theta for parameter in self.parameters])

    def theta_step(self, step: float) -> NDArray_f64:
        r"""Return the steps in :math:`\theta` of a step in the parameters, one per parameter.

        Args:
            step: the step, see :meth:`EstimatedParameter.theta_step`

        Returns:
            the step in :math:`\theta` of each parameter
        """
        return np.array([parameter.theta_step(step) for parameter in self.parameters])

    def initial_theta(self) -> NDArray_f64:
        r"""Return :math:`\theta` at the anchors, where the filter starts.

        Returns:
            initial :math:`\theta`
        """
        return np.array([parameter.initial_theta() for parameter in self.parameters])

    def to_physical(self, theta: NDArray_f64) -> Parameters:
        r"""Map :math:`\theta` to physical values, as passed to the forward solver.

        Args:
            theta: reparameterized parameters, in the order of :attr:`parameters`

        Returns:
            parameter name -> physical value
        """
        return {
            parameter.name: parameter.to_physical(float(value))
            for parameter, value in zip(self.parameters, theta, strict=True)
        }

    def with_anchor(self, values: Mapping[str, float]) -> Self:
        r"""Return the parameterization with other anchors.

        The standard deviations of :math:`\theta` stay. Used by the sensitivity analysis, which
        anchors :math:`\theta` at the nominal values.

        Args:
            values: the new anchors, as physical values, by parameter name

        Returns:
            new parameterization
        """
        return replace(
            self,
            parameters=[replace(p, anchor=values[p.name]) for p in self.parameters],
        )

    def recenter(self, theta: NDArray_f64) -> Self:
        r"""Return the parameterization with the anchors moved to ``theta``.

        Used between outer iterations: the next pass starts from the estimate, with the
        initial standard deviations of :math:`\theta`.

        Args:
            theta: estimated parameters, in the order of :attr:`parameters`

        Returns:
            new parameterization, whose anchors are the physical values of ``theta``
        """
        return replace(
            self,
            parameters=[
                replace(parameter, anchor=parameter.to_physical(float(value)))
                for parameter, value in zip(self.parameters, theta, strict=True)
            ],
        )

    def one_sigma_range(self) -> list[tuple[float, float]]:
        r"""Return the physical values at :math:`\theta` of the anchors
        :math:`\pm \sigma_\theta`.

        Shows what the uncertainty means in physical terms. Asymmetric around the anchor for
        ``log``.

        Returns:
            (lower, upper) physical value per parameter
        """
        return self.physical_range(self.initial_theta(), self.stddev_theta())

    def physical_range(
        self, theta: NDArray_f64, stddev_theta: NDArray_f64
    ) -> list[tuple[float, float]]:
        r"""Return the physical values at :math:`\theta \pm \sigma_\theta`.

        Args:
            theta: estimated parameters, in the order of :attr:`parameters`
            stddev_theta: their standard deviations

        Returns:
            (lower, upper) physical value per parameter
        """
        ranges = []
        for parameter, value, sigma in zip(self.parameters, theta, stddev_theta, strict=True):
            lower = parameter.to_physical(float(value - sigma))
            upper = parameter.to_physical(float(value + sigma))
            ranges.append((min(lower, upper), max(lower, upper)))

        return ranges


def build_parameterization(options: ParametersOptions, nominal: Parameters) -> Parameterization:
    """Check the selected parameters against the solver's and build the parameterization.

    Args:
        options: ``parameters`` section of the study file
        nominal: the forward solver's parameters and their nominal values

    Returns:
        the parameterization, with the parameters in the order of ``options.select``

    Raises:
        StudyFileError: if a selected parameter is unknown to the solver, or an initial
            estimate is not admissible for its reparameterization
    """
    unknown = [name for name in options.select if name not in nominal]
    if unknown:
        available = ", ".join(f"{name} = {value:g}" for name, value in nominal.items())
        raise StudyFileError(
            f"parameters.select: the forward solver has no parameter {', '.join(unknown)}. "
            f"Available: {available}"
        )

    parameters = []
    for name, prior in options.select.items():
        reparameterization = prior.reparameterization or options.reparameterization
        initial = nominal[name] if prior.initial is None else prior.initial

        # the options check that each reparameterization has its kind of stddev
        match reparameterization:
            case "log":  # first order, see ParameterPrior.relative_stddev
                stddev_theta = prior.relative_stddev / np.log(2)
            case "multiplicative":
                stddev_theta = prior.relative_stddev
            case "additive":
                stddev_theta = prior.stddev
            case _:
                assert_never(reparameterization)

        try:
            parameters.append(
                EstimatedParameter(name, reparameterization, initial, float(stddev_theta))
            )
        except ValueError as err:
            raise StudyFileError(f"parameters.select.{err}") from err

    return Parameterization(parameters)
