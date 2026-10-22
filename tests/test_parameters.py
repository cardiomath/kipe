"""Priors and reparameterization of the estimated parameters."""

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from kipe.options import ParametersOptions, StudyFileError
from kipe.parameters import EstimatedParameter, build_parameterization

AVAILABLE = ("a", "b", "c")  # the forward solver's parameters


def _build(data: dict):
    options = TypeAdapter(ParametersOptions).validate_python(data)
    return build_parameterization(options, AVAILABLE)


@pytest.mark.parametrize(
    ("reparameterization", "spread", "theta0"),
    [
        ("log", {"relative_stddev": 0.1}, 0.0),
        ("multiplicative", {"relative_stddev": 0.1}, 1.0),
        ("additive", {"stddev": 0.1}, 0.0),
    ],
)
def test_new_parameterization_anchored_at_prior_means(reparameterization, spread, theta0):
    """A new parameterization is anchored at the prior means: theta at the anchor maps to
    them."""
    parameterization = _build({
        "reparameterization": reparameterization,
        "priors": {"c": {"mean": 2.0, **spread}, "a": {"mean": 0.25, **spread}},
    })
    theta = parameterization.anchor_theta()

    np.testing.assert_allclose(theta, [theta0, theta0])
    assert parameterization.to_physical(theta) == pytest.approx({"c": 2.0, "a": 0.25})
    assert parameterization.names == ["c", "a"]  # order of the study file


@pytest.mark.parametrize(
    ("reparameterization", "spread", "theta", "phi"),
    [
        ("log", {"relative_stddev": 0.1}, 1.0, 6.0),
        ("log", {"relative_stddev": 0.1}, -1.0, 1.5),
        ("multiplicative", {"relative_stddev": 0.1}, 0.5, 1.5),
        ("additive", {"stddev": 0.1}, -0.5, 2.5),
    ],
)
def test_to_physical(reparameterization, spread, theta, phi):
    parameterization = _build({
        "reparameterization": reparameterization,
        "priors": {"c": {"mean": 3.0, **spread}},
    })

    assert parameterization.to_physical(np.array([theta])) == pytest.approx({"c": phi})


@pytest.mark.parametrize("reparameterization", ["log", "multiplicative"])
def test_relative_stddev_is_first_order_equivalent(reparameterization):
    """A small relative stddev gives about the same physical 1σ range for both maps."""
    parameterization = _build({
        "reparameterization": reparameterization,
        "priors": {"c": {"mean": 3.0, "relative_stddev": 0.01}},
    })
    np.testing.assert_allclose(parameterization.one_sigma_range(), [(2.97, 3.03)], rtol=1e-4)


def test_one_sigma_range():
    parameterization = _build({
        "reparameterization": "log",
        "priors": {
            # stddev_theta = 1: a factor of 2
            "c": {"mean": 3.0, "relative_stddev": np.log(2)},
            "a": {"reparameterization": "additive", "mean": 0.0, "stddev": 0.1},
        },
    })
    np.testing.assert_allclose(parameterization.one_sigma_range(), [(1.5, 6.0), (-0.1, 0.1)])


def test_reparameterization_per_parameter():
    """The section's reparameterization is the default; each parameter can override it."""
    parameterization = _build({
        "reparameterization": "log",
        "priors": {
            "c": {"mean": 3.0, "relative_stddev": 0.1},
            "a": {"reparameterization": "additive", "mean": 0.0, "stddev": 0.05},
        },
    })
    kinds = [parameter.reparameterization for parameter in parameterization.parameters]
    assert kinds == ["log", "additive"]
    np.testing.assert_allclose(parameterization.stddev_theta(), [0.1 / np.log(2), 0.05])


@pytest.mark.parametrize(
    ("reparameterization", "spread"),
    [
        ("log", {"relative_stddev": 0.5}),
        ("multiplicative", {"relative_stddev": 0.5}),
        ("additive", {"stddev": 0.5}),
    ],
)
def test_with_anchor_moves_the_anchor_to_the_estimate(reparameterization, spread):
    """Between outer iterations: after anchoring at the estimate, theta at the anchor maps to
    the estimate; the standard deviations of theta stay."""
    parameterization = _build({
        "reparameterization": reparameterization,
        "priors": {"c": {"mean": 3.0, **spread}, "a": {"mean": 0.2, **spread}},
    })
    theta = parameterization.anchor_theta() + np.array([0.5, -0.25])
    estimate = parameterization.to_physical(theta)

    anchored = parameterization.with_anchor(estimate)
    assert anchored.to_physical(anchored.anchor_theta()) == pytest.approx(estimate)
    np.testing.assert_allclose(anchored.stddev_theta(), parameterization.stddev_theta())

    original = parameterization.to_physical(parameterization.anchor_theta())
    assert original == pytest.approx({"c": 3.0, "a": 0.2})  # unchanged by with_anchor


def test_unknown_parameter_lists_available():
    with pytest.raises(StudyFileError, match=r"no parameter d\. Available: a, b, c"):
        _build({
            "reparameterization": "log",
            "priors": {"d": {"mean": 1.0, "relative_stddev": 1}},
        })


@pytest.mark.parametrize(
    ("reparameterization", "mean"),
    [("log", 0.0), ("multiplicative", 0.0)],
)
def test_inadmissible_mean(reparameterization, mean):
    with pytest.raises(StudyFileError, match=f"'{reparameterization}' needs a nonzero mean"):
        _build({
            "reparameterization": reparameterization,
            "priors": {"c": {"relative_stddev": 0.1, "mean": mean}},
        })


@pytest.mark.parametrize(
    "data",
    [
        {"priors": {"c": {"relative_stddev": 0.1}}},  # reparameterization has no default
        {"reparameterization": "none", "priors": {"c": {"relative_stddev": 0.1}}},
        {"reparameterization": "log", "priors": {}},  # nothing selected
        {"reparameterization": "log", "priors": {"c": {"relative_stddev": 0.0}}},
        {"reparameterization": "log", "priors": {"c": {"mean": 1.0}}},  # no stddev
        {"reparameterization": "log", "priors": {"c": {"stddev": 0.1}}},  # absolute for log
        {"reparameterization": "additive", "priors": {"c": {"relative_stddev": 0.1}}},
        {
            "reparameterization": "log",
            "priors": {"c": {"relative_stddev": 0.1, "stddev": 0.1}},  # both
        },
        {  # the former key
            "reparameterization": "log",
            "priors": {"c": {"initial": 1.0, "relative_stddev": 0.1}},
        },
    ],
)
def test_invalid_options(data):
    with pytest.raises(ValidationError):
        TypeAdapter(ParametersOptions).validate_python(data)


@pytest.mark.parametrize(
    ("reparameterization", "stddev_theta", "lower", "upper", "rtol"),
    [
        ("log", 0.1 / np.log(2), 2.0 * 0.99, 2.0 * 1.01, 1e-4),  # to first order
        ("multiplicative", 0.1, 2.0 * 0.99, 2.0 * 1.01, 1e-14),
        ("additive", 0.5, 2.0 - 0.005, 2.0 + 0.005, 1e-14),  # 0.01 of the stddev
    ],
)
def test_theta_step(reparameterization, stddev_theta, lower, upper, rtol):
    """A step of 0.01: 1 % of the parameter (log, multiplicative), 0.01 stddev (additive)."""
    parameter = EstimatedParameter("x", reparameterization, 2.0, stddev_theta)
    step, theta = parameter.theta_step(0.01), parameter.anchor_theta()
    assert parameter.to_physical(theta - step) == pytest.approx(lower, rel=rtol)
    assert parameter.to_physical(theta + step) == pytest.approx(upper, rel=rtol)
