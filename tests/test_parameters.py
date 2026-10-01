"""Selection, initial estimate and reparameterization of the estimated parameters."""

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from kipe.options import ParametersOptions, StudyFileError
from kipe.parameters import build_parameterization

NOMINAL = {"a": 0.2, "b": 0.2, "c": 3.0}


def _build(data: dict):
    options = TypeAdapter(ParametersOptions).validate_python(data)
    return build_parameterization(options, NOMINAL)


@pytest.mark.parametrize(
    ("reparameterization", "spread", "theta0"),
    [
        ("log", {"relative_stddev": 0.1}, 0.0),
        ("multiplicative", {"relative_stddev": 0.1}, 1.0),
        ("additive", {"stddev": 0.1}, 0.0),
    ],
)
def test_initial_theta_maps_to_initial_estimate(reparameterization, spread, theta0):
    """The initial theta maps to the initial estimates: nominal or ``initial``."""
    parameterization = _build({
        "reparameterization": reparameterization,
        "select": {"c": spread, "a": {**spread, "initial": 0.25}},
    })
    theta = parameterization.initial_theta()

    np.testing.assert_allclose(theta, [theta0, theta0])
    assert parameterization.to_physical(theta) == pytest.approx({"c": 3.0, "a": 0.25})
    assert parameterization.names == ["c", "a"]  # order of the study file


@pytest.mark.parametrize(
    ("reparameterization", "spread", "theta", "c"),
    [
        ("log", {"relative_stddev": 0.1}, 1.0, 6.0),
        ("log", {"relative_stddev": 0.1}, -1.0, 1.5),
        ("multiplicative", {"relative_stddev": 0.1}, 0.5, 1.5),
        ("additive", {"stddev": 0.1}, -0.5, 2.5),
    ],
)
def test_to_physical(reparameterization, spread, theta, c):
    parameterization = _build({
        "reparameterization": reparameterization,
        "select": {"c": spread},
    })

    assert parameterization.to_physical(np.array([theta])) == pytest.approx({"c": c})


@pytest.mark.parametrize("reparameterization", ["log", "multiplicative"])
def test_relative_stddev_is_first_order_equivalent(reparameterization):
    """A small relative stddev gives about the same physical 1σ range for both maps."""
    parameterization = _build({
        "reparameterization": reparameterization,
        "select": {"c": {"relative_stddev": 0.01}},
    })
    np.testing.assert_allclose(parameterization.one_sigma_range(), [(2.97, 3.03)], rtol=1e-4)


def test_one_sigma_range():
    parameterization = _build({
        "reparameterization": "log",
        "select": {
            "c": {"relative_stddev": np.log(2)},  # stddev_theta = 1: a factor of 2
            "a": {"reparameterization": "additive", "initial": 0.0, "stddev": 0.1},
        },
    })
    np.testing.assert_allclose(parameterization.one_sigma_range(), [(1.5, 6.0), (-0.1, 0.1)])


def test_reparameterization_per_parameter():
    """The section's reparameterization is the default; each parameter can override it."""
    parameterization = _build({
        "reparameterization": "log",
        "select": {
            "c": {"relative_stddev": 0.1},
            "a": {"reparameterization": "additive", "initial": 0.0, "stddev": 0.05},
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
def test_recenter_moves_initial_estimate_to_estimate(reparameterization, spread):
    """After recentering, the initial theta maps to the previous estimate."""
    parameterization = _build({
        "reparameterization": reparameterization,
        "select": {"c": spread, "a": spread},
    })
    theta = parameterization.initial_theta() + np.array([0.5, -0.25])
    estimate = parameterization.to_physical(theta)

    recentered = parameterization.recenter(theta)
    assert recentered.to_physical(recentered.initial_theta()) == pytest.approx(estimate)
    np.testing.assert_allclose(recentered.stddev_theta(), parameterization.stddev_theta())

    original = parameterization.to_physical(parameterization.initial_theta())
    assert original == pytest.approx({"c": 3.0, "a": 0.2})  # unchanged by recenter


def test_unknown_parameter_lists_available():
    with pytest.raises(StudyFileError, match=r"no parameter d\. Available: a = 0\.2, b = 0\.2"):
        _build({"reparameterization": "log", "select": {"d": {"relative_stddev": 1}}})


@pytest.mark.parametrize(
    ("reparameterization", "initial"),
    [("log", 0.0), ("log", -1.0), ("multiplicative", 0.0)],
)
def test_inadmissible_initial_estimate(reparameterization, initial):
    with pytest.raises(StudyFileError, match=f"'{reparameterization}' needs"):
        _build({
            "reparameterization": reparameterization,
            "select": {"c": {"relative_stddev": 0.1, "initial": initial}},
        })


@pytest.mark.parametrize(
    "data",
    [
        {"select": {"c": {"relative_stddev": 0.1}}},  # reparameterization has no default
        {"reparameterization": "none", "select": {"c": {"relative_stddev": 0.1}}},
        {"reparameterization": "log", "select": {}},  # nothing selected
        {"reparameterization": "log", "select": {"c": {"relative_stddev": 0.0}}},
        {"reparameterization": "log", "select": {"c": {"initial": 1.0}}},  # no stddev
        {"reparameterization": "log", "select": {"c": {"stddev": 0.1}}},  # absolute for log
        {"reparameterization": "additive", "select": {"c": {"relative_stddev": 0.1}}},
        {
            "reparameterization": "log",
            "select": {"c": {"relative_stddev": 0.1, "stddev": 0.1}},  # both
        },
        {"reparameterization": "log", "select": {"c": {"mean": 1.0, "relative_stddev": 0.1}}},
    ],
)
def test_invalid_options(data):
    with pytest.raises(ValidationError):
        TypeAdapter(ParametersOptions).validate_python(data)
