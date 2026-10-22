r"""Sensitivity analysis: ``kipe sensitivity``.

The Fisher information :math:`\bs{M}` of a study's measurements says how much they tell about
the estimated parameters, and which combinations of the parameters they barely determine. It
is evaluated at the nominal parameters, i.e., the truth of a twin experiment, and in
:math:`\bs{\theta}` of the ``parameters`` section, the space the filter estimates in.

Two forward runs per parameter, at :math:`\bs{\theta}_\mathrm{true} \pm h_\theta` along it, give
the noise-weighted sensitivities of the predicted data at each measurement time :math:`t^k` by
centered differences. :math:`h_\theta` is the step in :math:`\theta` of ``sensitivity.step``, a
relative change of the parameter, or a fraction of its prior standard deviation for ``additive``
parameters (:meth:`~kipe.parameters.EstimatedParameter.theta_step`):

.. math::

   \bs{\tilde{J}}^k = \bs{W}^{-1/2}
   \frac{\partial \mathcal{H}\big(\bs{\chi}^k(\bs{\theta})\big)}{\partial \bs{\theta}}
   \bigg|_{\bs{\theta} = \bs{\theta}_\mathrm{true}},
   \qquad
   \bs{M} = \sum_{k=1}^{N_T} \big(\bs{\tilde{J}}^k\big)^T \bs{\tilde{J}}^k,

with :math:`\bs{W}` the noise covariance of the measurements' ``noise.stddev``, which estimation
assumes too, and the stacked :math:`\bs{\tilde{J}}` of all times and measurements. The runs
themselves are noise-free.

- :math:`\bs{M}^{-1}` is the Cramér-Rao bound of the covariance of an estimate from the data
  alone. The filter's :math:`\bs{P}_\theta` also contains the prior: compare it with the
  Bayesian Cramér-Rao bound :math:`(\bs{M} + (\bs{P}^0_\theta)^{-1})^{-1}`, which includes
  the prior. For observations linear in :math:`\bs{\theta}`, the filter's :math:`\bs{P}_\theta`
  after each pass equals it: with ``additive`` and ``multiplicative``, observations linear in
  the parameters are also linear in :math:`\bs{\theta}`; with ``log``, they are not. It is close
  to :math:`\bs{M}^{-1}` only where the data dominate the prior (see below).
- For ``log`` and ``additive``, :math:`\bs{\theta}` here differs from the filter's only by a
  constant, so :math:`\bs{M}` is the same in both. ``multiplicative``,
  :math:`\phi_i = \phi_i^0 \theta_i`, scales by the anchors :math:`\phi_i^0` in the filter,
  i.e., the initial values, by the nominal values :math:`\phi_{\mathrm{true},i}` here: the
  filter's :math:`\bs{P}_\theta` corresponds to :math:`\bs{D} \bs{P}_\theta \bs{D}` here,
  :math:`\bs{D} = \operatorname{diag}(\phi_i^0 / \phi_{\mathrm{true},i})`. The correlations
  are the same in both. With several iterations, :math:`\phi_i^0` is the anchor of the last
  pass, the estimate of the one before. Its prior is relative to that anchor too, so the
  Bayesian bound to compare with is :math:`(\bs{M} + (\bs{D} \bs{P}^0_\theta \bs{D})^{-1})^{-1}`
  here, the reported one only for :math:`\bs{D} \approx \bs{1}`.
- The singular values :math:`s_1 \geq \dots \geq s_p` of :math:`\bs{\tilde{J}}` and its right
  singular vectors, the columns of :math:`\bs{V}` in
  :math:`\bs{\tilde{J}} = \bs{U} \operatorname{diag}(\bs{s}) \bs{V}^T`, are also the
  eigendecomposition of the Fisher information,
  :math:`\bs{M} = \bs{V} \operatorname{diag}(\bs{s})^2 \bs{V}^T`. Each vector is a combination
  of the parameters, determined well by the data for a large :math:`s_i`, poorly for a small
  one. The condition number :math:`\kappa = s_1 / s_p` summarizes this with the ratio of the
  best to the worst determined combination. The SVD of :math:`\bs{\tilde{J}}` is better
  conditioned than the eigendecomposition of :math:`\bs{M}`, whose condition number is
  :math:`\kappa^2`: the small eigenvalues of :math:`\bs{M}` lose accuracy first, so kipe
  decomposes :math:`\bs{\tilde{J}}`. For ``additive`` parameters, the decomposition depends on
  their units.

The runs are independent: :func:`run` computes one of them, e.g., as a task of a job array,
:func:`assemble` combines them. Output, in ``<output.path>/sensitivity/``:

- ``runs/<i>/``: the predicted data of run :math:`i`, one file per measurement in the format
  of the data files, with the run's log and provenance;
- ``sensitivity.npz``: the sensitivities, the Fisher information per measurement and in total
  (the information of several measurements adds up), the singular values and right singular
  vectors of the noise-weighted sensitivities (corresponding to the square-roots of eigenvalues
  and to the eigenvectors of the Fisher information, respectively), the Cramér-Rao covariance,
  standard deviations and correlations, and the Bayesian Cramér-Rao covariance and standard
  deviations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mpi4py import MPI

import numpy as np
from ruamel.yaml import YAML

from kipe._formatting import format_table
from kipe._types import NDArray_f64
from kipe.forward_solver import build_forward_solver
from kipe.measurements import (
    build_observation_operator,
    measurement_times,
    read_numpy,
    write_data,
)
from kipe.options import StudyFileError, StudyOptions
from kipe.parameters import Parameterization, build_parameterization
from kipe.provenance import recorded_run
from kipe.synthesis import observe_run

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sensitivity:
    r"""The result of a sensitivity analysis, in :math:`\theta`."""

    names: list[str]
    """Names of the estimated parameters."""

    theta: NDArray_f64
    r""":math:`\bs{\theta}_\mathrm{true}`, which maps to the nominal parameters."""

    stddev_theta: NDArray_f64
    r"""Prior standard deviations :math:`\sigma_\theta`, of the ``parameters`` section."""

    jacobians: dict[str, NDArray_f64]
    r"""Sensitivities :math:`\partial \mathcal{H} / \partial \bs{\theta}` of each measurement,
    not weighted with the noise: one row
    per entry of its data, the entries of all its times one after another; one column per
    parameter."""

    fims: dict[str, NDArray_f64]
    """Fisher information of each measurement."""

    fim: NDArray_f64
    """Fisher information of all measurements: the sum of theirs."""

    singular_values: NDArray_f64
    r"""Singular values of the noise-weighted sensitivities :math:`\bs{\tilde{J}}` in descending
    order, corresponding to the square roots of the eigenvalues of the Fisher information."""

    right_singular_vectors: NDArray_f64
    r"""Right singular vectors of the noise-weighted sensitivities :math:`\bs{\tilde{J}}`,
    column-wise, in the order of the singular values. They correspond to eigenvectors of the
    Fisher information."""

    condition_number: float
    r"""Condition number :math:`\kappa = s_1 / s_p` of the noise-weighted sensitivities of all
    times and measurements, from their singular values."""

    covariance: NDArray_f64
    r"""Cramér-Rao bound of the covariance of :math:`\theta`, the inverse Fisher information;
    infinite where it is singular."""

    posterior_covariance: NDArray_f64
    r"""Bayesian Cramér-Rao bound, with the prior:
    :math:`(\bs{M} + (\bs{P}^0_\theta)^{-1})^{-1}`, to compare with the filter's
    :math:`\bs{P}_\theta`."""

    @property
    def stddev(self) -> NDArray_f64:
        r"""Cramér-Rao standard deviations of :math:`\theta`."""
        return np.sqrt(np.diag(self.covariance))

    @property
    def posterior_stddev(self) -> NDArray_f64:
        r"""Standard deviations of :math:`\theta` of the Bayesian Cramér-Rao bound."""
        return np.sqrt(np.diag(self.posterior_covariance))

    @property
    def correlation(self) -> NDArray_f64:
        """Cramér-Rao correlations."""
        stddev = self.stddev
        return self.covariance / np.outer(stddev, stddev)


def run_count(study: StudyOptions) -> int:
    """Return the number of forward runs of the sensitivity analysis: two per parameter.

    Args:
        study: the study options, with parameters

    Returns:
        the number of runs
    """
    if study.parameters is None:
        raise StudyFileError("sensitivity needs a parameters section")

    return 2 * len(study.parameters.select)


def run_directory(study: StudyOptions, index: int) -> Path:
    """Return the output directory of a run, e.g., ``<output.path>/sensitivity/runs/03``.

    Args:
        study: the study options
        index: number of the run, from 0

    Returns:
        the directory, numbered with as many digits as the largest number

    Raises:
        StudyFileError: if the study has no such run
    """
    n = run_count(study)
    if not 0 <= index < n:
        raise StudyFileError(f"sensitivity run {index}: the study has runs 0 to {n - 1}")

    return Path(study.output.path) / "sensitivity" / "runs" / f"{index:0{len(str(n - 1))}d}"


def run(study: StudyOptions, index: int, comm: MPI.Comm = MPI.COMM_WORLD) -> None:
    r"""Compute one forward run of the sensitivity analysis, and write its predicted data.

    Run :math:`i` perturbs parameter :math:`i // 2`, by :math:`+h_\theta` for even :math:`i`, by
    :math:`-h_\theta` for odd :math:`i`, the step in :math:`\theta` of ``sensitivity.step``
    (:meth:`~kipe.parameters.EstimatedParameter.theta_step`).

    Args:
        study: the study options
        index: number of the run, from 0
        comm: communicator over which the model state is distributed

    Raises:
        StudyFileError: if the study lacks what the sensitivity analysis needs, or has no such
            run
    """
    directory = run_directory(study, index)
    if not study.measurements:
        raise StudyFileError("sensitivity needs a measurements section")

    solver = build_forward_solver(study.forward_solver)
    operators = {
        name: build_observation_operator(name, options, solver, comm)
        for name, options in study.measurements.items()
    }
    times = {name: _times(name, study) for name in study.measurements}

    parameterization = _at_nominal(study, solver.nominal_parameters())
    theta = _perturbed_theta(parameterization, study.sensitivity.step, index)
    physical = parameterization.to_physical(theta)
    label = _label(parameterization, index)

    info = {
        "run": index,
        "perturbed": label,
        "theta": theta.tolist(),
        "parameters": physical,
    }
    with recorded_run(directory, study, info, comm):
        logger.info("sensitivity run %d/%d: %s", index + 1, run_count(study), label)
        listed = ", ".join(f"{n} = {v:.6g}" for n, v in physical.items())
        logger.info("  %-14s%s", "parameters", listed)

        predicted = observe_run(solver, times, operators, physical)

        for measurement, values in predicted.items():
            path = directory / f"{measurement}.npz"
            write_data(path, operators[measurement], times[measurement], values, comm)

        logger.info("  %-14s%s", "wrote", directory)


def assemble(study: StudyOptions, comm: MPI.Comm = MPI.COMM_WORLD) -> Sensitivity:
    """Combine the runs of the sensitivity analysis; write and log the result.

    Every rank reads the runs; rank 0 writes ``sensitivity.npz``.

    Args:
        study: the study options
        comm: communicator of the run

    Returns:
        the result

    Raises:
        StudyFileError: if a run is missing or unfinished, or was computed with other
            parameters than the study's
    """
    n = run_count(study)
    if not study.measurements:
        raise StudyFileError("sensitivity needs a measurements section")

    output = Path(study.output.path) / "sensitivity"
    with recorded_run(output, study, comm=comm):
        solver = build_forward_solver(study.forward_solver)
        nominal = solver.nominal_parameters()
        parameterization = _at_nominal(study, nominal)
        _check_runs(study, parameterization)

        step = parameterization.theta_step(study.sensitivity.step)
        jacobians = {}
        for name in study.measurements:
            plus, minus = [
                [read_numpy(run_directory(study, i) / f"{name}.npz")[1] for i in range(s, n, 2)]
                for s in (0, 1)
            ]
            columns = [
                (p - m).ravel() / (2 * h) for p, m, h in zip(plus, minus, step, strict=True)
            ]
            jacobians[name] = np.column_stack(columns)

        sensitivity = _sensitivity(study, parameterization, jacobians)

        if comm.rank == 0:
            _write(output / "sensitivity.npz", sensitivity, study.sensitivity.step)
        _log(sensitivity, parameterization, study.sensitivity.step)
        logger.info("%-11s%s", "result", output / "sensitivity.npz")

    return sensitivity


def _at_nominal(study: StudyOptions, nominal: dict[str, float]) -> Parameterization:
    r"""Return the study's parameterization, anchored at the nominal parameters: its initial
    :math:`\bs{\theta}` maps to them, :math:`\bs{\theta}_\mathrm{true}`.

    Args:
        study: the study options, with parameters
        nominal: the solver's nominal parameters

    Returns:
        the parameterization
    """
    if study.parameters is None:
        raise StudyFileError("sensitivity needs a parameters section")

    parameterization = build_parameterization(study.parameters, nominal)
    return parameterization.with_anchor(nominal)


def _perturbed_theta(parameterization: Parameterization, step: float, index: int) -> NDArray_f64:
    r"""Return :math:`\theta` of a run.

    Args:
        parameterization: the parameterization, anchored at the nominal parameters
        step: the step, see :attr:`kipe.options.SensitivityOptions.step`
        index: number of the run

    Returns:
        :math:`\bs{\theta}_\mathrm{true}`, perturbed along parameter ``index // 2``
    """
    theta = parameterization.initial_theta()
    j = index // 2
    sign = 1.0 if index % 2 == 0 else -1.0
    theta[j] += sign * parameterization.theta_step(step)[j]
    return theta


def _label(parameterization: Parameterization, index: int) -> str:
    """Return the label of a run: the sign and name of the parameter it perturbs.

    Args:
        parameterization: the parameterization
        index: number of the run

    Returns:
        the label, e.g., ``-b`` for a downward step of parameter ``b``
    """
    name = parameterization.names[index // 2]
    return f"+{name}" if index % 2 == 0 else f"-{name}"


def _times(name: str, study: StudyOptions) -> NDArray_f64:
    """Return the times of a measurement, which the sensitivity analysis requires.

    Args:
        name: name of the measurement
        study: the study options

    Returns:
        the measurement times

    Raises:
        StudyFileError: if the measurement has no times
    """
    assert study.measurements is not None
    options = study.measurements[name]
    if options.times is None:
        raise StudyFileError(f"measurements.{name}: the sensitivity analysis needs 'times'")

    return measurement_times(options.times)


def _check_runs(study: StudyOptions, parameterization: Parameterization) -> None:
    """Check that all runs are finished, with the parameters the study asks for.

    Args:
        study: the study options
        parameterization: the parameterization, anchored at the nominal values

    Raises:
        StudyFileError: if a run is missing or unfinished, or has other parameters
    """
    missing, different = [], []
    for i in range(run_count(study)):
        path = run_directory(study, i) / "run.yaml"
        run_info = YAML(typ="safe").load(path) if path.exists() else None
        if not run_info or run_info.get("status") != "finished":
            missing.append(str(i))
        elif not np.array_equal(
            run_info["theta"], _perturbed_theta(parameterization, study.sensitivity.step, i)
        ):
            different.append(str(i))

    if missing:
        raise StudyFileError(
            f"sensitivity runs {', '.join(missing)} are missing or unfinished: run them with "
            "kipe sensitivity --run"
        )
    if different:
        raise StudyFileError(
            f"sensitivity runs {', '.join(different)} have other parameters than the study "
            "asks for, e.g., another step: run them again"
        )


def _sensitivity(
    study: StudyOptions, parameterization: Parameterization, jacobians: dict[str, NDArray_f64]
) -> Sensitivity:
    """Return the Fisher information and what follows from it.

    Args:
        study: the study options, with measurements
        parameterization: the parameterization, anchored at the nominal values
        jacobians: the sensitivities of each measurement

    Returns:
        the result
    """
    assert study.measurements is not None

    # noise-weighted Jacobians, of all times of each measurement
    J_tilde = {name: J / study.measurements[name].noise.stddev for name, J in jacobians.items()}
    fims = {name: J.T @ J for name, J in J_tilde.items()}
    fim = sum(fims.values(), start=np.zeros((len(parameterization.names),) * 2))

    # of all measurements, stacked
    _, singular_values, Vt = np.linalg.svd(
        np.vstack(list(J_tilde.values())), full_matrices=False
    )

    try:
        covariance = np.asarray(np.linalg.inv(fim), dtype=np.float64)
    except np.linalg.LinAlgError:
        logger.warning(
            "the Fisher information is singular: some parameters are not identifiable"
        )
        covariance = np.full_like(fim, np.inf)

    prior_information = np.diag(parameterization.stddev_theta() ** -2.0)  # (P_theta^0)^-1
    posterior_covariance = np.asarray(np.linalg.inv(fim + prior_information), dtype=np.float64)

    return Sensitivity(
        names=parameterization.names,
        theta=parameterization.initial_theta(),
        stddev_theta=parameterization.stddev_theta(),
        jacobians=jacobians,
        fims=fims,
        fim=fim,
        singular_values=singular_values,
        right_singular_vectors=Vt.T,
        condition_number=float(singular_values[0] / singular_values[-1]),
        covariance=covariance,
        posterior_covariance=posterior_covariance,
    )


def _write(path: Path, sensitivity: Sensitivity, step: float) -> None:
    """Write the result to ``sensitivity.npz``.

    Args:
        path: path to the file
        sensitivity: the result
        step: the step of the finite differences, ``sensitivity.step``
    """
    arrays: dict[str, Any] = {
        "names": np.array(sensitivity.names),
        "measurements": np.array(list(sensitivity.jacobians)),
        "step": step,
        "theta": sensitivity.theta,
        "stddev_theta": sensitivity.stddev_theta,
        "fim": sensitivity.fim,
        "singular_values": sensitivity.singular_values,
        "right_singular_vectors": sensitivity.right_singular_vectors,
        "covariance": sensitivity.covariance,
        "stddev": sensitivity.stddev,
        "posterior_covariance": sensitivity.posterior_covariance,
        "posterior_stddev": sensitivity.posterior_stddev,
        "correlation": sensitivity.correlation,
        "condition_number": sensitivity.condition_number,
    }
    arrays |= {f"J_{name}": J for name, J in sensitivity.jacobians.items()}
    arrays |= {f"fim_{name}": fim for name, fim in sensitivity.fims.items()}
    np.savez(path, **arrays)


def _log(sensitivity: Sensitivity, parameterization: Parameterization, step: float) -> None:
    """Log the result: the Cramér-Rao bounds, correlations and singular values.

    Args:
        sensitivity: the result
        parameterization: the parameterization, anchored at the nominal values
        step: the step of the finite differences, ``sensitivity.step``
    """
    logger.info("sensitivity at the nominal parameters (step %g)", step)
    ranges = parameterization.physical_range(sensitivity.theta, sensitivity.stddev)
    rows = [("parameter", "nominal", "prior σ_θ", "CR σ_θ", "CR 1σ range", "with prior σ_θ")]
    for i, name in enumerate(sensitivity.names):
        lower, upper = ranges[i]
        rows.append((
            name,
            f"{parameterization.parameters[i].anchor:.4g}",
            f"{sensitivity.stddev_theta[i]:.3g}",
            f"{sensitivity.stddev[i]:.3g}",
            f"[{lower:.4g}, {upper:.4g}]",
            f"{sensitivity.posterior_stddev[i]:.3g}",
        ))
    for line in format_table(rows, "<>>><>"):
        logger.info("  %s", line)
    logger.info(
        "  (CR: from the data alone; with prior: the Bayesian bound, as the filter's σ_θ)"
    )

    logger.info("")
    logger.info("  Cramér-Rao correlations")
    names = sensitivity.names
    correlations = [("", *names)] + [
        (name, *(f"{c:.2f}" for c in row))
        for name, row in zip(names, sensitivity.correlation, strict=True)
    ]
    for line in format_table(correlations, "<" + ">" * len(names)):
        logger.info("    %s", line)

    logger.info("")
    logger.info("  singular values of the weighted sensitivities (header), and the magnitudes")
    logger.info("  of their right singular vectors: the parameters of each combination")
    vectors = [("", *(f"{s:.3g}" for s in sensitivity.singular_values))] + [
        (name, *(f"{abs(v):.2f}" for v in row))
        for name, row in zip(sensitivity.names, sensitivity.right_singular_vectors, strict=True)
    ]
    for line in format_table(vectors, "<" + ">" * len(sensitivity.names)):
        logger.info("    %s", line)
    logger.info("  %-18s%.3g", "condition number", sensitivity.condition_number)
