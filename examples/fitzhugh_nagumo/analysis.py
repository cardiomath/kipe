"""Template: the estimates of a study over its noise realizations, against the bounds."""

# %% [markdown]
# # The estimates of a study over its noise realizations
#
# A template to copy and adapt: it reads the estimates of all noise realizations of a study
# (`kipe.results.read_realizations`) and compares their spread with the filter's uncertainty
# and, if the study has a sensitivity analysis, with the Cramér-Rao and Bayesian bounds. All
# in physical units, which are comparable across realizations (see `kipe.results`).
#
# ## Run this study
#
# E.g., in this directory:
# ```bash
# kipe synthesis study_realizations.yaml
# kipe estimation study_realizations.yaml
# kipe sensitivity study_realizations.yaml   # optional: the bounds
# ```
#
# Then run this file as a script, `python analysis.py study_realizations.yaml [--save DIR]`,
# or cell by cell, e.g., in VS Code or in Jupyter (with jupytext).
#
# ## Understanding the results
#
# The empirical confidence intervals and ellipsoids should converge
# to the Cramér-Rao bound as the number of realizations grow. This will only happen
# as long as `estimation.iterations` is large enough,
# so that the effect of the parameter prior mean is negligible.
#
# Still, the filter's standard deviations (and covariances) should be smaller than the empirical
# standard deviation and covariances, while they should approximately agrees with the
# Bayesian Cramér-Rao bound.
#
# The correlations tell a similar story: the filter's should agree with the Bayesian bound,
# while the empirical correlations and pairwise ellipses should agree with the Cramér-Rao bound.

# %%
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from kipe.forward_solver import build_forward_solver
from kipe.options import load_study
from kipe.parameters import build_parameterization
from kipe.results import read_realizations
from kipe.sensitivity import read_sensitivity

parser = argparse.ArgumentParser(description="The estimates of a study's noise realizations.")
parser.add_argument("study", nargs="?", default="study_realizations.yaml", help="study file")
parser.add_argument("--save", metavar="DIR", help="save the figures to DIR, don't show them")
parser.add_argument("-f", help=argparse.SUPPRESS)  # Jupyter kernel connection file
args, _ = parser.parse_known_args()

# colors that stay distinguishable with color vision deficiencies, in a fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# %% [markdown]
# ## Reading the results

# %%
study = load_study(args.study)
result = read_realizations(study)
names = result.names
truth = build_forward_solver(study.forward_solver).nominal_parameters()
true_values = np.array([truth[name] for name in names])

empirical = np.cov(result.parameters, rowvar=False)  # over the realizations
filter_covariance = result.covariance.mean(axis=0)  # mean over the realizations

# the bounds, if the study has a sensitivity analysis: from theta to physical units
bounds: dict[str, np.ndarray] = {}
path = Path(study.output.path) / "sensitivity" / "sensitivity.npz"
if path.exists():
    sensitivity = read_sensitivity(path)
    assert study.parameters is not None
    nominal = build_parameterization(study.parameters, names).with_anchor(truth)
    J = nominal.derivative(sensitivity.theta)
    if np.all(np.isfinite(sensitivity.covariance)):
        bounds["Cramér-Rao"] = np.outer(J, J) * sensitivity.covariance
    bounds["Bayesian"] = np.outer(J, J) * sensitivity.posterior_covariance

print(f"{len(result.realizations)} realizations of {args.study}")

# %% [markdown]
# ## Summary
#
# The bias of the mean estimate, and the standard deviations: of the estimates over the
# realizations (empirical), the filter's (mean over the realizations), and the bounds. A
# filter standard deviation clearly below the empirical one means an overconfident filter.

# %%
covariances = {"empirical": empirical, "filter": filter_covariance} | bounds
covariances = {
    label: covariances[label]
    for label in ("empirical", "Cramér-Rao", "filter", "Bayesian")
    if label in covariances
}
stddevs = {label: np.sqrt(np.diag(covariance)) for label, covariance in covariances.items()}

header = f"{'parameter':<10}{'truth':>10}{'bias':>9}" + "".join(f"{s:>12}" for s in stddevs)
print(header + f"{'filter/emp.':>13}")
for i, name in enumerate(names):
    bias = (result.parameters[:, i].mean() - true_values[i]) / true_values[i]
    row = f"{name:<10}{true_values[i]:>10.4g}{bias:>9.2%}"
    row += "".join(f"{s[i]:>12.3g}" for s in stddevs.values())
    print(row + f"{stddevs['filter'][i] / stddevs['empirical'][i]:>13.2f}")

# %% [markdown]
# ## The estimates over the realizations
#
# Each point is the final estimate of one realization; the bar shows their mean and standard
# deviation, the line the truth.

# %%
figure_estimates, axes = plt.subplots(
    1, len(names), figsize=(3 * len(names), 3.5), layout="constrained"
)
rng = np.random.default_rng(0)
for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), names, strict=True)):
    values = result.parameters[:, i]
    ax.scatter(rng.uniform(-0.15, 0.15, values.size), values, s=16, color=COLORS[0],
               alpha=0.7, label="realizations")  # fmt: skip
    ax.errorbar(0.4, values.mean(), yerr=values.std(ddof=1), fmt="o", color=COLORS[1],
                capsize=4, label="mean ± std")  # fmt: skip
    ax.axhline(true_values[i], color="0.2", lw=1, ls="--", label="truth")
    ax.set(title=name, xticks=[], xlim=(-0.5, 0.8))
    ax.grid(axis="y", color="0.9")
np.atleast_1d(axes)[0].legend(fontsize="small")

# %% [markdown]
# ## Consistency: the spread against the filter's uncertainty and the bounds
#
# Standard deviations relative to the truth, one panel per parameter, each with its own scale:
# the parameters can be determined to very different degrees.

# %%
figure_consistency, axes = plt.subplots(
    1, len(names), figsize=(3 * len(names), 3.5), layout="constrained"
)
for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), names, strict=True)):
    for k, (label, s) in enumerate(stddevs.items()):
        ax.bar(k, 100 * s[i] / abs(true_values[i]), 0.8, color=COLORS[k], label=label)
    ax.set(title=name, xticks=[])
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
np.atleast_1d(axes)[0].set_ylabel("standard deviation (% of the truth)")
figure_consistency.legend(
    *np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="outside right upper"
)

# %% [markdown]
# ## Correlations
#
# Of the estimates over the realizations, of the filter (mean over the realizations), and of
# the bounds. The filter can have the right spreads but miss correlations. Where the prior
# matters, the filter's correlations are those of the Bayesian bound, which includes it, and
# the estimates' those of the Cramér-Rao bound (see the summary).


# %%
def correlation(covariance: np.ndarray) -> np.ndarray:
    """Return the correlations of a covariance matrix."""
    stddev = np.sqrt(np.diag(covariance))
    return covariance / np.outer(stddev, stddev)


correlations = {"empirical": correlation(empirical), "filter": correlation(filter_covariance)}
correlations |= {label: correlation(covariance) for label, covariance in bounds.items()}

n = len(names)
figure_correlations, axes = plt.subplots(
    1, len(correlations), figsize=(3.2 * len(correlations), 3), layout="constrained"
)
for ax, (label, matrix) in zip(np.atleast_1d(axes), correlations.items(), strict=True):
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize="small",
                color="white" if abs(value) > 0.6 else "black")  # fmt: skip
    ax.set(title=label, xticks=range(n), yticks=range(n), xticklabels=names, yticklabels=names)
figure_correlations.colorbar(image, ax=axes, shrink=0.8)

# %% [markdown]
# ## Pairs: the estimates and the covariance ellipses
#
# For each pair of parameters, the estimates and the 2σ ellipses of the covariances, all
# centered at the mean estimate so that their shapes compare: size and orientation together,
# i.e., also whether the filter or a bound misses a correlation.


# %%
def ellipse(center: np.ndarray, covariance: np.ndarray, color: str, label: str) -> Ellipse:
    """Return the 2σ ellipse of a 2 x 2 covariance matrix."""
    values, vectors = np.linalg.eigh(covariance)  # ascending
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    width, height = 4 * np.sqrt(values[::-1])  # 2σ on either side
    return Ellipse(
        center, width, height, angle=angle, fill=False, color=color, lw=1.5, label=label
    )


pairs = list(combinations(range(n), 2))
columns = min(3, len(pairs))
rows = -(-len(pairs) // columns)
figure_pairs, axes = plt.subplots(
    rows, columns, figsize=(3.4 * columns, 3.2 * rows), layout="constrained", squeeze=False
)
for ax, (i, j) in zip(axes.flat, pairs, strict=False):  # the last row may have room to spare
    ax.scatter(result.parameters[:, i], result.parameters[:, j], s=12, color="0.55",
               label="realizations")  # fmt: skip
    center = result.parameters[:, [i, j]].mean(axis=0)
    for k, (label, covariance) in enumerate(covariances.items()):
        ax.add_patch(ellipse(center, covariance[np.ix_([i, j], [i, j])], COLORS[k], label))
    ax.scatter(*center, s=36, facecolors="none", edgecolors=COLORS[0], linewidths=1.5, zorder=3)
    ax.plot(true_values[i], true_values[j], "x", color="0.1", ms=8, label="truth")
    ax.set(xlabel=names[i], ylabel=names[j])
    ax.grid(color="0.9")
    ax.set_axisbelow(True)
for ax in axes.flat[len(pairs) :]:
    ax.set_visible(False)
figure_pairs.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside right upper")

# %%
if args.save:
    directory = Path(args.save)
    directory.mkdir(parents=True, exist_ok=True)
    figures = ["estimates", "consistency", "correlations", "pairs"]
    for figure, name in zip(
        (figure_estimates, figure_consistency, figure_correlations, figure_pairs),
        figures,
        strict=True,
    ):
        figure.savefig(directory / f"{name}.png", dpi=150)
else:
    plt.show()
