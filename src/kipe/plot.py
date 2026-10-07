"""Plots of the estimation history: ``kipe plot``.

One subplot per parameter: the estimate after each assimilation step with its physical 1σ
range, over the steps of all outer iterations. The history is read from ``history.csv``, which
the estimation writes as it runs, so a running estimation can be plotted too.

Needs matplotlib, an optional dependency: ``pip install 'kipe[plot]'``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kipe._types import NDArray_f64
from kipe.forward_solver import Parameters

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class PlotError(Exception):
    """The history cannot be plotted, e.g., because it does not exist (yet)."""


@dataclass(frozen=True)
class History:
    """The estimation history, as read from ``history.csv``."""

    names: list[str]
    """Names of the estimated parameters."""

    iteration: NDArray_f64
    """Index of the outer iteration of each step, shape ``(k,)``."""

    parameters: NDArray_f64
    """The estimate after each step, as physical values, shape ``(k, p)``."""

    lower: NDArray_f64
    """Lower end of the physical 1σ range, shape ``(k, p)``."""

    upper: NDArray_f64
    """Upper end of the physical 1σ range, shape ``(k, p)``."""


def read_history(path: str | Path) -> History:
    """Read an estimation history from ``history.csv``.

    Args:
        path: path to ``history.csv``

    Returns:
        the history

    Raises:
        PlotError: if the file does not exist or holds no steps yet
    """
    try:
        with open(path) as f:
            header = f.readline().strip().split(",")
            rows = f.readlines()
    except OSError as err:
        raise PlotError(f"no estimation history at {path}; run kipe estimation first") from err
    if not rows:
        raise PlotError(f"the estimation history {path} has no steps yet")
    data = np.loadtxt(rows, delimiter=",", ndmin=2)

    names = [column for column in header[2:] if f"theta_{column}" in header]
    column = {name: i for i, name in enumerate(header)}

    return History(
        names=names,
        iteration=data[:, column["iteration"]],
        parameters=data[:, [column[name] for name in names]],
        lower=data[:, [column[f"lower_{name}"] for name in names]],
        upper=data[:, [column[f"upper_{name}"] for name in names]],
    )


def plot_histories(
    figure: "Figure", histories: dict[str, History], truth: Parameters | None = None
) -> None:
    """Plot estimation histories into a figure, one subplot per parameter.

    Args:
        figure: the figure to draw into
        histories: label -> history; all must estimate the same parameters
        truth: true values, drawn as reference lines (e.g., in a twin experiment)

    Raises:
        PlotError: if the histories estimate different parameters
    """
    names = next(iter(histories.values())).names
    if any(history.names != names for history in histories.values()):
        raise PlotError("the histories estimate different parameters and cannot be compared")

    axes = figure.subplots(1, len(names), squeeze=False)[0]
    for j, (ax, name) in enumerate(zip(axes, names, strict=True)):
        for k, (label, history) in enumerate(histories.items()):
            steps = np.arange(len(history.iteration))
            color = f"C{k}"
            ax.fill_between(
                steps, history.lower[:, j], history.upper[:, j], color=color, alpha=0.2, lw=0
            )
            ax.plot(steps, history.parameters[:, j], color=color, label=label)

        # starts of the outer iterations, after the first
        first = next(iter(histories.values()))
        for start in np.flatnonzero(np.diff(first.iteration)) + 1:
            ax.axvline(start, color="0.7", lw=0.8)

        if truth is not None and name in truth:
            ax.axhline(truth[name], color="k", ls="--", lw=1, label="truth")

        ax.set_title(name)
        ax.set_xlabel("assimilation step")

    if len(histories) > 1 or truth is not None:
        axes[0].legend()
