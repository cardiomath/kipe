"""Plots of the estimation history: ``kipe plot``.

One subplot per parameter: the estimate after each assimilation step with its physical 1σ
range, over the steps of all outer iterations. The history is read from ``history.csv``, which
the estimation writes as it runs, so a running estimation can be plotted too. If the run's
plan is known from its provenance (:mod:`kipe.provenance`), the x axis spans the whole run
from the start, with all outer iterations marked.

Draws with matplotlib (:func:`plot_histories`) or as text in the terminal with plotext
(:func:`render_histories`), e.g., to follow an estimation on a cluster (:func:`watch_histories`).
Both are optional dependencies: ``pip install 'kipe[plot]'``.
"""

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from ruamel.yaml import YAML, YAMLError

from kipe._types import NDArray_f64
from kipe.forward_solver import Parameters

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# matplotlib's first colors, so that terminal and figure agree
_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
_GRAY = (128, 128, 128)
_TRUTH = (227, 119, 194)  # matplotlib's pink, not used for the estimates
_ROWS_PER_PARAMETER = 12


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

    steps_per_iteration: int | None = None
    """Number of steps of each outer iteration, from the run's plan; None if unknown."""

    iterations: int | None = None
    """Number of outer iterations, from the run's plan; None if unknown."""

    @property
    def steps(self) -> int:
        """Number of steps of the whole run if its plan is known, else of the history so far."""
        if self.steps_per_iteration is None or self.iterations is None:
            return len(self.iteration)
        return max(self.steps_per_iteration * self.iterations, len(self.iteration))

    def iteration_starts(self) -> list[int]:
        """Return the steps at which the outer iterations after the first start.

        Returns:
            all of them if the run's plan is known, else those reached so far
        """
        if self.steps_per_iteration is None or self.iterations is None:
            return (np.flatnonzero(np.diff(self.iteration)) + 1).tolist()
        return [k * self.steps_per_iteration for k in range(1, self.iterations)]


def read_history(path: str | Path) -> History:
    """Read an estimation history from ``history.csv``, and the run's plan next to it.

    The plan, the number of steps per outer iteration and of iterations, comes from the
    provenance of the run: the assimilation times in ``run.yaml``, the iterations in
    ``study.yaml`` (the options the run started with).

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
    if rows and not rows[-1].endswith("\n"):  # the estimation is writing this row right now
        rows.pop()
    if not rows:
        raise PlotError(f"the estimation history {path} has no steps yet")
    data = np.loadtxt(rows, delimiter=",", ndmin=2)

    names = [column for column in header[2:] if f"theta_{column}" in header]
    column = {name: i for i, name in enumerate(header)}

    steps_per_iteration, iterations = _read_plan(Path(path).parent)

    return History(
        names=names,
        iteration=data[:, column["iteration"]],
        parameters=data[:, [column[name] for name in names]],
        lower=data[:, [column[f"lower_{name}"] for name in names]],
        upper=data[:, [column[f"upper_{name}"] for name in names]],
        steps_per_iteration=steps_per_iteration,
        iterations=iterations,
    )


def _read_plan(directory: Path) -> tuple[int | None, int | None]:
    """Read the number of steps per outer iteration and of iterations of a run.

    Args:
        directory: output directory of the estimation

    Returns:
        the steps per iteration (the assimilation times and the initial state) and the
        iterations; None, None if the run's provenance is missing, e.g., for older runs
    """
    try:
        run = YAML(typ="safe").load(directory / "run.yaml")
        study = YAML(typ="safe").load(directory / "study.yaml")
        return len(run["assimilation_times"]) + 1, int(study["estimation"]["iterations"])
    except (OSError, YAMLError, KeyError, TypeError):
        return None, None


def plot_histories(
    figure: "Figure", histories: dict[str, History], truth: Parameters | None = None
) -> None:
    """Plot estimation histories into a figure, one subplot per parameter.

    With a single history, each subplot title shows the latest estimate and range; with a
    truth, also the true value.

    Args:
        figure: the figure to draw into
        histories: label -> history; all must estimate the same parameters
        truth: true values, drawn as reference lines (e.g., in a twin experiment)

    Raises:
        PlotError: if the histories estimate different parameters
    """
    names = _common_names(histories)
    last = _last_step(histories)
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
        for start in first.iteration_starts():
            ax.axvline(start, color="0.7", lw=0.8)

        if truth is not None and name in truth:
            ax.axhline(truth[name], color="k", ls="--", lw=1, label="truth")

        ax.set_title(_title(name, histories, truth))
        ax.set_xlabel("assimilation step")
        if last > 0:
            ax.set_xlim(0, last)

    if len(histories) > 1 or truth is not None:
        axes[0].legend()


def render_histories(
    histories: Mapping[str, History],
    truth: Parameters | None = None,
    height: int | None = None,
    color: bool = True,
) -> str:
    """Render estimation histories as text for the terminal, one subplot per parameter.

    Like :func:`plot_histories`, with the 1σ range drawn as gray lines below and above the
    estimate. With a single history, each subplot title shows the latest estimate and range.
    Several histories or a truth get a legend below the plot.

    Args:
        histories: label -> history; all must estimate the same parameters
        truth: true values, drawn as reference lines (e.g., in a twin experiment)
        height: number of lines, legend included; if not given, 12 per parameter
        color: whether to color the output with ANSI escape codes

    Returns:
        the rendered plot

    Raises:
        PlotError: if plotext is not installed, or the histories estimate different parameters
    """
    plotext = _import_plotext()
    names = _common_names(histories)

    # as in plot_histories: a legend with several histories or a truth
    legend = [(label, _color(k)) for k, label in enumerate(histories)]
    if truth is not None and any(name in truth for name in names):
        legend.append(("truth", _TRUTH))
    legend_rows = len(legend) if len(legend) > 1 else 0

    plotext.terminal.limit(height=False)  # taller than the terminal is fine, it scrolls
    figure = plotext.figure
    figure.clear()
    figure.subplots(len(names), 1)
    figure.theme("simple")  # the terminal's own colors; after subplots(), to reach them too
    if height is None:
        figure.plot_size(None, _ROWS_PER_PARAMETER * len(names))
    else:
        figure.plot_size(None, height - legend_rows)

    # all lines are signals: plotext's own reference lines paint a white background
    def draw(
        subplot: Any, x: Sequence[float], y: Sequence[float], rgb: tuple[int, int, int]
    ) -> None:
        marker = plotext.marker("braille", pixel=plotext.pixel(foreground=rgb))
        subplot.draw(subplot.signal(x, y, marker=marker).lines(True))

    last = _last_step(histories)
    for j, name in enumerate(names):
        subplot = figure.subplot(j + 1, 1)
        first = next(iter(histories.values()))
        low = min(float(history.lower[:, j].min()) for history in histories.values())
        high = max(float(history.upper[:, j].max()) for history in histories.values())

        # where lines share a cell, the last one drawn colors all its dots: first the reference
        # lines (the truth below the 1σ ranges, which jump across it at each outer iteration),
        # the estimates on top
        for start in first.iteration_starts():  # outer iterations
            draw(subplot, [float(start)] * 2, [low, high], _GRAY)

        if truth is not None and name in truth:
            draw(subplot, [0.0, float(last)], [truth[name]] * 2, _TRUTH)

        for history in histories.values():
            steps = np.arange(len(history.iteration)).tolist()
            draw(subplot, steps, history.lower[:, j].tolist(), _GRAY)
            draw(subplot, steps, history.upper[:, j].tolist(), _GRAY)

        for k, history in enumerate(histories.values()):
            steps = np.arange(len(history.iteration)).tolist()
            draw(subplot, steps, history.parameters[:, j].tolist(), _color(k))

        subplot.title(_title(name, histories, truth))
        if last > 0:
            subplot.ruler(0).lim(0, last)

    figure.subplot(len(names), 1).label("assimilation step", axis=0)

    lines = [figure.build().string(colorless=not color)]
    if legend_rows:  # below the plot, so that it covers nothing
        for label, (r, g, b) in legend:
            lines.append(f"\x1b[38;2;{r};{g};{b}m━━\x1b[0m {label}" if color else f"━━ {label}")

    return "\n".join(lines)


def watch_histories(
    paths: Mapping[str, Path], truth: Parameters | None = None, interval: float = 2.0
) -> None:
    """Redraw estimation histories in the terminal as they grow, until Ctrl+C.

    Fills the terminal; waits while a history does not exist yet or holds no steps.

    Args:
        paths: label -> path to ``history.csv``
        truth: true values, drawn as reference lines
        interval: seconds between redraws

    Raises:
        PlotError: if plotext is not installed, or the histories estimate different parameters
    """
    plotext = _import_plotext()
    color = sys.stdout.isatty()

    try:
        while True:
            _, rows = plotext.terminal.size(update=True)
            try:
                histories = {label: read_history(path) for label, path in paths.items()}
            except PlotError as err:
                text = f"kipe: {err}; waiting ..."
            else:
                text = render_histories(histories, truth, height=rows, color=color)

            plotext.terminal.clean(-1)  # clear the screen without flickering
            print(text, flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


def _common_names(histories: Mapping[str, History]) -> list[str]:
    """Return the names of the parameters that all histories estimate.

    Args:
        histories: label -> history

    Returns:
        names of the estimated parameters

    Raises:
        PlotError: if the histories estimate different parameters
    """
    names = next(iter(histories.values())).names
    if any(history.names != names for history in histories.values()):
        raise PlotError("the histories estimate different parameters and cannot be compared")

    return names


def _last_step(histories: Mapping[str, History]) -> int:
    """Return the last step of the x axis: of the longest run, or history if not planned.

    Args:
        histories: label -> history

    Returns:
        the last step
    """
    return max(history.steps for history in histories.values()) - 1


def _title(name: str, histories: Mapping[str, History], truth: Parameters | None) -> str:
    """Return the subplot title: with a single history, also the latest estimate and range.

    The true value is added as a number, since the plot hides the truth line wherever an
    estimate or a 1σ range covers it.

    Args:
        name: parameter name
        histories: label -> history
        truth: true values, if any

    Returns:
        the title
    """
    title = name
    if len(histories) == 1:
        (history,) = histories.values()
        j = history.names.index(name)
        value, lower, upper = (
            history.parameters[-1, j],
            history.lower[-1, j],
            history.upper[-1, j],
        )
        title += f" = {value:.4g}  [{lower:.4g}, {upper:.4g}]"

    if truth is not None and name in truth:
        title += f"  truth {truth[name]:.4g}"

    return title


def _color(k: int) -> tuple[int, int, int]:
    """Return the color of the k-th history, as in matplotlib's default color cycle.

    Args:
        k: index of the history

    Returns:
        RGB color
    """
    return _COLORS[k % len(_COLORS)]


def _import_plotext() -> ModuleType:
    """Import plotext, an optional dependency.

    Returns:
        the plotext module

    Raises:
        PlotError: if plotext is not installed
    """
    try:
        import plotext
    except ImportError as err:
        raise PlotError("kipe plot --terminal needs plotext: pip install 'kipe[plot]'") from err

    return plotext
