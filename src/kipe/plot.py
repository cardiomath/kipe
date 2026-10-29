"""Plots of the estimation history, ``kipe plot-estimation``, and of the sensitivity
analysis, ``kipe plot-sensitivity`` (:func:`plot_sensitivity`).

The estimation history:

One subplot per parameter: the estimate after each assimilation step with its physical 1σ
range, over the steps of all outer iterations. The history is read from ``history.csv``, which
the estimation writes as it runs, so a running estimation can be plotted too. If the run's
plan is known from its provenance (:mod:`kipe.provenance`), the x axis spans the whole run
from the start, with all outer iterations marked.

Draws with matplotlib (:func:`plot_histories`) or as text in the terminal with plotext
(:func:`render_histories`), e.g., to follow an estimation on a cluster (:func:`watch_histories`).
"""

import itertools
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ruamel.yaml import YAML, YAMLError

from kipe._types import NDArray_f64
from kipe.forward_solver import Parameters

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from kipe.sensitivity import Sensitivity


# matplotlib's first colors, so that terminal and figure agree
_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
_GRAY = (128, 128, 128)
_GREEN = (144, 238, 144)  # starts of the outer iterations
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


def plot_sensitivity(figure: "Figure", sensitivity: "Sensitivity") -> None:
    r"""Plot the result of a sensitivity analysis into a figure, in three panels.

    The singular values :math:`s_i` of the noise-weighted sensitivities over their index, with
    the condition number, and the noise floor :math:`s = 1` if it is within their range: below
    it, a unit change of :math:`\theta` along a singular vector changes the predicted data by
    less than the noise;
    the right singular vectors :math:`\bs{v}_i`, one row each, from the best to the worst
    determined combination of the parameters, each with its largest entry positive (the sign
    of a whole vector is arbitrary, the signs of its entries relative to each other are not);
    and the Cramér-Rao correlations, unless the Fisher information is singular.

    The heatmaps order the parameters by identifiability: first the one that dominates
    :math:`\bs{v}_1`, then the one of the others that dominates :math:`\bs{v}_2`, and so on.
    Where each combination is mainly one parameter, the diagonal of the singular vectors is
    dominant, and the order is the ranking from the best to the worst determined parameter.

    Args:
        figure: the figure to draw into
        sensitivity: the result, e.g., from :func:`kipe.sensitivity.read_sensitivity`
    """
    values, vectors, correlations = figure.subplots(1, 3)
    index = np.arange(1, len(sensitivity.names) + 1)

    values.semilogy(index, sensitivity.singular_values, "o-")
    values.grid(which="major", color="0.85")
    values.grid(which="minor", axis="y", color="0.93")
    low, high = values.get_ylim()
    if low <= 1.0 <= high:  # only where it does not stretch the axis
        values.axhline(1.0, color="k", ls=":", lw=1, label="noise floor, $s = 1$")
        values.legend(fontsize="small")
    values.set_xticks(index)
    values.set_xlim(0.5, len(index) + 0.5)
    values.set_xlabel("$i$")
    values.set_ylabel("singular value $s_i$")
    values.set_title(f"singular values, κ = {sensitivity.condition_number:.3g}")

    vectors_by_row = _largest_positive(sensitivity.right_singular_vectors.T)  # row i: v_i
    order = _by_identifiability(vectors_by_row)
    names = [sensitivity.names[j] for j in order]

    _matrix(
        vectors,
        vectors_by_row[:, order],
        (-1.0, 1.0),
        "RdBu_r",
        names,
        [f"$v_{i}$" for i in index],
    )
    vectors.set_xlabel("parameters, by identifiability")
    vectors.set_title("right singular vectors")

    if np.all(np.isfinite(sensitivity.covariance)):
        correlation = sensitivity.correlation[np.ix_(order, order)]
        _matrix(correlations, correlation, (-1.0, 1.0), "RdBu_r", names, names)
        correlations.set_xlabel("parameters, by identifiability")
        correlations.set_title("Cramér-Rao correlations")
    else:
        correlations.axis("off")
        correlations.text(
            0.5,
            0.5,
            "the Fisher information is singular:\nno Cramér-Rao correlations",
            ha="center",
            va="center",
        )


def _by_identifiability(vectors: NDArray_f64) -> list[int]:
    """Return the parameters in the order of the singular vectors they dominate.

    Greedy, from the best determined vector to the worst: each takes the parameter of its
    largest magnitude among those not taken yet.

    Args:
        vectors: the right singular vectors, one per row, from the best determined

    Returns:
        the indices of the parameters, in this order
    """
    order: list[int] = []
    for vector in np.abs(vectors):
        remaining = [j for j in range(len(vector)) if j not in order]
        order.append(max(remaining, key=lambda j: vector[j]))
    return order


def _largest_positive(vectors: NDArray_f64) -> NDArray_f64:
    """Return vectors with their signs flipped so that the largest entry of each is positive.

    Args:
        vectors: the vectors, one per row

    Returns:
        the vectors, each multiplied by the sign of its entry of the largest magnitude
    """
    largest = vectors[np.arange(len(vectors)), np.argmax(np.abs(vectors), axis=1)]
    return vectors * np.where(largest < 0, -1.0, 1.0)[:, np.newaxis]


def _matrix(
    ax: "Axes",
    matrix: NDArray_f64,
    limits: tuple[float, float],
    cmap: str,
    columns: list[str],
    rows: list[str],
) -> None:
    """Draw a matrix as a heatmap, with its values in the cells.

    Args:
        ax: the axes to draw into
        matrix: the matrix, shape ``(len(rows), len(columns))``
        limits: the values at the ends of the color map
        cmap: name of the color map
        columns: labels of the columns
        rows: labels of the rows
    """
    image = ax.imshow(matrix, cmap=cmap, vmin=limits[0], vmax=limits[1])
    # in an inset at the right edge, so that it has the height of the heatmap
    colorbar = ax.figure.colorbar(image, cax=ax.inset_axes((1.05, 0.0, 0.05, 1.0)))
    colorbar.set_ticks(np.linspace(*limits, 5).tolist())
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    end = max(abs(limits[0]), abs(limits[1]))
    for (i, j), value in np.ndenumerate(matrix):
        color = "white" if abs(value) > 0.6 * end else "black"  # on the dark colors
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize="small", color=color)


def render_histories(
    histories: Mapping[str, History],
    truth: Parameters | None = None,
    height: int | None = None,
    color: bool = True,
    columns: int = 1,
    all_iterations: bool = False,
) -> str:
    """Render estimation histories as text for the terminal, one subplot per parameter.

    Like :func:`plot_histories`, with the 1σ range drawn as gray lines below and above the
    estimate. With a single history, each subplot title shows the latest estimate and range.
    Several histories or a truth get a legend below the plot, and a last line explains the
    lines: the colored estimate, the gray 1σ range and, with all iterations, the green vertical
    lines at the starts of the outer iterations.

    By default, only the current (latest) outer iteration is drawn, which zooms in on where the
    estimate converges; its x axis counts the steps of that iteration.

    Args:
        histories: label -> history; all must estimate the same parameters
        truth: true values, drawn as reference lines (e.g., in a twin experiment)
        height: number of lines, legend included; if not given, 12 per parameter; never less
            than that, so that the plots keep their resolution in a small terminal
        color: whether to color the output with ANSI escape codes
        columns: number of subplots per row; the parameters fill the grid row by row
        all_iterations: whether to draw all outer iterations instead of the current one

    Returns:
        the rendered plot

    Raises:
        PlotError: if the histories estimate different parameters
    """
    import plotext

    names = _common_names(histories)

    # as in plot_histories: a legend with several histories or a truth
    legend = [(label, _color(k)) for k, label in enumerate(histories)]
    if truth is not None and any(name in truth for name in names):
        legend.append(("truth", _TRUTH))
    columns = max(1, min(columns, len(names)))
    grid_rows = -(-len(names) // columns)
    legend_rows = (len(legend) if len(legend) > 1 else 0) + 1  # plus the key to the line styles

    plotext.terminal.limit(height=False)  # taller than the terminal is fine, it scrolls
    figure = plotext.figure
    figure.clear()
    figure.subplots(grid_rows, columns)
    figure.theme("simple")  # the terminal's own colors; after subplots(), to reach them too
    rows = _ROWS_PER_PARAMETER * grid_rows
    if height is not None:
        rows = max(rows, height - legend_rows)
    figure.plot_size(None, rows)

    # all lines are signals: plotext's own reference lines paint a white background
    def draw(
        subplot: Any, x: Sequence[float], y: Sequence[float], rgb: tuple[int, int, int]
    ) -> None:
        marker = plotext.marker("braille", pixel=plotext.pixel(foreground=rgb))
        subplot.draw(subplot.signal(x, y, marker=marker).lines(True))

    segments = {label: _segments(history) for label, history in histories.items()}
    if not all_iterations:  # the current iteration, counted from its first step
        segments = {
            label: [(list(range(len(steps))), sl) for steps, sl in parts[-1:]]
            for label, parts in segments.items()
        }
    if all_iterations:
        last = _last_step(histories)
    else:
        last = (
            max(
                history.steps_per_iteration or len(segments[label][0][0])
                for label, history in histories.items()
            )
            - 1
        )
    for j, name in enumerate(names):
        subplot = figure.subplot(j // columns + 1, j % columns + 1)
        first = next(iter(histories.values()))
        low = min(float(history.lower[:, j].min()) for history in histories.values())
        high = max(float(history.upper[:, j].max()) for history in histories.values())

        # where lines share a cell, the last one drawn colors all its dots: first the reference
        # lines (the truth below the 1σ ranges, which jump across it at each outer iteration),
        # the estimates on top
        for start in first.iteration_starts() if all_iterations else []:  # outer iterations
            draw(subplot, [float(start)] * 2, [low, high], _GREEN)

        if truth is not None and name in truth:
            draw(subplot, [0.0, float(last)], [truth[name]] * 2, _TRUTH)

        # each outer iteration is drawn on its own: the estimate jumps back to the initial guess
        # at its start, and a line across the jump would pile dots up vertically
        for label, history in histories.items():
            for steps, sl in segments[label]:
                draw(subplot, steps, history.lower[sl, j].tolist(), _GRAY)
                draw(subplot, steps, history.upper[sl, j].tolist(), _GRAY)

        for k, (label, history) in enumerate(histories.items()):
            for steps, sl in segments[label]:
                draw(subplot, steps, history.parameters[sl, j].tolist(), _color(k))

        subplot.title(_title(name, histories, truth))
        if last > 0:
            subplot.ruler(0).lim(0, last)

    for j in range(len(names), grid_rows * columns):  # unused cells of the last row: no frame
        figure.subplot(j // columns + 1, j % columns + 1).axes(False)

    xlabel = "assimilation step"
    if not all_iterations:
        current = next(iter(histories.values()))
        reached = int(current.iteration[-1]) + 1
        xlabel = f"step of iteration {reached}/{current.iterations or reached}"
    for j in range(len(names)):  # the x axis label under the last plot of each column
        if j + columns >= len(names):
            figure.subplot(j // columns + 1, j % columns + 1).label(xlabel, axis=0)

    lines = [figure.build().string(colorless=not color)]

    def entry(label: str, rgb: tuple[int, int, int]) -> str:
        r, g, b = rgb
        return f"\x1b[38;2;{r};{g};{b}m━━\x1b[0m {label}" if color else f"━━ {label}"

    if len(legend) > 1:  # below the plot, so that it covers nothing
        lines.extend(entry(label, rgb) for label, rgb in legend)

    key = [entry("1σ range", _GRAY)]
    if all_iterations:
        key.append(entry("starts of iterations", _GREEN))
    lines.append("   ".join(key))

    return "\n".join(lines)


def watch_histories(
    paths: Mapping[str, Path],
    truth: Parameters | None = None,
    interval: float = 2.0,
    columns: int = 1,
    all_iterations: bool = False,
) -> None:
    """Redraw estimation histories in the terminal as they grow, until Ctrl+C.

    Fills the terminal, but each plot keeps at least the height it has with ``--terminal``;
    waits while a history does not exist yet or holds no steps.

    Args:
        paths: label -> path to ``history.csv``
        truth: true values, drawn as reference lines
        interval: seconds between redraws
        columns: number of subplots per row
        all_iterations: whether to draw all outer iterations instead of the current one

    Raises:
        PlotError: if the histories estimate different parameters
    """
    import plotext

    color = sys.stdout.isatty()

    try:
        while True:
            _, rows = plotext.terminal.size(update=True)
            try:
                histories = {label: read_history(path) for label, path in paths.items()}
            except PlotError as err:
                text = f"kipe: {err}; waiting ..."
            else:
                text = render_histories(
                    histories,
                    truth,
                    height=rows,
                    color=color,
                    columns=columns,
                    all_iterations=all_iterations,
                )

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


def _segments(history: History) -> list[tuple[list[int], slice]]:
    """Split a history into its outer iterations.

    Args:
        history: the history

    Returns:
        for each outer iteration, its step numbers and the slice of its rows
    """
    changes = (np.flatnonzero(np.diff(history.iteration)) + 1).tolist()
    bounds = [0, *changes, len(history.iteration)]
    return [(np.arange(a, b).tolist(), slice(a, b)) for a, b in itertools.pairwise(bounds)]


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
