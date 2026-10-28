"""Plots of the estimation history, ``kipe plot-estimation``, and of the sensitivity
analysis, ``kipe plot-sensitivity`` (:func:`plot_sensitivity`).

The estimation history:

One subplot per parameter: the estimate after each assimilation step with its physical 1σ
range, over physical time. The outer iterations are side by side, each over its own time axis,
from the initial state to the last assimilation time, so that steps at irregular times are
where they belong. The history is read from ``history.csv``, which the estimation writes as it
runs, so a running estimation can be plotted too. If the run's plan is known from its
provenance (:mod:`kipe.provenance`), the x axis spans the whole run from the start, with all
outer iterations.

Draws with matplotlib (:func:`plot_histories`) or as text in the terminal with plotext
(:func:`render_histories`), e.g., to follow an estimation on a cluster (:func:`watch_histories`).
"""

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ruamel.yaml import YAML, YAMLError

from kipe._types import NDArray_bool, NDArray_f64
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

    time: NDArray_f64
    """Time of each step, shape ``(k,)``; each outer iteration starts at the initial state's."""

    parameters: NDArray_f64
    """The estimate after each step, as physical values, shape ``(k, p)``."""

    lower: NDArray_f64
    """Lower end of the physical 1σ range, shape ``(k, p)``."""

    upper: NDArray_f64
    """Upper end of the physical 1σ range, shape ``(k, p)``."""

    end_time: float | None = None
    """The last assimilation time of each outer iteration, from the run's plan; None if
    unknown."""

    iterations: int | None = None
    """Number of outer iterations, from the run's plan; None if unknown."""

    @property
    def start_time(self) -> float:
        """Time of the initial state, at which each outer iteration starts."""
        return float(self.time[0])

    @property
    def span(self) -> float:
        """Time span of an outer iteration if the run's plan is known, else reached so far."""
        end = float(self.time.max()) if self.end_time is None else self.end_time
        return end - self.start_time

    @property
    def iteration_count(self) -> int:
        """Number of outer iterations if the run's plan is known, else reached so far."""
        reached = int(self.iteration.max()) + 1
        return reached if self.iterations is None else max(self.iterations, reached)


def read_history(path: str | Path) -> History:
    """Read an estimation history from ``history.csv``, and the run's plan next to it.

    The plan, the last assimilation time and the number of outer iterations, comes from the
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

    end_time, iterations = _read_plan(Path(path).parent)

    return History(
        names=names,
        iteration=data[:, column["iteration"]],
        time=data[:, column["time"]],
        parameters=data[:, [column[name] for name in names]],
        lower=data[:, [column[f"lower_{name}"] for name in names]],
        upper=data[:, [column[f"upper_{name}"] for name in names]],
        end_time=end_time,
        iterations=iterations,
    )


def _read_plan(directory: Path) -> tuple[float | None, int | None]:
    """Read the last assimilation time and the number of outer iterations of a run.

    Args:
        directory: output directory of the estimation

    Returns:
        the last assimilation time and the iterations; None, None if the run's provenance is
        missing, e.g., for older runs
    """
    try:
        run = YAML(typ="safe").load(directory / "run.yaml")
        study = YAML(typ="safe").load(directory / "study.yaml")
        return float(run["assimilation_times"][-1]), int(study["estimation"]["iterations"])
    except (OSError, YAMLError, KeyError, IndexError, TypeError):
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
    timeline = _Timeline.of(histories)
    axes = figure.subplots(1, len(names), squeeze=False)[0]
    for j, (ax, name) in enumerate(zip(axes, names, strict=True)):
        for k, (label, history) in enumerate(histories.items()):
            color = f"C{k}"
            for i, (x, rows) in enumerate(timeline.segments(history)):
                ax.fill_between(
                    x,
                    history.lower[rows, j],
                    history.upper[rows, j],
                    color=color,
                    alpha=0.2,
                    lw=0,
                )
                ax.plot(
                    x, history.parameters[rows, j], color=color, label=label if i == 0 else None
                )

        for boundary in timeline.boundaries():  # between the outer iterations
            ax.axvline(boundary, color="0.7", lw=0.8)

        if truth is not None and name in truth:
            ax.axhline(truth[name], color="k", ls="--", lw=1, label="truth")

        ax.set_title(_title(name, histories, truth))
        ax.set_xlabel("time")
        if timeline.span > 0:
            ax.set_xlim(timeline.start, timeline.end)
            ax.set_xticks(*timeline.ticks(bins=max(2, 6 // timeline.iterations)))
            if timeline.iterations > 1:  # above the plot, so that they cover nothing
                top = ax.secondary_xaxis("top")
                top.set_xticks(*timeline.labels())
                top.tick_params(length=0, labelsize="small", labelcolor="0.4")

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
    Several histories get a legend below the plot, and a last line explains the lines: the
    gray 1σ range, with all iterations the green vertical lines at the starts of the outer
    iterations, and with a truth its reference line.

    By default, only the current (latest) outer iteration is drawn, which zooms in on where the
    estimate converges, over the time of that iteration.

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

    # a legend with several histories; the truth is in the key to the lines
    legend = [(label, _color(k)) for k, label in enumerate(histories)]
    has_truth = truth is not None and any(name in truth for name in names)
    columns = max(1, min(columns, len(names)))
    grid_rows = -(-len(names) // columns)
    legend_rows = (len(legend) if len(legend) > 1 else 0) + 1  # plus the key to the line styles

    plotext.terminal.limit(height=False)  # taller than the terminal is fine, it scrolls
    figure = plotext.figure
    figure.clear()
    figure.subplots(grid_rows, columns)
    figure.theme("simple")  # the terminal's own colors; after subplots(), to reach them too
    plot_rows = _ROWS_PER_PARAMETER * grid_rows
    if height is not None:
        plot_rows = max(plot_rows, height - legend_rows)
    figure.plot_size(None, plot_rows)

    # all lines are signals: plotext's own reference lines paint a white background
    def draw(
        subplot: Any, x: Sequence[float], y: Sequence[float], rgb: tuple[int, int, int]
    ) -> None:
        marker = plotext.marker("braille", pixel=plotext.pixel(foreground=rgb))
        subplot.draw(subplot.signal(x, y, marker=marker).lines(True))

    # plotext has no subplots in a 1 x 1 grid: then the figure itself is the plot
    def panel(j: int) -> Any:
        return figure.subplot(j // columns + 1, j % columns + 1) if len(names) > 1 else figure

    # all outer iterations side by side, or the current one alone, over its own time
    timeline = _Timeline.of(histories)
    if not all_iterations:
        timeline = replace(timeline, iterations=1)

    def segments(history: History) -> list[tuple[NDArray_f64, NDArray_bool]]:
        if all_iterations:
            return timeline.segments(history)
        rows = history.iteration == history.iteration[-1]
        return [(history.time[rows], rows)]

    for j, name in enumerate(names):
        subplot = panel(j)
        low = min(float(history.lower[:, j].min()) for history in histories.values())
        high = max(float(history.upper[:, j].max()) for history in histories.values())

        # where lines share a cell, the last one drawn colors all its dots: first the reference
        # lines (the truth below the 1σ ranges, which jump across it at each outer iteration),
        # the estimates on top
        for boundary in timeline.boundaries():  # starts of the outer iterations
            draw(subplot, [boundary] * 2, [low, high], _GREEN)

        if truth is not None and name in truth:
            draw(subplot, [timeline.start, timeline.end], [truth[name]] * 2, _TRUTH)

        # each outer iteration is drawn on its own: the estimate jumps back to the initial guess
        # at its start, and a line across the jump would pile dots up vertically
        for history in histories.values():
            for x, rows in segments(history):
                draw(subplot, x.tolist(), history.lower[rows, j].tolist(), _GRAY)
                draw(subplot, x.tolist(), history.upper[rows, j].tolist(), _GRAY)

        for k, history in enumerate(histories.values()):
            for x, rows in segments(history):
                draw(subplot, x.tolist(), history.parameters[rows, j].tolist(), _color(k))

        subplot.title(_title(name, histories, truth))
        if timeline.span > 0:
            subplot.ruler(0).lim(timeline.start, timeline.end)
            bins = max(2, 8 // (timeline.iterations * columns))
            subplot.ruler(0).ticks(*timeline.ticks(bins=bins))

        if j < columns and timeline.span > 0 and timeline.iterations > 1:  # top row only
            top = subplot.ruler(0, "upper")
            top.lim(timeline.start, timeline.end)
            top.ticks(*timeline.labels(room=5 // columns))

    for j in range(len(names), grid_rows * columns):  # unused cells of the last row: no frame
        figure.subplot(j // columns + 1, j % columns + 1).axes(False)

    xlabel = "time"
    if not all_iterations:
        current = next(iter(histories.values()))
        reached = int(current.iteration[-1]) + 1
        xlabel = f"time in iteration {reached}/{current.iteration_count}"
    for j in range(len(names)):  # the x axis label under the last plot of each column
        if j + columns >= len(names):
            panel(j).label(xlabel, axis=0)

    lines = [figure.build().string(colorless=not color)]

    def entry(label: str, rgb: tuple[int, int, int]) -> str:
        r, g, b = rgb
        return f"\x1b[38;2;{r};{g};{b}m━━\x1b[0m {label}" if color else f"━━ {label}"

    if len(legend) > 1:  # below the plot, so that it covers nothing
        lines.extend(entry(label, rgb) for label, rgb in legend)

    key = [entry("1σ range", _GRAY)]
    if timeline.boundaries():
        key.append(entry("starts of iterations", _GREEN))
    if has_truth:
        key.append(entry("truth", _TRUTH))
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


@dataclass(frozen=True)
class _Timeline:
    """The x axis: the outer iterations side by side, each over the same span of time.

    Iteration ``i`` covers ``[start + i span, start + (i + 1) span]``; a step at time ``t`` of
    iteration ``i`` is at ``t + i span``.
    """

    start: float
    """Time of the initial state."""

    span: float
    """Time span of each outer iteration."""

    iterations: int
    """Number of outer iterations on the axis."""

    @classmethod
    def of(cls, histories: Mapping[str, History]) -> "_Timeline":
        """Return the timeline that covers all histories: the longest run, if planned.

        Args:
            histories: label -> history

        Returns:
            the timeline
        """
        start = min(history.start_time for history in histories.values())
        end = max(history.start_time + history.span for history in histories.values())
        iterations = max(history.iteration_count for history in histories.values())
        return cls(start, end - start, iterations)

    @property
    def end(self) -> float:
        """The end of the x axis."""
        return self.start + self.iterations * self.span

    def segments(self, history: History) -> list[tuple[NDArray_f64, NDArray_bool]]:
        """Return the positions and rows of a history's steps, per outer iteration.

        Each outer iteration is drawn as a line of its own, so that its first step does not
        connect to the previous iteration's last.

        Args:
            history: the history

        Returns:
            for each outer iteration reached: the positions of its steps on the x axis, and
            the mask of its rows in the history
        """
        segments = []
        for i in np.unique(history.iteration):
            rows = history.iteration == i
            segments.append((history.time[rows] + i * self.span, rows))
        return segments

    def boundaries(self) -> list[float]:
        """Return the boundaries between the outer iterations on the x axis."""
        return [self.start + i * self.span for i in range(1, self.iterations)]

    def ticks(self, bins: int) -> tuple[list[float], list[str]]:
        """Return the ticks of the x axis: the time, restarting in each outer iteration.

        Args:
            bins: the number of intervals between the ticks of an iteration, at least

        Returns:
            the positions of the ticks and their labels
        """
        step = _nice_step(self.span / bins)
        end = self.start + self.span
        # the multiples of the step within an iteration, short of its end; integers, since
        # ceil(-1e-9) = -0.0, labeled "-0"
        first, stop = (int(np.ceil(t / step - 1e-9)) for t in (self.start, end))
        times = step * np.arange(first, stop, dtype=float)

        positions: list[float] = []
        labels: list[str] = []
        for i in range(self.iterations):  # the end of an iteration is the start of the next
            for t in times:
                positions.append(float(t + i * self.span))
                labels.append(f"{t:.6g}")

        # the end of the last iteration, unless too close to the tick before it
        if not times.size or end - times[-1] >= step / 4:
            positions.append(self.end)
            labels.append(f"{end:.6g}")
        return positions, labels

    def labels(self, room: int = 5) -> tuple[list[float], list[str]]:
        """Return the labels of the outer iterations, at their centers on the x axis.

        Args:
            room: how many labels "iteration i" fit across the axis; with more iterations,
                the labels are just their numbers

        Returns:
            the positions of the labels and the labels
        """
        positions = [self.start + (i + 0.5) * self.span for i in range(self.iterations)]
        prefix = "iteration " if self.iterations <= room else ""
        return positions, [f"{prefix}{i + 1}" for i in range(self.iterations)]


def _nice_step(step: float) -> float:
    """Return the smallest step of 1, 2, or 5 times a power of 10 that is at least ``step``.

    Args:
        step: the smallest step

    Returns:
        the nice step
    """
    magnitude = 10.0 ** np.floor(np.log10(step))
    return float(
        next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= step * (1 - 1e-9))
    )


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
