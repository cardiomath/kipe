"""Text formatting for the command line and log output."""

from collections.abc import Sequence


def format_table(rows: Sequence[Sequence[str]], align: str | None = None) -> list[str]:
    """Align the cells of a table in columns, two spaces apart.

    Args:
        rows: the rows, the first one is the header; shorter rows are padded with empty cells
        align: one character per column, ``"<"`` (left) or ``">"`` (right); all left if not
            given

    Returns:
        the lines of the table, without trailing whitespace
    """
    n = max(len(row) for row in rows)
    padded = [list(row) + [""] * (n - len(row)) for row in rows]
    widths = [max(len(row[i]) for row in padded) for i in range(n)]
    alignments = align or "<" * n

    return [
        "  ".join(
            f"{cell:{a}{w}}" for cell, a, w in zip(row, alignments, widths, strict=True)
        ).rstrip()
        for row in padded
    ]
