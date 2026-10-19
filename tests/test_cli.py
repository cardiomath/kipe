"""Errors on the command line, on several ranks: printed once, or the run ends."""

import pytest

from kipe.cli import _report_error


class _Aborted(Exception):
    pass


class _Comm:
    """A communicator of two ranks, as one of them sees it after an error."""

    size = 2

    def __init__(self, rank: int, others_fail: bool) -> None:
        self.rank = rank
        self._others_fail = others_fail

    def Ibarrier(self) -> "_Request":
        return _Request(self._others_fail)

    def Abort(self, code: int) -> None:
        raise _Aborted(code)


class _Request:
    def __init__(self, complete: bool) -> None:
        self._complete = complete

    def Test(self) -> bool:
        return self._complete


@pytest.mark.parametrize(("rank", "printed"), [(0, "kipe: error: no data\n"), (1, "")])
def test_error_of_all_ranks(capsys, rank, printed):
    """All ranks have it: rank 0 prints it."""
    _report_error(ValueError("no data"), _Comm(rank, others_fail=True))
    assert capsys.readouterr().err == printed


def test_error_of_one_rank(capsys):
    """Only this rank has it: it prints it, with its rank, and aborts the run."""
    with pytest.raises(_Aborted):
        _report_error(ValueError("no data"), _Comm(1, others_fail=False), wait=0.05)
    assert capsys.readouterr().err == "kipe[1]: error: no data\n"
