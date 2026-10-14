"""Fixtures for all tests."""

import logging
from collections.abc import Iterator

import pytest


@pytest.fixture
def kipe_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """pytest's ``caplog``, also capturing the messages of ``kipe`` on the command line.

    ``kipe synthesis`` and ``kipe estimation`` keep their messages from propagating to the root
    logger, where ``caplog`` captures them; here, ``caplog`` listens on the ``kipe`` logger
    directly.
    """
    logger = logging.getLogger("kipe")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)
