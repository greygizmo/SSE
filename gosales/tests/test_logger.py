import logging
import sys
from uuid import uuid4

from gosales.utils.logger import get_logger


def _cleanup_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def test_get_logger_reuses_stream_handler():
    logger_name = f"test_logger_{uuid4()}"

    first_logger = get_logger(logger_name)
    second_logger = get_logger(logger_name)

    assert first_logger is second_logger

    stream_handlers = [
        handler
        for handler in first_logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1

    handler = stream_handlers[0]
    expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert handler.stream is sys.stdout
    assert handler.formatter is not None
    assert handler.formatter._fmt == expected_format

    _cleanup_handlers(first_logger)


def test_get_logger_ignores_non_stdout_stream_handlers():
    logger_name = f"test_logger_{uuid4()}"
    logger = logging.getLogger(logger_name)

    stderr_handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(stderr_handler)

    first_logger = get_logger(logger_name)
    second_logger = get_logger(logger_name)

    assert first_logger is second_logger is logger

    stream_handlers = [
        handler
        for handler in first_logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    stdout_handlers = [
        handler for handler in stream_handlers if getattr(handler, "stream", None) is sys.stdout
    ]

    assert stderr_handler in stream_handlers
    assert len(stdout_handlers) == 1

    stdout_handler = stdout_handlers[0]
    expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert stdout_handler.formatter is not None
    assert stdout_handler.formatter._fmt == expected_format

    _cleanup_handlers(first_logger)
