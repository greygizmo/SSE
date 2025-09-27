import logging
from uuid import uuid4

from gosales.utils.logger import get_logger


def test_get_logger_reuses_stream_handler():
    logger_name = f"test_logger_{uuid4()}"

    first_logger = get_logger(logger_name)
    second_logger = get_logger(logger_name)

    assert first_logger is second_logger

    stream_handlers = [
        handler for handler in first_logger.handlers if isinstance(handler, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1

    handler = stream_handlers[0]
    expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert handler.formatter is not None
    assert handler.formatter._fmt == expected_format

    for existing_handler in list(first_logger.handlers):
        first_logger.removeHandler(existing_handler)
