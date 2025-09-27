import logging
import sys

# ======================================================================================
#  Standard Logger
# ======================================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Initializes a standard logger with a custom format and color.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger instance.
    """

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create or reuse a handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = None
    for existing_handler in logger.handlers:
        if isinstance(existing_handler, logging.StreamHandler):
            handler = existing_handler
            break

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    # Ensure the handler uses the expected formatter configuration
    handler.setFormatter(formatter)

    return logger
