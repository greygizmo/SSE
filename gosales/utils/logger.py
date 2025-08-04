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

    # Create a handler
    handler = logging.StreamHandler(sys.stdout)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the formatter for the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger
