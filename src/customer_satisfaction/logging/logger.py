"""Application logging configuration."""

import logging


def get_logger(name: str) -> logging.Logger:
    "Create and configure a logger instance."

    logger = logging.getLogger(name)

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    return logger
