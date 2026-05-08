# sentinel/utils/logging.py

import logging
import sys
from typing import Optional


_SENTINEL_LOGGER_NAME = "sentinel"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger namespaced under 'sentinel'.

    Usage:
        logger = get_logger(__name__)
        logger.info("Scraper started")
    """
    logger_name = f"{_SENTINEL_LOGGER_NAME}.{name}" if name else _SENTINEL_LOGGER_NAME
    return logging.getLogger(logger_name)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root sentinel logger with a clean formatter.
    Should be called once at startup in core.py.

    level: "DEBUG", "INFO", "WARNING", "ERROR"
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(_SENTINEL_LOGGER_NAME)
    logger.setLevel(numeric_level)

    if logger.handlers:
        return  # already configured, don't add duplicate handlers

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False