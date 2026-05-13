"""Logging utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"


def get_logger(name: str, log_path: str | Path) -> logging.Logger:
    """Create a logger that writes to stdout and a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if getattr(logger, "_configured_log_path", None) == str(log_path):
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_path = Path(log_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger._configured_log_path = str(log_path)  # type: ignore[attr-defined]
    return logger
