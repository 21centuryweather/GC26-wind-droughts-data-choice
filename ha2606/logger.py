"""Logging configuration using loguru for the weasel package."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Time format for logging
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log directory
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Default log file path
DEFAULT_LOG_FILE = LOG_DIR / "weasel.log"


def elapsed_time(start_time: float, end_time: float) -> str:
    """
    Calculate and return the elapsed time in minutes between two timestamps.
    """
    return f"{(end_time - start_time) / 60:.4f}"


def now() -> str:
    """Return current timestamp as formatted string."""
    return datetime.now().strftime(TIME_FORMAT)


def setup_logger(
    log_file: Path | str = DEFAULT_LOG_FILE,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    rotation: str = "50 MB",
    retention: str = "30 days",
) -> logger:
    """
    Configure and return the loguru logger with file and console handlers.

    Args:
        log_file: Path to the log file.
        console_level: Minimum log level for console output.
        file_level: Minimum log level for file output.
        rotation: When to rotate the log file (e.g., "10 MB", "1 day").
        retention: How long to keep old log files.

    Returns:
        Configured loguru logger instance.
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stderr,
        level=console_level,
        format="<green>{time:" + TIME_FORMAT + "}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>L{line}</cyan> - "
        "<level>{message}</level>",
        colorize=True,
    )

    # File handler with rotation and retention
    logger.add(
        log_file,
        level=file_level,
        format="{time:" + TIME_FORMAT + "} | {level: <7} | {module}:{function}:L{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    return logger


# Initialize logger with default settings
setup_logger()


def get_logger():
    """
    Return the configured loguru logger instance.

    This function provides a simple interface to get the logger
    that's already configured by setup_logger().
    """
    return logger
