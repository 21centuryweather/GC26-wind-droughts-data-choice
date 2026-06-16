import logging
from datetime import datetime


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def elapsed_time(start_time: float, end_time: float) -> str:
    """
    Calculate and return the elapsed time in minutes between two timestamps.
    """
    return f"{(end_time - start_time) / 60:.4f}"


def now():
    return datetime.now().strftime(TIME_FORMAT)


def apply_formatter(handler, log_level):
    """
    Apply a standard formatter to a log handler.
    """
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        "{asctime} | {levelname:7s} | {module}:{funcName}:L{lineno} - {message}",
        TIME_FORMAT,
        style="{",
    )
    handler.setFormatter(formatter)
    return handler


class RankFilter(logging.Filter):
    """Filter that only allows logging from rank 0 in distributed training."""
    def __init__(self, rank: int = 0):
        super().__init__()
        self.rank = rank
    
    def filter(self, record):
        return self.rank == 0


def get_logger(name: str, log_name: str = 'weasel.log', rank: int = None) -> logging.Logger:
    """
    Create and return a logger with specified name, adding file and console handlers.
    
    Args:
        name: Logger name
        log_name: Log file name
        rank: Distributed rank. If provided, only rank 0 will log; others get NullHandler.
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # If rank is provided and not 0, use NullHandler (silent)
    if rank is not None and rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger
    
    file_handler = apply_formatter(logging.FileHandler(log_name), logging.DEBUG)
    console_handler = apply_formatter(logging.StreamHandler(), logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
