"""Setup a logger to be used in all modules in the library.

To use the logger, import it in any module and use it as follows:
    
    ```
    from aah.log import logger
    logger.info("Info message")
    logger.warning("Warning message")
    ```
"""

import logging
import os
from logging.config import dictConfig
from pathlib import Path

DEFAULT_LOGFILE = Path(__file__).resolve().parent.parent / "logs" / "logs.log"


def setup_logger(logfile: Path = DEFAULT_LOGFILE, log_level: str = None) -> logging.Logger:
    """Setup a logger to be used in all modules in the library.

    Sets up logging configuration with a console handler and a file handler.
    Console handler logs messages with INFO level, file handler logs WARNING level.
    The root logger is configured to use both handlers.

    Args:
        logfile: Path to the log file
        log_level: Override logging level (checks LOGLEVEL env var if None)

    Returns:
        logging.Logger: A configured logger object.

    Example:
        >>> logger = setup_logger()
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
    """
    # Determine log level from parameter, environment variable, or default
    if log_level is None:
        log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
    else:
        log_level = log_level.upper()
    
    # Validate log level
    if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        log_level = 'INFO'
    
    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": log_level,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(logfile),
                "formatter": "default",
                "level": "WARNING",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": log_level,
        },
    }

    dictConfig(logging_config)
    return logging.getLogger()


logger = setup_logger()
