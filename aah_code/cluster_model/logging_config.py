"""
Logging configuration for cluster model calculations.
Control verbosity via environment variable or direct configuration.
"""
import os
import logging
import sys


# Configure logging level from environment variable
VERBOSE = os.environ.get('CLUSTER_VERBOSE', '').lower() in ('true', '1', 'yes')
DEBUG = os.environ.get('CLUSTER_DEBUG', '').lower() in ('true', '1', 'yes')

# Set up logger as singleton
logger = logging.getLogger('cluster_model')

# Only configure if not already configured
if not logger.handlers:
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set level based on environment variables
    if DEBUG:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
    elif VERBOSE:
        logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
        console_handler.setLevel(logging.WARNING)
    
    # Create formatter
    formatter = logging.Formatter('%(levelname)-8s: %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)


def get_logger():
    """Get the configured logger instance."""
    return logger


def set_verbose(enabled=True):
    """Enable or disable verbose output programmatically."""
    global logger
    if enabled:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)


def set_debug(enabled=True):
    """Enable or disable debug output programmatically."""
    global logger
    if enabled:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)


# Convenience functions for different log levels
def debug(msg, *args, **kwargs):
    """Log a debug message."""
    logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Log an info message."""
    logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log a warning message."""
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log an error message."""
    logger.error(msg, *args, **kwargs)