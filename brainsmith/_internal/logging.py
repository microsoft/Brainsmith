# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Simple logging configuration using Python's standard logging with Rich.

This module provides basic logging setup for Brainsmith using standard
Python logging levels (DEBUG, INFO, WARNING, ERROR) with Rich for pretty output.

Key principle: Use what exists. Python's logging module already provides
everything we need - don't reinvent it.

Usage:
    from brainsmith._internal.logging import setup_logging

    # In CLI setup
    setup_logging(level="info")

    # In application code
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Processing...")
    logger.error("Failed!")
"""

import logging
import warnings


def setup_logging(level: str = "warning") -> None:
    """Configure Python logging with Rich handler.

    Maps string level ('error', 'warning', 'info', 'debug') to logging constants.
    """
    from rich.logging import RichHandler

    level_map = {
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    log_level = level_map.get(level.lower(), logging.WARNING)

    root = logging.getLogger()
    root.setLevel(log_level)

    if not root.handlers:
        handler = RichHandler(
            rich_tracebacks=(log_level == logging.DEBUG),
            show_path=False,
            markup=True,
            show_time=False
        )
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(log_level)

    # Suppress noisy FINN/QONNX loggers unless in debug mode
    if log_level > logging.DEBUG:
        logging.getLogger('finn').setLevel(logging.ERROR)
        logging.getLogger('qonnx').setLevel(logging.ERROR)
        warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'^finn\..*')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'^qonnx\..*')
