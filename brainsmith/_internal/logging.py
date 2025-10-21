# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Simple logging configuration using Python's standard logging with Rich.

This module provides basic logging setup for Brainsmith using standard
Python logging levels (DEBUG, INFO, WARNING, ERROR) with Rich for pretty output.

Key principle: Use what exists. Python's logging module already provides
everything we need - don't reinvent it.

Usage:
    from brainsmith._internal.logging import setup_logging, capture_finn_output

    # In CLI setup
    setup_logging(quiet=False, verbose=True, debug=False)

    # In application code
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Processing...")
    logger.error("Failed!")

    # For FINN builds
    with capture_finn_output():
        build_dataflow_cfg(model, config)
"""

import logging
import warnings


def setup_logging(level: str = "warning") -> None:
    """Configure Python logging with Rich handler.

    Args:
        level: Logging level as string: 'error', 'warning', 'info', or 'debug'
               Maps to Python logging levels (ERROR=40, WARNING=30, INFO=20, DEBUG=10)

    Example:
        setup_logging(level="debug")
        setup_logging(level="error")
    """
    from rich.logging import RichHandler

    # Map string to logging level
    level_map = {
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    log_level = level_map.get(level.lower(), logging.WARNING)

    # Get root logger and set level
    root = logging.getLogger()
    root.setLevel(log_level)

    # Add handler if first time, otherwise update existing handlers
    if not root.handlers:
        # First time - add RichHandler
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
        # Already configured - update handler levels
        for handler in root.handlers:
            handler.setLevel(log_level)

    # Suppress noisy FINN/QONNX loggers unless in debug mode
    if log_level > logging.DEBUG:
        logging.getLogger('finn').setLevel(logging.ERROR)
        logging.getLogger('qonnx').setLevel(logging.ERROR)
        warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'^finn\..*')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'^qonnx\..*')
