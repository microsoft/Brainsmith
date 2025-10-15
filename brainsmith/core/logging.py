# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Simple logging configuration using Python's standard logging with Rich.

This module provides basic logging setup for Brainsmith using standard
Python logging levels (DEBUG, INFO, WARNING, ERROR) with Rich for pretty output.

Key principle: Use what exists. Python's logging module already provides
everything we need - don't reinvent it.

Usage:
    from brainsmith.core.logging import setup_logging, capture_finn_output

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
import sys
import warnings
from contextlib import contextmanager
from io import StringIO
from typing import Iterator


def setup_logging(quiet: bool = False, verbose: bool = False, debug: bool = False) -> None:
    """Configure Python logging with Rich handler.

    Maps CLI flags to standard Python logging levels:
    - quiet:   ERROR (40) - Only show errors
    - normal:  WARNING (30) - Default, show warnings and errors
    - verbose: INFO (20) - Show informational messages
    - debug:   DEBUG (10) - Show everything including debug traces

    Flag precedence: debug > verbose > quiet > normal (default)

    Args:
        quiet: Enable quiet mode (minimal output)
        verbose: Enable verbose mode (show INFO logs)
        debug: Enable debug mode (show DEBUG logs and tracebacks)

    Example:
        setup_logging(debug=args.debug, verbose=args.verbose, quiet=args.quiet)
    """
    from rich.logging import RichHandler

    # Determine logging level from flags
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.WARNING

    # Configure root logger with RichHandler
    # Note: basicConfig only works if no handlers exist, so we need to
    # handle both first-time and subsequent calls
    root = logging.getLogger()

    if not root.handlers:
        # First time - use basicConfig
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=True,
                show_time=False
            )]
        )
    else:
        # Already configured - just update level
        root.setLevel(level)
        for handler in root.handlers:
            handler.setLevel(level)

    # Suppress noisy FINN/QONNX loggers and warnings unless in debug mode
    # Use module-based filtering (not content heuristics)
    if level > logging.DEBUG:
        logging.getLogger('finn').setLevel(logging.ERROR)
        logging.getLogger('qonnx').setLevel(logging.ERROR)
        warnings.filterwarnings('ignore', module='finn.*')
        warnings.filterwarnings('ignore', module='qonnx.*')


@contextmanager
def capture_finn_output() -> Iterator[None]:
    """Capture and control FINN subprocess output visibility.

    FINN must run with verbose=True (to prevent stdout redirection that breaks
    Rich console). This context manager captures FINN's output and controls
    visibility based on the current logging level.

    Behavior by logging level:
    - DEBUG/INFO: Pass through all output (verbose mode)
    - WARNING: Show output (includes build_dataflow.py's "Running step" messages)
    - ERROR: Suppress stdout, show only stderr

    The key insight: build_dataflow.py's own messages ("Running step X",
    "Completed successfully") are printed OUTSIDE of subprocess calls, so they
    naturally show through in WARNING mode without content parsing.

    Example:
        with capture_finn_output():
            build_dataflow_cfg(model, config)

    Yields:
        None
    """
    root_level = logging.getLogger().level

    # In verbose modes (INFO/DEBUG), pass through all output
    if root_level <= logging.INFO:
        yield
        return

    # In quiet/normal modes, capture output for filtering
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield
    finally:
        # Always restore stdout/stderr
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig

        # Control visibility based on logging level
        # Don't try to parse content - just show/hide based on level

        if root_level == logging.WARNING:
            # Normal mode: Show captured stdout
            # This includes build_dataflow.py's "Running step" messages
            output = stdout_capture.getvalue()
            if output.strip():
                print(output, file=stdout_orig, end='')

        # In ERROR mode: stdout is suppressed entirely

        # Always show stderr regardless of level (errors are important)
        err_output = stderr_capture.getvalue()
        if err_output.strip():
            print(err_output, file=stderr_orig, end='')
