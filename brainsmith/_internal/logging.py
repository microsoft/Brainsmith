# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging orchestration for Brainsmith and FINN integration.

Progressive disclosure logging: simple CLI surface, advanced config file customization.

Key principle: Let FINN log everything (verbose=False means "don't add your own handlers"),
then add our own handlers to control console vs file output.

Architecture:
    Root logger → file handler (everything, DEBUG)
    ├─ brainsmith.* → Rich console handler (controlled by level)
    ├─ finn.builder.* → console handler (progress messages)
    └─ finn.* → console handler (tool output, verbose mode only)

Verbosity levels:
    - quiet: No console output, file logs everything
    - normal: brainsmith.* + finn.builder.* (INFO), file logs everything
    - verbose: Above + finn.* (WARNING or per-tool), file logs everything
    - debug: Everything (DEBUG) with paths and tracebacks, file logs everything
"""

import logging
import re
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "normal",
    output_dir: Optional[Path] = None,
    config: Optional['LoggingConfig'] = None
) -> None:
    """Configure logging for Brainsmith and FINN.

    Args:
        level: Verbosity level (quiet | normal | verbose | debug)
        output_dir: Output directory for log file (None = no file logging)
        config: LoggingConfig instance (loaded from settings if None)
    """
    if config is None:
        from brainsmith.settings import get_config
        config = get_config().logging

    # Use config.level if not explicitly overridden
    if level == "normal" and config.level != "normal":
        level = config.level

    _setup_root(output_dir, config)
    _setup_brainsmith(level, config)
    _setup_finn(level, config)


def _setup_root(output_dir: Optional[Path], config: 'LoggingConfig') -> None:
    """Configure root logger with file handler.

    Root logger is permissive (DEBUG) and propagates everything to file.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Permissive gate

    # Add file handler if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "brainsmith.log"

        # Support rotation if configured
        if config.max_log_size_mb > 0:
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                log_file,
                maxBytes=config.max_log_size_mb * 1024 * 1024,
                backupCount=config.keep_backups,
            )
        else:
            handler = logging.FileHandler(log_file)

        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)


def _setup_brainsmith(level: str, config: 'LoggingConfig') -> None:
    """Configure brainsmith.* logger with Rich console handler."""
    if level == "quiet":
        return  # No console output

    from rich.logging import RichHandler

    logger = logging.getLogger("brainsmith")
    logger.setLevel(logging.DEBUG)  # Permissive gate
    logger.propagate = True  # Send to root for file logging

    # Console handler with Rich formatting
    handler = RichHandler(
        rich_tracebacks=(level == "debug"),
        show_path=(level == "debug"),
        markup=True,
        show_time=False,
    )

    # Set handler level (actual filtering)
    if level == "debug":
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)

    # Apply suppress filter if configured
    if config.suppress_patterns:
        handler.addFilter(_SuppressFilter(config.suppress_patterns))

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _setup_finn(level: str, config: 'LoggingConfig') -> None:
    """Configure FINN loggers (finn.builder.* and finn.*)."""
    if level == "quiet":
        return  # No console output

    # Always show finn.builder.* progress (unless quiet)
    builder_logger = logging.getLogger("finn.builder")
    builder_logger.setLevel(logging.DEBUG)
    builder_logger.propagate = True

    from rich.logging import RichHandler
    builder_handler = RichHandler(
        rich_tracebacks=False,
        show_path=False,
        markup=True,
        show_time=False,
    )
    builder_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    builder_handler.setFormatter(formatter)
    builder_logger.addHandler(builder_handler)

    # Show finn.* tool output in verbose/debug modes
    if level in ("verbose", "debug"):
        if config.finn_tools:
            _setup_finn_per_tool(level, config)
        else:
            _setup_finn_simple(level, config)


def _setup_finn_simple(level: str, config: 'LoggingConfig') -> None:
    """Configure FINN tool loggers with default levels."""
    finn_logger = logging.getLogger("finn")
    finn_logger.setLevel(logging.DEBUG)
    finn_logger.propagate = True

    from rich.logging import RichHandler
    handler = RichHandler(
        rich_tracebacks=(level == "debug"),
        show_path=(level == "debug"),
        markup=True,
        show_time=False,
    )

    # Default: WARNING for verbose, DEBUG for debug
    if level == "debug":
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.WARNING)

    if config.suppress_patterns:
        handler.addFilter(_SuppressFilter(config.suppress_patterns))

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    finn_logger.addHandler(handler)


def _setup_finn_per_tool(level: str, config: 'LoggingConfig') -> None:
    """Configure per-tool FINN loggers with custom levels."""
    from rich.logging import RichHandler

    for tool_name, log_level_str in config.finn_tools.items():
        logger = logging.getLogger(f"finn.{tool_name}")
        logger.setLevel(logging.DEBUG)  # Permissive gate
        logger.propagate = True

        handler = RichHandler(
            rich_tracebacks=(level == "debug"),
            show_path=(level == "debug"),
            markup=True,
            show_time=False,
        )

        # Parse level string to logging constant
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
        }
        handler_level = level_map.get(log_level_str.upper(), logging.WARNING)
        handler.setLevel(handler_level)

        if config.suppress_patterns:
            handler.addFilter(_SuppressFilter(config.suppress_patterns))

        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class _SuppressFilter(logging.Filter):
    """Filter to suppress console messages matching regex patterns.

    File handlers should NOT have this filter - they get everything.
    """

    def __init__(self, patterns: list[str]):
        super().__init__()
        self.patterns = [re.compile(p) for p in patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress matching messages."""
        message = record.getMessage()
        for pattern in self.patterns:
            if pattern.search(message):
                return False  # Suppress this message
        return True  # Allow this message


def get_finn_config() -> dict:
    """Get FINN build configuration for logging integration.

    Returns config that prevents FINN from adding its own handlers,
    allowing Brainsmith to control all logging.
    """
    return {
        "verbose": False,  # Don't add FINN's handlers
        "show_progress": True,  # We handle progress via finn.builder.*
    }
