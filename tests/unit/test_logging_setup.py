# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for logging setup and handler configuration.

Streamlined for CI/CD: 9 essential tests covering handler attachment,
file logging, filters, and FINN integration.

Arete Approach: No mocking. Real logging setup, real handlers.
"""

import logging
import pytest
import tempfile
from pathlib import Path

from brainsmith._internal.logging import (
    setup_logging,
    get_finn_config,
    _SuppressFilter
)
from brainsmith.settings.schema import LoggingConfig


# ============================================================================
# Test Utilities
# ============================================================================

def get_handler_count(logger_name: str) -> int:
    """Count handlers on a logger."""
    logger = logging.getLogger(logger_name)
    return len(logger.handlers)


def get_console_handlers(logger_name: str) -> list:
    """Get console handlers from a logger."""
    logger = logging.getLogger(logger_name)
    from rich.logging import RichHandler
    return [h for h in logger.handlers if isinstance(h, RichHandler)]


def get_file_handlers(logger_name: str) -> list:
    """Get file handlers from a logger."""
    logger = logging.getLogger(logger_name)
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def clear_all_handlers():
    """Clear all handlers from all loggers (test isolation)."""
    root = logging.getLogger()
    root.handlers.clear()

    for logger_name in ['brainsmith', 'finn', 'finn.builder', 'finn.vivado', 'finn.hls']:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state before each test."""
    clear_all_handlers()
    yield
    clear_all_handlers()


# ============================================================================
# CI/CD Test Suite: Logging Setup (9 tests)
# ============================================================================

class TestVerbosityLevels:
    """Test handler attachment at each verbosity level."""

    def test_quiet_no_console_handlers(self):
        """Quiet level: No console handlers attached."""
        setup_logging(level="quiet", output_dir=None)

        assert get_handler_count("brainsmith") == 0
        assert get_handler_count("finn.builder") == 0
        assert get_handler_count("finn") == 0

    def test_normal_brainsmith_and_builder(self):
        """Normal level: brainsmith.* + finn.builder.* handlers (default behavior)."""
        setup_logging(level="normal", output_dir=None)

        # brainsmith handler attached
        assert get_handler_count("brainsmith") > 0
        assert len(get_console_handlers("brainsmith")) > 0

        # finn.builder handler attached
        assert get_handler_count("finn.builder") > 0
        assert len(get_console_handlers("finn.builder")) > 0

    def test_verbose_includes_finn_tools(self):
        """Verbose level: Above + finn.* handler."""
        setup_logging(level="verbose", output_dir=None)

        assert get_handler_count("brainsmith") > 0
        assert get_handler_count("finn.builder") > 0
        assert get_handler_count("finn") > 0

    def test_debug_all_at_debug_level(self):
        """Debug level: All handlers with DEBUG level."""
        setup_logging(level="debug", output_dir=None)

        # Check brainsmith handler level
        brainsmith_handlers = get_console_handlers("brainsmith")
        assert len(brainsmith_handlers) > 0
        assert brainsmith_handlers[0].level == logging.DEBUG


class TestFileLogging:
    """Test file handler creation and comprehensive logging."""

    def test_file_handler_created(self):
        """File handler created when output_dir provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            setup_logging(level="normal", output_dir=output_dir)

            # File handler on root logger
            root_file_handlers = get_file_handlers("")
            assert len(root_file_handlers) > 0

            # Log file created
            log_file = output_dir / "brainsmith.log"
            assert log_file.exists()

    def test_file_captures_all_levels(self):
        """File handler captures DEBUG regardless of console level (critical for debugging)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            setup_logging(level="quiet", output_dir=output_dir)

            # File handler level should be DEBUG
            root_file_handlers = get_file_handlers("")
            assert len(root_file_handlers) > 0
            assert root_file_handlers[0].level == logging.DEBUG

            # Log some messages
            logger = logging.getLogger("brainsmith.test")
            logger.debug("Debug message")
            logger.info("Info message")

            # Check file contains both
            log_file = output_dir / "brainsmith.log"
            content = log_file.read_text()
            assert "Debug message" in content
            assert "Info message" in content


class TestAdvancedFeatures:
    """Test pattern suppression and per-tool configuration."""

    def test_suppress_filter_matches(self):
        """SuppressFilter suppresses matching patterns."""
        patterns = ["Compiling module", "Analyzing entity"]
        filter_obj = _SuppressFilter(patterns)

        # Create log records
        record_match = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Compiling module work.foo",
            args=(),
            exc_info=None
        )

        record_no_match = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Some other message",
            args=(),
            exc_info=None
        )

        # Filter should suppress matching record
        assert filter_obj.filter(record_match) is False
        assert filter_obj.filter(record_no_match) is True

    def test_per_tool_handlers(self):
        """Separate handlers created for each FINN tool."""
        config = LoggingConfig(
            level="verbose",
            finn_tools={
                "vivado": "WARNING",
                "hls": "INFO"
            }
        )

        setup_logging(level="verbose", output_dir=None, config=config)

        # Check handlers on tool loggers
        assert get_handler_count("finn.vivado") > 0
        assert get_handler_count("finn.hls") > 0

        # Verify correct levels
        vivado_handlers = get_console_handlers("finn.vivado")
        hls_handlers = get_console_handlers("finn.hls")

        assert len(vivado_handlers) > 0
        assert vivado_handlers[0].level == logging.WARNING

        assert len(hls_handlers) > 0
        assert hls_handlers[0].level == logging.INFO


class TestFINNIntegration:
    """Test FINN integration configuration."""

    def test_get_finn_config(self):
        """get_finn_config returns correct values for FINN integration."""
        config = get_finn_config()

        assert config["verbose"] is False
        assert config["show_progress"] is True
