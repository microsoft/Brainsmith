# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Essential tests for logging configuration."""

import logging
import pytest
import sys
from io import StringIO
from click.testing import CliRunner

from brainsmith.core.logging import setup_logging, capture_finn_output
from brainsmith.interface.cli import create_cli


@pytest.fixture
def reset_logging():
    """Reset logging system between tests."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)


def test_logging_levels_documentation(reset_logging):
    """Document expected logging levels for each verbosity mode.

    This test serves as documentation for what each flag does.
    """
    # Quiet mode: ERROR level
    setup_logging(quiet=True)
    assert logging.getLogger().level == logging.ERROR

    # Normal mode (default): WARNING level
    setup_logging()
    assert logging.getLogger().level == logging.WARNING

    # Verbose mode: INFO level
    setup_logging(verbose=True)
    assert logging.getLogger().level == logging.INFO

    # Debug mode: DEBUG level
    setup_logging(debug=True)
    assert logging.getLogger().level == logging.DEBUG


def test_capture_finn_output_passes_through_in_verbose(reset_logging):
    """Test capture_finn_output passes through in verbose/debug modes."""
    setup_logging(verbose=True)

    output_capture = StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = output_capture
        with capture_finn_output():
            print("FINN output")
        sys.stdout = original_stdout

        # In verbose mode, output passes through
        assert "FINN output" in output_capture.getvalue()
    finally:
        sys.stdout = original_stdout


def test_capture_finn_output_suppresses_in_quiet(reset_logging):
    """Test capture_finn_output suppresses stdout in quiet mode."""
    setup_logging(quiet=True)

    original_stdout = sys.stdout
    capture_output = StringIO()

    try:
        sys.stdout = capture_output
        with capture_finn_output():
            print("Should be suppressed")

        # In quiet mode, stdout is suppressed
        assert capture_output.getvalue() == ""
    finally:
        sys.stdout = original_stdout


def test_cli_flags_smoke_test():
    """Smoke test: Both CLIs accept verbosity flags without crashing."""
    runner = CliRunner()

    # Test smith CLI
    smith_cli = create_cli('smith', include_admin=False)
    result = runner.invoke(smith_cli, ['--quiet'])
    assert result.exit_code == 0
    result = runner.invoke(smith_cli, ['--verbose'])
    assert result.exit_code == 0
    result = runner.invoke(smith_cli, ['--debug'])
    assert result.exit_code == 0

    # Test brainsmith CLI
    brainsmith_cli = create_cli('brainsmith', include_admin=True)
    result = runner.invoke(brainsmith_cli, ['--quiet'])
    assert result.exit_code == 0
    result = runner.invoke(brainsmith_cli, ['--verbose'])
    assert result.exit_code == 0
    result = runner.invoke(brainsmith_cli, ['--debug'])
    assert result.exit_code == 0
