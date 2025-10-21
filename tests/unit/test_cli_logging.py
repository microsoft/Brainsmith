# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Essential tests for logging configuration."""

import logging
import pytest
import sys
from io import StringIO
from click.testing import CliRunner

from brainsmith._internal.logging import setup_logging
from brainsmith.cli.cli import create_cli


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

    This test serves as documentation for what each level does.
    """
    # Error level
    setup_logging(level='error')
    assert logging.getLogger().level == logging.ERROR

    # Warning level (default)
    setup_logging(level='warning')
    assert logging.getLogger().level == logging.WARNING

    # Info level
    setup_logging(level='info')
    assert logging.getLogger().level == logging.INFO

    # Debug level
    setup_logging(level='debug')
    assert logging.getLogger().level == logging.DEBUG


def test_cli_flags_smoke_test():
    """Smoke test: Both CLIs accept log level flags without crashing."""
    runner = CliRunner()

    # Test smith CLI
    smith_cli = create_cli('smith', include_admin=False)
    result = runner.invoke(smith_cli, ['--logs', 'error'])
    assert result.exit_code == 0
    result = runner.invoke(smith_cli, ['--logs', 'info'])
    assert result.exit_code == 0
    result = runner.invoke(smith_cli, ['--logs', 'debug'])
    assert result.exit_code == 0

    # Test brainsmith CLI
    brainsmith_cli = create_cli('brainsmith', include_admin=True)
    result = runner.invoke(brainsmith_cli, ['--logs', 'error'])
    assert result.exit_code == 0
    result = runner.invoke(brainsmith_cli, ['--logs', 'info'])
    assert result.exit_code == 0
    result = runner.invoke(brainsmith_cli, ['--logs', 'debug'])
    assert result.exit_code == 0
