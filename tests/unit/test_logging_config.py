# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for logging configuration schema and loading.

Streamlined for CI/CD: 5 essential tests covering schema, YAML loading,
and environment variable precedence.

Arete Approach: No mocking. Real config loading, real environment variables.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from brainsmith.settings.loader import load_config
from brainsmith.settings.schema import LoggingConfig

# ============================================================================
# Test Helpers
# ============================================================================

def minimal_config(logging_level: str = "normal") -> str:
    """Minimal valid brainsmith.yaml with logging config."""
    return f"""xilinx_path: /tools/Xilinx
xilinx_version: '2024.2'
logging:
  level: {logging_level}
"""


# ============================================================================
# CI/CD Test Suite: Configuration (5 tests)
# ============================================================================

class TestLoggingConfigEssentials:
    """Essential logging config tests for CI/CD."""

    def test_default_logging_config(self):
        """Default LoggingConfig has expected values."""
        config = LoggingConfig()

        assert config.level == "normal"
        assert config.finn_tools is None
        assert config.suppress_patterns is None
        assert config.max_log_size_mb == 0
        assert config.keep_backups == 3

    @pytest.mark.parametrize("level", ["quiet", "normal", "verbose", "debug"])
    def test_valid_log_levels(self, level):
        """All valid log levels are accepted."""
        config = LoggingConfig(level=level)
        assert config.level == level

    def test_system_config_includes_logging(self):
        """SystemConfig includes logging field with default factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "brainsmith.yaml"
            config_file.write_text(minimal_config())

            config = load_config(project_file=config_file)

            assert hasattr(config, 'logging')
            assert isinstance(config.logging, LoggingConfig)
            assert config.logging.level == "normal"

    def test_load_from_yaml(self):
        """Load logging section from YAML config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "brainsmith.yaml"
            config_file.write_text(minimal_config("verbose"))

            config = load_config(project_file=config_file)

            assert config.logging.level == "verbose"

    def test_env_var_override(self):
        """BSMITH_LOG_LEVEL env var overrides YAML (validates precedence)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "brainsmith.yaml"
            config_file.write_text(minimal_config("normal"))

            with mock.patch.dict(os.environ, {'BSMITH_LOG_LEVEL': 'debug'}):
                config = load_config(project_file=config_file)
                assert config.logging.level == "debug"
