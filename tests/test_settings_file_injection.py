# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for settings priority chain (Critical Issue #2).

Focused on testing the complex priority override chain, not trivial file loading.
"""

import pytest
from pathlib import Path
from brainsmith.settings import SystemConfig, load_config


def test_project_file_overrides_user_file(tmp_path):
    """Verify project file has higher priority than user file."""
    # Create user config
    user_config = tmp_path / "user.yaml"
    user_config.write_text("""
netron_port: 5555
default_workers: 2
""")

    # Create project config that overrides netron_port
    project_config = tmp_path / "project.yaml"
    project_config.write_text("""
netron_port: 6666
""")

    config = load_config(user_file=user_config, project_file=project_config)

    # Project file should override netron_port
    assert config.netron_port == 6666
    # User file value for default_workers should still be used
    assert config.default_workers == 2


def test_cli_overrides_project_file(tmp_path):
    """Verify CLI args override project file values."""
    # Create project config
    project_config = tmp_path / "project.yaml"
    project_config.write_text("""
netron_port: 8888
""")

    # CLI override should win
    config = load_config(project_file=project_config, netron_port=9090)

    assert config.netron_port == 9090


def test_priority_order_complete(tmp_path):
    """Verify complete priority chain: CLI > env > project > user > defaults."""
    import os

    # Setup files
    user_file = tmp_path / "user.yaml"
    user_file.write_text("netron_port: 1111\ndefault_workers: 11")

    project_file = tmp_path / "project.yaml"
    project_file.write_text("netron_port: 2222")

    # Setup environment variable
    os.environ['BSMITH_DEFAULT_WORKERS'] = '22'

    try:
        # Load with CLI override
        config = load_config(
            user_file=user_file,
            project_file=project_file,
            netron_port=3333  # CLI override
        )

        # Priority verification:
        assert config.netron_port == 3333      # CLI wins
        assert config.default_workers == 22    # Env wins (project didn't set it)

    finally:
        os.environ.pop('BSMITH_DEFAULT_WORKERS', None)
