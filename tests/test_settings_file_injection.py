# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for custom file path injection fix (Critical Issue #2)."""

import pytest
from pathlib import Path
from brainsmith.settings import SystemConfig, load_config, get_default_config


def test_custom_project_file_loads_correctly(tmp_path):
    """Verify custom project file path is used when provided."""
    # Create a custom config file
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("""
build_dir: custom_build
netron_port: 9999
""")

    config = load_config(project_file=config_file)

    # Custom values from file should be loaded
    assert config.build_dir == tmp_path / "custom_build"
    assert config.netron_port == 9999


def test_custom_user_file_loads_correctly(tmp_path):
    """Verify custom user file path is used when provided."""
    # Create a custom user config file
    user_config_file = tmp_path / "custom_user_config.yaml"
    user_config_file.write_text("""
netron_port: 7777
""")

    # Disable project file auto-discovery to test user file in isolation
    config = load_config(
        user_file=user_config_file,
        project_file=Path('/dev/null')  # Non-existent file
    )

    # Custom value from user file should be loaded
    assert config.netron_port == 7777


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


def test_get_default_config_ignores_files():
    """Verify get_default_config returns only defaults without loading any files."""
    config = get_default_config()

    # Should have default values only
    assert config.build_dir.name == "build"
    assert config.netron_port == 8080
    assert config.default_workers == 4


def test_nonexistent_custom_file_handled_gracefully(tmp_path):
    """Verify nonexistent custom file paths don't break config loading."""
    nonexistent = tmp_path / "does_not_exist.yaml"

    # Should not raise error, just use defaults
    config = load_config(project_file=nonexistent)

    # Should have default values
    assert config.netron_port == 8080


def test_relative_paths_in_custom_file_resolve_to_file_location(tmp_path):
    """Verify relative paths in custom config file resolve to config file's directory."""
    # Create a subdirectory for the config
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    config_file = config_dir / "test_config.yaml"
    config_file.write_text("""
build_dir: my_build
""")

    config = load_config(project_file=config_file)

    # Relative path should resolve to config file's directory
    assert config.build_dir == config_dir / "my_build"


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
