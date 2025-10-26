# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for FINN configuration external integration.

Focused on FINN integration points and environment handling, not static structure.
"""

import pytest
from pathlib import Path
from brainsmith.settings import SystemConfig, load_config


def test_finn_paths_resolve_relative_to_project_dir(tmp_path):
    """Verify relative FINN paths resolve to project_dir."""
    # Create a temporary config file
    (tmp_path / ".brainsmith").mkdir()
    config_file = tmp_path / ".brainsmith" / "config.yaml"
    config_file.write_text("""
finn_root: finn_custom
finn_build_dir: build_custom
finn_deps_dir: deps_custom
""")

    config = load_config(project_file=config_file)

    # Relative paths should resolve to project dir (tmp_path, not tmp_path/.brainsmith)
    assert config.finn_root == tmp_path / "finn_custom"
    assert config.finn_build_dir == tmp_path / "build_custom"
    assert config.finn_deps_dir == tmp_path / "deps_custom"


def test_finn_env_var_overrides():
    """Verify FINN paths can be set via environment variables."""
    import os

    # Set flat env vars (no nested delimiter)
    os.environ['BSMITH_FINN_ROOT'] = '/env/finn'
    os.environ['BSMITH_FINN_BUILD_DIR'] = '/env/build'
    os.environ['BSMITH_FINN_DEPS_DIR'] = '/env/deps'

    try:
        config = SystemConfig()

        assert config.finn_root == Path('/env/finn')
        assert config.finn_build_dir == Path('/env/build')
        assert config.finn_deps_dir == Path('/env/deps')
    finally:
        # Clean up
        os.environ.pop('BSMITH_FINN_ROOT', None)
        os.environ.pop('BSMITH_FINN_BUILD_DIR', None)
        os.environ.pop('BSMITH_FINN_DEPS_DIR', None)


def test_finn_export_to_environment():
    """Verify FINN paths are exported correctly to environment variables."""
    config = SystemConfig()
    env_dict = config.export_to_environment(export=False)

    # Should export FINN environment variables
    assert 'FINN_ROOT' in env_dict
    assert 'FINN_BUILD_DIR' in env_dict
    assert 'FINN_DEPS_DIR' in env_dict

    # Values should match config
    assert env_dict['FINN_ROOT'] == str(config.finn_root)
    assert env_dict['FINN_BUILD_DIR'] == str(config.finn_build_dir)
    assert env_dict['FINN_DEPS_DIR'] == str(config.finn_deps_dir)
