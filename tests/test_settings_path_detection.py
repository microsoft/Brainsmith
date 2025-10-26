# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for path field detection and CLI path resolution.

Focused on integration tests and common use cases, not static validation.
"""

import pytest
from pathlib import Path
from brainsmith.settings import load_config
from brainsmith.settings.loader import _is_path_field, _resolve_cli_paths


def test_resolve_cli_paths_handles_path_fields():
    """Verify CLI path resolution works with naming convention detection."""
    cli_overrides = {
        'build_dir': 'custom_build',
        'xilinx_path': '/abs/path',
        'netron_port': 8888,  # Not a path field
    }

    resolved = _resolve_cli_paths(cli_overrides)

    # Path fields should be resolved
    assert Path(resolved['build_dir']).is_absolute()
    assert resolved['build_dir'].endswith('custom_build')
    assert resolved['xilinx_path'] == '/abs/path'  # Already absolute

    # Non-path fields should pass through unchanged
    assert resolved['netron_port'] == 8888


def test_resolve_cli_paths_handles_relative_paths():
    """Verify relative CLI paths resolve to CWD."""
    cli_overrides = {
        'build_dir': 'my_build',
        'finn_root': 'deps/finn',
    }

    resolved = _resolve_cli_paths(cli_overrides)

    # Should resolve to CWD
    cwd = Path.cwd()
    assert resolved['build_dir'] == str((cwd / 'my_build').resolve())
    assert resolved['finn_root'] == str((cwd / 'deps' / 'finn').resolve())


def test_cli_override_path_resolution_integration():
    """Integration test: CLI path overrides resolve correctly."""
    config = load_config(build_dir='test_build')

    # Should resolve relative to CWD
    assert config.build_dir.is_absolute()
    assert config.build_dir.name == 'test_build'
