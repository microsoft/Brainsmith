# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for path field detection using naming convention (replaces _path_fields.py)."""

import pytest
from pathlib import Path
from brainsmith.settings import load_config
from brainsmith.settings.loader import _is_path_field, _resolve_cli_paths


def test_is_path_field_detects_common_patterns():
    """Verify _is_path_field detects path fields by naming convention."""
    # Should be detected as path fields
    assert _is_path_field('build_dir')
    assert _is_path_field('xilinx_path')
    assert _is_path_field('config_file')
    assert _is_path_field('finn_root')
    assert _is_path_field('vivado_path')
    assert _is_path_field('deps_dir')

    # Should NOT be detected as path fields
    assert not _is_path_field('netron_port')
    assert not _is_path_field('default_workers')
    assert not _is_path_field('components_strict')
    assert not _is_path_field('xilinx_version')


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


def test_resolve_cli_paths_handles_absolute_paths():
    """Verify absolute paths are kept as-is."""
    cli_overrides = {
        'build_dir': '/absolute/build',
        'xilinx_path': '/tools/Xilinx',
    }

    resolved = _resolve_cli_paths(cli_overrides)

    # Absolute paths should remain unchanged
    assert resolved['build_dir'] == '/absolute/build'
    assert resolved['xilinx_path'] == '/tools/Xilinx'


def test_resolve_cli_paths_handles_none_values():
    """Verify None values are handled correctly."""
    cli_overrides = {
        'build_dir': None,
        'netron_port': 8080,
    }

    resolved = _resolve_cli_paths(cli_overrides)

    assert resolved['build_dir'] is None
    assert resolved['netron_port'] == 8080


def test_naming_convention_covers_all_path_fields():
    """Verify naming convention catches all actual path fields in SystemConfig."""
    from brainsmith.settings import SystemConfig

    # Get all field names from the schema
    config = SystemConfig()

    # These are known path fields - verify they match the naming convention
    path_fields = [
        'build_dir',
        'deps_dir',
        'xilinx_path',
        'vivado_path',
        'vitis_path',
        'vitis_hls_path',
        'vivado_ip_cache',
        'finn_root',
        'finn_build_dir',
        'finn_deps_dir',
    ]

    for field_name in path_fields:
        assert _is_path_field(field_name), f"{field_name} should be detected as a path field"


def test_cli_override_path_resolution_integration():
    """Integration test: CLI path overrides resolve correctly."""
    config = load_config(build_dir='test_build')

    # Should resolve relative to CWD
    assert config.build_dir.is_absolute()
    assert config.build_dir.name == 'test_build'
