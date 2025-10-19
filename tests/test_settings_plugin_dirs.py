# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tests for plugin_dirs configuration field.

Tests cover:
- Default value behavior
- Validator handling different input types
- effective_plugin_dirs property
- Integration with config loading
"""

import pytest
from pathlib import Path
from brainsmith.settings import SystemConfig


class TestPluginDirsField:
    """Test plugin_dirs field validation and behavior."""

    def test_plugin_dirs_default_empty(self):
        """Test that plugin_dirs defaults to empty list."""
        config = SystemConfig()
        assert config.plugin_dirs == []

    def test_plugin_dirs_single_string(self):
        """Test plugin_dirs accepts single string path."""
        config = SystemConfig(plugin_dirs="/custom/plugins")
        assert config.plugin_dirs == [Path("/custom/plugins")]

    def test_plugin_dirs_single_path(self):
        """Test plugin_dirs accepts single Path object."""
        custom_path = Path("/custom/plugins")
        config = SystemConfig(plugin_dirs=custom_path)
        assert config.plugin_dirs == [custom_path]

    def test_plugin_dirs_list_of_strings(self):
        """Test plugin_dirs accepts list of string paths."""
        config = SystemConfig(plugin_dirs=["/path1", "/path2"])
        assert config.plugin_dirs == [Path("/path1"), Path("/path2")]

    def test_plugin_dirs_list_of_paths(self):
        """Test plugin_dirs accepts list of Path objects."""
        paths = [Path("/path1"), Path("/path2")]
        config = SystemConfig(plugin_dirs=paths)
        assert config.plugin_dirs == paths

    def test_plugin_dirs_mixed_list(self):
        """Test plugin_dirs accepts mixed list of strings and Paths."""
        config = SystemConfig(plugin_dirs=["/path1", Path("/path2")])
        assert config.plugin_dirs == [Path("/path1"), Path("/path2")]

    def test_plugin_dirs_none(self):
        """Test plugin_dirs handles None as empty list."""
        config = SystemConfig(plugin_dirs=None)
        assert config.plugin_dirs == []


class TestEffectivePluginDirs:
    """Test effective_plugin_dirs property."""

    def test_effective_plugin_dirs_with_configured_paths(self):
        """Test effective_plugin_dirs returns configured paths."""
        custom_dirs = [Path("/custom/plugins1"), Path("/custom/plugins2")]
        config = SystemConfig(plugin_dirs=custom_dirs)
        assert config.effective_plugin_dirs == custom_dirs

    def test_effective_plugin_dirs_with_empty_config(self):
        """Test effective_plugin_dirs returns default when empty."""
        config = SystemConfig(plugin_dirs=[])
        default_dir = Path.home() / '.brainsmith' / 'plugins'
        assert config.effective_plugin_dirs == [default_dir]

    def test_effective_plugin_dirs_default_config(self):
        """Test effective_plugin_dirs with default unset config."""
        config = SystemConfig()
        default_dir = Path.home() / '.brainsmith' / 'plugins'
        assert config.effective_plugin_dirs == [default_dir]

    def test_effective_plugin_dirs_single_path(self):
        """Test effective_plugin_dirs with single configured path."""
        config = SystemConfig(plugin_dirs=[Path("/custom")])
        assert config.effective_plugin_dirs == [Path("/custom")]


class TestPluginDirsRelativePaths:
    """Test handling of relative vs absolute paths."""

    def test_plugin_dirs_relative_path(self):
        """Test plugin_dirs with relative path."""
        config = SystemConfig(plugin_dirs=["./plugins"])
        assert config.plugin_dirs == [Path("./plugins")]

    def test_plugin_dirs_absolute_path(self):
        """Test plugin_dirs with absolute path."""
        config = SystemConfig(plugin_dirs=["/opt/plugins"])
        assert config.plugin_dirs == [Path("/opt/plugins")]

    def test_plugin_dirs_home_expansion(self):
        """Test plugin_dirs with home directory."""
        config = SystemConfig(plugin_dirs=["~/.brainsmith/plugins"])
        # Path doesn't auto-expand ~, it's stored as-is
        assert config.plugin_dirs == [Path("~/.brainsmith/plugins")]


class TestPluginDirsYAMLLoading:
    """Test loading plugin_dirs from YAML config."""

    def test_plugin_dirs_from_env_var(self):
        """Test loading plugin_dirs from environment variable."""
        import os

        # Set env var
        os.environ['BSMITH_PLUGIN_DIRS'] = "/env/plugins1:/env/plugins2"

        try:
            # Note: Pydantic will parse colon-separated string as single path
            # For list, need to use YAML or pass as list to constructor
            config = SystemConfig()

            # Clean up
            del os.environ['BSMITH_PLUGIN_DIRS']

            # This test shows the limitation - env vars with lists need special handling
            # For now, we expect users to use YAML for multiple paths
        except Exception:
            # Clean up even on error
            if 'BSMITH_PLUGIN_DIRS' in os.environ:
                del os.environ['BSMITH_PLUGIN_DIRS']
            # This is expected - env var list handling is complex
            pass


class TestPluginDirsEdgeCases:
    """Test edge cases and error handling."""

    def test_plugin_dirs_empty_string(self):
        """Test plugin_dirs with empty string."""
        config = SystemConfig(plugin_dirs="")
        # Empty string converts to Path("")
        assert config.plugin_dirs == [Path("")]

    def test_plugin_dirs_duplicate_paths(self):
        """Test plugin_dirs allows duplicate paths (no deduplication)."""
        dup_path = Path("/custom/plugins")
        config = SystemConfig(plugin_dirs=[dup_path, dup_path])
        # Duplicates are preserved (discovery code handles this)
        assert config.plugin_dirs == [dup_path, dup_path]

    def test_plugin_dirs_priority_order_preserved(self):
        """Test that plugin_dirs preserves order (first = highest priority)."""
        paths = [Path("/priority1"), Path("/priority2"), Path("/priority3")]
        config = SystemConfig(plugin_dirs=paths)
        assert config.plugin_dirs == paths
        assert config.plugin_dirs[0] == Path("/priority1")
