# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tests for the lazy plugin loader.

Tests cover:
- Lazy import with caching
- Multi-source discovery (core, user, entry points)
- Step lookup and listing
- Kernel lookup and metadata
- Backend resolution
- Error handling
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the loader module
from brainsmith import loader


class TestLazyImport:
    """Test the core lazy_import function."""

    def test_lazy_import_success(self):
        """Test successful lazy import."""
        # Import a known module
        result = loader.lazy_import('brainsmith.dse.types.SegmentStatus')

        # Should get the actual class
        assert result is not None
        assert result.__name__ == 'SegmentStatus'

    def test_lazy_import_caching(self):
        """Test that imports are cached."""
        # Clear cache
        loader._import_cache.clear()

        # First import
        path = 'brainsmith.dse.types.SegmentStatus'
        result1 = loader.lazy_import(path)

        # Should be in cache
        assert path in loader._import_cache

        # Second import should return cached value
        result2 = loader.lazy_import(path)
        assert result1 is result2

    def test_lazy_import_invalid_module(self):
        """Test error handling for invalid module."""
        with pytest.raises(ImportError) as exc_info:
            loader.lazy_import('nonexistent.module.Class')

        assert 'Could not import' in str(exc_info.value)

    def test_lazy_import_invalid_attribute(self):
        """Test error handling for invalid attribute."""
        with pytest.raises(ImportError) as exc_info:
            loader.lazy_import('brainsmith.dse.types.NonexistentClass')

        assert 'Could not import' in str(exc_info.value)


class TestStepLookup:
    """Test step lookup functions."""

    def test_has_step_core(self):
        """Test has_step for core steps."""
        # Should have core Brainsmith steps
        assert loader.has_step('qonnx_to_finn')
        assert loader.has_step('specialize_layers')

    def test_has_step_nonexistent(self):
        """Test has_step for nonexistent step."""
        assert not loader.has_step('this_step_definitely_does_not_exist')

    def test_get_step_core(self):
        """Test get_step for core steps."""
        # Get a core step
        step_fn = loader.get_step('qonnx_to_finn')

        # Should be callable
        assert callable(step_fn)
        assert step_fn.__name__ == 'qonnx_to_finn_step'

    def test_get_step_nonexistent(self):
        """Test get_step error for nonexistent step."""
        with pytest.raises(KeyError) as exc_info:
            loader.get_step('this_step_does_not_exist')

        assert 'not found' in str(exc_info.value)

    def test_list_steps(self):
        """Test list_steps returns all steps."""
        steps = loader.list_steps()

        # Should be a list
        assert isinstance(steps, list)

        # Should contain core steps
        assert 'qonnx_to_finn' in steps
        assert 'specialize_layers' in steps
        assert 'infer_kernels' in steps

        # Should be sorted
        assert steps == sorted(steps)

    def test_step_caching(self):
        """Test that step imports are cached."""
        # Clear cache
        loader._import_cache.clear()

        # Get step twice
        step1 = loader.get_step('qonnx_to_finn')
        step2 = loader.get_step('qonnx_to_finn')

        # Should be same object (cached)
        assert step1 is step2


class TestKernelLookup:
    """Test kernel lookup functions."""

    def test_get_kernel(self):
        """Test get_kernel for core kernels."""
        # Get a Brainsmith kernel
        LayerNorm = loader.get_kernel('LayerNorm')

        # Should be a class
        assert LayerNorm is not None
        assert LayerNorm.__name__ == 'LayerNorm'

    def test_get_kernel_finn(self):
        """Test get_kernel for FINN kernels."""
        # Get a FINN kernel
        MVAU = loader.get_kernel('MVAU')

        assert MVAU is not None
        assert MVAU.__name__ == 'MVAU'

    def test_get_kernel_nonexistent(self):
        """Test get_kernel error for nonexistent kernel."""
        with pytest.raises(KeyError) as exc_info:
            loader.get_kernel('NonexistentKernel')

        assert 'not found' in str(exc_info.value)

    def test_list_kernels(self):
        """Test list_kernels returns all kernels."""
        kernels = loader.list_kernels()

        # Should contain Brainsmith kernels
        assert 'LayerNorm' in kernels
        assert 'Softmax' in kernels

        # Should contain FINN kernels
        assert 'MVAU' in kernels
        assert 'Thresholding' in kernels

        # Should be sorted
        assert kernels == sorted(kernels)


class TestKernelMetadata:
    """Test kernel metadata (InferTransform) functionality."""

    def test_get_kernel_infer(self):
        """Test getting InferTransform from kernel metadata."""
        # Get InferTransform for LayerNorm
        InferLayerNorm = loader.get_kernel_infer('LayerNorm')

        assert InferLayerNorm is not None
        assert InferLayerNorm.__name__ == 'InferLayerNorm'

    def test_get_kernel_infer_finn(self):
        """Test getting InferTransform for FINN kernel."""
        # MVAU has InferTransform
        InferMVAU = loader.get_kernel_infer('MVAU')

        assert InferMVAU is not None
        # Should be InferQuantizedMatrixVectorActivation
        assert 'Infer' in InferMVAU.__name__

    def test_get_kernel_infer_no_infer(self):
        """Test error when kernel has no InferTransform."""
        # StreamingDataWidthConverter has no InferTransform
        with pytest.raises(KeyError) as exc_info:
            loader.get_kernel_infer('StreamingDataWidthConverter')

        assert 'no InferTransform' in str(exc_info.value)


class TestBackendResolution:
    """Test backend resolution."""

    def test_get_backend_hls(self):
        """Test getting HLS backend."""
        # Get HLS backend for LayerNorm
        LayerNormHLS = loader.get_backend('LayerNorm', 'hls')

        assert LayerNormHLS is not None
        assert 'LayerNorm' in LayerNormHLS.__name__

    def test_get_backend_rtl(self):
        """Test getting RTL backend."""
        # MVAU has RTL backend
        MVAU_RTL = loader.get_backend('MVAU', 'rtl')

        assert MVAU_RTL is not None
        assert 'MVAU' in MVAU_RTL.__name__

    def test_get_backend_invalid_language(self):
        """Test error for invalid backend language."""
        with pytest.raises(KeyError) as exc_info:
            loader.get_backend('LayerNorm', 'vhdl')  # LayerNorm only has HLS

        assert 'not available' in str(exc_info.value)

    def test_list_backends(self):
        """Test listing available backends for a kernel."""
        # LayerNorm only has HLS
        backends = loader.list_backends('LayerNorm')
        assert backends == ['hls']

        # MVAU has both HLS and RTL
        backends = loader.list_backends('MVAU')
        assert set(backends) == {'hls', 'rtl'}


class TestUserPluginDiscovery:
    """Test user plugin discovery from ~/.brainsmith/plugins/."""

    def test_user_plugin_discovery_no_directory(self):
        """Test that missing user plugin directory doesn't break anything."""
        # Reset discovery state
        loader._user_plugins_scanned = False
        loader._user_plugins = {'steps': {}, 'kernels': {}}

        # Mock _get_plugin_dirs to return non-existent path
        with patch.object(loader, '_get_plugin_dirs', return_value=[Path('/tmp/nonexistent_brainsmith_plugins')]):
            # Should not crash
            loader._discover_user_plugins()

            # Should mark as scanned
            assert loader._user_plugins_scanned

    @patch('brainsmith.loader.Path')
    def test_user_plugin_discovery_structure(self, mock_path):
        """Test user plugin directory structure scanning."""
        # This is a structural test - actual file-based testing would require fixtures
        # Reset discovery
        loader._user_plugins_scanned = False
        loader._user_plugins = {'steps': {}, 'kernels': {}}

        # Mock a plugin directory that doesn't exist
        mock_plugin_path = MagicMock()
        mock_plugin_path.exists.return_value = False

        with patch.object(loader, '_get_plugin_dirs', return_value=[mock_plugin_path]):
            loader._discover_user_plugins()

            # Should have scanned (even if nothing found)
            assert loader._user_plugins_scanned


class TestEntryPointDiscovery:
    """Test entry point discovery for pip-installed extensions."""

    def test_entry_point_discovery(self):
        """Test entry point discovery doesn't crash."""
        # Reset discovery
        loader._entry_points_scanned = False
        loader._entry_point_plugins = {'steps': {}, 'kernels': {}}

        # Should not crash even with no entry points
        loader._discover_entry_points()

        # Should mark as scanned
        assert loader._entry_points_scanned

    @patch('brainsmith.loader.entry_points')
    def test_entry_point_discovery_with_plugins(self, mock_entry_points):
        """Test entry point discovery with mocked plugins."""
        # Reset discovery
        loader._entry_points_scanned = False
        loader._entry_point_plugins = {'steps': {}, 'kernels': {}}

        # Mock entry points
        mock_step_ep = MagicMock()
        mock_step_ep.name = 'custom_step'

        mock_kernel_ep = MagicMock()
        mock_kernel_ep.name = 'CustomKernel'

        def mock_eps(group):
            if group == 'brainsmith.steps':
                return [mock_step_ep]
            elif group == 'brainsmith.kernels':
                return [mock_kernel_ep]
            return []

        mock_entry_points.side_effect = mock_eps

        # Discover
        loader._discover_entry_points()

        # Should have found the plugins
        assert 'custom_step' in loader._entry_point_plugins['steps']
        assert 'CustomKernel' in loader._entry_point_plugins['kernels']


class TestPriorityOrder:
    """Test that plugin sources are checked in correct priority order."""

    def test_core_takes_precedence(self):
        """Test that core plugins are found before user/entry point plugins."""
        # Reset user/entry point discovery
        loader._user_plugins_scanned = False
        loader._entry_points_scanned = False
        loader._user_plugins = {'steps': {}, 'kernels': {}}
        loader._entry_point_plugins = {'steps': {}, 'kernels': {}}

        # Get a core step - should not trigger user/entry point discovery
        step = loader.get_step('qonnx_to_finn')
        assert step is not None

        # User plugins should not have been scanned (core was sufficient)
        assert not loader._user_plugins_scanned
        assert not loader._entry_points_scanned

    def test_user_plugins_before_entry_points(self):
        """Test that user plugins are checked before entry points."""
        # Reset discovery
        loader._user_plugins_scanned = False
        loader._entry_points_scanned = False

        # Try to get a nonexistent step
        try:
            loader.get_step('definitely_nonexistent_step')
        except KeyError:
            pass

        # Both should have been scanned
        assert loader._user_plugins_scanned
        assert loader._entry_points_scanned


class TestConfigIntegration:
    """Test integration with settings system."""

    def test_get_plugin_dirs_from_config(self):
        """Test that _get_plugin_dirs uses config.effective_plugin_dirs."""
        from brainsmith.settings import SystemConfig

        # Create config with custom plugin_dirs
        custom_dirs = [Path("/custom/plugins1"), Path("/custom/plugins2")]
        config = SystemConfig(plugin_dirs=custom_dirs)

        # Mock get_config in settings module (where it's imported from)
        with patch('brainsmith.settings.get_config', return_value=config):
            dirs = loader._get_plugin_dirs()
            assert dirs == custom_dirs

    def test_get_plugin_dirs_default_fallback(self):
        """Test fallback to default when config not available."""
        # Mock get_config to raise exception
        with patch('brainsmith.settings.get_config', side_effect=Exception("Config not available")):
            dirs = loader._get_plugin_dirs()
            # Should fall back to default
            assert dirs == [Path.home() / '.brainsmith' / 'plugins']

    def test_get_plugin_dirs_empty_config(self):
        """Test that empty plugin_dirs in config uses default."""
        from brainsmith.settings import SystemConfig

        # Create config with empty plugin_dirs
        config = SystemConfig(plugin_dirs=[])

        # Mock get_config in settings module
        with patch('brainsmith.settings.get_config', return_value=config):
            dirs = loader._get_plugin_dirs()
            # effective_plugin_dirs should return default
            assert dirs == [Path.home() / '.brainsmith' / 'plugins']


class TestErrorMessages:
    """Test that error messages are helpful."""

    def test_step_not_found_error_message(self):
        """Test that step not found error provides helpful info."""
        with pytest.raises(KeyError) as exc_info:
            loader.get_step('nonexistent_step')

        error_msg = str(exc_info.value)
        assert 'not found' in error_msg
        # Should suggest available steps
        assert 'Available steps' in error_msg or 'qonnx_to_finn' in error_msg

    def test_kernel_not_found_error_message(self):
        """Test that kernel not found error provides helpful info."""
        with pytest.raises(KeyError) as exc_info:
            loader.get_kernel('NonexistentKernel')

        error_msg = str(exc_info.value)
        assert 'not found' in error_msg
        assert 'Available kernels' in error_msg or 'LayerNorm' in error_msg

    def test_backend_not_found_error_message(self):
        """Test that backend not found error shows available backends."""
        with pytest.raises(KeyError) as exc_info:
            loader.get_backend('LayerNorm', 'rtl')  # LayerNorm only has HLS

        error_msg = str(exc_info.value)
        assert 'not available' in error_msg
        assert 'Available' in error_msg
