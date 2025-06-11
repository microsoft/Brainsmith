"""
Tests for Kernel Registry

Tests the refactored KernelRegistry with unified BaseRegistry interface.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from brainsmith.libraries.kernels.registry import KernelRegistry, get_kernel_registry
from brainsmith.libraries.kernels.types import KernelPackage, OperatorType, BackendType
from brainsmith.core.registry.exceptions import ComponentNotFoundError, ValidationError


class TestKernelRegistry:
    """Test suite for KernelRegistry unified interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = KernelRegistry(search_dirs=[self.temp_dir])
        
        # Create mock kernel package
        self.mock_kernel = KernelPackage(
            name="test_kernel",
            operator_type="Convolution",  # String, not enum
            backend="RTL",  # String, not enum
            version="1.0.0",
            author="Test Author",
            license="MIT",
            description="Test kernel package",
            parameters={"param1": "value1"},
            files={"main": "main.py", "config": "config.yaml"},
            performance={"latency": "1ms"},
            validation={"verified": True, "last_tested": "2024-01-01"},
            repository={"url": "https://github.com/test/test"},
            package_path=os.path.join(self.temp_dir, "test_kernel")
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_kernel_dir(self, kernel_name: str, kernel_data: dict):
        """Create a mock kernel directory with kernel.yaml."""
        kernel_dir = os.path.join(self.temp_dir, kernel_name)
        os.makedirs(kernel_dir, exist_ok=True)
        
        kernel_yaml_path = os.path.join(kernel_dir, "kernel.yaml")
        with open(kernel_yaml_path, 'w') as f:
            yaml.dump(kernel_data, f)
        
        return kernel_dir
    
    def test_initialization(self):
        """Test registry initialization."""
        assert isinstance(self.registry, KernelRegistry)
        assert self.registry.search_dirs == [self.temp_dir]
        assert hasattr(self.registry, 'kernel_cache')  # Backward compatibility
        assert self.registry.kernel_cache is self.registry._cache
    
    def test_discover_components_empty_dir(self):
        """Test discovery with empty directory."""
        components = self.registry.discover_components()
        assert components == {}
        assert isinstance(components, dict)
    
    def test_discover_components_with_kernels(self):
        """Test discovery with kernel packages."""
        # Create mock kernel directories
        kernel_data_1 = {
            'name': 'conv_kernel',
            'operator_type': 'Convolution',
            'backend': 'RTL',
            'version': '1.0.0',
            'description': 'Convolution kernel'
        }
        kernel_data_2 = {
            'name': 'pool_kernel',
            'operator_type': 'Pool',
            'backend': 'HLS',
            'version': '2.0.0',
            'description': 'Pooling kernel'
        }
        
        self._create_mock_kernel_dir('conv_kernel', kernel_data_1)
        self._create_mock_kernel_dir('pool_kernel', kernel_data_2)
        
        components = self.registry.discover_components()
        
        assert len(components) == 2
        assert 'conv_kernel' in components
        assert 'pool_kernel' in components
        assert components['conv_kernel'].name == 'conv_kernel'
        assert components['pool_kernel'].operator_type == OperatorType.POOL
    
    def test_discover_components_caching(self):
        """Test that discover_components caches results."""
        kernel_data = {
            'name': 'test_kernel',
            'operator_type': 'Convolution',
            'backend': 'RTL',
            'version': '1.0.0'
        }
        self._create_mock_kernel_dir('test_kernel', kernel_data)
        
        # First call
        components1 = self.registry.discover_components()
        
        # Second call without rescan should return cached result
        components2 = self.registry.discover_components()
        
        assert components1 is components2  # Same object reference
        
        # Force rescan
        components3 = self.registry.discover_components(rescan=True)
        assert len(components3) == len(components1)
    
    def test_get_component(self):
        """Test getting specific component."""
        kernel_data = {
            'name': 'test_kernel',
            'operator_type': 'Convolution',
            'backend': 'RTL',
            'version': '1.0.0'
        }
        self._create_mock_kernel_dir('test_kernel', kernel_data)
        
        # Get existing component
        component = self.registry.get_component('test_kernel')
        assert component is not None
        assert component.name == 'test_kernel'
        
        # Get non-existent component
        component = self.registry.get_component('non_existent')
        assert component is None
    
    def test_find_components_by_type(self):
        """Test finding components by operator type."""
        kernel_data_1 = {
            'name': 'conv1',
            'operator_type': 'Convolution',
            'backend': 'RTL',
            'version': '1.0.0'
        }
        kernel_data_2 = {
            'name': 'conv2',
            'operator_type': 'Convolution',
            'backend': 'HLS',
            'version': '1.0.0'
        }
        kernel_data_3 = {
            'name': 'pool1',
            'operator_type': 'Pool',
            'backend': 'RTL',
            'version': '1.0.0'
        }
        
        self._create_mock_kernel_dir('conv1', kernel_data_1)
        self._create_mock_kernel_dir('conv2', kernel_data_2)
        self._create_mock_kernel_dir('pool1', kernel_data_3)
        
        # Find convolution kernels
        conv_kernels = self.registry.find_components_by_type(OperatorType.CONVOLUTION)
        assert len(conv_kernels) == 2
        assert all(k.operator_type == OperatorType.CONVOLUTION for k in conv_kernels)
        
        # Find pooling kernels
        pool_kernels = self.registry.find_components_by_type(OperatorType.POOL)
        assert len(pool_kernels) == 1
        assert pool_kernels[0].operator_type == OperatorType.POOL
    
    def test_find_kernels_by_backend(self):
        """Test finding kernels by backend type."""
        kernel_data_1 = {
            'name': 'rtl_kernel',
            'operator_type': 'Convolution',
            'backend': 'RTL',
            'version': '1.0.0'
        }
        kernel_data_2 = {
            'name': 'hls_kernel',
            'operator_type': 'Convolution',
            'backend': 'HLS',
            'version': '1.0.0'
        }
        
        self._create_mock_kernel_dir('rtl_kernel', kernel_data_1)
        self._create_mock_kernel_dir('hls_kernel', kernel_data_2)
        
        # Find RTL kernels
        rtl_kernels = self.registry.find_kernels_by_backend(BackendType.RTL)
        assert len(rtl_kernels) == 1
        assert rtl_kernels[0].backend == BackendType.RTL
        
        # Find HLS kernels
        hls_kernels = self.registry.find_kernels_by_backend(BackendType.HLS)
        assert len(hls_kernels) == 1
        assert hls_kernels[0].backend == BackendType.HLS
    
    def test_list_component_names(self):
        """Test listing component names."""
        kernel_data_1 = {'name': 'kernel1', 'operator_type': 'Convolution', 'backend': 'RTL', 'version': '1.0.0'}
        kernel_data_2 = {'name': 'kernel2', 'operator_type': 'Pool', 'backend': 'HLS', 'version': '1.0.0'}
        
        self._create_mock_kernel_dir('kernel1', kernel_data_1)
        self._create_mock_kernel_dir('kernel2', kernel_data_2)
        
        names = self.registry.list_component_names()
        assert set(names) == {'kernel1', 'kernel2'}
    
    def test_list_operator_types(self):
        """Test listing operator types."""
        kernel_data_1 = {'name': 'conv', 'operator_type': 'Convolution', 'backend': 'RTL', 'version': '1.0.0'}
        kernel_data_2 = {'name': 'pool', 'operator_type': 'Pool', 'backend': 'RTL', 'version': '1.0.0'}
        
        self._create_mock_kernel_dir('conv', kernel_data_1)
        self._create_mock_kernel_dir('pool', kernel_data_2)
        
        operator_types = self.registry.list_operator_types()
        assert OperatorType.CONVOLUTION in operator_types
        assert OperatorType.POOL in operator_types
    
    def test_list_backend_types(self):
        """Test listing backend types."""
        kernel_data_1 = {'name': 'rtl_kernel', 'operator_type': 'Convolution', 'backend': 'RTL', 'version': '1.0.0'}
        kernel_data_2 = {'name': 'hls_kernel', 'operator_type': 'Convolution', 'backend': 'HLS', 'version': '1.0.0'}
        
        self._create_mock_kernel_dir('rtl_kernel', kernel_data_1)
        self._create_mock_kernel_dir('hls_kernel', kernel_data_2)
        
        backend_types = self.registry.list_backend_types()
        assert BackendType.RTL in backend_types
        assert BackendType.HLS in backend_types
    
    def test_get_component_info(self):
        """Test getting component info."""
        kernel_data = {
            'name': 'test_kernel',
            'operator_type': 'Convolution',
            'backend': 'RTL',
            'version': '1.0.0',
            'description': 'Test description',
            'files': {'main': 'main.py'},
            'validation': {'verified': True, 'last_tested': '2024-01-01'}
        }
        self._create_mock_kernel_dir('test_kernel', kernel_data)
        
        info = self.registry.get_component_info('test_kernel')
        assert info is not None
        assert info['name'] == 'test_kernel'
        assert info['type'] == 'kernel'
        assert info['operator_type'] == 'Convolution'
        assert info['backend'] == 'RTL'
        assert info['verified'] == True
    
    def test_validate_component(self):
        """Test component validation."""
        # Create kernel with all required files
        kernel_data = {
            'name': 'valid_kernel',
            'operator_type': 'Convolution',
            'backend': 'RTL',
            'version': '1.0.0',
            'files': {'main': 'main.py'}
        }
        kernel_dir = self._create_mock_kernel_dir('valid_kernel', kernel_data)
        
        # Create the required file
        with open(os.path.join(kernel_dir, 'main.py'), 'w') as f:
            f.write('# Mock main file')
        
        is_valid, errors = self.registry.validate_component('valid_kernel')
        assert is_valid == True
        assert len(errors) == 0
        
        # Test validation of non-existent kernel
        is_valid, errors = self.registry.validate_component('non_existent')
        assert is_valid == False
        assert len(errors) > 0
    
    def test_refresh_cache(self):
        """Test cache refresh."""
        kernel_data = {'name': 'test_kernel', 'operator_type': 'Convolution', 'backend': 'RTL', 'version': '1.0.0'}
        self._create_mock_kernel_dir('test_kernel', kernel_data)
        
        # Populate cache
        self.registry.discover_components()
        assert len(self.registry._cache) > 0
        
        # Refresh cache
        self.registry.refresh_cache()
        assert len(self.registry._cache) == 0
    
    def test_health_check(self):
        """Test registry health check."""
        # Should work even with empty registry
        status = self.registry.health_check()
        assert 'registry_type' in status
        assert 'cache_size' in status
        assert 'search_dirs' in status
    
    def test_get_default_dirs(self):
        """Test getting default directories."""
        registry = KernelRegistry()  # No search_dirs provided
        default_dirs = registry._get_default_dirs()
        assert len(default_dirs) > 0
        assert all(isinstance(d, str) for d in default_dirs)


class TestKernelRegistryConvenience:
    """Test convenience functions."""
    
    def test_get_kernel_registry(self):
        """Test global registry getter."""
        registry1 = get_kernel_registry()
        registry2 = get_kernel_registry()
        assert registry1 is registry2  # Should return same instance
    
    @patch('brainsmith.libraries.kernels.registry.get_kernel_registry')
    def test_discover_all_kernels(self, mock_get_registry):
        """Test discover_all_kernels convenience function."""
        from brainsmith.libraries.kernels.registry import discover_all_kernels
        
        mock_registry = Mock()
        mock_registry.discover_components.return_value = {'test': 'kernel'}
        mock_get_registry.return_value = mock_registry
        
        result = discover_all_kernels(rescan=True)
        
        mock_registry.discover_components.assert_called_once_with(True)
        assert result == {'test': 'kernel'}
    
    @patch('brainsmith.libraries.kernels.registry.get_kernel_registry')
    def test_get_kernel_by_name(self, mock_get_registry):
        """Test get_kernel_by_name convenience function."""
        from brainsmith.libraries.kernels.registry import get_kernel_by_name
        
        mock_registry = Mock()
        mock_registry.get_component.return_value = 'mock_kernel'
        mock_get_registry.return_value = mock_registry
        
        result = get_kernel_by_name('test_kernel')
        
        mock_registry.get_component.assert_called_once_with('test_kernel')
        assert result == 'mock_kernel'
    
    @patch('brainsmith.libraries.kernels.registry.get_kernel_registry')
    def test_find_kernels_for_operator(self, mock_get_registry):
        """Test find_kernels_for_operator convenience function."""
        from brainsmith.libraries.kernels.registry import find_kernels_for_operator
        
        mock_registry = Mock()
        mock_registry.find_components_by_type.return_value = ['kernel1', 'kernel2']
        mock_get_registry.return_value = mock_registry
        
        result = find_kernels_for_operator(OperatorType.CONVOLUTION)
        
        mock_registry.find_components_by_type.assert_called_once_with(OperatorType.CONVOLUTION)
        assert result == ['kernel1', 'kernel2']
    
    @patch('brainsmith.libraries.kernels.registry.get_kernel_registry')
    def test_list_available_kernels(self, mock_get_registry):
        """Test list_available_kernels convenience function."""
        from brainsmith.libraries.kernels.registry import list_available_kernels
        
        mock_registry = Mock()
        mock_registry.list_component_names.return_value = ['kernel1', 'kernel2']
        mock_get_registry.return_value = mock_registry
        
        result = list_available_kernels()
        
        mock_registry.list_component_names.assert_called_once()
        assert result == ['kernel1', 'kernel2']
    
    @patch('brainsmith.libraries.kernels.registry.get_kernel_registry')
    def test_refresh_kernel_registry(self, mock_get_registry):
        """Test refresh_kernel_registry convenience function."""
        from brainsmith.libraries.kernels.registry import refresh_kernel_registry
        
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        refresh_kernel_registry()
        
        mock_registry.refresh_cache.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])