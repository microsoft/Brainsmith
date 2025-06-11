"""
Tests for Blueprint Library Registry

Tests the refactored BlueprintLibraryRegistry with unified BaseRegistry interface.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from brainsmith.libraries.blueprints.registry import (
    BlueprintLibraryRegistry, 
    BlueprintInfo, 
    BlueprintCategory,
    get_blueprint_library_registry
)
from brainsmith.core.registry.exceptions import ComponentNotFoundError, ValidationError


class TestBlueprintLibraryRegistry:
    """Test suite for BlueprintLibraryRegistry unified interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = BlueprintLibraryRegistry(search_dirs=[self.temp_dir])
        
        # Create mock blueprint info
        self.mock_blueprint = BlueprintInfo(
            name="test_blueprint",
            category=BlueprintCategory.BASIC,
            file_path=os.path.join(self.temp_dir, "basic", "test_blueprint.yaml"),
            version="1.0",
            description="Test blueprint",
            model_type="cnn",
            target_platform="fpga",
            parameters={"param1": "value1"},
            targets={"target1": "config1"}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_blueprint_file(self, category: str, blueprint_name: str, blueprint_data: dict):
        """Create a mock blueprint YAML file."""
        category_dir = os.path.join(self.temp_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        blueprint_file = os.path.join(category_dir, f"{blueprint_name}.yaml")
        with open(blueprint_file, 'w') as f:
            yaml.dump(blueprint_data, f)
        
        return blueprint_file
    
    def test_initialization(self):
        """Test registry initialization."""
        assert isinstance(self.registry, BlueprintLibraryRegistry)
        assert self.registry.search_dirs == [self.temp_dir]
        assert hasattr(self.registry, 'blueprint_cache')  # Backward compatibility
        assert self.registry.blueprint_cache is self.registry._cache
    
    def test_discover_components_empty_dir(self):
        """Test discovery with empty directory."""
        components = self.registry.discover_components()
        assert components == {}
        assert isinstance(components, dict)
    
    def test_discover_components_with_blueprints(self):
        """Test discovery with blueprint files."""
        # Create mock blueprint files
        blueprint_data_1 = {
            'name': 'cnn_accelerator',
            'description': 'CNN accelerator blueprint',
            'model_type': 'cnn',
            'target_platform': 'fpga',
            'version': '1.0',
            'parameters': {
                'pe_count': {'range': [1, 64], 'default': 8},
                'simd_width': {'range': [1, 32], 'default': 4}
            }
        }
        blueprint_data_2 = {
            'name': 'mobilenet_accelerator',
            'description': 'MobileNet accelerator blueprint',
            'model_type': 'mobilenet',
            'target_platform': 'fpga',
            'version': '2.0',
            'parameters': {
                'depth_multiplier': {'values': [0.25, 0.5, 0.75, 1.0], 'default': 1.0}
            }
        }
        
        self._create_mock_blueprint_file('basic', 'cnn_accelerator', blueprint_data_1)
        self._create_mock_blueprint_file('advanced', 'mobilenet_accelerator', blueprint_data_2)
        
        components = self.registry.discover_components()
        
        assert len(components) == 2
        assert 'cnn_accelerator' in components
        assert 'mobilenet_accelerator' in components
        assert components['cnn_accelerator'].category == BlueprintCategory.BASIC
        assert components['mobilenet_accelerator'].category == BlueprintCategory.ADVANCED
    
    def test_discover_components_caching(self):
        """Test that discover_components caches results."""
        blueprint_data = {
            'name': 'test_blueprint',
            'description': 'Test blueprint',
            'model_type': 'test',
            'target_platform': 'test',
            'version': '1.0'
        }
        self._create_mock_blueprint_file('basic', 'test_blueprint', blueprint_data)
        
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
        blueprint_data = {
            'name': 'test_blueprint',
            'description': 'Test blueprint',
            'model_type': 'test',
            'target_platform': 'test',
            'version': '1.0'
        }
        self._create_mock_blueprint_file('basic', 'test_blueprint', blueprint_data)
        
        # Get existing component
        component = self.registry.get_component('test_blueprint')
        assert component is not None
        assert component.name == 'test_blueprint'
        
        # Get non-existent component
        component = self.registry.get_component('non_existent')
        assert component is None
    
    def test_find_components_by_type(self):
        """Test finding components by blueprint category."""
        blueprint_data_1 = {
            'name': 'basic1',
            'description': 'Basic blueprint 1',
            'model_type': 'cnn',
            'target_platform': 'fpga',
            'version': '1.0'
        }
        blueprint_data_2 = {
            'name': 'basic2',
            'description': 'Basic blueprint 2',
            'model_type': 'rnn',
            'target_platform': 'fpga',
            'version': '1.0'
        }
        blueprint_data_3 = {
            'name': 'advanced1',
            'description': 'Advanced blueprint 1',
            'model_type': 'transformer',
            'target_platform': 'fpga',
            'version': '1.0'
        }
        
        self._create_mock_blueprint_file('basic', 'basic1', blueprint_data_1)
        self._create_mock_blueprint_file('basic', 'basic2', blueprint_data_2)
        self._create_mock_blueprint_file('advanced', 'advanced1', blueprint_data_3)
        
        # Find basic blueprints
        basic_blueprints = self.registry.find_components_by_type(BlueprintCategory.BASIC)
        assert len(basic_blueprints) == 2
        assert all(b.category == BlueprintCategory.BASIC for b in basic_blueprints)
        
        # Find advanced blueprints
        advanced_blueprints = self.registry.find_components_by_type(BlueprintCategory.ADVANCED)
        assert len(advanced_blueprints) == 1
        assert advanced_blueprints[0].category == BlueprintCategory.ADVANCED
    
    def test_find_blueprints_by_model_type(self):
        """Test finding blueprints by model type."""
        blueprint_data_1 = {
            'name': 'cnn1',
            'description': 'CNN blueprint 1',
            'model_type': 'cnn',
            'target_platform': 'fpga',
            'version': '1.0'
        }
        blueprint_data_2 = {
            'name': 'cnn2',
            'description': 'CNN blueprint 2',
            'model_type': 'CNN',  # Different case
            'target_platform': 'asic',
            'version': '1.0'
        }
        blueprint_data_3 = {
            'name': 'rnn1',
            'description': 'RNN blueprint',
            'model_type': 'rnn',
            'target_platform': 'fpga',
            'version': '1.0'
        }
        
        self._create_mock_blueprint_file('basic', 'cnn1', blueprint_data_1)
        self._create_mock_blueprint_file('basic', 'cnn2', blueprint_data_2)
        self._create_mock_blueprint_file('advanced', 'rnn1', blueprint_data_3)
        
        # Find CNN blueprints (case insensitive)
        cnn_blueprints = self.registry.find_blueprints_by_model_type('cnn')
        assert len(cnn_blueprints) == 2
        
        # Find RNN blueprints
        rnn_blueprints = self.registry.find_blueprints_by_model_type('rnn')
        assert len(rnn_blueprints) == 1
    
    def test_find_blueprints_by_platform(self):
        """Test finding blueprints by target platform."""
        blueprint_data_1 = {
            'name': 'fpga1',
            'description': 'FPGA blueprint 1',
            'model_type': 'cnn',
            'target_platform': 'fpga',
            'version': '1.0'
        }
        blueprint_data_2 = {
            'name': 'fpga2',
            'description': 'FPGA blueprint 2',
            'model_type': 'rnn',
            'target_platform': 'xilinx_fpga',
            'version': '1.0'
        }
        blueprint_data_3 = {
            'name': 'asic1',
            'description': 'ASIC blueprint',
            'model_type': 'cnn',
            'target_platform': 'asic',
            'version': '1.0'
        }
        
        self._create_mock_blueprint_file('basic', 'fpga1', blueprint_data_1)
        self._create_mock_blueprint_file('basic', 'fpga2', blueprint_data_2)
        self._create_mock_blueprint_file('advanced', 'asic1', blueprint_data_3)
        
        # Find FPGA blueprints (partial matching)
        fpga_blueprints = self.registry.find_blueprints_by_platform('fpga')
        assert len(fpga_blueprints) == 2
        
        # Find ASIC blueprints
        asic_blueprints = self.registry.find_blueprints_by_platform('asic')
        assert len(asic_blueprints) == 1
    
    def test_list_component_names(self):
        """Test listing component names."""
        blueprint_data_1 = {'name': 'blueprint1', 'description': 'Blueprint 1', 'model_type': 'cnn', 'target_platform': 'fpga', 'version': '1.0'}
        blueprint_data_2 = {'name': 'blueprint2', 'description': 'Blueprint 2', 'model_type': 'rnn', 'target_platform': 'asic', 'version': '1.0'}
        
        self._create_mock_blueprint_file('basic', 'blueprint1', blueprint_data_1)
        self._create_mock_blueprint_file('advanced', 'blueprint2', blueprint_data_2)
        
        names = self.registry.list_component_names()
        assert set(names) == {'blueprint1', 'blueprint2'}
    
    def test_list_categories(self):
        """Test listing categories."""
        blueprint_data_1 = {'name': 'basic1', 'description': 'Basic blueprint', 'model_type': 'cnn', 'target_platform': 'fpga', 'version': '1.0'}
        blueprint_data_2 = {'name': 'advanced1', 'description': 'Advanced blueprint', 'model_type': 'rnn', 'target_platform': 'fpga', 'version': '1.0'}
        
        self._create_mock_blueprint_file('basic', 'basic1', blueprint_data_1)
        self._create_mock_blueprint_file('advanced', 'advanced1', blueprint_data_2)
        
        categories = self.registry.list_categories()
        assert BlueprintCategory.BASIC in categories
        assert BlueprintCategory.ADVANCED in categories
    
    def test_list_model_types(self):
        """Test listing model types."""
        blueprint_data_1 = {'name': 'cnn_bp', 'description': 'CNN blueprint', 'model_type': 'cnn', 'target_platform': 'fpga', 'version': '1.0'}
        blueprint_data_2 = {'name': 'rnn_bp', 'description': 'RNN blueprint', 'model_type': 'rnn', 'target_platform': 'fpga', 'version': '1.0'}
        
        self._create_mock_blueprint_file('basic', 'cnn_bp', blueprint_data_1)
        self._create_mock_blueprint_file('basic', 'rnn_bp', blueprint_data_2)
        
        model_types = self.registry.list_model_types()
        assert 'cnn' in model_types
        assert 'rnn' in model_types
    
    def test_list_platforms(self):
        """Test listing platforms."""
        blueprint_data_1 = {'name': 'fpga_bp', 'description': 'FPGA blueprint', 'model_type': 'cnn', 'target_platform': 'fpga', 'version': '1.0'}
        blueprint_data_2 = {'name': 'asic_bp', 'description': 'ASIC blueprint', 'model_type': 'cnn', 'target_platform': 'asic', 'version': '1.0'}
        
        self._create_mock_blueprint_file('basic', 'fpga_bp', blueprint_data_1)
        self._create_mock_blueprint_file('basic', 'asic_bp', blueprint_data_2)
        
        platforms = self.registry.list_platforms()
        assert 'fpga' in platforms
        assert 'asic' in platforms
    
    def test_get_component_info(self):
        """Test getting component info."""
        blueprint_data = {
            'name': 'test_blueprint',
            'description': 'Test description',
            'model_type': 'cnn',
            'target_platform': 'fpga',
            'version': '1.5',
            'parameters': {'param1': 'value1', 'param2': 'value2'},
            'targets': {'target1': 'config1'}
        }
        self._create_mock_blueprint_file('basic', 'test_blueprint', blueprint_data)
        
        info = self.registry.get_component_info('test_blueprint')
        assert info is not None
        assert info['name'] == 'test_blueprint'
        assert info['type'] == 'blueprint'
        assert info['category'] == 'basic'
        assert info['version'] == '1.5'
        assert info['model_type'] == 'cnn'
        assert info['target_platform'] == 'fpga'
        assert info['parameter_count'] == 2
        assert info['has_targets'] == True
    
    def test_load_blueprint_yaml(self):
        """Test loading blueprint YAML content."""
        blueprint_data = {
            'name': 'test_blueprint',
            'description': 'Test blueprint',
            'model_type': 'cnn',
            'target_platform': 'fpga',
            'version': '1.0',
            'parameters': {'param1': 'value1'}
        }
        self._create_mock_blueprint_file('basic', 'test_blueprint', blueprint_data)
        
        yaml_content = self.registry.load_blueprint_yaml('test_blueprint')
        assert yaml_content is not None
        assert yaml_content['name'] == 'test_blueprint'
        assert yaml_content['parameters']['param1'] == 'value1'
        
        # Test non-existent blueprint
        yaml_content = self.registry.load_blueprint_yaml('non_existent')
        assert yaml_content is None
    
    def test_validate_component(self):
        """Test component validation."""
        # Create valid blueprint
        valid_blueprint_data = {
            'name': 'valid_blueprint',
            'description': 'Valid blueprint',
            'model_type': 'cnn',
            'target_platform': 'fpga',
            'version': '1.0',
            'parameters': {
                'pe_count': {'range': [1, 64], 'default': 8},
                'simd_width': {'values': [1, 2, 4, 8], 'default': 4}
            }
        }
        self._create_mock_blueprint_file('basic', 'valid_blueprint', valid_blueprint_data)
        
        is_valid, errors = self.registry.validate_component('valid_blueprint')
        assert is_valid == True
        assert len(errors) == 0
        
        # Create invalid blueprint (missing required fields)
        invalid_blueprint_data = {
            'version': '1.0',
            'parameters': 'invalid_structure'  # Should be dict
        }
        self._create_mock_blueprint_file('basic', 'invalid_blueprint', invalid_blueprint_data)
        
        is_valid, errors = self.registry.validate_component('invalid_blueprint')
        assert is_valid == False
        assert len(errors) > 0
        
        # Test validation of non-existent blueprint
        is_valid, errors = self.registry.validate_component('non_existent')
        assert is_valid == False
        assert len(errors) > 0
    
    def test_refresh_cache(self):
        """Test cache refresh."""
        blueprint_data = {'name': 'test_blueprint', 'description': 'Test', 'model_type': 'cnn', 'target_platform': 'fpga', 'version': '1.0'}
        self._create_mock_blueprint_file('basic', 'test_blueprint', blueprint_data)
        
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
        assert 'total_components' in status
        assert 'valid_components' in status
    
    def test_get_default_dirs(self):
        """Test getting default directories."""
        registry = BlueprintLibraryRegistry()  # No search_dirs provided
        default_dirs = registry._get_default_dirs()
        assert len(default_dirs) > 0
        assert all(isinstance(d, str) for d in default_dirs)


class TestBlueprintLibraryRegistryConvenience:
    """Test convenience functions."""
    
    def test_get_blueprint_library_registry(self):
        """Test global registry getter."""
        registry1 = get_blueprint_library_registry()
        registry2 = get_blueprint_library_registry()
        assert registry1 is registry2  # Should return same instance
    
    @patch('brainsmith.libraries.blueprints.registry.get_blueprint_library_registry')
    def test_discover_all_blueprints(self, mock_get_registry):
        """Test discover_all_blueprints convenience function."""
        from brainsmith.libraries.blueprints.registry import discover_all_blueprints
        
        mock_registry = Mock()
        mock_registry.discover_components.return_value = {'test': 'blueprint'}
        mock_get_registry.return_value = mock_registry
        
        result = discover_all_blueprints(rescan=True)
        
        mock_registry.discover_components.assert_called_once_with(True)
        assert result == {'test': 'blueprint'}
    
    @patch('brainsmith.libraries.blueprints.registry.get_blueprint_library_registry')
    def test_get_blueprint_by_name(self, mock_get_registry):
        """Test get_blueprint_by_name convenience function."""
        from brainsmith.libraries.blueprints.registry import get_blueprint_by_name
        
        mock_registry = Mock()
        mock_registry.get_component.return_value = 'mock_blueprint'
        mock_get_registry.return_value = mock_registry
        
        result = get_blueprint_by_name('test_blueprint')
        
        mock_registry.get_component.assert_called_once_with('test_blueprint')
        assert result == 'mock_blueprint'
    
    @patch('brainsmith.libraries.blueprints.registry.get_blueprint_library_registry')
    def test_find_blueprints_by_category(self, mock_get_registry):
        """Test find_blueprints_by_category convenience function."""
        from brainsmith.libraries.blueprints.registry import find_blueprints_by_category
        
        mock_registry = Mock()
        mock_registry.find_components_by_type.return_value = ['blueprint1', 'blueprint2']
        mock_get_registry.return_value = mock_registry
        
        result = find_blueprints_by_category(BlueprintCategory.BASIC)
        
        mock_registry.find_components_by_type.assert_called_once_with(BlueprintCategory.BASIC)
        assert result == ['blueprint1', 'blueprint2']
    
    @patch('brainsmith.libraries.blueprints.registry.get_blueprint_library_registry')
    def test_list_available_blueprints(self, mock_get_registry):
        """Test list_available_blueprints convenience function."""
        from brainsmith.libraries.blueprints.registry import list_available_blueprints
        
        mock_registry = Mock()
        mock_registry.list_component_names.return_value = ['blueprint1', 'blueprint2']
        mock_get_registry.return_value = mock_registry
        
        result = list_available_blueprints()
        
        mock_registry.list_component_names.assert_called_once()
        assert result == ['blueprint1', 'blueprint2']
    
    @patch('brainsmith.libraries.blueprints.registry.get_blueprint_library_registry')
    def test_refresh_blueprint_library_registry(self, mock_get_registry):
        """Test refresh_blueprint_library_registry convenience function."""
        from brainsmith.libraries.blueprints.registry import refresh_blueprint_library_registry
        
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        refresh_blueprint_library_registry()
        
        mock_registry.refresh_cache.assert_called_once()


class TestBlueprintCategory:
    """Test BlueprintCategory enum."""
    
    def test_blueprint_category_values(self):
        """Test BlueprintCategory enum values."""
        assert BlueprintCategory.BASIC.value == "basic"
        assert BlueprintCategory.ADVANCED.value == "advanced"
        assert BlueprintCategory.EXPERIMENTAL.value == "experimental"
        assert BlueprintCategory.CUSTOM.value == "custom"


class TestBlueprintInfo:
    """Test BlueprintInfo dataclass."""
    
    def test_blueprint_info_creation(self):
        """Test BlueprintInfo creation."""
        info = BlueprintInfo(
            name="test",
            category=BlueprintCategory.BASIC,
            file_path="/path/to/test.yaml"
        )
        
        assert info.name == "test"
        assert info.category == BlueprintCategory.BASIC
        assert info.file_path == "/path/to/test.yaml"
        assert info.version == "1.0"  # Default value
        assert info.parameters == {}  # Default value set by __post_init__
        assert info.targets == {}  # Default value set by __post_init__
    
    def test_blueprint_info_with_parameters(self):
        """Test BlueprintInfo with explicit parameters."""
        params = {"param1": "value1", "param2": "value2"}
        targets = {"target1": "config1"}
        
        info = BlueprintInfo(
            name="test",
            category=BlueprintCategory.BASIC,
            file_path="/path/to/test.yaml",
            parameters=params,
            targets=targets
        )
        
        assert info.parameters == params
        assert info.targets == targets


if __name__ == '__main__':
    pytest.main([__file__])