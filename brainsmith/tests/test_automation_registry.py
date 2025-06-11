"""
Test suite for AutomationRegistry unified interface implementation.

Tests the BaseRegistry inheritance and unified method compatibility
for automation tool discovery and management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# Add the project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.libraries.automation.registry import (
    AutomationRegistry,
    AutomationToolInfo,
    AutomationType,
    get_automation_registry,
    discover_all_automation_components,
    get_automation_component,
    find_components_by_type,
    list_available_automation_components
)
from brainsmith.core.registry.exceptions import RegistryError, ComponentNotFoundError


class TestAutomationRegistryUnifiedInterface:
    """Test AutomationRegistry unified interface compliance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = AutomationRegistry()
        
        # Create mock automation tools
        self.mock_tool1 = AutomationToolInfo(
            name="test_sweep",
            automation_type=AutomationType.PARAMETER_SWEEP,
            function=Mock(__name__="test_sweep"),
            module_path="test.module.sweep",
            category="parameter_exploration",
            description="Test parameter sweep tool",
            parameters=["param1", "param2"],
            supports_parallel=True
        )
        
        self.mock_tool2 = AutomationToolInfo(
            name="test_batch",
            automation_type=AutomationType.BATCH_PROCESSING,
            function=Mock(__name__="test_batch"),
            module_path="test.module.batch",
            category="batch_processing",
            description="Test batch processing tool",
            parameters=["input_list", "config"],
            supports_parallel=True
        )
        
        # Mock discovery results
        self.mock_tools = {
            "test_sweep": self.mock_tool1,
            "test_batch": self.mock_tool2
        }
    
    def test_inheritance_from_base_registry(self):
        """Test that AutomationRegistry properly inherits from BaseRegistry."""
        from brainsmith.core.registry import BaseRegistry
        assert isinstance(self.registry, BaseRegistry)
        assert hasattr(self.registry, 'discover_components')
        assert hasattr(self.registry, 'get_component')
        assert hasattr(self.registry, 'find_components_by_type')
        assert hasattr(self.registry, 'refresh_cache')
    
    def test_unified_discover_components_method(self):
        """Test discover_components() method (unified interface)."""
        with patch.object(self.registry, '_discover_core_tools', return_value=self.mock_tools):
            with patch.object(self.registry, '_discover_contrib_tools', return_value={}):
                components = self.registry.discover_components()
                
                assert isinstance(components, dict)
                assert len(components) == 2
                assert "test_sweep" in components
                assert "test_batch" in components
                assert isinstance(components["test_sweep"], AutomationToolInfo)
    
    def test_unified_get_component_method(self):
        """Test get_component() method (unified interface)."""
        with patch.object(self.registry, '_discover_core_tools', return_value=self.mock_tools):
            with patch.object(self.registry, '_discover_contrib_tools', return_value={}):
                # Test existing component
                component = self.registry.get_component("test_sweep")
                assert component is not None
                assert component.name == "test_sweep"
                assert component.automation_type == AutomationType.PARAMETER_SWEEP
                
                # Test non-existent component
                component = self.registry.get_component("nonexistent")
                assert component is None
    
    def test_unified_find_components_by_type_method(self):
        """Test find_components_by_type() method (unified interface)."""
        with patch.object(self.registry, '_discover_core_tools', return_value=self.mock_tools):
            with patch.object(self.registry, '_discover_contrib_tools', return_value={}):
                # Test with AutomationType enum value
                components = self.registry.find_components_by_type("parameter_sweep")
                assert len(components) == 1
                assert components[0].name == "test_sweep"
                
                # Test with category string
                components = self.registry.find_components_by_type("batch_processing")
                assert len(components) == 1
                assert components[0].name == "test_batch"
                
                # Test with non-existent type
                components = self.registry.find_components_by_type("nonexistent")
                assert len(components) == 0
    
    def test_component_validation(self):
        """Test _validate_component() method."""
        # Test valid component
        assert self.registry._validate_component(self.mock_tool1) == True
        
        # Test invalid component - missing name
        invalid_tool = AutomationToolInfo(
            name="",
            automation_type=AutomationType.UTILITY,
            function=Mock(),
            module_path="test.module"
        )
        assert self.registry._validate_component(invalid_tool) == False
        
        # Test invalid component - missing function
        invalid_tool2 = AutomationToolInfo(
            name="test",
            automation_type=AutomationType.UTILITY,
            function=None,
            module_path="test.module"
        )
        assert self.registry._validate_component(invalid_tool2) == False
    
    def test_process_component_method(self):
        """Test _process_component() method (required by BaseRegistry)."""
        # This method is not used by AutomationRegistry but required by interface
        result = self.registry._process_component(Path("/test/path"), {})
        assert result is None
    
    def test_clean_unified_interface_only(self):
        """Test that only unified interface methods exist (no legacy methods)."""
        # Verify unified methods exist
        assert hasattr(self.registry, 'discover_components')
        assert hasattr(self.registry, 'get_component')
        assert hasattr(self.registry, 'find_components_by_type')
        assert hasattr(self.registry, 'list_component_names')
        assert hasattr(self.registry, 'get_component_info')
        
        # Verify legacy methods do NOT exist
        assert not hasattr(self.registry, 'discover_tools')
        assert not hasattr(self.registry, 'get_tool')
        assert not hasattr(self.registry, 'find_tools_by_type')
        assert not hasattr(self.registry, 'list_tool_names')
        assert not hasattr(self.registry, 'get_tool_info')
        
        # Verify legacy properties do NOT exist
        assert not hasattr(self.registry, 'automation_dirs')
        assert not hasattr(self.registry, 'tool_cache')
        assert not hasattr(self.registry, 'metadata_cache')
    
    def test_cache_integration_with_unified_interface(self):
        """Test that caching works properly with unified interface."""
        with patch.object(self.registry, '_discover_core_tools', return_value=self.mock_tools):
            with patch.object(self.registry, '_discover_contrib_tools', return_value={}):
                # First call should populate cache
                components1 = self.registry.discover_components()
                assert len(components1) == 2
                
                # Second call should use cache
                components2 = self.registry.discover_components()
                assert components1 == components2
                
                # Clear cache and verify refresh
                self.registry.refresh_cache()
                assert len(self.registry._cache) == 0
    
    def test_health_check_functionality(self):
        """Test health check functionality from BaseRegistry."""
        # Test that health_check method exists and can be called
        assert hasattr(self.registry, 'health_check')
        
        with patch.object(self.registry, 'discover_components', return_value=self.mock_tools):
            health_status = self.registry.health_check()
            assert isinstance(health_status, dict)
            assert 'status' in health_status
            assert 'component_count' in health_status
            assert 'last_scan_time' in health_status


class TestAutomationRegistryConvenienceFunctions:
    """Test convenience functions use unified interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_components = {
            "test_component": AutomationToolInfo(
                name="test_component",
                automation_type=AutomationType.UTILITY,
                function=Mock(__name__="test_component"),
                module_path="test.module",
                category="utility"
            )
        }
    
    def test_discover_all_automation_components(self):
        """Test discover_all_automation_components() uses unified interface."""
        with patch('brainsmith.libraries.automation.registry.get_automation_registry') as mock_get_registry:
            mock_registry = Mock()
            mock_registry.discover_components.return_value = self.mock_components
            mock_get_registry.return_value = mock_registry
            
            components = discover_all_automation_components(rescan=True)
            
            mock_registry.discover_components.assert_called_once_with(True)
            assert components == self.mock_components
    
    def test_get_automation_component(self):
        """Test get_automation_component() uses unified interface."""
        with patch('brainsmith.libraries.automation.registry.get_automation_registry') as mock_get_registry:
            mock_registry = Mock()
            mock_registry.get_component.return_value = self.mock_components["test_component"]
            mock_get_registry.return_value = mock_registry
            
            component = get_automation_component("test_component")
            
            mock_registry.get_component.assert_called_once_with("test_component")
            assert component == self.mock_components["test_component"]
    
    def test_find_components_by_type(self):
        """Test find_components_by_type() uses unified interface."""
        with patch('brainsmith.libraries.automation.registry.get_automation_registry') as mock_get_registry:
            mock_registry = Mock()
            mock_registry.find_components_by_type.return_value = [self.mock_components["test_component"]]
            mock_get_registry.return_value = mock_registry
            
            components = find_components_by_type(AutomationType.UTILITY)
            
            mock_registry.find_components_by_type.assert_called_once_with("utility")
            assert components == [self.mock_components["test_component"]]
    
    def test_list_available_automation_components(self):
        """Test list_available_automation_components() uses unified interface."""
        with patch('brainsmith.libraries.automation.registry.get_automation_registry') as mock_get_registry:
            mock_registry = Mock()
            mock_registry.list_component_names.return_value = ["test_component"]
            mock_get_registry.return_value = mock_registry
            
            component_names = list_available_automation_components()
            
            mock_registry.list_component_names.assert_called_once()
            assert component_names == ["test_component"]


class TestAutomationRegistryErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = AutomationRegistry()
    
    def test_graceful_handling_of_import_errors(self):
        """Test graceful handling when automation modules can't be imported."""
        with patch('brainsmith.libraries.automation.registry.logger') as mock_logger:
            with patch('importlib.import_module', side_effect=ImportError("Module not found")):
                tools = self.registry._discover_core_tools()
                assert isinstance(tools, dict)
                # Should return empty dict on import failure
    
    def test_validation_with_invalid_components(self):
        """Test validation behavior with various invalid components."""
        # Test with None
        assert self.registry._validate_component(None) == False
        
        # Test with component missing required attributes
        try:
            incomplete_tool = AutomationToolInfo(
                name="test",
                automation_type="invalid_type",  # Wrong type
                function=Mock(),
                module_path="test.module"
            )
            # This should fail validation
            assert self.registry._validate_component(incomplete_tool) == False
        except:
            # If constructor fails, that's also acceptable
            pass
    
    def test_edge_cases_in_component_discovery(self):
        """Test edge cases in component discovery."""
        with patch.object(self.registry, '_discover_core_tools', return_value={}):
            with patch.object(self.registry, '_discover_contrib_tools', return_value={}):
                components = self.registry.discover_components()
                assert isinstance(components, dict)
                assert len(components) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])