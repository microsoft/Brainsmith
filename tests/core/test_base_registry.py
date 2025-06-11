"""
Unit tests for BaseRegistry abstract base class.

Tests the core functionality of the unified registry interface including
abstract method contracts, common functionality, and error handling.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, List, Any
from dataclasses import dataclass

from brainsmith.core.registry import BaseRegistry, ComponentInfo
from brainsmith.core.registry.exceptions import ComponentNotFoundError, ValidationError


@dataclass
class MockComponent:
    """Mock component for testing."""
    name: str
    description: str
    component_type: str
    version: str = "1.0.0"


class MockComponentInfo(ComponentInfo):
    """Mock ComponentInfo implementation."""
    
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description


class MockRegistry(BaseRegistry[MockComponent]):
    """Mock registry implementation for testing."""
    
    def __init__(self, search_dirs=None, config_manager=None, mock_components=None):
        super().__init__(search_dirs, config_manager)
        self.mock_components = mock_components or {}
        self.discovery_call_count = 0
    
    def discover_components(self, rescan: bool = False) -> Dict[str, MockComponent]:
        """Mock implementation of component discovery."""
        if not rescan and self._cache:
            return self._cache
        
        # Only increment count when actually doing discovery
        self.discovery_call_count += 1
        
        # Simulate component discovery
        self._cache = self.mock_components.copy()
        return self._cache
    
    def find_components_by_type(self, component_type: str) -> List[MockComponent]:
        """Mock implementation of type-based search."""
        components = self.discover_components()
        return [comp for comp in components.values() if comp.component_type == component_type]
    
    def _get_default_dirs(self) -> List[str]:
        """Mock default directories."""
        return ["/mock/registry/dir"]
    
    def _extract_info(self, component: MockComponent) -> Dict[str, Any]:
        """Mock info extraction."""
        return {
            'name': component.name,
            'description': component.description,
            'type': component.component_type,
            'version': component.version
        }
    
    def _validate_component_implementation(self, component: MockComponent) -> tuple[bool, List[str]]:
        """Mock validation logic."""
        errors = []
        if not component.name:
            errors.append("Component name is required")
        if not component.description:
            errors.append("Component description is required")
        return len(errors) == 0, errors


class TestBaseRegistryInitialization:
    """Test BaseRegistry initialization and configuration."""
    
    def test_default_initialization(self):
        """Test registry initialization with defaults."""
        registry = MockRegistry()
        
        assert registry.search_dirs == ["/mock/registry/dir"]
        assert registry.config_manager is None
        assert len(registry._cache) == 0
        assert len(registry._metadata_cache) == 0
    
    def test_custom_search_dirs(self):
        """Test registry initialization with custom search directories."""
        custom_dirs = ["/custom/dir1", "/custom/dir2"]
        registry = MockRegistry(search_dirs=custom_dirs)
        
        assert registry.search_dirs == custom_dirs
    
    def test_config_manager_injection(self):
        """Test registry initialization with config manager."""
        mock_config = Mock()
        registry = MockRegistry(config_manager=mock_config)
        
        assert registry.config_manager == mock_config


class TestComponentDiscovery:
    """Test component discovery functionality."""
    
    def test_discover_components_caching(self):
        """Test that component discovery uses caching properly."""
        mock_components = {
            "comp1": MockComponent("comp1", "Component 1", "type1"),
            "comp2": MockComponent("comp2", "Component 2", "type2")
        }
        registry = MockRegistry(mock_components=mock_components)
        
        # First call should populate cache
        result1 = registry.discover_components()
        assert registry.discovery_call_count == 1
        assert len(result1) == 2
        
        # Second call should use cache
        result2 = registry.discover_components()
        assert registry.discovery_call_count == 1  # No additional call
        assert result1 == result2
    
    def test_discover_components_rescan(self):
        """Test forced rescan functionality."""
        mock_components = {
            "comp1": MockComponent("comp1", "Component 1", "type1")
        }
        registry = MockRegistry(mock_components=mock_components)
        
        # First discovery
        registry.discover_components()
        assert registry.discovery_call_count == 1
        
        # Force rescan
        registry.discover_components(rescan=True)
        assert registry.discovery_call_count == 2
    
    def test_get_component_existing(self):
        """Test retrieving an existing component."""
        mock_component = MockComponent("test_comp", "Test Component", "test_type")
        mock_components = {"test_comp": mock_component}
        registry = MockRegistry(mock_components=mock_components)
        
        result = registry.get_component("test_comp")
        assert result == mock_component
    
    def test_get_component_nonexistent(self):
        """Test retrieving a non-existent component."""
        registry = MockRegistry()
        
        result = registry.get_component("nonexistent")
        assert result is None
    
    def test_get_component_required_existing(self):
        """Test required component retrieval for existing component."""
        mock_component = MockComponent("test_comp", "Test Component", "test_type")
        mock_components = {"test_comp": mock_component}
        registry = MockRegistry(mock_components=mock_components)
        
        result = registry.get_component_required("test_comp")
        assert result == mock_component
    
    def test_get_component_required_nonexistent(self):
        """Test required component retrieval raises exception for missing component."""
        registry = MockRegistry()
        
        with pytest.raises(ComponentNotFoundError) as exc_info:
            registry.get_component_required("nonexistent")
        
        assert "nonexistent" in str(exc_info.value)
        assert "MockRegistry" in str(exc_info.value)


class TestComponentSearch:
    """Test component search functionality."""
    
    def test_list_component_names(self):
        """Test listing all component names."""
        mock_components = {
            "comp1": MockComponent("comp1", "Component 1", "type1"),
            "comp2": MockComponent("comp2", "Component 2", "type2")
        }
        registry = MockRegistry(mock_components=mock_components)
        
        names = registry.list_component_names()
        assert set(names) == {"comp1", "comp2"}
    
    def test_find_components_by_type(self):
        """Test finding components by type."""
        mock_components = {
            "comp1": MockComponent("comp1", "Component 1", "type1"),
            "comp2": MockComponent("comp2", "Component 2", "type1"),
            "comp3": MockComponent("comp3", "Component 3", "type2")
        }
        registry = MockRegistry(mock_components=mock_components)
        
        type1_components = registry.find_components_by_type("type1")
        assert len(type1_components) == 2
        assert all(comp.component_type == "type1" for comp in type1_components)
    
    def test_find_components_by_attribute(self):
        """Test finding components by arbitrary attributes."""
        mock_components = {
            "comp1": MockComponent("comp1", "Component 1", "type1", "1.0.0"),
            "comp2": MockComponent("comp2", "Component 2", "type1", "2.0.0"),
            "comp3": MockComponent("comp3", "Component 3", "type2", "1.0.0")
        }
        registry = MockRegistry(mock_components=mock_components)
        
        version_1_components = registry.find_components_by_attribute("version", "1.0.0")
        assert len(version_1_components) == 2
        assert all(comp.version == "1.0.0" for comp in version_1_components)


class TestComponentValidation:
    """Test component validation functionality."""
    
    def test_validate_component_valid(self):
        """Test validation of a valid component."""
        mock_component = MockComponent("test_comp", "Test Component", "test_type")
        mock_components = {"test_comp": mock_component}
        registry = MockRegistry(mock_components=mock_components)
        
        is_valid, errors = registry.validate_component("test_comp")
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_component_invalid(self):
        """Test validation of an invalid component."""
        mock_component = MockComponent("", "", "test_type")  # Invalid: empty name and description
        mock_components = {"test_comp": mock_component}
        registry = MockRegistry(mock_components=mock_components)
        
        is_valid, errors = registry.validate_component("test_comp")
        assert is_valid is False
        assert len(errors) == 2
        assert "Component name is required" in errors
        assert "Component description is required" in errors
    
    def test_validate_component_nonexistent(self):
        """Test validation of non-existent component."""
        registry = MockRegistry()
        
        is_valid, errors = registry.validate_component("nonexistent")
        assert is_valid is False
        assert len(errors) == 1
        assert "not found" in errors[0]
    
    def test_validate_all_components(self):
        """Test validation of all components."""
        mock_components = {
            "valid_comp": MockComponent("valid_comp", "Valid Component", "type1"),
            "invalid_comp": MockComponent("", "", "type2")  # Invalid
        }
        registry = MockRegistry(mock_components=mock_components)
        
        results = registry.validate_all_components()
        assert len(results) == 2
        assert results["valid_comp"][0] is True
        assert results["invalid_comp"][0] is False


class TestCacheManagement:
    """Test cache management functionality."""
    
    def test_refresh_cache(self):
        """Test cache refresh functionality."""
        mock_components = {
            "comp1": MockComponent("comp1", "Component 1", "type1")
        }
        registry = MockRegistry(mock_components=mock_components)
        
        # Populate cache
        registry.discover_components()
        assert len(registry._cache) == 1
        
        # Clear cache
        registry.refresh_cache()
        assert len(registry._cache) == 0
        assert len(registry._metadata_cache) == 0


class TestHealthCheck:
    """Test health check functionality."""
    
    def test_health_check_success(self):
        """Test health check with all valid components."""
        mock_components = {
            "comp1": MockComponent("comp1", "Component 1", "type1"),
            "comp2": MockComponent("comp2", "Component 2", "type2")
        }
        registry = MockRegistry(mock_components=mock_components)
        
        health = registry.health_check()
        assert health["total_components"] == 2
        assert health["valid_components"] == 2
        assert health["success_rate"] == 100.0
        assert len(health["errors"]) == 0
        assert health["registry_type"] == "MockRegistry"
    
    def test_health_check_with_failures(self):
        """Test health check with some invalid components."""
        mock_components = {
            "valid_comp": MockComponent("valid_comp", "Valid Component", "type1"),
            "invalid_comp": MockComponent("", "", "type2")  # Invalid
        }
        registry = MockRegistry(mock_components=mock_components)
        
        health = registry.health_check()
        assert health["total_components"] == 2
        assert health["valid_components"] == 1
        assert health["success_rate"] == 50.0
        assert len(health["errors"]) > 0
    
    def test_health_check_exception_handling(self):
        """Test health check handles exceptions gracefully."""
        registry = MockRegistry()
        
        # Mock discover_components to raise an exception
        with patch.object(registry, 'discover_components', side_effect=Exception("Discovery failed")):
            health = registry.health_check()
            
            assert health["total_components"] == 0
            assert health["valid_components"] == 0
            assert health["success_rate"] == 0
            assert "Discovery failed" in health["errors"][0]


class TestComponentInfo:
    """Test ComponentInfo functionality."""
    
    def test_component_info_extraction(self):
        """Test component info extraction."""
        mock_component = MockComponent("test_comp", "Test Component", "test_type", "2.0.0")
        mock_components = {"test_comp": mock_component}
        registry = MockRegistry(mock_components=mock_components)
        
        info = registry.get_component_info("test_comp")
        assert info is not None
        assert info["name"] == "test_comp"
        assert info["description"] == "Test Component"
        assert info["type"] == "test_type"
        assert info["version"] == "2.0.0"
    
    def test_component_info_nonexistent(self):
        """Test component info for non-existent component."""
        registry = MockRegistry()
        
        info = registry.get_component_info("nonexistent")
        assert info is None


if __name__ == "__main__":
    pytest.main([__file__])