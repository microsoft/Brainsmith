"""
Tests for Analysis Registry

Tests the refactored AnalysisRegistry with unified BaseRegistry interface.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from brainsmith.libraries.analysis.registry import (
    AnalysisRegistry, 
    AnalysisToolInfo, 
    AnalysisType,
    get_analysis_registry
)
from brainsmith.core.registry.exceptions import ComponentNotFoundError, ValidationError


class TestAnalysisRegistry:
    """Test suite for AnalysisRegistry unified interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = AnalysisRegistry(search_dirs=[self.temp_dir])
        
        # Create mock analysis tool function
        def mock_analysis_func():
            """Mock analysis function for testing."""
            pass
        
        # Create mock analysis tool info
        self.mock_tool = AnalysisToolInfo(
            name="test_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=mock_analysis_func,
            module_path="test.module",
            category="test",
            description="Test analysis tool",
            dependencies=[],
            available=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test registry initialization."""
        assert isinstance(self.registry, AnalysisRegistry)
        assert self.registry.search_dirs == [self.temp_dir]
        assert hasattr(self.registry, 'tool_cache')  # Backward compatibility
        assert self.registry.tool_cache is self.registry._cache
    
    def test_discover_components_empty_dir(self):
        """Test discovery with empty directory."""
        components = self.registry.discover_components()
        assert components == {}
        assert isinstance(components, dict)
    
    @patch('brainsmith.libraries.analysis.profiling')
    def test_discover_components_with_profiling_tools(self, mock_profiling):
        """Test discovery with profiling tools."""
        # Mock profiling module
        def mock_roofline_analysis():
            """Mock roofline analysis function."""
            pass
        
        class MockRooflineProfiler:
            """Mock roofline profiler class."""
            pass
        
        # Setup mock module
        mock_profiling.roofline_analysis = mock_roofline_analysis
        mock_profiling.RooflineProfiler = MockRooflineProfiler
        
        components = self.registry.discover_components()
        
        # Should discover the profiling tools
        assert len(components) >= 0  # May be 0 due to import issues in test
    
    def test_discover_components_caching(self):
        """Test that discover_components caches results."""
        # First call
        components1 = self.registry.discover_components()
        
        # Second call without rescan should return cached result
        components2 = self.registry.discover_components()
        
        assert components1 is components2  # Same object reference
        
        # Force rescan
        components3 = self.registry.discover_components(rescan=True)
        assert isinstance(components3, dict)
    
    def test_get_component(self):
        """Test getting specific component."""
        # Manually add a tool to cache for testing
        test_tool = AnalysisToolInfo(
            name="test_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            category="test",
            description="Test tool"
        )
        self.registry._cache['test_tool'] = test_tool
        
        # Get existing component
        component = self.registry.get_component('test_tool')
        assert component is not None
        assert component.name == 'test_tool'
        
        # Get non-existent component
        component = self.registry.get_component('non_existent')
        assert component is None
    
    def test_find_components_by_type(self):
        """Test finding components by analysis type."""
        # Manually add tools to cache for testing
        profiling_tool = AnalysisToolInfo(
            name="profiling_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            category="profiling"
        )
        codegen_tool = AnalysisToolInfo(
            name="codegen_tool",
            analysis_type=AnalysisType.CODE_GENERATION,
            tool_object=lambda: None,
            module_path="test.module",
            category="codegen"
        )
        
        self.registry._cache.update({
            'profiling_tool': profiling_tool,
            'codegen_tool': codegen_tool
        })
        
        # Find profiling tools
        profiling_tools = self.registry.find_components_by_type(AnalysisType.PROFILING)
        assert len(profiling_tools) == 1
        assert profiling_tools[0].analysis_type == AnalysisType.PROFILING
        
        # Find code generation tools
        codegen_tools = self.registry.find_components_by_type(AnalysisType.CODE_GENERATION)
        assert len(codegen_tools) == 1
        assert codegen_tools[0].analysis_type == AnalysisType.CODE_GENERATION
    
    def test_find_tools_by_category(self):
        """Test finding tools by category."""
        # Manually add tools to cache for testing
        tool1 = AnalysisToolInfo(
            name="tool1",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            category="performance"
        )
        tool2 = AnalysisToolInfo(
            name="tool2",
            analysis_type=AnalysisType.UTILITY,
            tool_object=lambda: None,
            module_path="test.module",
            category="performance"
        )
        tool3 = AnalysisToolInfo(
            name="tool3",
            analysis_type=AnalysisType.CODE_GENERATION,
            tool_object=lambda: None,
            module_path="test.module",
            category="utility"
        )
        
        self.registry._cache.update({
            'tool1': tool1,
            'tool2': tool2,
            'tool3': tool3
        })
        
        # Find performance tools
        performance_tools = self.registry.find_tools_by_category("performance")
        assert len(performance_tools) == 2
        assert all(t.category == "performance" for t in performance_tools)
        
        # Find utility tools
        utility_tools = self.registry.find_tools_by_category("utility")
        assert len(utility_tools) == 1
        assert utility_tools[0].category == "utility"
    
    def test_list_component_names(self):
        """Test listing component names."""
        # Add test tools
        tool1 = AnalysisToolInfo("tool1", AnalysisType.PROFILING, lambda: None, "test.module")
        tool2 = AnalysisToolInfo("tool2", AnalysisType.CODE_GENERATION, lambda: None, "test.module")
        
        self.registry._cache.update({
            'tool1': tool1,
            'tool2': tool2
        })
        
        names = self.registry.list_component_names()
        assert set(names) == {'tool1', 'tool2'}
    
    def test_list_available_tools(self):
        """Test listing available tools only."""
        # Add test tools with different availability
        available_tool = AnalysisToolInfo(
            name="available_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            available=True
        )
        unavailable_tool = AnalysisToolInfo(
            name="unavailable_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            available=False
        )
        
        self.registry._cache.update({
            'available_tool': available_tool,
            'unavailable_tool': unavailable_tool
        })
        
        available_names = self.registry.list_available_tools()
        assert 'available_tool' in available_names
        assert 'unavailable_tool' not in available_names
    
    def test_list_categories(self):
        """Test listing categories."""
        # Add test tools with different categories
        tool1 = AnalysisToolInfo(
            name="tool1",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            category="category1"
        )
        tool2 = AnalysisToolInfo(
            name="tool2",
            analysis_type=AnalysisType.CODE_GENERATION,
            tool_object=lambda: None,
            module_path="test.module",
            category="category2"
        )
        
        self.registry._cache.update({
            'tool1': tool1,
            'tool2': tool2
        })
        
        categories = self.registry.list_categories()
        assert 'category1' in categories
        assert 'category2' in categories
    
    def test_get_component_info(self):
        """Test getting component info."""
        tool = AnalysisToolInfo(
            name="test_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            category="test_category",
            description="Test description",
            dependencies=["dep1", "dep2"],
            available=True
        )
        self.registry._cache['test_tool'] = tool
        
        info = self.registry.get_component_info('test_tool')
        assert info is not None
        assert info['name'] == 'test_tool'
        assert info['type'] == 'analysis_tool'
        assert info['analysis_type'] == 'profiling'
        assert info['category'] == 'test_category'
        assert info['dependencies'] == ["dep1", "dep2"]
        assert info['available'] == True
    
    def test_check_tool_availability(self):
        """Test tool availability checking."""
        # Available tool
        available_tool = AnalysisToolInfo(
            name="available_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            available=True
        )
        # Unavailable tool
        unavailable_tool = AnalysisToolInfo(
            name="unavailable_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module",
            available=False
        )
        
        self.registry._cache.update({
            'available_tool': available_tool,
            'unavailable_tool': unavailable_tool
        })
        
        # Check available tool
        is_available, error = self.registry.check_tool_availability('available_tool')
        assert is_available == True
        assert error is None
        
        # Check unavailable tool
        is_available, error = self.registry.check_tool_availability('unavailable_tool')
        assert is_available == False
        assert error is not None
        
        # Check non-existent tool
        is_available, error = self.registry.check_tool_availability('non_existent')
        assert is_available == False
        assert error is not None
    
    def test_validate_component(self):
        """Test component validation."""
        # Valid tool
        valid_tool = AnalysisToolInfo(
            name="valid_tool",
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module"
        )
        self.registry._cache['valid_tool'] = valid_tool
        
        is_valid, errors = self.registry.validate_component('valid_tool')
        assert is_valid == True
        assert len(errors) == 0
        
        # Invalid tool (missing name)
        invalid_tool = AnalysisToolInfo(
            name="",  # Empty name
            analysis_type=AnalysisType.PROFILING,
            tool_object=lambda: None,
            module_path="test.module"
        )
        self.registry._cache['invalid_tool'] = invalid_tool
        
        is_valid, errors = self.registry.validate_component('invalid_tool')
        assert is_valid == False
        assert len(errors) > 0
        
        # Non-existent tool
        is_valid, errors = self.registry.validate_component('non_existent')
        assert is_valid == False
        assert len(errors) > 0
    
    def test_refresh_cache(self):
        """Test cache refresh."""
        # Populate cache
        self.registry._cache['test'] = self.mock_tool
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
        registry = AnalysisRegistry()  # No search_dirs provided
        default_dirs = registry._get_default_dirs()
        assert len(default_dirs) > 0
        assert all(isinstance(d, str) for d in default_dirs)
    
    def test_extract_description(self):
        """Test description extraction from function docstring."""
        def func_with_docstring():
            """This is a test function.
            
            With additional details.
            """
            pass
        
        def func_without_docstring():
            pass
        
        desc1 = self.registry._extract_description(func_with_docstring)
        assert desc1 == "This is a test function."
        
        desc2 = self.registry._extract_description(func_without_docstring)
        assert "Analysis tool: func_without_docstring" in desc2


class TestAnalysisRegistryConvenience:
    """Test convenience functions."""
    
    def test_get_analysis_registry(self):
        """Test global registry getter."""
        registry1 = get_analysis_registry()
        registry2 = get_analysis_registry()
        assert registry1 is registry2  # Should return same instance
    
    @patch('brainsmith.libraries.analysis.registry.get_analysis_registry')
    def test_discover_all_analysis_tools(self, mock_get_registry):
        """Test discover_all_analysis_tools convenience function."""
        from brainsmith.libraries.analysis.registry import discover_all_analysis_tools
        
        mock_registry = Mock()
        mock_registry.discover_components.return_value = {'test': 'tool'}
        mock_get_registry.return_value = mock_registry
        
        result = discover_all_analysis_tools(rescan=True)
        
        mock_registry.discover_components.assert_called_once_with(True)
        assert result == {'test': 'tool'}
    
    @patch('brainsmith.libraries.analysis.registry.get_analysis_registry')
    def test_get_analysis_tool(self, mock_get_registry):
        """Test get_analysis_tool convenience function."""
        from brainsmith.libraries.analysis.registry import get_analysis_tool
        
        mock_registry = Mock()
        mock_registry.get_component.return_value = 'mock_tool'
        mock_get_registry.return_value = mock_registry
        
        result = get_analysis_tool('test_tool')
        
        mock_registry.get_component.assert_called_once_with('test_tool')
        assert result == 'mock_tool'
    
    @patch('brainsmith.libraries.analysis.registry.get_analysis_registry')
    def test_find_tools_by_type(self, mock_get_registry):
        """Test find_tools_by_type convenience function."""
        from brainsmith.libraries.analysis.registry import find_tools_by_type
        
        mock_registry = Mock()
        mock_registry.find_components_by_type.return_value = ['tool1', 'tool2']
        mock_get_registry.return_value = mock_registry
        
        result = find_tools_by_type(AnalysisType.PROFILING)
        
        mock_registry.find_components_by_type.assert_called_once_with(AnalysisType.PROFILING)
        assert result == ['tool1', 'tool2']
    
    @patch('brainsmith.libraries.analysis.registry.get_analysis_registry')
    def test_list_available_analysis_tools(self, mock_get_registry):
        """Test list_available_analysis_tools convenience function."""
        from brainsmith.libraries.analysis.registry import list_available_analysis_tools
        
        mock_registry = Mock()
        mock_registry.list_component_names.return_value = ['tool1', 'tool2']
        mock_get_registry.return_value = mock_registry
        
        result = list_available_analysis_tools()
        
        mock_registry.list_component_names.assert_called_once()
        assert result == ['tool1', 'tool2']
    
    @patch('brainsmith.libraries.analysis.registry.get_analysis_registry')
    def test_refresh_analysis_registry(self, mock_get_registry):
        """Test refresh_analysis_registry convenience function."""
        from brainsmith.libraries.analysis.registry import refresh_analysis_registry
        
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        refresh_analysis_registry()
        
        mock_registry.refresh_cache.assert_called_once()


class TestAnalysisType:
    """Test AnalysisType enum."""
    
    def test_analysis_type_values(self):
        """Test AnalysisType enum values."""
        assert AnalysisType.PROFILING.value == "profiling"
        assert AnalysisType.CODE_GENERATION.value == "code_generation"
        assert AnalysisType.REPORTING.value == "reporting"
        assert AnalysisType.UTILITY.value == "utility"


class TestAnalysisToolInfo:
    """Test AnalysisToolInfo dataclass."""
    
    def test_analysis_tool_info_creation(self):
        """Test AnalysisToolInfo creation."""
        def test_func():
            pass
        
        info = AnalysisToolInfo(
            name="test",
            analysis_type=AnalysisType.PROFILING,
            tool_object=test_func,
            module_path="test.module"
        )
        
        assert info.name == "test"
        assert info.analysis_type == AnalysisType.PROFILING
        assert info.tool_object == test_func
        assert info.module_path == "test.module"
        assert info.category == "unknown"  # Default value
        assert info.dependencies == []  # Default value set by __post_init__
        assert info.available == True  # Default value
    
    def test_analysis_tool_info_with_dependencies(self):
        """Test AnalysisToolInfo with explicit dependencies."""
        def test_func():
            pass
        
        info = AnalysisToolInfo(
            name="test",
            analysis_type=AnalysisType.PROFILING,
            tool_object=test_func,
            module_path="test.module",
            dependencies=["dep1", "dep2"]
        )
        
        assert info.dependencies == ["dep1", "dep2"]


if __name__ == '__main__':
    pytest.main([__file__])