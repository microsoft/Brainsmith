"""
Analysis Tools Registry System

Auto-discovery and management of analysis tools in the BrainSmith libraries.
Provides registration, caching, and lookup functionality for profiling tools,
code generators, and other analysis utilities.

BREAKING CHANGE: Now uses unified BaseRegistry interface with standardized method names.
"""

import os
import inspect
import logging
from typing import Dict, List, Optional, Set, Callable, Union, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from brainsmith.core.registry import BaseRegistry, ComponentInfo

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis tools available."""
    PROFILING = "profiling"
    CODE_GENERATION = "code_generation"
    REPORTING = "reporting"
    UTILITY = "utility"


@dataclass
class AnalysisToolInfo:
    """Information about a discovered analysis tool."""
    name: str
    analysis_type: AnalysisType
    tool_object: Union[Callable, type, Any]
    module_path: str
    category: str = "unknown"
    description: str = ""
    dependencies: List[str] = None
    available: bool = True
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class AnalysisRegistry(BaseRegistry[AnalysisToolInfo]):
    """Registry for auto-discovery and management of analysis tools."""
    
    def __init__(self, search_dirs: Optional[List[str]] = None, config_manager=None):
        """
        Initialize analysis registry.
        
        Args:
            search_dirs: List of directories to search for analysis tools.
                        If None, uses default analysis directories.
            config_manager: Optional configuration manager.
        """
        super().__init__(search_dirs, config_manager)
        # For backward compatibility, maintain tool_cache reference
        self.tool_cache = self._cache
        self.metadata_cache = self._metadata_cache
    
    def discover_components(self, rescan: bool = False) -> Dict[str, AnalysisToolInfo]:
        """
        Discover all available analysis tools.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping tool names to AnalysisToolInfo objects
        """
        if not rescan and self._cache:
            return self._cache
        
        discovered = {}
        
        # Discover profiling tools
        profiling_tools = self._discover_profiling_tools()
        discovered.update(profiling_tools)
        
        # Discover code generation tools
        codegen_tools = self._discover_codegen_tools()
        discovered.update(codegen_tools)
        
        # Cache the results
        self._cache = discovered
        self.tool_cache = self._cache  # Maintain backward compatibility reference
        
        self._log_info(f"Discovered {len(discovered)} analysis tools")
        return discovered
    
    def find_components_by_type(self, analysis_type: AnalysisType) -> List[AnalysisToolInfo]:
        """
        Find tools by analysis type.
        
        Args:
            analysis_type: Type of analysis to search for
            
        Returns:
            List of matching AnalysisToolInfo objects
        """
        tools = self.discover_components()
        matches = []
        
        for tool in tools.values():
            if tool.analysis_type == analysis_type:
                matches.append(tool)
        
        return matches
    
    def find_tools_by_category(self, category: str) -> List[AnalysisToolInfo]:
        """
        Find tools by category.
        
        Args:
            category: Category to search for
            
        Returns:
            List of matching AnalysisToolInfo objects
        """
        tools = self.discover_components()
        matches = []
        
        for tool in tools.values():
            if tool.category == category:
                matches.append(tool)
        
        return matches
    
    def list_available_tools(self) -> List[str]:
        """Get list of available tool names (excluding unavailable ones)."""
        tools = self.discover_components()
        return [name for name, tool in tools.items() if tool.available]
    
    def list_categories(self) -> Set[str]:
        """Get set of all available categories."""
        tools = self.discover_components()
        return {tool.category for tool in tools.values()}
    
    def check_tool_availability(self, tool_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if a tool is available for use.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tuple of (is_available, error_message)
        """
        tool = self.get_component(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found"
        
        if not tool.available:
            return False, f"Tool '{tool_name}' is not available (missing dependencies)"
        
        return True, None
    
    def _get_default_dirs(self) -> List[str]:
        """Get default search directories for analysis registry."""
        current_dir = Path(__file__).parent
        return [str(current_dir)]
    
    def _extract_info(self, component: AnalysisToolInfo) -> Dict[str, Any]:
        """Extract standardized info from analysis tool component."""
        return {
            'name': component.name,
            'type': 'analysis_tool',
            'analysis_type': component.analysis_type.value,
            'category': component.category,
            'description': component.description,
            'module_path': component.module_path,
            'dependencies': component.dependencies,
            'available': component.available,
            'tool_type': type(component.tool_object).__name__
        }
    
    def _validate_component_implementation(self, component: AnalysisToolInfo) -> tuple[bool, List[str]]:
        """Analysis tool-specific validation logic."""
        errors = []
        
        # Validate basic fields
        if not component.name:
            errors.append("Analysis tool name is required")
        
        if not component.analysis_type:
            errors.append("Analysis type is required")
        
        if not component.tool_object:
            errors.append("Tool object is required")
        elif not callable(component.tool_object) and not inspect.isclass(component.tool_object):
            errors.append("Tool object must be callable or a class")
        
        # Validate dependencies if specified
        if component.dependencies:
            # Check if dependencies are available (basic check)
            for dep in component.dependencies:
                try:
                    __import__(dep)
                except ImportError:
                    errors.append(f"Dependency '{dep}' not available")
        
        return len(errors) == 0, errors
    
    def _discover_profiling_tools(self) -> Dict[str, AnalysisToolInfo]:
        """Discover profiling tools."""
        tools = {}
        
        try:
            # Import profiling module
            import brainsmith.libraries.analysis.profiling as profiling_module
            
            # Check for roofline analysis function
            if hasattr(profiling_module, 'roofline_analysis'):
                tools['roofline_analysis'] = AnalysisToolInfo(
                    name='roofline_analysis',
                    analysis_type=AnalysisType.PROFILING,
                    tool_object=profiling_module.roofline_analysis,
                    module_path='brainsmith.libraries.analysis.profiling',
                    category='performance',
                    description='Roofline analysis for model performance profiling',
                    available=profiling_module.roofline_analysis is not None
                )
            
            # Check for RooflineProfiler class
            if hasattr(profiling_module, 'RooflineProfiler'):
                tools['roofline_profiler'] = AnalysisToolInfo(
                    name='roofline_profiler',
                    analysis_type=AnalysisType.PROFILING,
                    tool_object=profiling_module.RooflineProfiler,
                    module_path='brainsmith.libraries.analysis.profiling',
                    category='performance',
                    description='High-level roofline profiler interface',
                    available=profiling_module.RooflineProfiler is not None
                )
            
            # Discover other profiling functions
            for name, obj in inspect.getmembers(profiling_module):
                if (inspect.isfunction(obj) and not name.startswith('_') and 
                    name not in ['roofline_analysis']):
                    tools[name] = AnalysisToolInfo(
                        name=name,
                        analysis_type=AnalysisType.PROFILING,
                        tool_object=obj,
                        module_path='brainsmith.libraries.analysis.profiling',
                        category='profiling',
                        description=self._extract_description(obj)
                    )
                    
        except ImportError:
            self._log_debug("Could not import profiling module")
        
        return tools
    
    def _discover_codegen_tools(self) -> Dict[str, AnalysisToolInfo]:
        """Discover code generation tools."""
        tools = {}
        
        try:
            # Import tools module
            import brainsmith.libraries.analysis.tools as tools_module
            
            # Check for hw kernel generation
            if hasattr(tools_module, 'generate_hw_kernel'):
                tools['generate_hw_kernel'] = AnalysisToolInfo(
                    name='generate_hw_kernel',
                    analysis_type=AnalysisType.CODE_GENERATION,
                    tool_object=tools_module.generate_hw_kernel,
                    module_path='brainsmith.libraries.analysis.tools',
                    category='code_generation',
                    description='Hardware kernel code generation tool',
                    available=tools_module.generate_hw_kernel is not None
                )
            
            # Look for hw_kernel_gen submodule
            try:
                import brainsmith.libraries.analysis.tools.hw_kernel_gen as hkg_module
                
                # Discover generators
                if hasattr(hkg_module, 'generators'):
                    generators_module = hkg_module.generators
                    
                    for name, obj in inspect.getmembers(generators_module):
                        if inspect.isclass(obj) and name.endswith('Generator'):
                            tool_name = name.lower().replace('generator', '_gen')
                            tools[tool_name] = AnalysisToolInfo(
                                name=tool_name,
                                analysis_type=AnalysisType.CODE_GENERATION,
                                tool_object=obj,
                                module_path=f'brainsmith.libraries.analysis.tools.hw_kernel_gen.generators',
                                category='code_generation',
                                description=self._extract_description(obj)
                            )
                
                # Check for main hkg interface
                if hasattr(hkg_module, 'hkg'):
                    hkg_obj = hkg_module.hkg
                    for name, obj in inspect.getmembers(hkg_obj):
                        if inspect.isfunction(obj) and not name.startswith('_'):
                            tools[f'hkg_{name}'] = AnalysisToolInfo(
                                name=f'hkg_{name}',
                                analysis_type=AnalysisType.CODE_GENERATION,
                                tool_object=obj,
                                module_path='brainsmith.libraries.analysis.tools.hw_kernel_gen.hkg',
                                category='code_generation',
                                description=self._extract_description(obj)
                            )
                            
            except ImportError:
                self._log_debug("Could not import hw_kernel_gen module")
            
            # Discover other tools
            for name, obj in inspect.getmembers(tools_module):
                if (inspect.isfunction(obj) and not name.startswith('_') and 
                    name not in ['generate_hw_kernel']):
                    tools[name] = AnalysisToolInfo(
                        name=name,
                        analysis_type=AnalysisType.UTILITY,
                        tool_object=obj,
                        module_path='brainsmith.libraries.analysis.tools',
                        category='utility',
                        description=self._extract_description(obj)
                    )
                    
        except ImportError:
            self._log_debug("Could not import tools module")
        
        return tools
    
    def _extract_description(self, obj: Union[Callable, type]) -> str:
        """Extract description from object docstring."""
        docstring = obj.__doc__ or ""
        
        # Get first line of docstring as description
        first_line = docstring.split('\n')[0].strip()
        return first_line if first_line else f"Analysis tool: {getattr(obj, '__name__', 'unknown')}"


# Global registry instance
_global_registry = None


def get_analysis_registry() -> AnalysisRegistry:
    """Get the global analysis registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AnalysisRegistry()
    return _global_registry


# BREAKING CHANGE: Updated convenience functions to use new unified interface
def discover_all_analysis_tools(rescan: bool = False) -> Dict[str, AnalysisToolInfo]:
    """
    Discover all available analysis tools.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping tool names to AnalysisToolInfo objects
    """
    registry = get_analysis_registry()
    return registry.discover_components(rescan)


def get_analysis_tool(tool_name: str) -> Optional[AnalysisToolInfo]:
    """
    Get an analysis tool by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        AnalysisToolInfo object or None if not found
    """
    registry = get_analysis_registry()
    return registry.get_component(tool_name)


def find_tools_by_type(analysis_type: AnalysisType) -> List[AnalysisToolInfo]:
    """
    Find all tools of a specific type.
    
    Args:
        analysis_type: Type of analysis
        
    Returns:
        List of matching AnalysisToolInfo objects
    """
    registry = get_analysis_registry()
    return registry.find_components_by_type(analysis_type)


def list_available_analysis_tools() -> List[str]:
    """
    Get list of all available analysis tool names.
    
    Returns:
        List of tool names
    """
    registry = get_analysis_registry()
    return registry.list_component_names()


def refresh_analysis_registry():
    """Refresh the analysis registry cache."""
    registry = get_analysis_registry()
    registry.refresh_cache()