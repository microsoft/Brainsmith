"""
Automation Registry System

Auto-discovery and management of automation tools in the BrainSmith libraries.
Provides registration, caching, and lookup functionality for batch processing,
parameter sweeps, and other automation utilities.
"""

import os
import inspect
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AutomationType(Enum):
    """Types of automation tools available."""
    BATCH_PROCESSING = "batch_processing"
    PARAMETER_SWEEP = "parameter_sweep"
    RESULT_ANALYSIS = "result_analysis"
    WORKFLOW = "workflow"
    UTILITY = "utility"


@dataclass
class AutomationToolInfo:
    """Information about a discovered automation tool."""
    name: str
    automation_type: AutomationType
    function: Callable
    module_path: str
    category: str = "unknown"
    description: str = ""
    parameters: List[str] = None
    supports_parallel: bool = False
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []


class AutomationRegistry:
    """Registry for auto-discovery and management of automation tools."""
    
    def __init__(self, automation_dirs: Optional[List[str]] = None):
        """
        Initialize automation registry.
        
        Args:
            automation_dirs: List of directories to search for automation tools.
                           Defaults to current automation directory
        """
        if automation_dirs is None:
            # Default to the current automation directory
            current_dir = Path(__file__).parent
            automation_dirs = [str(current_dir)]
        
        self.automation_dirs = automation_dirs
        self.tool_cache = {}
        self.metadata_cache = {}
        
        logger.info(f"Automation registry initialized with dirs: {self.automation_dirs}")
    
    def discover_tools(self, rescan: bool = False) -> Dict[str, AutomationToolInfo]:
        """
        Discover all available automation tools.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping tool names to AutomationToolInfo objects
        """
        if self.tool_cache and not rescan:
            return self.tool_cache
        
        discovered = {}
        
        # Discover core automation functions
        core_tools = self._discover_core_tools()
        discovered.update(core_tools)
        
        # Discover contrib tools
        contrib_tools = self._discover_contrib_tools()
        discovered.update(contrib_tools)
        
        # Cache the results
        self.tool_cache = discovered
        
        logger.info(f"Discovered {len(discovered)} automation tools")
        return discovered
    
    def _discover_core_tools(self) -> Dict[str, AutomationToolInfo]:
        """Discover core automation tools."""
        tools = {}
        
        try:
            # Import automation module
            import brainsmith.libraries.automation as automation_module
            
            # Discover parameter_sweep function
            if hasattr(automation_module, 'parameter_sweep'):
                tools['parameter_sweep'] = AutomationToolInfo(
                    name='parameter_sweep',
                    automation_type=AutomationType.PARAMETER_SWEEP,
                    function=automation_module.parameter_sweep,
                    module_path='brainsmith.libraries.automation.sweep',
                    category='parameter_exploration',
                    description='Run forge() with different parameter combinations',
                    parameters=['model_path', 'blueprint_path', 'param_ranges', 'max_workers'],
                    supports_parallel=True
                )
            
            # Discover batch_process function
            if hasattr(automation_module, 'batch_process'):
                tools['batch_process'] = AutomationToolInfo(
                    name='batch_process',
                    automation_type=AutomationType.BATCH_PROCESSING,
                    function=automation_module.batch_process,
                    module_path='brainsmith.libraries.automation.batch',
                    category='batch_processing',
                    description='Process multiple model/blueprint pairs in batch',
                    parameters=['model_blueprint_pairs', 'common_config', 'max_workers'],
                    supports_parallel=True
                )
            
            # Discover find_best function
            if hasattr(automation_module, 'find_best'):
                tools['find_best'] = AutomationToolInfo(
                    name='find_best',
                    automation_type=AutomationType.RESULT_ANALYSIS,
                    function=automation_module.find_best,
                    module_path='brainsmith.libraries.automation.sweep',
                    category='result_analysis',
                    description='Find optimal result by specified metric',
                    parameters=['results', 'metric', 'maximize'],
                    supports_parallel=False
                )
            
            # Discover aggregate_stats function
            if hasattr(automation_module, 'aggregate_stats'):
                tools['aggregate_stats'] = AutomationToolInfo(
                    name='aggregate_stats',
                    automation_type=AutomationType.RESULT_ANALYSIS,
                    function=automation_module.aggregate_stats,
                    module_path='brainsmith.libraries.automation.sweep',
                    category='result_analysis',
                    description='Generate statistical summary of results',
                    parameters=['results'],
                    supports_parallel=False
                )
            
            # Also check individual modules for additional functions
            try:
                from brainsmith.libraries.automation import sweep, batch
                
                # Check sweep module for additional functions
                for name, obj in inspect.getmembers(sweep):
                    if (inspect.isfunction(obj) and not name.startswith('_') and 
                        name not in ['parameter_sweep', 'find_best', 'aggregate_stats']):
                        tools[name] = AutomationToolInfo(
                            name=name,
                            automation_type=self._classify_function_type(name, obj),
                            function=obj,
                            module_path='brainsmith.libraries.automation.sweep',
                            category='sweep_utilities',
                            description=self._extract_description(obj),
                            parameters=self._extract_parameters(obj)
                        )
                
                # Check batch module for additional functions
                for name, obj in inspect.getmembers(batch):
                    if (inspect.isfunction(obj) and not name.startswith('_') and 
                        name not in ['batch_process']):
                        tools[name] = AutomationToolInfo(
                            name=name,
                            automation_type=self._classify_function_type(name, obj),
                            function=obj,
                            module_path='brainsmith.libraries.automation.batch',
                            category='batch_utilities',
                            description=self._extract_description(obj),
                            parameters=self._extract_parameters(obj)
                        )
                        
            except ImportError:
                logger.debug("Could not import individual automation modules")
            
        except ImportError:
            logger.debug("Could not import automation module")
        
        return tools
    
    def _discover_contrib_tools(self) -> Dict[str, AutomationToolInfo]:
        """Discover contrib automation tools."""
        tools = {}
        
        for automation_dir in self.automation_dirs:
            contrib_dir = os.path.join(automation_dir, "contrib")
            
            if not os.path.exists(contrib_dir):
                continue
            
            # Look for Python files in contrib directory
            for item in os.listdir(contrib_dir):
                if item.endswith('.py') and item != '__init__.py':
                    module_name = item[:-3]
                    
                    try:
                        # Import contrib module
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            f"contrib.{module_name}", 
                            os.path.join(contrib_dir, item)
                        )
                        
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Discover functions in contrib module
                            for name, obj in inspect.getmembers(module):
                                if inspect.isfunction(obj) and not name.startswith('_'):
                                    tool_name = f"contrib_{name}"
                                    tools[tool_name] = AutomationToolInfo(
                                        name=tool_name,
                                        automation_type=self._classify_function_type(name, obj),
                                        function=obj,
                                        module_path=f'brainsmith.libraries.automation.contrib.{module_name}',
                                        category='contrib',
                                        description=self._extract_description(obj),
                                        parameters=self._extract_parameters(obj)
                                    )
                                    
                    except Exception as e:
                        logger.debug(f"Could not load contrib module {module_name}: {e}")
        
        return tools
    
    def get_tool(self, tool_name: str) -> Optional[AutomationToolInfo]:
        """
        Get a specific automation tool by name.
        
        Args:
            tool_name: Name of the automation tool
            
        Returns:
            AutomationToolInfo object or None if not found
        """
        tools = self.discover_tools()
        return tools.get(tool_name)
    
    def find_tools_by_type(self, automation_type: AutomationType) -> List[AutomationToolInfo]:
        """
        Find tools by automation type.
        
        Args:
            automation_type: Type of automation to search for
            
        Returns:
            List of matching AutomationToolInfo objects
        """
        tools = self.discover_tools()
        matches = []
        
        for tool in tools.values():
            if tool.automation_type == automation_type:
                matches.append(tool)
        
        return matches
    
    def find_parallel_tools(self) -> List[AutomationToolInfo]:
        """Find tools that support parallel execution."""
        tools = self.discover_tools()
        return [tool for tool in tools.values() if tool.supports_parallel]
    
    def list_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        tools = self.discover_tools()
        return list(tools.keys())
    
    def list_categories(self) -> Set[str]:
        """Get set of all available categories."""
        tools = self.discover_tools()
        return {tool.category for tool in tools.values()}
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """
        Get summary information about a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with tool summary or None if not found
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return None
        
        return {
            'name': tool.name,
            'type': tool.automation_type.value,
            'category': tool.category,
            'description': tool.description,
            'module_path': tool.module_path,
            'parameters': tool.parameters,
            'supports_parallel': tool.supports_parallel,
            'function_name': tool.function.__name__
        }
    
    def refresh_cache(self):
        """Refresh the tool cache by clearing it."""
        self.tool_cache.clear()
        self.metadata_cache.clear()
        logger.info("Automation registry cache refreshed")
    
    def _classify_function_type(self, name: str, func: Callable) -> AutomationType:
        """Classify function type based on name and signature."""
        name_lower = name.lower()
        
        if 'batch' in name_lower or 'bulk' in name_lower:
            return AutomationType.BATCH_PROCESSING
        elif 'sweep' in name_lower or 'param' in name_lower:
            return AutomationType.PARAMETER_SWEEP
        elif 'find' in name_lower or 'best' in name_lower or 'stat' in name_lower or 'aggregate' in name_lower:
            return AutomationType.RESULT_ANALYSIS
        elif 'workflow' in name_lower or 'pipeline' in name_lower:
            return AutomationType.WORKFLOW
        else:
            return AutomationType.UTILITY
    
    def _extract_description(self, func: Callable) -> str:
        """Extract description from function docstring."""
        docstring = func.__doc__ or ""
        
        # Get first line of docstring as description
        first_line = docstring.split('\n')[0].strip()
        return first_line if first_line else f"Automation function: {func.__name__}"
    
    def _extract_parameters(self, func: Callable) -> List[str]:
        """Extract parameter names from function signature."""
        try:
            sig = inspect.signature(func)
            return list(sig.parameters.keys())
        except (ValueError, TypeError):
            return []


# Global registry instance
_global_registry = None


def get_automation_registry() -> AutomationRegistry:
    """Get the global automation registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AutomationRegistry()
    return _global_registry


# Convenience functions for common operations
def discover_all_automation_tools(rescan: bool = False) -> Dict[str, AutomationToolInfo]:
    """
    Discover all available automation tools.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping tool names to AutomationToolInfo objects
    """
    registry = get_automation_registry()
    return registry.discover_tools(rescan)


def get_automation_tool(tool_name: str) -> Optional[AutomationToolInfo]:
    """
    Get an automation tool by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        AutomationToolInfo object or None if not found
    """
    registry = get_automation_registry()
    return registry.get_tool(tool_name)


def find_tools_by_type(automation_type: AutomationType) -> List[AutomationToolInfo]:
    """
    Find all tools of a specific type.
    
    Args:
        automation_type: Type of automation
        
    Returns:
        List of matching AutomationToolInfo objects
    """
    registry = get_automation_registry()
    return registry.find_tools_by_type(automation_type)


def list_available_automation_tools() -> List[str]:
    """
    Get list of all available automation tool names.
    
    Returns:
        List of tool names
    """
    registry = get_automation_registry()
    return registry.list_tool_names()


def refresh_automation_registry():
    """Refresh the automation registry cache."""
    registry = get_automation_registry()
    registry.refresh_cache()