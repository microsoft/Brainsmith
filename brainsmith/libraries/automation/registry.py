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
from brainsmith.core.registry import BaseRegistry, ComponentInfo

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


class AutomationRegistry(BaseRegistry[AutomationToolInfo]):
    """Registry for auto-discovery and management of automation tools."""
    
    def __init__(self, search_dirs: Optional[List[str]] = None, config_manager=None):
        """
        Initialize automation registry.
        
        Args:
            search_dirs: List of directories to search for automation tools.
                        If None, uses default automation directories.
            config_manager: Optional configuration manager.
        """
        super().__init__(search_dirs, config_manager)
    
    def discover_components(self, rescan: bool = False) -> Dict[str, AutomationToolInfo]:
        """
        Discover all available automation tools.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping component names to AutomationToolInfo objects
        """
        if self._cache and not rescan:
            return self._cache
        
        discovered = {}
        
        # Discover core automation functions
        core_tools = self._discover_core_tools()
        discovered.update(core_tools)
        
        # Discover contrib tools
        contrib_tools = self._discover_contrib_tools()
        discovered.update(contrib_tools)
        
        # Cache the results
        self._cache = discovered
        
        self._log_debug(f"Discovered {len(discovered)} automation tools")
        return discovered
    
    def _validate_component(self, component: AutomationToolInfo) -> bool:
        """Validate an automation tool component."""
        try:
            if not component.name or not isinstance(component.name, str):
                return False
            if not component.function or not callable(component.function):
                return False
            if not component.module_path or not isinstance(component.module_path, str):
                return False
            if not isinstance(component.automation_type, AutomationType):
                return False
            return True
        except Exception as e:
            self._log_debug(f"Component validation failed: {e}")
            return False
    
    def _process_component(self, tool_path: Path, metadata: Dict) -> Optional[AutomationToolInfo]:
        """Process a discovered automation tool into AutomationToolInfo."""
        # AutomationRegistry discovers functions dynamically, not from file paths
        # This method is required by BaseRegistry but not used in this implementation
        return None
    
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
                self._log_debug("Could not import individual automation modules")
            
        except ImportError:
            self._log_debug("Could not import automation module")
        
        return tools
    
    def _discover_contrib_tools(self) -> Dict[str, AutomationToolInfo]:
        """Discover contrib automation tools."""
        tools = {}
        
        for automation_dir in self.search_dirs:
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
                        self._log_debug(f"Could not load contrib module {module_name}: {e}")
        
        return tools
    
    def get_component(self, component_name: str) -> Optional[AutomationToolInfo]:
        """
        Get a specific automation tool by name.
        
        Args:
            component_name: Name of the automation tool
            
        Returns:
            AutomationToolInfo object or None if not found
        """
        components = self.discover_components()
        return components.get(component_name)
    
    def find_components_by_type(self, component_type: str) -> List[AutomationToolInfo]:
        """
        Find tools by type.
        
        Args:
            component_type: Type or category to search for (AutomationType value or category string)
            
        Returns:
            List of matching AutomationToolInfo objects
        """
        components = self.discover_components()
        matches = []
        
        for component in components.values():
            # Check if matches AutomationType enum value
            if component.automation_type.value == component_type:
                matches.append(component)
            # Check if matches category string
            elif component.category == component_type:
                matches.append(component)
        
        return matches
    
    def find_parallel_tools(self) -> List[AutomationToolInfo]:
        """Find tools that support parallel execution."""
        components = self.discover_components()
        return [component for component in components.values() if component.supports_parallel]
    
    def list_component_names(self) -> List[str]:
        """Get list of all available component names."""
        components = self.discover_components()
        return list(components.keys())
    
    def list_categories(self) -> Set[str]:
        """Get set of all available categories."""
        components = self.discover_components()
        return {component.category for component in components.values()}
    
    def get_component_info(self, component_name: str) -> Optional[Dict]:
        """
        Get summary information about a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dictionary with component summary or None if not found
        """
        component = self.get_component(component_name)
        if not component:
            return None
        
        return {
            'name': component.name,
            'type': component.automation_type.value,
            'category': component.category,
            'description': component.description,
            'module_path': component.module_path,
            'parameters': component.parameters,
            'supports_parallel': component.supports_parallel,
            'function_name': component.function.__name__
        }
    
    def refresh_cache(self):
        """Refresh the component cache by clearing it."""
        self._cache.clear()
        self._metadata_cache.clear()
        self._log_debug("Automation registry cache refreshed")
    
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
    
    def _get_default_dirs(self) -> List[str]:
        """Get default search directories for automation tools."""
        current_dir = Path(__file__).parent
        return [str(current_dir)]
    
    def _extract_info(self, component: AutomationToolInfo) -> Dict[str, Any]:
        """Extract standardized info from automation tool component."""
        return {
            'name': component.name,
            'type': component.automation_type.value,
            'category': component.category,
            'description': component.description,
            'module_path': component.module_path,
            'parameters': component.parameters,
            'supports_parallel': component.supports_parallel,
            'function_name': component.function.__name__ if component.function else None
        }
    
    def _validate_component_implementation(self, component: AutomationToolInfo) -> tuple[bool, List[str]]:
        """Registry-specific validation logic for automation tools."""
        errors = []
        
        try:
            # Validate name
            if not component.name or not isinstance(component.name, str):
                errors.append("Component name must be a non-empty string")
            
            # Validate function
            if not component.function or not callable(component.function):
                errors.append("Component function must be callable")
            
            # Validate module path
            if not component.module_path or not isinstance(component.module_path, str):
                errors.append("Component module_path must be a non-empty string")
            
            # Validate automation type
            if not isinstance(component.automation_type, AutomationType):
                errors.append("Component automation_type must be a valid AutomationType")
            
            # Validate parameters list
            if component.parameters and not isinstance(component.parameters, list):
                errors.append("Component parameters must be a list")
            
            # Validate supports_parallel
            if not isinstance(component.supports_parallel, bool):
                errors.append("Component supports_parallel must be a boolean")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {e}"]


# Global registry instance
_global_registry = None


def get_automation_registry() -> AutomationRegistry:
    """Get the global automation registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AutomationRegistry()
    return _global_registry


# Convenience functions for common operations
def discover_all_automation_components(rescan: bool = False) -> Dict[str, AutomationToolInfo]:
    """
    Discover all available automation components.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping component names to AutomationToolInfo objects
    """
    registry = get_automation_registry()
    return registry.discover_components(rescan)


def get_automation_component(component_name: str) -> Optional[AutomationToolInfo]:
    """
    Get an automation component by name.
    
    Args:
        component_name: Name of the component
        
    Returns:
        AutomationToolInfo object or None if not found
    """
    registry = get_automation_registry()
    return registry.get_component(component_name)


def find_components_by_type(automation_type: AutomationType) -> List[AutomationToolInfo]:
    """
    Find all components of a specific type.
    
    Args:
        automation_type: Type of automation
        
    Returns:
        List of matching AutomationToolInfo objects
    """
    registry = get_automation_registry()
    return registry.find_components_by_type(automation_type.value)


def list_available_automation_components() -> List[str]:
    """
    Get list of all available automation component names.
    
    Returns:
        List of component names
    """
    registry = get_automation_registry()
    return registry.list_component_names()


def refresh_automation_registry():
    """Refresh the automation registry cache."""
    registry = get_automation_registry()
    registry.refresh_cache()