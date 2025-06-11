"""
Transform Registry System

Auto-discovery and management of transformation operations and steps.
Provides unified access to both custom operations and FINN transformation steps.

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


class TransformType(Enum):
    """Types of transforms available."""
    OPERATION = "operation"
    STEP = "step"
    COMBINED = "combined"


@dataclass
class TransformInfo:
    """Information about a discovered transform."""
    name: str
    transform_type: TransformType
    function: Callable
    module_path: str
    category: str = "unknown"
    description: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class TransformRegistry(BaseRegistry[TransformInfo]):
    """Registry for auto-discovery and management of transforms."""
    
    def __init__(self, search_dirs: Optional[List[str]] = None, config_manager=None):
        """
        Initialize transform registry.
        
        Args:
            search_dirs: List of directories to search for transforms.
                        If None, uses default transform directories.
            config_manager: Optional configuration manager.
        """
        super().__init__(search_dirs, config_manager)
        # For backward compatibility, maintain transform_cache reference
        self.transform_cache = self._cache
        self.metadata_cache = self._metadata_cache
    
    def discover_components(self, rescan: bool = False) -> Dict[str, TransformInfo]:
        """
        Discover all available transforms (operations and steps).
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping transform names to TransformInfo objects
        """
        if not rescan:
            return self._cache
        
        discovered = {}
        
        # Discover operations
        operations = self._discover_operations()
        discovered.update(operations)
        
        # Discover steps
        steps = self._discover_steps()
        discovered.update(steps)
        
        # Cache the results
        self._cache = discovered
        self.transform_cache = self._cache  # Maintain backward compatibility reference
        
        self._log_info(f"Discovered {len(discovered)} transforms")
        return discovered
    
    def find_components_by_type(self, transform_type: TransformType) -> List[TransformInfo]:
        """
        Find transforms by type.
        
        Args:
            transform_type: Type of transform to search for
            
        Returns:
            List of matching TransformInfo objects
        """
        transforms = self.discover_components()
        matches = []
        
        for transform in transforms.values():
            if transform.transform_type == transform_type:
                matches.append(transform)
        
        return matches
    
    def find_transforms_by_category(self, category: str) -> List[TransformInfo]:
        """
        Find transforms by category.
        
        Args:
            category: Category to search for
            
        Returns:
            List of matching TransformInfo objects
        """
        transforms = self.discover_components()
        matches = []
        
        for transform in transforms.values():
            if transform.category == category:
                matches.append(transform)
        
        return matches
    
    def list_categories(self) -> Set[str]:
        """Get set of all available categories."""
        transforms = self.discover_components()
        return {transform.category for transform in transforms.values()}
    
    def validate_dependencies(self, transform_names: List[str]) -> List[str]:
        """
        Validate dependencies for a list of transforms.
        
        Args:
            transform_names: List of transform names
            
        Returns:
            List of validation errors
        """
        transforms = self.discover_components()
        errors = []
        
        for transform_name in transform_names:
            if transform_name not in transforms:
                errors.append(f"Transform '{transform_name}' not found")
                continue
            
            transform = transforms[transform_name]
            for dep in transform.dependencies:
                if dep not in transform_names:
                    errors.append(f"Transform '{transform_name}' requires '{dep}'")
                elif transform_names.index(dep) > transform_names.index(transform_name):
                    errors.append(f"Dependency '{dep}' must come before '{transform_name}'")
        
        return errors
    
    def _get_default_dirs(self) -> List[str]:
        """Get default search directories for transform registry."""
        current_dir = Path(__file__).parent
        return [str(current_dir)]
    
    def _extract_info(self, component: TransformInfo) -> Dict[str, Any]:
        """Extract standardized info from transform component."""
        return {
            'name': component.name,
            'type': 'transform',
            'transform_type': component.transform_type.value,
            'category': component.category,
            'description': component.description,
            'module_path': component.module_path,
            'dependencies': component.dependencies,
            'function': component.function.__name__
        }
    
    def _validate_component_implementation(self, component: TransformInfo) -> tuple[bool, List[str]]:
        """Transform-specific validation logic."""
        errors = []
        
        # Validate basic fields
        if not component.name:
            errors.append("Transform name is required")
        
        if not component.transform_type:
            errors.append("Transform type is required")
        
        if not component.function:
            errors.append("Transform function is required")
        elif not callable(component.function):
            errors.append("Transform function must be callable")
        
        # Validate dependencies exist
        if component.dependencies:
            transforms = self.discover_components()
            for dep in component.dependencies:
                if dep not in transforms:
                    errors.append(f"Dependency '{dep}' not found")
        
        return len(errors) == 0, errors
    
    def _discover_operations(self) -> Dict[str, TransformInfo]:
        """Discover transform operations."""
        operations = {}
        
        for transform_dir in self.search_dirs:
            operations_dir = os.path.join(transform_dir, "operations")
            
            if not os.path.exists(operations_dir):
                continue
            
            # Import operations module to access functions
            try:
                import brainsmith.libraries.transforms.operations as ops_module
                
                for name, obj in inspect.getmembers(ops_module):
                    if inspect.isfunction(obj) and not name.startswith('_'):
                        transform_info = TransformInfo(
                            name=name,
                            transform_type=TransformType.OPERATION,
                            function=obj,
                            module_path=f"brainsmith.libraries.transforms.operations.{obj.__module__.split('.')[-1]}",
                            category="operation",
                            description=self._extract_description(obj)
                        )
                        operations[name] = transform_info
                        
                # Also check submodules
                for item in os.listdir(operations_dir):
                    if item.endswith('.py') and item != '__init__.py':
                        module_name = item[:-3]
                        try:
                            module = getattr(ops_module, module_name, None)
                            if module:
                                for name, obj in inspect.getmembers(module):
                                    if inspect.isfunction(obj) and not name.startswith('_'):
                                        transform_info = TransformInfo(
                                            name=name,
                                            transform_type=TransformType.OPERATION,
                                            function=obj,
                                            module_path=f"brainsmith.libraries.transforms.operations.{module_name}",
                                            category="operation",
                                            description=self._extract_description(obj)
                                        )
                                        operations[name] = transform_info
                        except (ImportError, AttributeError):
                            self._log_debug(f"Could not import operations module: {module_name}")
                            
            except ImportError:
                self._log_debug("Could not import operations module")
        
        return operations
    
    def _discover_steps(self) -> Dict[str, TransformInfo]:
        """Discover transform steps."""
        steps = {}
        
        try:
            # Import the steps module which has discovery built-in
            import brainsmith.libraries.transforms.steps as steps_module
            
            # Use the existing discover_all_steps function
            if hasattr(steps_module, 'discover_all_steps'):
                discovered_steps = steps_module.discover_all_steps()
                
                for step_name, step_func in discovered_steps.items():
                    # Extract metadata if available
                    metadata = None
                    if hasattr(steps_module, 'extract_step_metadata'):
                        try:
                            metadata = steps_module.extract_step_metadata(step_func)
                        except:
                            pass
                    
                    transform_info = TransformInfo(
                        name=step_name,
                        transform_type=TransformType.STEP,
                        function=step_func,
                        module_path=f"brainsmith.libraries.transforms.steps.{step_func.__module__.split('.')[-1]}",
                        category=metadata.category if metadata else "step",
                        description=metadata.description if metadata else self._extract_description(step_func),
                        dependencies=metadata.dependencies if metadata else []
                    )
                    steps[step_name] = transform_info
            
        except ImportError:
            self._log_debug("Could not import steps module")
        
        return steps
    
    def _extract_description(self, func: Callable) -> str:
        """Extract description from function docstring."""
        docstring = func.__doc__ or ""
        
        # Get first line of docstring as description
        first_line = docstring.split('\n')[0].strip()
        return first_line if first_line else f"Transform function: {func.__name__}"


# Global registry instance
_global_registry = None


def get_transform_registry() -> TransformRegistry:
    """Get the global transform registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = TransformRegistry()
    return _global_registry


# BREAKING CHANGE: Updated convenience functions to use new unified interface
def discover_all_transforms(rescan: bool = False) -> Dict[str, TransformInfo]:
    """
    Discover all available transforms.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping transform names to TransformInfo objects
    """
    registry = get_transform_registry()
    return registry.discover_components(rescan)


def get_transform_by_name(transform_name: str) -> Optional[TransformInfo]:
    """
    Get a transform by name.
    
    Args:
        transform_name: Name of the transform
        
    Returns:
        TransformInfo object or None if not found
    """
    registry = get_transform_registry()
    return registry.get_component(transform_name)


def find_transforms_by_type(transform_type: TransformType) -> List[TransformInfo]:
    """
    Find all transforms of a specific type.
    
    Args:
        transform_type: Type of transform
        
    Returns:
        List of matching TransformInfo objects
    """
    registry = get_transform_registry()
    return registry.find_components_by_type(transform_type)


def list_available_transforms() -> List[str]:
    """
    Get list of all available transform names.
    
    Returns:
        List of transform names
    """
    registry = get_transform_registry()
    return registry.list_component_names()


def refresh_transform_registry():
    """Refresh the transform registry cache."""
    registry = get_transform_registry()
    registry.refresh_cache()