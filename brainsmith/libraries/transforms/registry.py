"""
Transform Registry System

Auto-discovery and management of transformation operations and steps.
Provides unified access to both custom operations and FINN transformation steps.
"""

import os
import inspect
import logging
from typing import Dict, List, Optional, Set, Callable, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

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


class TransformRegistry:
    """Registry for auto-discovery and management of transforms."""
    
    def __init__(self, transform_dirs: Optional[List[str]] = None):
        """
        Initialize transform registry.
        
        Args:
            transform_dirs: List of directories to search for transforms.
                          Defaults to current transforms directory
        """
        if transform_dirs is None:
            # Default to the current transforms directory
            current_dir = Path(__file__).parent
            transform_dirs = [str(current_dir)]
        
        self.transform_dirs = transform_dirs
        self.transform_cache = {}
        self.metadata_cache = {}
        
        logger.info(f"Transform registry initialized with dirs: {self.transform_dirs}")
    
    def discover_transforms(self, rescan: bool = False) -> Dict[str, TransformInfo]:
        """
        Discover all available transforms (operations and steps).
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping transform names to TransformInfo objects
        """
        if self.transform_cache and not rescan:
            return self.transform_cache
        
        discovered = {}
        
        # Discover operations
        operations = self._discover_operations()
        discovered.update(operations)
        
        # Discover steps
        steps = self._discover_steps()
        discovered.update(steps)
        
        # Cache the results
        self.transform_cache = discovered
        
        logger.info(f"Discovered {len(discovered)} transforms")
        return discovered
    
    def _discover_operations(self) -> Dict[str, TransformInfo]:
        """Discover transform operations."""
        operations = {}
        
        for transform_dir in self.transform_dirs:
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
                            logger.debug(f"Could not import operations module: {module_name}")
                            
            except ImportError:
                logger.debug("Could not import operations module")
        
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
            logger.debug("Could not import steps module")
        
        return steps
    
    def get_transform(self, transform_name: str) -> Optional[TransformInfo]:
        """
        Get a specific transform by name.
        
        Args:
            transform_name: Name of the transform
            
        Returns:
            TransformInfo object or None if not found
        """
        transforms = self.discover_transforms()
        return transforms.get(transform_name)
    
    def find_transforms_by_type(self, transform_type: TransformType) -> List[TransformInfo]:
        """
        Find transforms by type.
        
        Args:
            transform_type: Type of transform to search for
            
        Returns:
            List of matching TransformInfo objects
        """
        transforms = self.discover_transforms()
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
        transforms = self.discover_transforms()
        matches = []
        
        for transform in transforms.values():
            if transform.category == category:
                matches.append(transform)
        
        return matches
    
    def list_transform_names(self) -> List[str]:
        """Get list of all available transform names."""
        transforms = self.discover_transforms()
        return list(transforms.keys())
    
    def list_categories(self) -> Set[str]:
        """Get set of all available categories."""
        transforms = self.discover_transforms()
        return {transform.category for transform in transforms.values()}
    
    def get_transform_info(self, transform_name: str) -> Optional[Dict]:
        """
        Get summary information about a transform.
        
        Args:
            transform_name: Name of the transform
            
        Returns:
            Dictionary with transform summary or None if not found
        """
        transform = self.get_transform(transform_name)
        if not transform:
            return None
        
        return {
            'name': transform.name,
            'type': transform.transform_type.value,
            'category': transform.category,
            'description': transform.description,
            'module_path': transform.module_path,
            'dependencies': transform.dependencies,
            'function': transform.function.__name__
        }
    
    def validate_dependencies(self, transform_names: List[str]) -> List[str]:
        """
        Validate dependencies for a list of transforms.
        
        Args:
            transform_names: List of transform names
            
        Returns:
            List of validation errors
        """
        transforms = self.discover_transforms()
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
    
    def refresh_cache(self):
        """Refresh the transform cache by clearing it."""
        self.transform_cache.clear()
        self.metadata_cache.clear()
        logger.info("Transform registry cache refreshed")
    
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


# Convenience functions for common operations
def discover_all_transforms(rescan: bool = False) -> Dict[str, TransformInfo]:
    """
    Discover all available transforms.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping transform names to TransformInfo objects
    """
    registry = get_transform_registry()
    return registry.discover_transforms(rescan)


def get_transform_by_name(transform_name: str) -> Optional[TransformInfo]:
    """
    Get a transform by name.
    
    Args:
        transform_name: Name of the transform
        
    Returns:
        TransformInfo object or None if not found
    """
    registry = get_transform_registry()
    return registry.get_transform(transform_name)


def find_transforms_by_type(transform_type: TransformType) -> List[TransformInfo]:
    """
    Find all transforms of a specific type.
    
    Args:
        transform_type: Type of transform
        
    Returns:
        List of matching TransformInfo objects
    """
    registry = get_transform_registry()
    return registry.find_transforms_by_type(transform_type)


def list_available_transforms() -> List[str]:
    """
    Get list of all available transform names.
    
    Returns:
        List of transform names
    """
    registry = get_transform_registry()
    return registry.list_transform_names()


def refresh_transform_registry():
    """Refresh the transform registry cache."""
    registry = get_transform_registry()
    registry.refresh_cache()