"""
Step Collections

Provides natural access to FINN build step plugins.
"""

import logging
from typing import Dict, Optional, List, TYPE_CHECKING

from .base import BaseCollection
from .wrappers import StepWrapper

if TYPE_CHECKING:
    from ..core.data_models import PluginInfo
    from ..core.registry import PluginRegistry
    from ..core.loader import PluginLoader

logger = logging.getLogger(__name__)


class CategoryStepCollection:
    """
    Collection of steps in a specific category.
    
    Provides access to steps by category:
        validation_steps = steps.validation
        metadata_steps = steps.metadata
    """
    
    def __init__(self, category: str, registry: 'PluginRegistry', 
                 loader: 'PluginLoader'):
        self.category = category
        self.registry = registry
        self.loader = loader
        self._wrapper_cache: Dict[str, StepWrapper] = {}
    
    def __getattr__(self, name: str) -> StepWrapper:
        """Get step by name from this category."""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
        if name in self._wrapper_cache:
            return self._wrapper_cache[name]
        
        # Find step in registry
        steps = self.registry.list_steps(self.category)
        
        for step in steps:
            if step.name == name:
                wrapper = StepWrapper(step, self.loader)
                self._wrapper_cache[name] = wrapper
                return wrapper
        
        # Not found in category
        available = [s.name for s in steps]
        raise AttributeError(
            f"Step '{name}' not found in category '{self.category}'. "
            f"Available steps: {available}"
        )
    
    def list_steps(self) -> List[str]:
        """List all steps in this category."""
        steps = self.registry.list_steps(self.category)
        return sorted([s.name for s in steps])
    
    def __dir__(self) -> List[str]:
        """Support tab completion."""
        return self.list_steps()
    
    def __repr__(self) -> str:
        return f"CategoryStepCollection({self.category})"


class StepCollection(BaseCollection):
    """
    Main step collection providing natural access to FINN build steps.
    
    Usage:
        steps = StepCollection(registry, loader)
        
        # Direct access to steps
        model = steps.shell_metadata_handover(model, cfg)
        model = steps.qonnx_to_finn(model, cfg)
        
        # Access by category
        validation_step = steps.validation.some_validation_step
        metadata_step = steps.metadata.shell_metadata_handover
    """
    
    def __init__(self, registry: 'PluginRegistry', loader: 'PluginLoader'):
        super().__init__(registry, loader)
        self._category_collections: Dict[str, CategoryStepCollection] = {}
    
    @property
    def plugin_type(self) -> str:
        return "step"
    
    def _create_wrapper(self, plugin_info: 'PluginInfo') -> StepWrapper:
        """Create step wrapper."""
        return StepWrapper(plugin_info, self.loader)
    
    def __getattr__(self, name: str) -> any:
        """
        Get step by name or category collection.
        
        First checks if name is a category, then checks for step name.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
        # Check if it's a category
        categories = self.list_categories()
        if name in categories:
            if name not in self._category_collections:
                self._category_collections[name] = CategoryStepCollection(
                    name, self.registry, self.loader
                )
            return self._category_collections[name]
        
        # Try to get step directly
        return self._get_wrapper(name)
    
    def list_categories(self) -> List[str]:
        """List all step categories."""
        all_steps = self.registry.list_steps()
        categories = set(s.metadata.get('category', 'unknown') for s in all_steps)
        return sorted(categories)
    
    def list_by_category(self) -> Dict[str, List[str]]:
        """List all steps organized by category."""
        result = {}
        
        for category in self.list_categories():
            steps = self.registry.list_steps(category)
            result[category] = sorted([s.name for s in steps])
        
        return result
    
    def list_with_dependencies(self) -> Dict[str, List[str]]:
        """List all steps with their dependencies."""
        result = {}
        
        for step in self.registry.list_steps():
            deps = step.metadata.get('dependencies', [])
            if deps:
                result[step.name] = deps
        
        return result
    
    def validate_order(self, step_names: List[str]) -> List[str]:
        """
        Validate step execution order based on dependencies.
        
        Returns list of errors if any.
        """
        errors = []
        step_set = set(step_names)
        
        for i, step_name in enumerate(step_names):
            try:
                step_info = self.registry.get_plugin(step_name)
                if step_info and step_info.plugin_type == "step":
                    deps = step_info.metadata.get('dependencies', [])
                    
                    for dep in deps:
                        if dep not in step_set:
                            errors.append(f"Step '{step_name}' requires '{dep}' which is not in list")
                        elif dep in step_names[i:]:
                            errors.append(f"Step '{step_name}' depends on '{dep}' which comes after it")
            except:
                # Step might not be in registry, skip validation
                pass
        
        return errors
    
    def __dir__(self) -> List[str]:
        """Support tab completion with step names and categories."""
        # Get all step names
        step_names = [s.name for s in self.registry.list_steps()]
        
        # Add category names
        categories = self.list_categories()
        
        return sorted(step_names + categories)
    
    def __repr__(self) -> str:
        categories = self.list_categories()
        return f"StepCollection(categories={categories})"