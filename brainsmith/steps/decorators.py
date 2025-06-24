"""
FINN Step Decorator

Simple decorator for registering FINN build steps with minimal metadata.
"""

import functools
from typing import List, Callable, Any

def finn_step(name: str, category: str = "unknown", dependencies: List[str] = None, 
              description: str = ""):
    """
    Decorator to register a FINN build step.
    
    Args:
        name: Step name for lookup (e.g., "shell_metadata_handover")
        category: Step category (e.g., "metadata", "validation", "hardware")
        dependencies: List of step names this step depends on
        description: Brief description of what the step does
        
    Example:
        @finn_step(
            name="shell_metadata_handover",
            category="metadata",
            dependencies=[],
            description="Extract metadata for shell integration"
        )
        def shell_metadata_handover_step(model, cfg):
            # Implementation
            return model
    """
    if dependencies is None:
        dependencies = []
        
    def decorator(func: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
        # Store metadata on the function
        func._finn_step_name = name
        func._finn_step_category = category
        func._finn_step_dependencies = dependencies
        func._finn_step_description = description
        
        # Register with global registry
        from .registry import FinnStepRegistry
        registry = FinnStepRegistry()
        registry.register(name, func, category, dependencies, description)
        
        @functools.wraps(func)
        def wrapper(model, cfg):
            return func(model, cfg)
        
        # Copy metadata to wrapper
        wrapper._finn_step_name = name
        wrapper._finn_step_category = category
        wrapper._finn_step_dependencies = dependencies
        wrapper._finn_step_description = description
        
        return wrapper
    
    return decorator