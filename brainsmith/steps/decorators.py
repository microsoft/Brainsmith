"""
FINN Step Decorator

Simple decorator for registering FINN build steps with minimal metadata.
Steps now access transforms directly via apply_transform() helper.
"""

import functools
import warnings
from typing import List, Callable, Any

def finn_step(name: str, category: str = "unknown", dependencies: List[str] = None, 
              description: str = "", transforms: List[str] = None):
    """
    Decorator to register a FINN build step.
    
    Args:
        name: Step name for lookup (e.g., "shell_metadata_handover")
        category: Step category (e.g., "metadata", "validation", "hardware")
        dependencies: List of step names this step depends on
        description: Brief description of what the step does
        transforms: DEPRECATED - Steps now access transforms via apply_transform()
        
    Example:
        @finn_step(
            name="shell_metadata_handover",
            category="metadata",
            dependencies=[],
            description="Extract metadata for shell integration"
        )
        def shell_metadata_handover_step(model, cfg):
            # Use apply_transform() to access transforms directly
            model = apply_transform(model, "ExtractShellIntegrationMetadata", ...)
            return model
    """
    if dependencies is None:
        dependencies = []
    
    # Deprecation warning for transforms parameter
    if transforms is not None and len(transforms) > 0:
        warnings.warn(
            "The 'transforms' parameter in @finn_step is deprecated. "
            "Steps should use apply_transform() to access transforms directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
    def decorator(func: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
        # Store metadata on the function (no longer storing transforms)
        func._finn_step_name = name
        func._finn_step_category = category
        func._finn_step_dependencies = dependencies
        func._finn_step_description = description
        
        # Register with global registry (no longer passing transforms)
        from .registry import FinnStepRegistry
        registry = FinnStepRegistry()
        registry.register(name, func, category, dependencies, description)
        
        # Simple wrapper - all steps now use (model, cfg) signature
        @functools.wraps(func)
        def wrapper(model, cfg):
            return func(model, cfg)
        
        # Copy metadata to wrapper (no longer storing transforms)
        wrapper._finn_step_name = name
        wrapper._finn_step_category = category
        wrapper._finn_step_dependencies = dependencies
        wrapper._finn_step_description = description
        
        return wrapper
    
    return decorator