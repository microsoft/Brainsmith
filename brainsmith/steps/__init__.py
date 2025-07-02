"""
FINN Build Steps Registration System

Registration system for FINN build steps that take (model, cfg) -> model.
Provides backward compatibility with existing get_step() interface while using
the unified plugin system.

Usage:
    from brainsmith.steps import get_step
    
    # Get step function by name
    step_fn = get_step("shell_metadata_handover")
    model = step_fn(model, cfg)
    
    # List all available steps
    from brainsmith.steps import list_finn_steps
    steps = list_finn_steps()
"""

# Step decorator is available in the unified plugin system:
# from brainsmith.core.plugins import step
from .transform_resolver import resolve_transforms, validate_transform_dependencies, TransformResolutionError

def get_step(name: str):
    """
    Get FINN build step by name using unified plugin system.
    
    Args:
        name: Step name (e.g., "shell_metadata_handover")
        
    Returns:
        Step function with signature (model, cfg) -> model
    """
    from brainsmith.core.plugins import get_registry
    registry = get_registry()
    
    # Look for steps in unified plugin registry
    all_plugins = registry.list_all_plugins()
    for plugin in all_plugins:
        metadata = plugin['metadata']
        if metadata.get('type') == 'step' and metadata.get('name') == name:
            return plugin['class']
    
    # Fallback to FINN built-in steps
    try:
        from finn.builder.build_dataflow_steps import __dict__ as finn_steps
        if name in finn_steps and callable(finn_steps[name]):
            return finn_steps[name]
    except ImportError:
        pass
    
    raise ValueError(f"FINN step '{name}' not found in plugin registry or FINN built-ins")

def list_finn_steps():
    """List all registered FINN build steps from unified plugin system."""
    from brainsmith.core.plugins import get_registry
    registry = get_registry()
    
    step_names = []
    all_plugins = registry.list_all_plugins()
    for plugin in all_plugins:
        metadata = plugin['metadata']
        if metadata.get('type') == 'step':
            step_names.append(metadata.get('name'))
    
    return sorted(step_names)

def register_step(name: str, func, category: str = "unknown", 
                  dependencies=None, description: str = "", transforms=None):
    """
    DEPRECATED: Use @step decorator instead.
    
    Programmatically register a FINN step. This function is deprecated
    in favor of using the @step decorator from brainsmith.core.plugins.
    """
    import warnings
    warnings.warn(
        "register_step() is deprecated. Use @step decorator instead:\n"
        "from brainsmith.core.plugins import step\n"
        "@step(name='my_step', category='metadata')",
        DeprecationWarning,
        stacklevel=2
    )

def get_step_transforms(name: str):
    """
    DEPRECATED: Steps no longer declare transforms in registration.
    
    Steps now access transforms directly via the unified plugin system.
    """
    import warnings
    warnings.warn(
        "get_step_transforms() is deprecated. Steps now access transforms directly "
        "via 'from brainsmith.core.plugins import transforms as tfm'",
        DeprecationWarning,
        stacklevel=2
    )
    return []

# Import all step modules to trigger registration
from . import bert_steps

__all__ = [
    # Note: step decorator available as: from brainsmith.core.plugins import step
    "get_step", 
    "list_finn_steps", 
    "register_step",  # DEPRECATED - use @step decorator instead
    "get_step_transforms",  # DEPRECATED - steps access transforms directly
    "resolve_transforms",
    "validate_transform_dependencies",
    "TransformResolutionError"
]