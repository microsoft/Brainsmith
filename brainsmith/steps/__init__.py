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
from typing import Optional

def get_step(name: str, framework: Optional[str] = None):
    """
    Get build step by name using unified plugin system with optional framework qualification.
    
    Args:
        name: Step name (e.g., "shell_metadata_handover") or framework-qualified name (e.g., "finn:qonnx_to_finn")
        framework: Optional framework filter (e.g., "finn", "brainsmith")
        
    Returns:
        Step function with signature (model, cfg) -> model
        
    Examples:
        get_step("qonnx_to_finn")                    # → Last registered version
        get_step("qonnx_to_finn", "brainsmith")      # → Brainsmith version specifically  
        get_step("finn:qonnx_to_finn")               # → FINN version (framework parsed from name)
    """
    from brainsmith.core.plugins import get_registry
    
    # Parse framework qualification from name if present
    parsed_framework = framework
    parsed_name = name
    
    if ':' in name and framework is None:
        parts = name.split(':', 1)
        if len(parts) == 2:
            parsed_framework, parsed_name = parts[0].strip(), parts[1].strip()
    
    registry = get_registry()
    
    # If a framework was specified (either via parameter or qualifier), validate it exists
    if parsed_framework is not None:
        # Check if framework exists in registry
        available_frameworks = list(registry.framework_steps.keys())
        if parsed_framework not in available_frameworks:
            raise ValueError(
                f"Framework '{parsed_framework}' not found. "
                f"Available frameworks: {sorted(available_frameworks)}"
            )
        
        # Try to get step from specified framework
        step_function = registry.get_step(parsed_name, parsed_framework)
        if step_function:
            return step_function
        
        # If framework is 'finn', also check built-in steps
        if parsed_framework == 'finn':
            from finn.builder.build_dataflow_steps import __dict__ as finn_steps
            # Try with step_ prefix for legacy compatibility
            step_name_with_prefix = f"step_{parsed_name}"
            if step_name_with_prefix in finn_steps and callable(finn_steps[step_name_with_prefix]):
                return finn_steps[step_name_with_prefix]
            # Try without prefix
            if parsed_name in finn_steps and callable(finn_steps[parsed_name]):
                return finn_steps[parsed_name]
        
        # Step not found in specified framework
        raise ValueError(f"Step '{parsed_name}' not found in {parsed_framework} framework")
    
    # No framework specified - use default resolution
    step_function = registry.get_step(parsed_name)
    if step_function:
        return step_function
    
    # Fallback to FINN built-in steps (only if no framework specified)
    from finn.builder.build_dataflow_steps import __dict__ as finn_steps
    # Try with step_ prefix for legacy compatibility
    step_name_with_prefix = f"step_{parsed_name}"
    if step_name_with_prefix in finn_steps and callable(finn_steps[step_name_with_prefix]):
        return finn_steps[step_name_with_prefix]
    # Try without prefix
    if parsed_name in finn_steps and callable(finn_steps[parsed_name]):
        return finn_steps[parsed_name]
    
    # Step not found anywhere
    raise ValueError(f"Step '{parsed_name}' not found in plugin registry or FINN built-ins")

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
]