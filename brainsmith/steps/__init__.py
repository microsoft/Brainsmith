# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    Get build step by name using unified plugin system.
    
    Args:
        name: Step name (e.g., "shell_metadata_handover")
        framework: Ignored for compatibility
        
    Returns:
        Step function with signature (model, cfg) -> model
    """
    from brainsmith.core.plugins import get_step as _get_step
    
    step_function = _get_step(name)
    if step_function:
        return step_function
    
    # Fallback to FINN built-in steps
    try:
        from finn.builder.build_dataflow_steps import __dict__ as finn_steps
        # Try with step_ prefix for legacy compatibility
        step_name_with_prefix = f"step_{name}"
        if step_name_with_prefix in finn_steps and callable(finn_steps[step_name_with_prefix]):
            return finn_steps[step_name_with_prefix]
        # Try without prefix
        if name in finn_steps and callable(finn_steps[name]):
            return finn_steps[name]
    except ImportError as e:
        # Arete: Fail with clear error about missing FINN
        raise ImportError(
            f"Cannot access FINN built-in steps: {e}\n"
            "Install FINN with: pip install git+https://github.com/Xilinx/finn.git"
        ) from e
    
    # Step not found anywhere
    raise ValueError(f"Step '{name}' not found in plugin registry or FINN built-ins")

def list_finn_steps():
    """List all registered FINN build steps from unified plugin system."""
    from brainsmith.core.plugins import get_registry
    registry = get_registry()
    
    # Get all step names from the registry
    all_steps = registry.all('step')
    return sorted(all_steps.keys())

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
from . import core_steps
from . import bert_custom_steps
from . import kernel_inference

__all__ = [
    # Note: step decorator available as: from brainsmith.core.plugins import step
    "get_step", 
    "list_finn_steps", 
    "register_step",  # DEPRECATED - use @step decorator instead
    "get_step_transforms",  # DEPRECATED - steps access transforms directly
]