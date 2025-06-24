"""
FINN Build Steps Registration System

Minimal registration system for FINN build steps that take (model, cfg) -> model.
Provides backward compatibility with existing get_step() interface while using
a simple decorator-based registration approach.

Usage:
    from brainsmith.finn_steps import get_step
    
    # Get step function by name
    step_fn = get_step("shell_metadata_handover")
    model = step_fn(model, cfg)
    
    # List all available steps
    from brainsmith.finn_steps import list_finn_steps
    steps = list_finn_steps()
"""

from .decorators import finn_step
from .registry import FinnStepRegistry

# Create global registry instance
_registry = FinnStepRegistry()

def get_step(name: str):
    """
    Get FINN build step by name with legacy fallback.
    
    Args:
        name: Step name (e.g., "shell_metadata_handover")
        
    Returns:
        Step function with signature (model, cfg) -> model
    """
    return _registry.get_step(name)

def list_finn_steps():
    """List all registered FINN build steps."""
    return _registry.list_steps()

def register_step(name: str, func, category: str = "unknown", 
                  dependencies=None, description: str = ""):
    """Programmatically register a FINN step."""
    _registry.register(name, func, category, dependencies or [], description)

# Import all step modules to trigger registration
from . import steps

__all__ = ["finn_step", "get_step", "list_finn_steps", "register_step"]