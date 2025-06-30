"""
BrainSmith Transforms Library - Registry Dictionary Pattern

Simple, explicit transform discovery using registry dictionary.  
No magical filesystem scanning - transforms explicitly registered.

Main Functions:
- get_transform(name): Get transform function by name with fail-fast errors
- list_transforms(): List all available transform names

Example Usage:
    from brainsmith.libraries.transforms import get_transform, list_transforms
    
    # List available transforms
    transforms = list_transforms()  # ['cleanup', 'streamlining', 'infer_hardware', ...]
    
    # Get specific transform
    cleanup_fn = get_transform('cleanup')
    model = cleanup_fn(model, config)
"""

from typing import List, Callable

# Direct imports - fail immediately if missing
from .steps.cleanup import cleanup_step, cleanup_advanced_step
from .steps.conversion import qonnx_to_finn_step
from .steps.streamlining import streamlining_step
from .steps.hardware import infer_hardware_step
from .steps.optimizations import constrain_folding_and_set_pumped_compute_step
from .steps.validation import generate_reference_io_step
from .steps.metadata import shell_metadata_handover_step
from .steps.bert import remove_head_step, remove_tail_step

# Registry maps transform names to their functions
AVAILABLE_TRANSFORMS = {
    "cleanup": cleanup_step,
    "cleanup_advanced": cleanup_advanced_step,
    "qonnx_to_finn": qonnx_to_finn_step,
    "streamlining": streamlining_step,
    "infer_hardware": infer_hardware_step,
    "constrain_folding_and_set_pumped_compute": constrain_folding_and_set_pumped_compute_step,
    "generate_reference_io": generate_reference_io_step,
    "shell_metadata_handover": shell_metadata_handover_step,
    "remove_head": remove_head_step,
    "remove_tail": remove_tail_step,
}

def get_transform(name: str) -> Callable:
    """
    Get transform function by name. Fails fast if not found.
    
    Args:
        name: Transform name to retrieve
        
    Returns:
        Transform function
        
    Raises:
        KeyError: If transform not found (with available options)
    """
    if name not in AVAILABLE_TRANSFORMS:
        available = ", ".join(sorted(AVAILABLE_TRANSFORMS.keys()))
        raise KeyError(f"Transform '{name}' not found. Available: {available}")
    
    return AVAILABLE_TRANSFORMS[name]

def list_transforms() -> List[str]:
    """
    List all available transform names.
    
    Returns:
        List of transform names
    """
    return list(AVAILABLE_TRANSFORMS.keys())



# Import operations module
from . import operations

# Export all public functions and types
__all__ = [
    # Registry functions
    'get_transform',
    'list_transforms',
    'AVAILABLE_TRANSFORMS',
    
    # Transform step functions
    'cleanup_step',
    'cleanup_advanced_step',
    'qonnx_to_finn_step',
    'streamlining_step',
    'infer_hardware_step',
    'constrain_folding_and_set_pumped_compute_step',
    'generate_reference_io_step',
    'shell_metadata_handover_step',
    'remove_head_step',
    'remove_tail_step',
    
    # Operations module
    'operations'
]

# Module metadata
__version__ = "2.0.0"  # Bumped for registry refactoring
__author__ = "BrainSmith Development Team"