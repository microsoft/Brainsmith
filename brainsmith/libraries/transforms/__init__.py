"""
Transform Libraries

Auto-discovery and management of transformation operations and steps.
Provides unified access to both custom operations and FINN transformation steps.

Main exports:
- TransformRegistry: Registry for transform discovery and management
- discover_all_transforms: Discover all available transforms
- get_transform_by_name: Get specific transform by name
- find_transforms_by_type: Find transforms by type (operation/step)
"""

# Import registry system
from .registry import (
    TransformRegistry,
    TransformType,
    TransformInfo,
    get_transform_registry,
    discover_all_transforms,
    get_transform_by_name,
    find_transforms_by_type,
    list_available_transforms,
    refresh_transform_registry
)

# Import operations module
from . import operations

# Import steps functionality (existing) - handle missing dependencies gracefully
try:
    from .steps import (
        # Core step functions
        cleanup_step,
        cleanup_advanced_step,
        qonnx_to_finn_step,
        streamlining_step,
        infer_hardware_step,
        constrain_folding_and_set_pumped_compute_step,
        generate_reference_io_step,
        shell_metadata_handover_step,
        remove_head_step,
        remove_tail_step,
        
        # Step discovery
        get_step,
        validate_step_sequence,
        discover_all_steps,
        extract_step_metadata,
        StepMetadata
    )
    _STEPS_AVAILABLE = True
except ImportError as e:
    # Handle missing dependencies (qonnx, finn, etc.) gracefully
    _STEPS_AVAILABLE = False
    
    # Define minimal stubs for missing functions
    def cleanup_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def cleanup_advanced_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def qonnx_to_finn_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def streamlining_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def infer_hardware_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def constrain_folding_and_set_pumped_compute_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def generate_reference_io_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def shell_metadata_handover_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def remove_head_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def remove_tail_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def get_step(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def validate_step_sequence(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def discover_all_steps(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    def extract_step_metadata(*args, **kwargs):
        raise ImportError(f"Steps functionality not available: {e}")
    
    # Define minimal StepMetadata stub
    class StepMetadata:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Steps functionality not available: {e}")

__all__ = [
    # Registry system
    "TransformRegistry",
    "TransformType",
    "TransformInfo",
    "get_transform_registry",
    "discover_all_transforms",
    "get_transform_by_name",
    "find_transforms_by_type",
    "list_available_transforms",
    "refresh_transform_registry",
    
    # Operations module
    "operations",
    
    # Step functions
    "cleanup_step",
    "cleanup_advanced_step", 
    "qonnx_to_finn_step",
    "streamlining_step",
    "infer_hardware_step",
    "constrain_folding_and_set_pumped_compute_step",
    "generate_reference_io_step",
    "shell_metadata_handover_step",
    "remove_head_step",
    "remove_tail_step",
    
    # Step discovery
    "get_step",
    "validate_step_sequence",
    "discover_all_steps",
    "extract_step_metadata",
    "StepMetadata"
]