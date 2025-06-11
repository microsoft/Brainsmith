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

# Import steps functionality (existing)
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