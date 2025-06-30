"""
Simple step management for BrainSmith FINN transformations.
North Star-aligned: Direct function access without enterprise registry.

Main Functions:
- get_step(): Get step function by name with FINN fallback
- validate_step_sequence(): Validate step dependencies
- discover_all_steps(): Discover all available steps

Example Usage:
    from brainsmith.steps import cleanup_step, streamlining_step
    
    # Direct usage
    model = cleanup_step(model, cfg)
    model = streamlining_step(model, cfg)
    
    # FINN compatibility
    step_fn = get_step("cleanup")
    model = step_fn(model, cfg)
"""

import re
import inspect
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

# Import all steps by functionality
from .cleanup import cleanup_step, cleanup_advanced_step, fix_dynamic_dimensions_step
from .conversion import qonnx_to_finn_step
from .streamlining import streamlining_step
from .hardware import infer_hardware_step
from .optimizations import constrain_folding_and_set_pumped_compute_step
from .validation import generate_reference_io_step
from .metadata import shell_metadata_handover_step
from .bert import remove_head_step, remove_tail_step
from .preprocessing import onnx_preprocessing_step


@dataclass
class StepMetadata:
    """Simple metadata extracted from function docstrings."""
    name: str
    category: str
    description: str
    dependencies: List[str]


def extract_step_metadata(func: Callable) -> StepMetadata:
    """Extract metadata from function docstring."""
    docstring = func.__doc__ or ""
    
    # Parse structured docstring fields
    category = _extract_field(docstring, "Category") or "unknown"
    dependencies = _parse_list_field(docstring, "Dependencies") or []
    description = _extract_field(docstring, "Description") or ""
    
    # Generate step name from function name
    name = func.__name__.replace('_step', '')
    
    return StepMetadata(
        name=name,
        category=category,
        description=description,
        dependencies=dependencies
    )


def _extract_field(docstring: str, field_name: str) -> Optional[str]:
    """Extract a field value from docstring."""
    pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, docstring, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _parse_list_field(docstring: str, field_name: str) -> List[str]:
    """Parse a list field from docstring (e.g., Dependencies: [dep1, dep2])."""
    field_value = _extract_field(docstring, field_name)
    if not field_value:
        return []
    
    # Handle both [item1, item2] and [] formats
    if field_value.strip() == "[]":
        return []
    
    # Extract items from [item1, item2] format
    list_match = re.search(r'\[(.*?)\]', field_value)
    if list_match:
        items = list_match.group(1).split(',')
        return [item.strip().strip('"\'') for item in items if item.strip()]
    
    return []


def discover_all_steps(additional_paths: Optional[List[str]] = None) -> Dict[str, Callable]:
    """
    Discover all step functions from functional modules.
    
    Args:
        additional_paths: Additional paths for community steps (future)
        
    Returns:
        Dictionary mapping step names to functions
    """
    steps = {}
    
    # Import current module to access all step functions
    import brainsmith.steps as steps_module
    
    # Find all functions ending with '_step'
    for name, obj in inspect.getmembers(steps_module):
        if inspect.isfunction(obj) and name.endswith('_step'):
            step_name = name.replace('_step', '')
            steps[step_name] = obj
    
    return steps


def get_step(name: str) -> Callable:
    """
    Get step function by name with FINN fallback for compatibility.
    Maintains existing DataflowBuildConfig integration.
    """
    # Check BrainSmith legacy steps first
    steps = discover_all_steps()
    if name in steps:
        return steps[name]
    
    # Check new FINN steps system
    try:
        from brainsmith.steps import get_step as finn_get_step
        return finn_get_step(name)
    except (ImportError, ValueError):
        pass
    
    # Final fallback to FINN built-in steps
    from finn.builder.build_dataflow_steps import __dict__ as finn_steps
    if name in finn_steps and callable(finn_steps[name]):
        return finn_steps[name]
    
    raise ValueError(f"Step '{name}' not found in BrainSmith legacy, FINN steps, or FINN built-in")


def validate_step_sequence(step_names: List[str]) -> List[str]:
    """
    Validate step sequence and return any errors.
    
    Args:
        step_names: List of step names to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    steps = discover_all_steps()
    errors = []
    
    for step_name in step_names:
        # Check if step exists (either in BrainSmith or FINN)
        try:
            get_step(step_name)
        except ValueError:
            errors.append(f"Step '{step_name}' not found")
            continue
            
        # Check dependencies for BrainSmith steps
        if step_name in steps:
            metadata = extract_step_metadata(steps[step_name])
            for dep in metadata.dependencies:
                if dep not in step_names:
                    errors.append(f"Step '{step_name}' requires '{dep}'")
                elif step_names.index(dep) > step_names.index(step_name):
                    errors.append(f"Dependency '{dep}' must come before '{step_name}'")
    
    return errors


# Export all public functions and steps
__all__ = [
    # Preprocessing operations
    'onnx_preprocessing_step',
    # Cleanup operations
    'cleanup_step', 'cleanup_advanced_step', 'fix_dynamic_dimensions_step',
    # Conversion operations  
    'qonnx_to_finn_step',
    # Streamlining operations
    'streamlining_step',
    # Hardware operations
    'infer_hardware_step',
    # Optimization operations
    'constrain_folding_and_set_pumped_compute_step',
    # Validation operations
    'generate_reference_io_step',
    # Metadata operations
    'shell_metadata_handover_step',
    # BERT-specific operations
    'remove_head_step', 'remove_tail_step',
    # Discovery functions
    'get_step', 'validate_step_sequence', 'discover_all_steps',
    'extract_step_metadata', 'StepMetadata'
]