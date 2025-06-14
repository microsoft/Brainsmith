"""
Kernel and Transform Selection Logic for Design Space Exploration

This module provides utilities for processing kernel and transform selections
from blueprints and generating all valid combinations for DSE exploration.
"""

from typing import List, Dict, Tuple, Set, Optional
import itertools
from dataclasses import dataclass


@dataclass
class KernelSelection:
    """Represents a kernel selection configuration"""
    available_kernels: List[str]
    mutually_exclusive_groups: List[List[str]]
    operation_mappings: Dict[str, List[str]]


@dataclass
class TransformSelection:
    """Represents a transform pipeline configuration"""
    core_pipeline: List[str]
    optional_transforms: List[str]
    mutually_exclusive_groups: List[List[str]]
    hooks: Dict[str, List[str]]  # Future 4-hooks support


def enumerate_kernel_combinations(kernel_selection: KernelSelection) -> List[List[str]]:
    """Generate all valid kernel combinations considering mutual exclusivity."""
    available = kernel_selection.available_kernels
    exclusive_groups = kernel_selection.mutually_exclusive_groups
    
    if not available:
        return []
    
    # Start with all available kernels
    base_choice = set(available)
    
    # Handle mutual exclusivity
    choices = []
    if not exclusive_groups:
        choices = [list(base_choice)]
    else:
        # Generate combinations for each exclusivity group
        group_combinations = []
        for group in exclusive_groups:
            # Only include kernels that are actually in available list
            valid_group = [k for k in group if k in available]
            if valid_group:
                group_combinations.append(valid_group)
        
        if not group_combinations:
            # No valid exclusive groups, return all available
            choices = [list(base_choice)]
        else:
            # Create Cartesian product of exclusive choices
            for combination in itertools.product(*group_combinations):
                choice = base_choice.copy()
                # Remove excluded kernels from each group
                for group in exclusive_groups:
                    valid_group = [k for k in group if k in available]
                    if valid_group:
                        selected = next((k for k in combination if k in valid_group), None)
                        if selected:
                            for kernel in valid_group:
                                if kernel != selected:
                                    choice.discard(kernel)
                choices.append(list(sorted(choice)))
    
    # Remove duplicates while preserving order
    unique_choices = []
    seen = set()
    for choice in choices:
        choice_tuple = tuple(sorted(choice))
        if choice_tuple not in seen:
            seen.add(choice_tuple)
            unique_choices.append(choice)
    
    return unique_choices


def enumerate_transform_pipelines(transform_selection: TransformSelection) -> List[List[str]]:
    """Generate all valid transform pipeline variants."""
    core = transform_selection.core_pipeline
    optional = transform_selection.optional_transforms
    exclusive_groups = transform_selection.mutually_exclusive_groups
    
    variants = []
    
    # Generate optional combinations (2^n combinations)
    for r in range(len(optional) + 1):
        for optional_combo in itertools.combinations(optional, r):
            # Generate exclusive group combinations
            if not exclusive_groups:
                pipeline = core + list(optional_combo)
                variants.append(pipeline)
            else:
                # Handle case where some exclusive groups might be empty
                valid_groups = [group for group in exclusive_groups if group]
                if not valid_groups:
                    pipeline = core + list(optional_combo)
                    variants.append(pipeline)
                else:
                    for exclusive_combo in itertools.product(*valid_groups):
                        pipeline = core + list(optional_combo) + list(exclusive_combo)
                        variants.append(pipeline)
    
    # Remove duplicates while preserving order
    unique_variants = []
    seen = set()
    for variant in variants:
        variant_tuple = tuple(variant)
        if variant_tuple not in seen:
            seen.add(variant_tuple)
            unique_variants.append(variant)
    
    return unique_variants


def validate_kernel_selection(kernel_selection: KernelSelection) -> Tuple[bool, List[str]]:
    """Validate kernel selection against registry."""
    try:
        from brainsmith.libraries.kernels import list_kernels
        available_in_registry = list_kernels()
    except ImportError:
        return False, ["Could not import kernel registry"]
    
    errors = []
    
    # Validate available kernels
    for kernel in kernel_selection.available_kernels:
        if kernel not in available_in_registry:
            errors.append(f"Kernel '{kernel}' not found in registry. Available: {', '.join(available_in_registry)}")
    
    # Validate mutual exclusivity groups
    for group in kernel_selection.mutually_exclusive_groups:
        for kernel in group:
            if kernel not in kernel_selection.available_kernels:
                errors.append(f"Kernel '{kernel}' in exclusivity group not in available kernels")
            if kernel not in available_in_registry:
                errors.append(f"Kernel '{kernel}' in exclusivity group not found in registry")
    
    return len(errors) == 0, errors


def validate_transform_selection(transform_selection: TransformSelection) -> Tuple[bool, List[str]]:
    """Validate transform selection against registry."""
    try:
        from brainsmith.libraries.transforms import list_transforms
        available_in_registry = list_transforms()
    except ImportError:
        return False, ["Could not import transform registry"]
    
    errors = []
    
    # Collect all transforms to validate
    all_transforms = (
        transform_selection.core_pipeline +
        transform_selection.optional_transforms +
        [t for group in transform_selection.mutually_exclusive_groups for t in group]
    )
    
    # Validate against registry
    for transform in all_transforms:
        if transform not in available_in_registry:
            errors.append(f"Transform '{transform}' not found in registry. Available: {', '.join(available_in_registry)}")
    
    return len(errors) == 0, errors