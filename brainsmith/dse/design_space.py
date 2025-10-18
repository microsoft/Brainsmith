# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Clean Design Space Implementation

This module defines the new DesignSpace that holds resolved plugin objects
from the blueprint, ready for tree construction.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union
from brainsmith.registry import has_step, list_all_steps
from brainsmith.dse._constants import is_skip


@dataclass
class DesignSpace:
    """
    Design space with resolved plugin objects.
    
    This is a clean intermediate representation between blueprint YAML
    and the execution tree. All plugin names have been resolved to
    actual classes from the registry.
    """
    model_path: str
    steps: List[Union[str, List[Optional[str]]]]  # Direct steps with variations
    kernel_backends: List[Tuple[str, List[Type]]]  # [(kernel_name, [Backend classes])]
    max_combinations: int = 100000  # Maximum allowed design space combinations
    
    def __post_init__(self):
        """Validate design space after initialization."""
        self.validate_steps()
        self.validate_size()
    
    def validate_steps(self) -> None:
        """Validate that all referenced steps exist in the registry."""
        invalid_steps = []
        
        def check_step(step: Optional[str]) -> None:
            """Check if a single step is valid."""
            if step and not is_skip(step) and not has_step(step):
                invalid_steps.append(step)
        
        # Check all steps, including those in branch points
        for step_spec in self.steps:
            if isinstance(step_spec, list):
                # Branch point - check each option
                for option in step_spec:
                    check_step(option)
            else:
                # Single step
                check_step(step_spec)
        
        if invalid_steps:
            available_steps = list_all_steps()
            error_msg = (
                f"Invalid steps found: {', '.join(invalid_steps)}\n\n"
                f"Available steps: {', '.join(available_steps)}"
            )
            raise ValueError(error_msg)
    
    def validate_size(self) -> None:
        """Validate that design space doesn't exceed max combinations."""
        estimated_size = self._estimate_combinations()
        if estimated_size > self.max_combinations:
            raise ValueError(
                f"Design space too large: {estimated_size:,} combinations exceeds "
                f"limit of {self.max_combinations:,}"
            )
    
    def _estimate_combinations(self) -> int:
        """Estimate total number of combinations without building tree."""
        total = 1
        
        # Step variations
        for step in self.steps:
            if isinstance(step, list):
                # Branch point - count non-skip options and skip option (if present)
                non_skip_count = sum(1 for opt in step if not is_skip(opt))
                has_skip_option = any(is_skip(opt) for opt in step)
                valid_options = non_skip_count + (1 if has_skip_option else 0)
                total *= max(1, valid_options)
        
        # Kernel backends (no branching in current design)
        # Each kernel has exactly one backend selected
        
        return total
    
    def get_kernel_summary(self) -> str:
        """Get human-readable summary of kernels and backends."""
        lines = []
        for kernel_name, backend_classes in self.kernel_backends:
            backend_names = [cls.__name__ for cls in backend_classes]
            lines.append(f"  {kernel_name}: {', '.join(backend_names)}")
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"DesignSpace(\n"
            f"  model: {self.model_path}\n"
            f"  steps: {len(self.steps)}\n"
            f"  kernels: {len(self.kernel_backends)}\n"
            f")"
        )


def find_step_index(steps: List[Union[str, List[Optional[str]]]], step_name: str) -> int:
    """Find index of step in steps list, checking inside branch points.

    Args:
        steps: List of step specifications
        step_name: Name of step to find

    Returns:
        Index of the step

    Raises:
        ValueError: If step not found in steps list
    """
    for i, step in enumerate(steps):
        if step == step_name:
            return i
        if isinstance(step, list) and step_name in step:
            return i

    # Step not found - create helpful error message
    available = []
    for s in steps:
        if isinstance(s, str):
            available.append(s)
        elif isinstance(s, list):
            available.append(f"[{', '.join(str(opt) for opt in s)}]")

    raise ValueError(
        f"Step '{step_name}' not found in steps list.\n"
        f"Available steps: {', '.join(available)}"
    )


def slice_steps(
    steps: List[Union[str, List[Optional[str]]]],
    start_step: Optional[str] = None,
    stop_step: Optional[str] = None
) -> List[Union[str, List[Optional[str]]]]:
    """Slice steps list by step names.

    Handles both string steps and branch point lists.
    Slicing is inclusive on both ends.

    Args:
        steps: List of step specifications
        start_step: Name of step to start from (inclusive), None for beginning
        stop_step: Name of step to stop at (inclusive), None for end

    Returns:
        Sliced steps list

    Raises:
        ValueError: If start_step or stop_step not found
    """
    start_idx = find_step_index(steps, start_step) if start_step else 0
    stop_idx = find_step_index(steps, stop_step) if stop_step else len(steps) - 1

    if start_idx > stop_idx:
        raise ValueError(
            f"Invalid step range: start_step '{start_step}' (index {start_idx}) "
            f"comes after stop_step '{stop_step}' (index {stop_idx})"
        )

    return steps[start_idx:stop_idx + 1]
