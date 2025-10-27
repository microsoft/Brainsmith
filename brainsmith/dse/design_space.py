# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Clean Design Space Implementation

This module defines the new GlobalDesignSpace that holds resolved component objects
from the blueprint, ready for tree construction.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union
from brainsmith.dse._constants import SKIP_INDICATOR


@dataclass
class GlobalDesignSpace:
    """Design space with resolved component objects."""
    model_path: str
    steps: List[Union[str, List[Optional[str]]]]  # Direct steps with variations

    # Kernel backends: [(kernel_name, [Backend classes])]
    # Currently all registered backends are included automatically.
    # TODO: Future - support backend filtering in blueprint YAML
    #   Example: {'LayerNorm': ['hls']} → only HLS backends
    kernel_backends: List[Tuple[str, List[Type]]]

    max_combinations: int = 100000  # Maximum allowed design space combinations
    
    def __post_init__(self):
        """Validate design space after initialization."""
        self._validate_size()

    def _validate_size(self) -> None:
        """Ensure design space doesn't exceed size limits."""
        combination_count = 1
        for step_spec in self.steps:
            if isinstance(step_spec, list):
                # Branch point - multiply by number of options
                combination_count *= len(step_spec)

        if combination_count > self.max_combinations:
            raise ValueError(
                f"Design space too large: {combination_count:,} combinations exceeds "
                f"limit of {self.max_combinations:,}"
            )
    
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
            f"GlobalDesignSpace(\n"
            f"  model: {self.model_path}\n"
            f"  steps: {len(self.steps)}\n"
            f"  kernels: {len(self.kernel_backends)}\n"
            f")"
        )


def _find_step_index(steps: List[Union[str, List[Optional[str]]]], step_name: str) -> int:
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
    available = [
        s if isinstance(s, str) else f"[{', '.join(str(opt) for opt in s)}]"
        for s in steps
    ]
    raise ValueError(
        f"Step '{step_name}' not found in steps list.\n"
        f"Available steps: {', '.join(available)}"
    )


def _slice_steps(
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
    start_idx = _find_step_index(steps, start_step) if start_step else 0
    stop_idx = _find_step_index(steps, stop_step) if stop_step else len(steps) - 1

    if start_idx > stop_idx:
        raise ValueError(
            f"Invalid step range: start_step '{start_step}' (index {start_idx}) "
            f"comes after stop_step '{stop_step}' (index {stop_idx})"
        )

    return steps[start_idx:stop_idx + 1]
