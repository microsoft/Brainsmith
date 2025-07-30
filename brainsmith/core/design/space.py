# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Clean Design Space Implementation

This module defines the new DesignSpace that holds resolved plugin objects
from the blueprint, ready for tree construction.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union


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
                # Branch point - multiply by number of options
                # Account for skip options (~, None, or empty string)
                valid_options = sum(1 for opt in step if opt and opt != "~")
                if any(opt in [None, "~", ""] for opt in step):
                    valid_options += 1  # Add one for skip branch
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