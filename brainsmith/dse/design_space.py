# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Clean Design Space Implementation

This module defines the new GlobalDesignSpace that holds resolved component objects
from the blueprint, ready for tree construction.
"""

from dataclasses import dataclass


@dataclass
class GlobalDesignSpace:
    """Design space ready for DSE tree construction.

    Attributes:
        model_path: Path to ONNX model file
        steps: Pipeline steps with optional variations (list = branch point)
        kernel_backends: Kernel names mapped to backend classes
        max_combinations: Maximum allowed design space size (default: 100,000)

    Validates size on initialization and provides kernel summary formatting.
    """
    model_path: str
    steps: list[str | list[str | None]]  # Direct steps with variations

    # Kernel backends: [(kernel_name, [Backend classes])]
    # Currently all registered backends are included automatically.
    # TODO: Future - support backend filtering in blueprint YAML
    #   Example: {'LayerNorm': ['hls']} → only HLS backends
    kernel_backends: list[tuple[str, list[type]]]

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
