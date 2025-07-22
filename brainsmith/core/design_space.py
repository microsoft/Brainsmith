"""
Clean Design Space Implementation

This module defines the new DesignSpace that holds resolved plugin objects
from the blueprint, ready for tree construction.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
from enum import Enum

from .execution_tree import TransformStage


class OutputStage(Enum):
    """Target output stage for the compilation process."""
    COMPILE_AND_PACKAGE = "compile_and_package"
    SYNTHESIZE_BITSTREAM = "synthesize_bitstream"
    GENERATE_REPORTS = "generate_reports"


@dataclass
class GlobalConfig:
    """Configuration that applies to all exploration runs."""
    output_stage: OutputStage = OutputStage.COMPILE_AND_PACKAGE
    working_directory: str = "work"
    save_intermediate_models: bool = False
    max_combinations: int = field(default_factory=lambda: 
        int(os.environ.get("BRAINSMITH_MAX_COMBINATIONS", "100000")))
    timeout_minutes: int = field(default_factory=lambda: 
        int(os.environ.get("BRAINSMITH_TIMEOUT_MINUTES", "60")))


@dataclass
class DesignSpace:
    """
    Design space with resolved plugin objects.
    
    This is a clean intermediate representation between blueprint YAML
    and the execution tree. All plugin names have been resolved to
    actual classes from the registry.
    """
    model_path: str
    transform_stages: Dict[str, TransformStage]
    kernel_backends: List[Tuple[str, List[Type]]]  # [(kernel_name, [Backend classes])]
    build_pipeline: List[str]
    global_config: GlobalConfig
    
    def validate_size(self) -> None:
        """Validate that design space doesn't exceed max combinations."""
        estimated_size = self._estimate_combinations()
        if estimated_size > self.global_config.max_combinations:
            raise ValueError(
                f"Design space too large: {estimated_size:,} combinations exceeds "
                f"limit of {self.global_config.max_combinations:,}"
            )
    
    def _estimate_combinations(self) -> int:
        """Estimate total number of combinations without building tree."""
        total = 1
        
        # Transform stages
        for stage in self.transform_stages.values():
            stage_combos = len(stage.get_combinations())
            if stage_combos > 0:
                total *= stage_combos
        
        # Kernel backends (no branching in new design)
        # Each kernel has exactly one backend selected
        
        return total
    
    def get_stage_names(self) -> List[str]:
        """Get all transform stage names in pipeline order."""
        stage_names = []
        for step in self.build_pipeline:
            if step.startswith("{") and step.endswith("}"):
                stage_name = step[1:-1]
                if stage_name in self.transform_stages:
                    stage_names.append(stage_name)
        return stage_names
    
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
            f"  stages: {list(self.transform_stages.keys())}\n"
            f"  kernels: {len(self.kernel_backends)}\n"
            f"  pipeline: {len(self.build_pipeline)} steps\n"
            f")"
        )