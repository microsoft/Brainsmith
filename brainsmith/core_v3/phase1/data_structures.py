"""
Core data structures for DSE V3 Design Space Constructor.

This module defines all the data structures used to represent design spaces,
including hardware compiler configurations, processing options, search settings,
and global parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import itertools


class SearchStrategy(Enum):
    """Available search strategies for design space exploration."""
    EXHAUSTIVE = "exhaustive"
    # Future strategies can be added here:
    # ADAPTIVE = "adaptive"
    # ML_GUIDED = "ml_guided"
    # RANDOM_SAMPLING = "random_sampling"


class OutputStage(Enum):
    """Target output stage for the compilation process."""
    DATAFLOW_GRAPH = "dataflow_graph"
    RTL = "rtl"
    STITCHED_IP = "stitched_ip"


# Kernel and Transform options are represented as simple tuples:
# Kernel: (name: str, backends: List[str])
# Transform: name: str
# Empty string represents a skipped/optional element


@dataclass
class ProcessingStep:
    """
    Represents a pre/post-processing step configuration.
    
    Attributes:
        name: Step name
        type: Either "preprocessing" or "postprocessing"
        parameters: Configuration parameters for this step
        enabled: Whether this step is enabled
    """
    name: str
    type: str  # "preprocessing" or "postprocessing"
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def __str__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.name} ({self.type}, {status})"


@dataclass
class HWCompilerSpace:
    """
    Hardware compiler configuration space.
    
    This defines all possible hardware configurations including kernels,
    transforms, build steps, and compiler flags. The space can contain
    alternatives that will be explored during DSE.
    
    Attributes:
        kernels: List of kernel configurations (can be nested for alternatives)
        transforms: Transform configurations (flat list or phase-based dict)
        build_steps: Fixed sequence of build steps
        config_flags: Fixed configuration flags for the compiler
    """
    kernels: List[Union[str, tuple, list]]
    transforms: Union[List[Union[str, list]], Dict[str, List[Union[str, list]]]]
    build_steps: List[str]
    config_flags: Dict[str, Any] = field(default_factory=dict)
    
    def get_kernel_combinations(self) -> List[List[tuple]]:
        """
        Generate all valid kernel combinations from the kernel space.
        
        Returns:
            List of kernel combinations, where each kernel is a tuple (name, backends)
            Empty name means the kernel is skipped (optional)
        """
        combinations = []
        for item in self.kernels:
            combinations.append(self._parse_kernel_item(item))
        
        # Generate cartesian product of all kernel options
        return list(itertools.product(*combinations))
    
    def _parse_kernel_item(self, item) -> List[tuple]:
        """Parse a single kernel configuration item into (name, backends) tuples."""
        options = []
        
        if isinstance(item, str):
            # Simple string: auto-import all backends
            options.append((item, ["*"]))
        
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], list):
            # Tuple/List format: (name, backends) or [name, backends]
            name, backends = item
            if name and isinstance(name, str) and name.startswith("~"):
                # Optional kernel - add both enabled and disabled
                options.append((name[1:], backends))
                options.append(("", []))  # Empty name = skipped
            elif name and isinstance(name, str):
                options.append((name, backends))
            else:
                # This is actually a mutually exclusive list, not a tuple
                for subitem in item:
                    options.extend(self._parse_kernel_item(subitem))
        
        elif isinstance(item, list):
            # List format: mutually exclusive options
            for subitem in item:
                if subitem is None or subitem == "~":
                    # None option (skip this kernel)
                    options.append(("", []))
                else:
                    options.extend(self._parse_kernel_item(subitem))
        
        return options
    
    def get_transform_combinations(self) -> List[List[str]]:
        """
        Generate all valid transform combinations from the transform space.
        
        Returns:
            List of transform combinations, where each transform is a string name
            Empty string means the transform is skipped (optional)
        """
        if isinstance(self.transforms, dict):
            # Phase-based transforms
            all_combinations = []
            for phase, transforms in self.transforms.items():
                phase_combinations = []
                for item in transforms:
                    phase_combinations.append(self._parse_transform_item(item))
                all_combinations.extend(phase_combinations)
            return list(itertools.product(*all_combinations))
        else:
            # Flat list of transforms
            combinations = []
            for item in self.transforms:
                combinations.append(self._parse_transform_item(item))
            return list(itertools.product(*combinations))
    
    def _parse_transform_item(self, item) -> List[str]:
        """Parse a single transform configuration item into transform names."""
        options = []
        
        if isinstance(item, str):
            if item.startswith("~"):
                # Optional transform - add both enabled and disabled options
                options.append(item[1:])  # Transform name without ~
                options.append("")  # Empty = skipped
            else:
                options.append(item)
        
        elif isinstance(item, list):
            # List format: mutually exclusive options
            for subitem in item:
                if subitem is None or subitem == "~":
                    # None option (skip this transform)
                    options.append("")
                else:
                    options.extend(self._parse_transform_item(subitem))
        
        return options


@dataclass
class ProcessingSpace:
    """
    Pre/post-processing configuration space.
    
    Attributes:
        preprocessing: List of preprocessing step alternatives
        postprocessing: List of postprocessing step alternatives
    """
    preprocessing: List[List[ProcessingStep]] = field(default_factory=list)
    postprocessing: List[List[ProcessingStep]] = field(default_factory=list)
    
    def get_preprocessing_combinations(self) -> List[List[ProcessingStep]]:
        """Get all preprocessing combinations."""
        if not self.preprocessing:
            return [[]]
        return list(itertools.product(*self.preprocessing))
    
    def get_postprocessing_combinations(self) -> List[List[ProcessingStep]]:
        """Get all postprocessing combinations."""
        if not self.postprocessing:
            return [[]]
        return list(itertools.product(*self.postprocessing))


@dataclass
class SearchConstraint:
    """
    A constraint on the search space.
    
    Attributes:
        metric: Name of the metric to constrain (e.g., "lut_utilization")
        operator: Comparison operator ("<=", ">=", "==", "<", ">")
        value: Target value for the constraint
    """
    metric: str
    operator: str
    value: Union[float, int]
    
    def __str__(self) -> str:
        return f"{self.metric} {self.operator} {self.value}"
    
    def evaluate(self, metric_value: Union[float, int]) -> bool:
        """
        Evaluate whether a metric value satisfies this constraint.
        
        Args:
            metric_value: The actual metric value to check
            
        Returns:
            True if the constraint is satisfied, False otherwise
        """
        if self.operator == "<=":
            return metric_value <= self.value
        elif self.operator == ">=":
            return metric_value >= self.value
        elif self.operator == "==":
            return metric_value == self.value
        elif self.operator == "<":
            return metric_value < self.value
        elif self.operator == ">":
            return metric_value > self.value
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass
class SearchConfig:
    """
    Configuration for design space exploration.
    
    Attributes:
        strategy: Search strategy to use
        constraints: List of constraints to apply
        max_evaluations: Maximum number of configurations to evaluate
        timeout_minutes: Maximum time to spend on exploration
        parallel_builds: Number of parallel builds to run
    """
    strategy: SearchStrategy
    constraints: List[SearchConstraint] = field(default_factory=list)
    max_evaluations: Optional[int] = None
    timeout_minutes: Optional[int] = None
    parallel_builds: int = 1
    
    def __str__(self) -> str:
        constraint_str = f", {len(self.constraints)} constraints" if self.constraints else ""
        return f"{self.strategy.value} search{constraint_str}"


@dataclass
class GlobalConfig:
    """
    Global configuration parameters that apply to all exploration runs.
    
    Attributes:
        output_stage: Target output stage (dataflow_graph, rtl, stitched_ip)
        working_directory: Directory for build artifacts
        cache_results: Whether to cache build results
        save_artifacts: Whether to save build artifacts
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    output_stage: OutputStage
    working_directory: str
    cache_results: bool = True
    save_artifacts: bool = True
    log_level: str = "INFO"
    
    def __str__(self) -> str:
        return f"Output: {self.output_stage.value}, Dir: {self.working_directory}"


@dataclass
class DesignSpace:
    """
    Complete definition of the exploration space.
    
    This is the main output of the Design Space Constructor phase,
    containing all information needed for exploration.
    
    Attributes:
        model_path: Path to the ONNX model
        hw_compiler_space: Hardware compiler configuration space
        processing_space: Pre/post-processing configuration space
        search_config: Search strategy and constraints
        global_config: Global parameters
    """
    model_path: str
    hw_compiler_space: HWCompilerSpace
    processing_space: ProcessingSpace
    search_config: SearchConfig
    global_config: GlobalConfig
    
    def get_total_combinations(self) -> int:
        """
        Calculate the total number of possible configurations.
        
        Returns:
            Total number of unique configurations in the design space
        """
        total = 1
        
        # Kernel combinations
        kernel_combos = self.hw_compiler_space.get_kernel_combinations()
        total *= len(kernel_combos) if kernel_combos else 1
        
        # Transform combinations
        transform_combos = self.hw_compiler_space.get_transform_combinations()
        total *= len(transform_combos) if transform_combos else 1
        
        # Processing combinations
        preproc_combos = self.processing_space.get_preprocessing_combinations()
        total *= len(preproc_combos) if preproc_combos else 1
        
        postproc_combos = self.processing_space.get_postprocessing_combinations()
        total *= len(postproc_combos) if postproc_combos else 1
        
        return total
    
    def __str__(self) -> str:
        return (
            f"DesignSpace(\n"
            f"  Model: {self.model_path}\n"
            f"  Total combinations: {self.get_total_combinations():,}\n"
            f"  Search: {self.search_config}\n"
            f"  Global: {self.global_config}\n"
            f")"
        )