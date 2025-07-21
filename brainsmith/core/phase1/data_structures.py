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
            # Tuple/List format: [name, backends] - single kernel with backend list
            name, backends = item
            if name and isinstance(name, str) and name.startswith("~"):
                # Optional kernel - add both enabled and disabled
                options.append((name[1:], backends))  # Kernel with all backends
                options.append(("", []))  # Empty name = skipped
            elif name and isinstance(name, str):
                # Single kernel option with its backends
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
    
    def get_transform_combinations_by_stage(self) -> List[Dict[str, List[str]]]:
        """
        Generate all valid transform combinations preserving stage information.
        
        Returns:
            List of transform combinations as dictionaries mapping stage -> transforms
        """
        if isinstance(self.transforms, dict):
            # Phase-based transforms - preserve stage info
            stage_combinations = {}
            for stage, transforms in self.transforms.items():
                stage_options = []
                for item in transforms:
                    stage_options.append(self._parse_transform_item(item))
                if stage_options:
                    stage_combinations[stage] = list(itertools.product(*stage_options))
                else:
                    stage_combinations[stage] = [[]]  # Empty list for this stage
            
            # Generate all combinations across stages
            all_combos = []
            stage_names = list(stage_combinations.keys())
            if stage_names:
                for combo in itertools.product(*[stage_combinations[s] for s in stage_names]):
                    combo_dict = {}
                    for i, stage in enumerate(stage_names):
                        # Filter out empty transforms
                        stage_transforms = [t for t in combo[i] if t]
                        if stage_transforms:
                            combo_dict[stage] = stage_transforms
                    all_combos.append(combo_dict)
            return all_combos if all_combos else [{}]
        else:
            # Flat list - put all in "default" stage
            combinations = []
            for item in self.transforms:
                combinations.append(self._parse_transform_item(item))
            
            all_combos = []
            for combo in itertools.product(*combinations):
                # Filter out empty transforms
                active_transforms = [t for t in combo if t]
                if active_transforms:
                    all_combos.append({"default": active_transforms})
                else:
                    all_combos.append({})
            return all_combos if all_combos else [{}]
    
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
class GlobalConfig:
    """
    Global configuration parameters that apply to all exploration runs.
    
    Attributes:
        output_stage: Target output stage (dataflow_graph, rtl, stitched_ip) - maintained for backward compatibility
        working_directory: Directory for build artifacts
        cache_results: Whether to cache build results
        save_artifacts: Whether to save build artifacts
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        max_combinations: Maximum allowed design space combinations (overrides global default)
        timeout_minutes: Default timeout for DSE jobs in minutes (overrides global default)
        start_step: Optional starting step for partial builds
        stop_step: Optional stopping step for partial builds
        input_type: Optional semantic input type specification
        output_type: Optional semantic output type specification
    """
    output_stage: OutputStage = OutputStage.RTL
    working_directory: str = "/tmp/brainsmith"
    cache_results: bool = True
    save_artifacts: bool = True
    log_level: str = "INFO"
    max_combinations: Optional[int] = None
    timeout_minutes: Optional[int] = None
    start_step: Optional[str] = None
    stop_step: Optional[str] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    
    def __str__(self) -> str:
        return f"Output: {self.output_stage.value}, Dir: {self.working_directory}"


@dataclass
class BuildMetrics:
    """
    Metrics collected from a build execution.
    
    This represents performance and resource utilization metrics
    from a successful build.
    
    Attributes:
        throughput: Inferences per second
        latency: Inference latency in microseconds
        clock_frequency: Operating frequency in MHz
        lut_utilization: LUT utilization (0.0-1.0)
        dsp_utilization: DSP utilization (0.0-1.0)
        bram_utilization: BRAM utilization (0.0-1.0)
        total_power: Total power consumption in Watts
        accuracy: Model accuracy (0.0-1.0)
        custom: Additional custom metrics
    """
    throughput: float                    # inferences/sec
    latency: float                       # microseconds
    clock_frequency: float               # MHz
    lut_utilization: float              # 0.0-1.0
    dsp_utilization: float              # 0.0-1.0
    bram_utilization: float             # 0.0-1.0
    total_power: float                  # Watts
    accuracy: float                     # 0.0-1.0
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DesignSpace:
    """Complete design space specification.
    
    Represents all possible hardware configurations to explore.
    Always uses exhaustive search up to max_combinations limit.
    """
    model_path: str
    hw_compiler_space: HWCompilerSpace
    global_config: GlobalConfig
    
    # Direct limits (no SearchConfig)
    max_combinations: int = 100000
    timeout_minutes: int = 60
    
    def __post_init__(self):
        """Validate design space size on creation."""
        estimated_size = self._estimate_size()
        if estimated_size > self.max_combinations:
            raise ValueError(
                f"Design space too large: {estimated_size:,} combinations exceeds "
                f"limit of {self.max_combinations:,}. Reduce design space or increase "
                f"max_combinations."
            )
    
    def _estimate_size(self) -> int:
        """Estimate total number of combinations."""
        size = 1
        
        # Kernels: product of backend choices
        kernel_combos = self.hw_compiler_space.get_kernel_combinations()
        size *= len(kernel_combos) if kernel_combos else 1
        
        # Transforms: product of choices per stage
        if isinstance(self.hw_compiler_space.transforms, dict):
            for stage, transforms in self.hw_compiler_space.transforms.items():
                stage_choices = 1
                for t in transforms:
                    if isinstance(t, list):
                        # Mutually exclusive options
                        stage_choices *= len([opt for opt in t if opt != "~"])
                    else:
                        # Required transform
                        stage_choices *= 1
                size *= stage_choices
        else:
            # Flat list of transforms
            transform_combos = self.hw_compiler_space.get_transform_combinations()
            size *= len(transform_combos) if transform_combos else 1
        
        return size
    
    def get_total_combinations(self) -> int:
        """Calculate the total number of possible configurations.
        
        Returns:
            Total number of unique configurations in the design space
        """
        return self._estimate_size()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hw_compiler_space": {
                "kernels": self.hw_compiler_space.kernels,
                "transforms": self.hw_compiler_space.transforms,
                "build_steps": self.hw_compiler_space.build_steps,
                "config_flags": self.hw_compiler_space.config_flags,
            },
            "global_config": {
                "output_stage": self.global_config.output_stage.value,
                "working_directory": self.global_config.working_directory,
                "cache_results": self.global_config.cache_results,
                "save_artifacts": self.global_config.save_artifacts,
                "log_level": self.global_config.log_level,
                "max_combinations": self.global_config.max_combinations,
                "timeout_minutes": self.global_config.timeout_minutes,
                "start_step": self.global_config.start_step,
                "stop_step": self.global_config.stop_step,
                "input_type": self.global_config.input_type,
                "output_type": self.global_config.output_type,
            },
            "max_combinations": self.max_combinations,
            "timeout_minutes": self.timeout_minutes,
        }