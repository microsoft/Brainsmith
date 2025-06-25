# Phase 1: Design Space Constructor - Detailed Design

## Overview

The Design Space Constructor is responsible for transforming user inputs (ONNX model + Blueprint YAML) into a structured, validated Design Space object. This phase establishes the foundation for the entire DSE system by defining what configurations will be explored.

## Key Simplifications

This design incorporates several simplifications based on feedback:

1. **Removed ModelInfo/ModelAnalyzer** - The ONNX model path is passed directly through the system
2. **Simplified Kernels** - Kernels are now tuples of `(kernel_name, list_of_backends)` with no parameters
3. **Simplified Transforms** - Transforms are just names (strings) with optional prefix `~`
4. **Advanced Features** - Support for:
   - Kernel/transform ordering (list order matters)
   - Optional elements (prefix with `~`)
   - Mutually exclusive kernel groups (using nested lists)
   - Both flat and phase-based transform organization

## Core Components

### 1. Data Structures

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

# Simplified kernel and transform definitions
KernelConfig = Tuple[str, List[str]]  # (kernel_name, [backend1, backend2, ...])
TransformName = str  # Just the transform name

# Hardware compiler configuration space
@dataclass
class HWCompilerSpace:
    """FINN-specific configuration space"""
    kernels: List[Union[KernelConfig, str]]      # List of kernel configs or names
    transforms: List[Union[TransformName, Dict]]  # List of transform names or phase dicts
    build_steps: List[str]                        # Fixed sequence of build steps
    config_flags: Dict[str, Any]                  # Fixed configuration flags

# Processing configuration space
@dataclass
class ProcessingStep:
    """Single processing step configuration"""
    name: str
    type: str  # "preprocessing" or "postprocessing"
    parameters: Dict[str, Any]
    enabled: bool = True

@dataclass
class ProcessingSpace:
    """Pre/post processing configuration space"""
    preprocessing: List[List[ProcessingStep]]   # Alternatives per step
    postprocessing: List[List[ProcessingStep]]  # Alternatives per step

# Search configuration
class SearchStrategy(Enum):
    EXHAUSTIVE = "exhaustive"
    # Future: ADAPTIVE, ML_GUIDED, etc.

@dataclass
class SearchConstraint:
    """Single constraint on the search space"""
    metric: str              # e.g., "lut_utilization"
    operator: str            # "<=", ">=", "=="
    value: Union[float, int]

@dataclass
class SearchConfig:
    """Configuration for design space exploration"""
    strategy: SearchStrategy
    constraints: List[SearchConstraint]
    max_evaluations: Optional[int] = None
    timeout_minutes: Optional[int] = None
    parallel_builds: int = 1

# Global configuration
class OutputStage(Enum):
    DATAFLOW_GRAPH = "dataflow_graph"
    RTL = "rtl"
    STITCHED_IP = "stitched_ip"

@dataclass
class GlobalConfig:
    """Fixed parameters for all exploration runs"""
    output_stage: OutputStage
    working_directory: str
    cache_results: bool = True
    save_artifacts: bool = True
    log_level: str = "INFO"
    
# Main Design Space
@dataclass
class DesignSpace:
    """Complete definition of the exploration space"""
    model_path: str  # Direct path to ONNX model
    hw_compiler_space: HWCompilerSpace
    processing_space: ProcessingSpace
    search_config: SearchConfig
    global_config: GlobalConfig
    
    def get_total_combinations(self) -> int:
        """Calculate total number of possible configurations"""
        total = 1
        
        # Kernel combinations (considering mutually exclusive groups)
        kernel_groups = self._group_kernels()
        for group in kernel_groups:
            if isinstance(group, list):  # Mutually exclusive group
                group_total = sum(len(backends) if isinstance(k, tuple) else 1 
                                for k in group)
                total *= group_total
            else:  # Single kernel
                if isinstance(group, tuple):
                    total *= len(group[1])
        
        # Transform combinations (only optional transforms create alternatives)
        for transform in self.hw_compiler_space.transforms:
            if isinstance(transform, str) and transform.startswith("~"):
                total *= 2  # Can be on or off
        
        # Processing combinations
        for prep_options in self.processing_space.preprocessing:
            total *= len(prep_options)
        for post_options in self.processing_space.postprocessing:
            total *= len(post_options)
        
        return total
    
    def _group_kernels(self) -> List[Union[KernelConfig, str, List]]:
        """Group kernels, handling mutually exclusive options"""
        groups = []
        i = 0
        kernels = self.hw_compiler_space.kernels
        
        while i < len(kernels):
            if isinstance(kernels[i], list):  # Mutually exclusive group
                groups.append(kernels[i])
                i += 1
            else:
                groups.append(kernels[i])
                i += 1
        
        return groups
```

### 2. Blueprint YAML Schema

```yaml
# blueprint_schema.yaml
version: "3.0"
name: "BERT Base Exploration"
description: "Design space exploration for BERT base model"

# Hardware compiler configuration space
hw_compiler:
  # Kernel configurations - simplified format
  kernels:
    # Simple kernel name (auto-imports all registered backends)
    - "matmul"
    
    # Kernel with explicit backend list
    - ("elementwise_binary", ["rtl", "hls"])
    
    # Mutually exclusive kernel group
    - [
        "attention",  # Uses all registered backends
        ("flash_attention", ["cuda", "triton"]),
        ("flash_attention_v2", ["triton"])
      ]
    
    # Optional kernel with specific backends
    - ("~layernorm", ["rtl", "hls"])
    
    # More examples
    - "gemm"
    - ("softmax", ["hls"])
  
  # Transform configurations - simplified format
  transforms:
    # Flat list of transforms (ordered)
    - "quantization"
    - "~folding"           # Optional transform
    - "memory_optimization"
    - "~buffer_insertion"  # Optional
    
    # OR phase-based organization
    transforms_phased:
      pre_hw:
        - "quantization"
        - "~graph_optimization"
      post_hw:
        - "folding"
        - "~memory_optimization"
        - "buffer_insertion"
  
  # Fixed build configuration
  build_steps:
    - "ConvertToHW"
    - "InsertDWC"
    - "InsertFIFO"
    - "GiveUniqueNodeNames"
    - "PrepareIP"
    - "HLSSynthIP"
    - "CreateStitchedIP"
  
  config_flags:
    target_device: "xczu7ev-ffvc1156-2-e"
    target_clock_ns: 3.33  # 300 MHz
    synth_directive: "Performance"

# Processing configuration space
processing:
  preprocessing:
    - name: "graph_optimization"
      type: "model_transform"
      options:
        - enabled: true
          passes: ["const_fold", "dead_code_elimination"]
        - enabled: false
    
    - name: "input_normalization"
      type: "data_transform"
      options:
        - method: "standard"
          mean: 0.5
          std: 0.5
        - method: "none"
  
  postprocessing:
    - name: "performance_analysis"
      type: "analysis"
      options:
        - enabled: true
          detailed_report: true
    
    - name: "accuracy_validation"
      type: "validation"
      options:
        - enabled: true
          dataset_size: 1000
        - enabled: false

# Search configuration
search:
  strategy: "exhaustive"
  
  constraints:
    - metric: "lut_utilization"
      operator: "<="
      value: 0.85
    
    - metric: "throughput"
      operator: ">="
      value: 1000  # inferences/sec
    
    - metric: "latency"
      operator: "<="
      value: 10    # milliseconds
  
  # Optional exploration limits
  max_evaluations: 100
  timeout_minutes: 720  # 12 hours
  parallel_builds: 4

# Global configuration
global:
  output_stage: "rtl"
  working_directory: "./exploration_builds"
  cache_results: true
  save_artifacts: true
  log_level: "INFO"
```

### 3. Forge API Implementation

```python
# forge.py
from pathlib import Path
from typing import Tuple
import os
import yaml
from .parser import BlueprintParser
from .validator import DesignSpaceValidator

class ForgeAPI:
    """Main API for constructing design spaces"""
    
    def __init__(self):
        self.parser = BlueprintParser()
        self.validator = DesignSpaceValidator()
    
    def forge(self, model_path: str, blueprint_path: str) -> DesignSpace:
        """
        Construct a validated DesignSpace from model and blueprint.
        
        Args:
            model_path: Path to ONNX model file
            blueprint_path: Path to Blueprint YAML file
            
        Returns:
            Validated DesignSpace object
            
        Raises:
            ModelLoadError: If ONNX model cannot be loaded
            BlueprintParseError: If blueprint is invalid
            ValidationError: If design space is invalid
        """
        # Validate model path exists
        if not os.path.exists(model_path):
            raise ModelLoadError(f"Model file not found: {model_path}")
        
        # Load and parse blueprint
        design_space = self._parse_blueprint(blueprint_path, model_path)
        
        # Validate design space
        self._validate_design_space(design_space)
        
        # Log summary
        self._log_summary(design_space)
        
        return design_space
    
    def _parse_blueprint(self, blueprint_path: str, model_path: str) -> DesignSpace:
        """Parse blueprint YAML into DesignSpace"""
        try:
            with open(blueprint_path, 'r') as f:
                blueprint_data = yaml.safe_load(f)
            
            return self.parser.parse(blueprint_data, model_path)
        except Exception as e:
            raise BlueprintParseError(f"Failed to parse blueprint: {e}")
    
    def _validate_design_space(self, design_space: DesignSpace):
        """Validate the constructed design space"""
        validation_result = self.validator.validate(design_space)
        
        if not validation_result.is_valid:
            raise ValidationError(
                f"Design space validation failed:\n" +
                "\n".join(f"- {error}" for error in validation_result.errors)
            )
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                print(f"Warning: {warning}")
    
    def _log_summary(self, design_space: DesignSpace):
        """Log design space summary"""
        total_combinations = design_space.get_total_combinations()
        print(f"\nDesign Space Summary:")
        print(f"- Model: {Path(design_space.model_path).name}")
        print(f"- Total combinations: {total_combinations:,}")
        print(f"- Search strategy: {design_space.search_config.strategy.value}")
        print(f"- Constraints: {len(design_space.search_config.constraints)}")
        print(f"- Output stage: {design_space.global_config.output_stage.value}")
```

### 4. Blueprint Parser

```python
# parser.py
from typing import Dict, List, Any, Union, Tuple
from .data_structures import *
from .kernel_registry import KernelRegistry  # Assumed to exist

class BlueprintParser:
    """Parse Blueprint YAML into structured DesignSpace"""
    
    SUPPORTED_VERSION = "3.0"
    
    def __init__(self):
        self.kernel_registry = KernelRegistry()
    
    def parse(self, blueprint_data: Dict[str, Any], model_path: str) -> DesignSpace:
        """Parse blueprint data into DesignSpace object"""
        # Validate version
        self._validate_version(blueprint_data)
        
        # Parse each section
        hw_compiler_space = self._parse_hw_compiler(blueprint_data.get("hw_compiler", {}))
        processing_space = self._parse_processing(blueprint_data.get("processing", {}))
        search_config = self._parse_search(blueprint_data.get("search", {}))
        global_config = self._parse_global(blueprint_data.get("global", {}))
        
        return DesignSpace(
            model_path=model_path,
            hw_compiler_space=hw_compiler_space,
            processing_space=processing_space,
            search_config=search_config,
            global_config=global_config
        )
    
    def _validate_version(self, blueprint_data: Dict[str, Any]):
        """Ensure blueprint version is supported"""
        version = blueprint_data.get("version")
        if version != self.SUPPORTED_VERSION:
            raise ValueError(f"Unsupported blueprint version: {version}. Expected: {self.SUPPORTED_VERSION}")
    
    def _parse_hw_compiler(self, hw_data: Dict[str, Any]) -> HWCompilerSpace:
        """Parse hardware compiler configuration space"""
        # Parse kernels with new simplified format
        kernels = self._parse_kernels(hw_data.get("kernels", []))
        
        # Parse transforms - check for both flat and phased formats
        if "transforms" in hw_data:
            transforms = self._parse_transforms_flat(hw_data["transforms"])
        elif "transforms_phased" in hw_data:
            transforms = self._parse_transforms_phased(hw_data["transforms_phased"])
        else:
            transforms = []
        
        return HWCompilerSpace(
            kernels=kernels,
            transforms=transforms,
            build_steps=hw_data.get("build_steps", []),
            config_flags=hw_data.get("config_flags", {})
        )
    
    def _parse_kernels(self, kernels_data: List) -> List[Union[KernelConfig, str, List]]:
        """Parse kernel configurations in the new format"""
        parsed_kernels = []
        
        for kernel_spec in kernels_data:
            if isinstance(kernel_spec, str):
                # Simple kernel name - check for optional prefix
                if kernel_spec.startswith("~"):
                    kernel_name = kernel_spec[1:]
                    # Get all registered backends for this kernel
                    backends = self.kernel_registry.get_backends(kernel_name)
                    parsed_kernels.append(("~" + kernel_name, backends))
                else:
                    # Non-optional kernel - use all registered backends
                    backends = self.kernel_registry.get_backends(kernel_spec)
                    parsed_kernels.append((kernel_spec, backends))
                    
            elif isinstance(kernel_spec, tuple) and len(kernel_spec) == 2:
                # Kernel with explicit backend list
                kernel_name, backends = kernel_spec
                parsed_kernels.append((kernel_name, backends))
                
            elif isinstance(kernel_spec, list):
                # Mutually exclusive kernel group
                parsed_group = []
                for item in kernel_spec:
                    if isinstance(item, str):
                        backends = self.kernel_registry.get_backends(item)
                        parsed_group.append((item, backends))
                    elif isinstance(item, tuple):
                        parsed_group.append(item)
                parsed_kernels.append(parsed_group)
                
            else:
                raise ValueError(f"Invalid kernel specification: {kernel_spec}")
        
        return parsed_kernels
    
    def _parse_transforms_flat(self, transforms_data: List[str]) -> List[str]:
        """Parse flat list of transform names"""
        return transforms_data  # Simple pass-through for flat format
    
    def _parse_transforms_phased(self, phases_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Parse phase-organized transforms"""
        return phases_data  # Return as-is for phase-based organization
    
    def _parse_processing(self, proc_data: Dict[str, Any]) -> ProcessingSpace:
        """Parse processing configuration space"""
        preprocessing = self._parse_processing_steps(
            proc_data.get("preprocessing", []), "preprocessing"
        )
        postprocessing = self._parse_processing_steps(
            proc_data.get("postprocessing", []), "postprocessing"
        )
        
        return ProcessingSpace(
            preprocessing=preprocessing,
            postprocessing=postprocessing
        )
    
    def _parse_processing_steps(self, steps_data: List[Dict], step_type: str) -> List[List[ProcessingStep]]:
        """Parse a list of processing steps"""
        steps = []
        
        for step_config in steps_data:
            step_options = []
            step_name = step_config["name"]
            
            for option in step_config.get("options", []):
                step_options.append(ProcessingStep(
                    name=step_name,
                    type=step_type,
                    parameters=option,
                    enabled=option.get("enabled", True)
                ))
            
            steps.append(step_options)
        
        return steps
    
    def _parse_search(self, search_data: Dict[str, Any]) -> SearchConfig:
        """Parse search configuration"""
        # Parse strategy
        strategy_str = search_data.get("strategy", "exhaustive")
        strategy = SearchStrategy(strategy_str)
        
        # Parse constraints
        constraints = []
        for constraint_data in search_data.get("constraints", []):
            constraints.append(SearchConstraint(
                metric=constraint_data["metric"],
                operator=constraint_data["operator"],
                value=constraint_data["value"]
            ))
        
        return SearchConfig(
            strategy=strategy,
            constraints=constraints,
            max_evaluations=search_data.get("max_evaluations"),
            timeout_minutes=search_data.get("timeout_minutes"),
            parallel_builds=search_data.get("parallel_builds", 1)
        )
    
    def _parse_global(self, global_data: Dict[str, Any]) -> GlobalConfig:
        """Parse global configuration"""
        output_stage_str = global_data.get("output_stage", "rtl")
        output_stage = OutputStage(output_stage_str)
        
        return GlobalConfig(
            output_stage=output_stage,
            working_directory=global_data.get("working_directory", "./builds"),
            cache_results=global_data.get("cache_results", True),
            save_artifacts=global_data.get("save_artifacts", True),
            log_level=global_data.get("log_level", "INFO")
        )


# kernel_registry.py (stub for example)
class KernelRegistry:
    """Registry for available kernel backends"""
    
    def __init__(self):
        # In real implementation, this would discover available backends
        self.registry = {
            "matmul": ["rtl", "hls", "dsp"],
            "attention": ["rtl", "hls"],
            "flash_attention": ["cuda", "triton"],
            "flash_attention_v2": ["triton"],
            "layernorm": ["rtl", "hls"],
            "gemm": ["rtl", "hls", "dsp"],
            "softmax": ["hls", "rtl"],
            "elementwise_binary": ["rtl", "hls"],
        }
    
    def get_backends(self, kernel_name: str) -> List[str]:
        """Get all registered backends for a kernel"""
        # Remove optional prefix if present
        if kernel_name.startswith("~"):
            kernel_name = kernel_name[1:]
        
        return self.registry.get(kernel_name, [])
```

### 5. Design Space Validator

```python
# validator.py
from dataclasses import dataclass
from typing import List
import os
from .data_structures import DesignSpace

@dataclass
class ValidationResult:
    """Result of design space validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class DesignSpaceValidator:
    """Validate design space for correctness and feasibility"""
    
    # Warning thresholds
    MAX_COMBINATIONS_WARNING = 1000
    MAX_COMBINATIONS_ERROR = 10000
    
    def validate(self, design_space: DesignSpace) -> ValidationResult:
        """Perform comprehensive validation of design space"""
        errors = []
        warnings = []
        
        # Validate model path
        self._validate_model_path(design_space.model_path, errors, warnings)
        
        # Validate HW compiler space
        self._validate_hw_compiler(design_space.hw_compiler_space, errors, warnings)
        
        # Validate processing space
        self._validate_processing(design_space.processing_space, errors, warnings)
        
        # Validate search configuration
        self._validate_search(design_space.search_config, errors, warnings)
        
        # Validate global configuration
        self._validate_global(design_space.global_config, errors, warnings)
        
        # Check total combinations
        self._check_combinations(design_space, errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_model_path(self, model_path: str, errors: List[str], warnings: List[str]):
        """Validate model path"""
        if not model_path:
            errors.append("Model path is required")
        elif not os.path.exists(model_path):
            errors.append(f"Model file not found: {model_path}")
        elif not model_path.endswith('.onnx'):
            warnings.append("Model file should have .onnx extension")
    
    def _validate_hw_compiler(self, hw_space: HWCompilerSpace, errors: List[str], warnings: List[str]):
        """Validate hardware compiler configuration"""
        # Check kernels
        if not hw_space.kernels:
            errors.append("At least one kernel configuration is required")
        
        # Validate kernel specifications
        for i, kernel_spec in enumerate(hw_space.kernels):
            if isinstance(kernel_spec, str):
                # Simple kernel name - ok
                pass
            elif isinstance(kernel_spec, tuple):
                if len(kernel_spec) != 2:
                    errors.append(f"Kernel tuple {i} must have exactly 2 elements (name, backends)")
                else:
                    kernel_name, backends = kernel_spec
                    if not isinstance(backends, list) or not backends:
                        errors.append(f"Kernel '{kernel_name}' must have at least one backend")
            elif isinstance(kernel_spec, list):
                # Mutually exclusive group
                if not kernel_spec:
                    errors.append(f"Mutually exclusive kernel group {i} cannot be empty")
                # Validate each item in the group
                for item in kernel_spec:
                    if isinstance(item, tuple) and len(item) == 2:
                        _, backends = item
                        if not isinstance(backends, list) or not backends:
                            errors.append(f"Kernel in group {i} must have at least one backend")
            else:
                errors.append(f"Invalid kernel specification at index {i}")
        
        # Validate transforms
        if isinstance(hw_space.transforms, list):
            # Flat format - check for valid transform names
            for transform in hw_space.transforms:
                if not isinstance(transform, str):
                    errors.append(f"Transform must be a string, got {type(transform)}")
        elif isinstance(hw_space.transforms, dict):
            # Phased format - validate phases
            for phase, transforms in hw_space.transforms.items():
                if not isinstance(transforms, list):
                    errors.append(f"Phase '{phase}' must contain a list of transforms")
                for transform in transforms:
                    if not isinstance(transform, str):
                        errors.append(f"Transform in phase '{phase}' must be a string")
        
        # Check build steps
        if not hw_space.build_steps:
            errors.append("Build steps are required")
        
        # Validate build step sequence
        required_steps = ["ConvertToHW", "PrepareIP"]
        for step in required_steps:
            if step not in hw_space.build_steps:
                warnings.append(f"Build step '{step}' is typically required")
    
    def _validate_processing(self, proc_space: ProcessingSpace, errors: List[str], warnings: List[str]):
        """Validate processing configuration"""
        # Check for empty processing steps
        all_steps = proc_space.preprocessing + proc_space.postprocessing
        if not any(all_steps):
            warnings.append("No processing steps defined")
    
    def _validate_search(self, search_config: SearchConfig, errors: List[str], warnings: List[str]):
        """Validate search configuration"""
        # Validate constraints
        for constraint in search_config.constraints:
            if constraint.operator not in ["<=", ">=", "==", "<", ">"]:
                errors.append(f"Invalid constraint operator: {constraint.operator}")
        
        # Check parallel builds
        if search_config.parallel_builds < 1:
            errors.append("Parallel builds must be at least 1")
        elif search_config.parallel_builds > 32:
            warnings.append(f"High parallel builds ({search_config.parallel_builds}) may cause resource issues")
    
    def _validate_global(self, global_config: GlobalConfig, errors: List[str], warnings: List[str]):
        """Validate global configuration"""
        if not global_config.working_directory:
            errors.append("Working directory is required")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if global_config.log_level not in valid_log_levels:
            errors.append(f"Invalid log level: {global_config.log_level}")
    
    def _check_combinations(self, design_space: DesignSpace, errors: List[str], warnings: List[str]):
        """Check total number of combinations"""
        total = design_space.get_total_combinations()
        
        if total > self.MAX_COMBINATIONS_ERROR:
            errors.append(
                f"Design space has {total:,} combinations, exceeding maximum of {self.MAX_COMBINATIONS_ERROR:,}"
            )
        elif total > self.MAX_COMBINATIONS_WARNING:
            warnings.append(
                f"Design space has {total:,} combinations, which may take significant time to explore"
            )
        elif total == 0:
            errors.append("Design space has no valid combinations")
```


## Error Handling

```python
# exceptions.py
class BrainsmithError(Exception):
    """Base exception for Brainsmith errors"""
    pass

class ModelLoadError(BrainsmithError):
    """Error loading ONNX model"""
    pass

class BlueprintParseError(BrainsmithError):
    """Error parsing Blueprint YAML"""
    pass

class ValidationError(BrainsmithError):
    """Design space validation error"""
    pass
```

## Usage Example

```python
from brainsmith.core.forge import ForgeAPI

# Initialize the Forge API
forge = ForgeAPI()

# Construct design space from model and blueprint
try:
    design_space = forge.forge(
        model_path="models/bert_base.onnx",
        blueprint_path="blueprints/bert_exploration.yaml"
    )
    
    print(f"\nSuccessfully created design space with {design_space.get_total_combinations()} configurations")
    
    # Example: Inspecting the parsed design space
    print("\nKernel configurations:")
    for i, kernel in enumerate(design_space.hw_compiler_space.kernels):
        if isinstance(kernel, str):
            print(f"  {i}: {kernel} (all backends)")
        elif isinstance(kernel, tuple):
            name, backends = kernel
            print(f"  {i}: {name} -> {backends}")
        elif isinstance(kernel, list):
            print(f"  {i}: Mutually exclusive group:")
            for item in kernel:
                if isinstance(item, str):
                    print(f"     - {item} (all backends)")
                else:
                    name, backends = item
                    print(f"     - {name} -> {backends}")
    
    print("\nTransform configurations:")
    if isinstance(design_space.hw_compiler_space.transforms, list):
        for transform in design_space.hw_compiler_space.transforms:
            optional = " (optional)" if transform.startswith("~") else ""
            print(f"  - {transform}{optional}")
    else:
        for phase, transforms in design_space.hw_compiler_space.transforms.items():
            print(f"  {phase}:")
            for transform in transforms:
                optional = " (optional)" if transform.startswith("~") else ""
                print(f"    - {transform}{optional}")
    
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except BlueprintParseError as e:
    print(f"Failed to parse blueprint: {e}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Complete Blueprint Example

```yaml
# Example showing all supported formats
version: "3.0"
name: "Advanced BERT Exploration"

hw_compiler:
  kernels:
    # Simple kernel (uses all registered backends)
    - "matmul"
    
    # Kernel with specific backends
    - ("gemm", ["rtl", "hls"])
    
    # Optional kernel
    - ("~softmax", ["hls"])
    
    # Mutually exclusive group (choose one)
    - [
        "attention",                              # All backends
        ("flash_attention", ["cuda", "triton"]),  # Specific backends
        ("flash_attention_v2", ["triton"])        # Single backend
      ]
    
    # Another simple kernel
    - "layernorm"
    
    # Optional with specific backends
    - ("~elementwise_add", ["rtl", "hls"])
  
  # Flat transform list (simple, ordered)
  transforms:
    - "quantization"
    - "~folding"              # Optional
    - "memory_optimization"
    - "~graph_cleanup"        # Optional
  
  # OR use phased transforms for more control
  # transforms_phased:
  #   pre_hardware:
  #     - "quantization"
  #     - "~graph_optimization"
  #   post_hardware:
  #     - "folding"
  #     - "~memory_optimization"
  
  build_steps:
    - "ConvertToHW"
    - "InsertDWC"
    - "InsertFIFO"
    - "PrepareIP"
    - "HLSSynthIP"
  
  config_flags:
    target_device: "xczu7ev-ffvc1156-2-e"
    target_clock_ns: 3.33

# Rest of configuration remains the same...
```

## Testing Strategy

### Unit Tests
- Test kernel parsing (simple names, tuples, mutually exclusive groups)
- Test transform parsing (flat vs phased, optional transforms)
- Test validation rules
- Test error handling

### Integration Tests
- Test complete forge pipeline with sample models/blueprints
- Test edge cases (empty spaces, invalid combinations)
- Test large design spaces

### Example Tests
```python
def test_kernel_parsing():
    """Test parsing of different kernel formats"""
    parser = BlueprintParser()
    
    # Test simple kernel name
    kernels = parser._parse_kernels(["matmul"])
    assert len(kernels) == 1
    assert kernels[0][0] == "matmul"
    assert len(kernels[0][1]) > 0  # Has backends from registry
    
    # Test kernel with explicit backends
    kernels = parser._parse_kernels([("gemm", ["rtl", "hls"])])
    assert kernels[0] == ("gemm", ["rtl", "hls"])
    
    # Test optional kernel
    kernels = parser._parse_kernels(["~softmax"])
    assert kernels[0][0] == "~softmax"
    
    # Test mutually exclusive group
    kernels = parser._parse_kernels([[
        "attention",
        ("flash_attention", ["cuda"])
    ]])
    assert isinstance(kernels[0], list)
    assert len(kernels[0]) == 2

def test_transform_parsing():
    """Test parsing of transform configurations"""
    parser = BlueprintParser()
    
    # Test flat format
    hw_data = {"transforms": ["quantization", "~folding", "memory_opt"]}
    hw_space = parser._parse_hw_compiler(hw_data)
    assert hw_space.transforms == ["quantization", "~folding", "memory_opt"]
    
    # Test phased format
    hw_data = {
        "transforms_phased": {
            "pre_hw": ["quantization", "~graph_opt"],
            "post_hw": ["folding"]
        }
    }
    hw_space = parser._parse_hw_compiler(hw_data)
    assert isinstance(hw_space.transforms, dict)
    assert "pre_hw" in hw_space.transforms
    assert "~graph_opt" in hw_space.transforms["pre_hw"]

def test_design_space_combinations():
    """Test calculation of total combinations"""
    design_space = DesignSpace(
        model_path="test.onnx",
        hw_compiler_space=HWCompilerSpace(
            kernels=[
                ("matmul", ["rtl", "hls"]),  # 2 options
                [("attention", ["rtl"]), ("flash_attention", ["cuda", "triton"])]  # 3 options
            ],
            transforms=["quantization", "~folding"],  # folding creates 2 options
            build_steps=[],
            config_flags={}
        ),
        processing_space=ProcessingSpace([], []),
        search_config=SearchConfig(SearchStrategy.EXHAUSTIVE, []),
        global_config=GlobalConfig(OutputStage.RTL, "./builds")
    )
    
    # 2 (matmul) * 3 (attention group) * 2 (optional folding) = 12
    assert design_space.get_total_combinations() == 12
```

## Next Steps

1. **Implementation Priority**:
   - Core data structures
   - Blueprint parser
   - Basic validation
   - Model analyzer
   - Forge API integration

2. **Future Enhancements**:
   - Blueprint inheritance support
   - Advanced validation rules
   - Design space optimization hints
   - Visual design space explorer

3. **Integration Points**:
   - Phase 2 will consume DesignSpace objects
   - Validation can be extended with backend-specific rules
   - Parser can be extended for new configuration types