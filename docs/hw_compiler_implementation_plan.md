# Brainsmith Hardware Compiler Implementation Plan

## Overview

This document provides a detailed implementation plan for refactoring the Brainsmith hardware compiler into a modular, extensible system that supports both Python library usage and CLI interfaces.

## File Structure

```
brainsmith/
├── core/
│   ├── __init__.py              # Export main classes
│   ├── config.py                # Configuration classes
│   ├── compiler.py              # Main HardwareCompiler class
│   ├── result.py                # Result classes
│   ├── preprocessor.py          # Model preprocessing
│   ├── builder.py               # Dataflow building
│   ├── postprocessor.py         # Output processing
│   └── hw_compiler.py           # Legacy forge() function
├── cli/
│   ├── __init__.py
│   ├── main.py                  # CLI entry point
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── compile.py           # Compile command
│   │   └── blueprints.py        # Blueprint management
│   └── config/
│       ├── __init__.py
│       └── loader.py            # Configuration file loading
└── __init__.py                  # Main package exports
```

## Implementation Specifications

### 1. Configuration System (`brainsmith/core/config.py`)

```python
"""
Configuration classes for the Brainsmith hardware compiler.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class CompilerConfig:
    """Configuration for hardware compilation jobs."""
    
    # Required parameters
    blueprint: str
    output_dir: str
    
    # Build directory management
    build_dir: Optional[str] = None  # Will use BSMITH_BUILD_DIR if None
    
    # Model preprocessing options
    save_intermediate: bool = True
    
    # FINN build configuration
    target_fps: int = 3000
    clk_period_ns: float = 3.33
    folding_config_file: Optional[str] = None
    stop_step: Optional[str] = None
    
    # Verification options
    run_fifo_sizing: bool = False
    fifosim_n_inferences: int = 2
    verification_atol: float = 1e-1
    
    # Hardware target
    board: str = "V80"
    generate_dcp: bool = True
    
    # Advanced options
    standalone_thresholds: bool = True
    split_large_fifos: bool = True
    
    # Model-specific metadata (for handover)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        if self.build_dir is None:
            self.build_dir = os.environ.get("BSMITH_BUILD_DIR")
            if self.build_dir is None:
                raise ValueError("build_dir must be specified or BSMITH_BUILD_DIR environment variable must be set")
        
        # Ensure paths are absolute
        self.output_dir = os.path.abspath(self.output_dir)
        self.build_dir = os.path.abspath(self.build_dir)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CompilerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CompilerConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_args(cls, args, blueprint: str) -> 'CompilerConfig':
        """Create config from argparse arguments (for backward compatibility)."""
        return cls(
            blueprint=blueprint,
            output_dir=getattr(args, 'output', './build'),
            save_intermediate=getattr(args, 'save_intermediate', True),
            target_fps=getattr(args, 'fps', 3000),
            clk_period_ns=getattr(args, 'clk', 3.33),
            folding_config_file=getattr(args, 'param', None),
            stop_step=getattr(args, 'stop_step', None),
            run_fifo_sizing=getattr(args, 'run_fifo_sizing', False),
            fifosim_n_inferences=getattr(args, 'fifosim_n_inferences', 2),
            verification_atol=getattr(args, 'verification_atol', 1e-1),
            board=getattr(args, 'board', 'V80'),
            generate_dcp=getattr(args, 'dcp', True),
            standalone_thresholds=getattr(args, 'standalone_thresholds', True),
            split_large_fifos=getattr(args, 'split_large_fifos', True),
            model_metadata={
                'num_hidden_layers': getattr(args, 'num_hidden_layers', None)
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
```

### 2. Result Classes (`brainsmith/core/result.py`)

```python
"""
Result classes for compilation operations.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class CompilerResult:
    """Result of a hardware compilation operation."""
    
    success: bool
    output_dir: str
    config: 'CompilerConfig'
    
    # Build artifacts
    final_model_path: Optional[str] = None
    build_artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Metadata and logs
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timing information
    build_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def add_log(self, message: str) -> None:
        """Add a log message."""
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.add_log(f"ERROR: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        self.add_log(f"WARNING: {warning}")
    
    def add_artifact(self, name: str, path: str) -> None:
        """Add a build artifact."""
        self.build_artifacts[name] = path
        self.add_log(f"Generated artifact: {name} -> {path}")
    
    def start_timing(self) -> None:
        """Start timing the build."""
        self.start_time = time.time()
    
    def end_timing(self) -> None:
        """End timing the build."""
        if self.start_time is not None:
            self.end_time = time.time()
            self.build_time = self.end_time - self.start_time
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        summary = [
            f"Compilation {status}",
            f"Blueprint: {self.config.blueprint}",
            f"Output: {self.output_dir}",
            f"Build time: {self.build_time:.2f}s"
        ]
        
        if self.errors:
            summary.append(f"Errors: {len(self.errors)}")
        if self.warnings:
            summary.append(f"Warnings: {len(self.warnings)}")
        
        return "\n".join(summary)


@dataclass
class BlueprintValidationResult:
    """Result of blueprint validation."""
    
    blueprint_name: str
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    step_count: int = 0
    missing_steps: List[str] = field(default_factory=list)
```

### 3. Model Preprocessor (`brainsmith/core/preprocessor.py`)

```python
"""
Model preprocessing components.
"""

import onnx
import numpy as np
from typing import Tuple, Optional
from onnxsim import simplify
from qonnx.util.cleanup import cleanup
import tempfile
import os

from .config import CompilerConfig
from .result import CompilerResult


class ModelPreprocessor:
    """Handles model preprocessing operations."""
    
    def __init__(self, config: CompilerConfig):
        self.config = config
    
    def preprocess(self, model: onnx.ModelProto, result: CompilerResult) -> onnx.ModelProto:
        """
        Preprocess the model for compilation.
        
        Args:
            model: Input ONNX model
            result: Result object to update with logs
            
        Returns:
            Preprocessed ONNX model
        """
        result.add_log("Starting model preprocessing")
        
        # Simplify model
        result.add_log("Simplifying model")
        model, check = simplify(model)
        if not check:
            result.add_error("Unable to simplify the model")
            return model
        
        # Save intermediate model if requested
        if self.config.save_intermediate:
            model_dir = os.path.join(self.config.build_dir, self.config.output_dir.split('/')[-1], "intermediate_models")
            os.makedirs(model_dir, exist_ok=True)
            simp_path = os.path.join(model_dir, "simp.onnx")
            onnx.save(model, simp_path)
            result.add_artifact("simplified_model", simp_path)
            result.add_log(f"Saved simplified model to {simp_path}")
        
        # Cleanup model
        result.add_log("Cleaning up model")
        build_dir = os.path.join(self.config.build_dir, self.config.output_dir.split('/')[-1])
        os.makedirs(build_dir, exist_ok=True)
        
        # Use temporary file for cleanup if intermediate saving is disabled
        if self.config.save_intermediate:
            input_path = simp_path
        else:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                onnx.save(model, tmp.name)
                input_path = tmp.name
        
        output_path = os.path.join(build_dir, "df_input.onnx")
        cleanup(in_file=input_path, out_file=output_path)
        result.add_artifact("cleaned_model", output_path)
        
        # Clean up temporary file if used
        if not self.config.save_intermediate and os.path.exists(input_path):
            os.unlink(input_path)
        
        # Load and return cleaned model
        cleaned_model = onnx.load(output_path)
        result.add_log("Model preprocessing completed")
        return cleaned_model
    
    def generate_test_data(self, model: onnx.ModelProto, result: CompilerResult) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate test input and expected output data.
        
        Args:
            model: Preprocessed ONNX model
            result: Result object to update with logs
            
        Returns:
            Tuple of (input_data, expected_output) or (None, None) if generation fails
        """
        result.add_log("Generating test data")
        
        # TODO: Implement general test data generation
        # For now, this is a placeholder that expects the caller to provide test data
        result.add_warning("Test data generation not yet implemented - using placeholder")
        
        return None, None
```

### 4. Dataflow Builder (`brainsmith/core/builder.py`)

```python
"""
FINN dataflow building components.
"""

import onnx
import numpy as np
from typing import List, Callable, Optional
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from .config import CompilerConfig
from .result import CompilerResult
from ..blueprints import get_blueprint_steps


class DataflowBuilder:
    """Handles FINN dataflow build process."""
    
    def __init__(self, config: CompilerConfig):
        self.config = config
    
    def get_build_steps(self, result: CompilerResult) -> List[Callable]:
        """Get build steps for the configured blueprint."""
        result.add_log(f"Loading blueprint: {self.config.blueprint}")
        
        try:
            steps = get_blueprint_steps(self.config.blueprint)
            result.add_log(f"Loaded {len(steps)} build steps")
            return steps
        except Exception as e:
            result.add_error(f"Failed to load blueprint '{self.config.blueprint}': {e}")
            return []
    
    def build(self, model_path: str, input_data: Optional[np.ndarray], 
              expected_output: Optional[np.ndarray], result: CompilerResult) -> bool:
        """
        Execute the FINN dataflow build.
        
        Args:
            model_path: Path to the cleaned input model
            input_data: Test input data (optional)
            expected_output: Expected output data (optional)
            result: Result object to update
            
        Returns:
            True if build succeeded, False otherwise
        """
        result.add_log("Starting dataflow build")
        
        # Get build steps
        steps = self.get_build_steps(result)
        if not steps:
            return False
        
        # Setup build directory
        build_dir = os.path.join(self.config.build_dir, self.config.output_dir.split('/')[-1])
        
        # Create FINN build configuration
        verify_steps = []
        if input_data is not None and expected_output is not None:
            # Save test data
            input_path = os.path.join(build_dir, "input.npy")
            output_path = os.path.join(build_dir, "expected_output.npy")
            np.save(input_path, input_data)
            np.save(output_path, expected_output)
            result.add_artifact("test_input", input_path)
            result.add_artifact("test_output", output_path)
            
            # Enable verification
            verify_steps = [
                build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
                build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
            ]
        
        df_cfg = build_cfg.DataflowBuildConfig(
            standalone_thresholds=self.config.standalone_thresholds,
            steps=steps,
            target_fps=self.config.target_fps,
            output_dir=build_dir,
            synth_clk_period_ns=self.config.clk_period_ns,
            folding_config_file=self.config.folding_config_file,
            stop_step=self.config.stop_step,
            auto_fifo_depths=self.config.run_fifo_sizing,
            fifosim_n_inferences=self.config.fifosim_n_inferences,
            verification_atol=self.config.verification_atol,
            split_large_fifos=self.config.split_large_fifos,
            stitched_ip_gen_dcp=self.config.generate_dcp,
            board=self.config.board,
            generate_outputs=[build_cfg.DataflowOutputType.STITCHED_IP],
            verify_input_npy=os.path.join(build_dir, "input.npy") if input_data is not None else None,
            verify_expected_output_npy=os.path.join(build_dir, "expected_output.npy") if expected_output is not None else None,
            verify_save_full_context=self.config.save_intermediate,
            verify_steps=verify_steps,
        )
        
        try:
            result.add_log("Executing FINN build")
            build.build_dataflow_cfg(model_path, df_cfg)
            result.add_log("FINN build completed successfully")
            return True
        except Exception as e:
            result.add_error(f"FINN build failed: {e}")
            return False
```

### 5. Output Processor (`brainsmith/core/postprocessor.py`)

```python
"""
Output processing components.
"""

import os
import json
import shutil
from typing import List, Callable, Dict, Any

from .config import CompilerConfig
from .result import CompilerResult


class OutputProcessor:
    """Handles post-build output processing."""
    
    def __init__(self, config: CompilerConfig):
        self.config = config
    
    def process_outputs(self, steps: List[Callable], result: CompilerResult) -> None:
        """
        Process build outputs and generate final artifacts.
        
        Args:
            steps: List of build steps that were executed
            result: Result object to update
        """
        result.add_log("Processing build outputs")
        
        build_dir = os.path.join(self.config.build_dir, self.config.output_dir.split('/')[-1])
        model_dir = os.path.join(build_dir, "intermediate_models")
        
        # Copy final model
        self._copy_final_model(steps, build_dir, model_dir, result)
        
        # Generate handover metadata
        self._generate_handover_metadata(build_dir, result)
        
        # Collect build artifacts
        self._collect_artifacts(build_dir, result)
        
        result.add_log("Output processing completed")
    
    def _copy_final_model(self, steps: List[Callable], build_dir: str, 
                         model_dir: str, result: CompilerResult) -> None:
        """Copy the final model to the output location."""
        if self.config.stop_step is None and steps:
            final_step = steps[-1].__name__
        else:
            final_step = self.config.stop_step
        
        if final_step:
            src_path = os.path.join(model_dir, f"{final_step}.onnx")
            dst_path = os.path.join(build_dir, "output.onnx")
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                result.final_model_path = dst_path
                result.add_artifact("final_model", dst_path)
                result.add_log(f"Copied final model: {final_step}.onnx -> output.onnx")
            else:
                result.add_warning(f"Final model not found: {src_path}")
    
    def _generate_handover_metadata(self, build_dir: str, result: CompilerResult) -> None:
        """Generate shell handover metadata."""
        handover_file = os.path.join(build_dir, "stitched_ip", "shell_handover.json")
        
        if os.path.exists(handover_file):
            try:
                with open(handover_file, "r") as fp:
                    handover = json.load(fp)
                
                # Add model-specific metadata
                handover.update(self.config.model_metadata)
                
                with open(handover_file, "w") as fp:
                    json.dump(handover, fp, indent=4)
                
                result.add_artifact("handover_metadata", handover_file)
                result.add_log("Updated shell handover metadata")
                
            except Exception as e:
                result.add_warning(f"Failed to update handover metadata: {e}")
    
    def _collect_artifacts(self, build_dir: str, result: CompilerResult) -> None:
        """Collect important build artifacts."""
        # Common artifacts to look for
        artifacts = {
            "stitched_ip": "stitched_ip",
            "reports": "reports",
            "vivado_project": "*.xpr",
            "bitstream": "*.bit",
            "dcp_file": "*.dcp"
        }
        
        for name, pattern in artifacts.items():
            # Simple existence check for directories
            if not pattern.startswith("*"):
                artifact_path = os.path.join(build_dir, pattern)
                if os.path.exists(artifact_path):
                    result.add_artifact(name, artifact_path)
```

### 6. Main Compiler Class (`brainsmith/core/compiler.py`)

```python
"""
Main hardware compiler class.
"""

import onnx
import os
import time
from typing import Optional
import numpy as np

from .config import CompilerConfig
from .result import CompilerResult
from .preprocessor import ModelPreprocessor
from .builder import DataflowBuilder
from .postprocessor import OutputProcessor


class HardwareCompiler:
    """Main hardware compiler orchestrator."""
    
    def __init__(self, config: CompilerConfig):
        self.config = config
        self.preprocessor = ModelPreprocessor(config)
        self.builder = DataflowBuilder(config)
        self.postprocessor = OutputProcessor(config)
    
    def compile(self, model: onnx.ModelProto, 
                input_data: Optional[np.ndarray] = None,
                expected_output: Optional[np.ndarray] = None) -> CompilerResult:
        """
        Compile an ONNX model to hardware.
        
        Args:
            model: Input ONNX model
            input_data: Optional test input data
            expected_output: Optional expected output data
            
        Returns:
            CompilerResult with build information
        """
        # Create result object
        result = CompilerResult(
            success=False,
            output_dir=self.config.output_dir,
            config=self.config
        )
        
        result.start_timing()
        result.add_log(f"Starting compilation with blueprint: {self.config.blueprint}")
        
        try:
            # Preprocess model
            preprocessed_model = self.preprocessor.preprocess(model, result)
            
            # Generate test data if not provided
            if input_data is None or expected_output is None:
                result.add_log("No test data provided, attempting to generate")
                gen_input, gen_output = self.preprocessor.generate_test_data(preprocessed_model, result)
                if input_data is None:
                    input_data = gen_input
                if expected_output is None:
                    expected_output = gen_output
            
            # Build dataflow
            build_dir = os.path.join(self.config.build_dir, self.config.output_dir.split('/')[-1])
            model_path = os.path.join(build_dir, "df_input.onnx")
            
            build_success = self.builder.build(model_path, input_data, expected_output, result)
            
            if build_success:
                # Process outputs
                steps = self.builder.get_build_steps(result)
                self.postprocessor.process_outputs(steps, result)
                result.success = True
                result.add_log("Compilation completed successfully")
            else:
                result.add_error("Build failed")
                
        except Exception as e:
            result.add_error(f"Compilation failed with exception: {e}")
        
        result.end_timing()
        return result
    
    def compile_from_file(self, model_path: str,
                         input_data: Optional[np.ndarray] = None,
                         expected_output: Optional[np.ndarray] = None) -> CompilerResult:
        """
        Compile an ONNX model from file.
        
        Args:
            model_path: Path to ONNX model file
            input_data: Optional test input data
            expected_output: Optional expected output data
            
        Returns:
            CompilerResult with build information
        """
        if not os.path.exists(model_path):
            result = CompilerResult(
                success=False,
                output_dir=self.config.output_dir,
                config=self.config
            )
            result.add_error(f"Model file not found: {model_path}")
            return result
        
        model = onnx.load(model_path)
        return self.compile(model, input_data, expected_output)
```

### 7. Package Interface (`brainsmith/core/__init__.py`)

```python
"""
Brainsmith core compiler interface.
"""

from .config import CompilerConfig
from .result import CompilerResult, BlueprintValidationResult
from .compiler import HardwareCompiler
from .hw_compiler import forge  # Legacy interface

__all__ = [
    'CompilerConfig',
    'CompilerResult', 
    'BlueprintValidationResult',
    'HardwareCompiler',
    'forge'
]
```

### 8. Main Package Interface (`brainsmith/__init__.py`)

```python
"""
Brainsmith - Neural Network to FPGA Compiler
"""

import logging
import os
import onnx
import numpy as np
from typing import Optional

# Set default logging handler
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

# Import main classes
from .core import CompilerConfig, CompilerResult, HardwareCompiler
from .blueprints import list_blueprints, load_blueprint

__version__ = "1.0.0"

__all__ = [
    'CompilerConfig',
    'CompilerResult', 
    'HardwareCompiler',
    'compile_model',
    'compile_model_from_file',
    'list_blueprints',
    'load_blueprint'
]


def compile_model(model: onnx.ModelProto,
                  blueprint: str,
                  output_dir: str,
                  **kwargs) -> CompilerResult:
    """
    Simple interface for compiling ONNX models.
    
    Args:
        model: ONNX model to compile
        blueprint: Blueprint name to use
        output_dir: Output directory for build artifacts
        **kwargs: Additional configuration options
        
    Returns:
        CompilerResult with build information
    """
    config = CompilerConfig(
        blueprint=blueprint,
        output_dir=output_dir,
        **kwargs
    )
    
    compiler = HardwareCompiler(config)
    return compiler.compile(model)


def compile_model_from_file(model_path: str,
                           blueprint: str,
                           output_dir: str,
                           **kwargs) -> CompilerResult:
    """
    Simple interface for compiling ONNX models from file.
    
    Args:
        model_path: Path to ONNX model file
        blueprint: Blueprint name to use
        output_dir: Output directory for build artifacts
        **kwargs: Additional configuration options
        
    Returns:
        CompilerResult with build information
    """
    config = CompilerConfig(
        blueprint=blueprint,
        output_dir=output_dir,
        **kwargs
    )
    
    compiler = HardwareCompiler(config)
    return compiler.compile_from_file(model_path)
```

### 9. Legacy Compatibility (`brainsmith/core/hw_compiler.py`)

Update the existing file to use the new system:

```python
"""
Legacy hardware compiler interface for backward compatibility.
"""

import onnx
from .config import CompilerConfig
from .compiler import HardwareCompiler


def forge(blueprint: str, model: onnx.ModelProto, args) -> None:
    """
    Legacy interface for backward compatibility.
    
    Args:
        blueprint: Blueprint name
        model: ONNX model
        args: Legacy arguments object
        
    Raises:
        RuntimeError: If compilation fails
    """
    # Convert legacy args to new config
    config = CompilerConfig.from_args(args, blueprint)
    
    # Run new compiler
    compiler = HardwareCompiler(config)
    result = compiler.compile(model)
    
    # Handle legacy behavior (no return value, raises on error)
    if not result.success:
        error_msg = "\n".join(result.errors) if result.errors else "Unknown compilation error"
        raise RuntimeError(f"Compilation failed: {error_msg}")
```

## Usage Examples

### Library Usage

```python
# Simple interface
import brainsmith
import onnx

model = onnx.load("my_model.onnx")
result = brainsmith.compile_model(
    model=model,
    blueprint="bert",
    output_dir="./build",
    target_fps=3000,
    board="V80"
)

if result.success:
    print(f"Compilation successful! Output: {result.final_model_path}")
else:
    print(f"Compilation failed: {result.errors}")

# Advanced interface
config = brainsmith.CompilerConfig(
    blueprint="bert",
    output_dir="./build",
    target_fps=3000,
    board="V80",
    save_intermediate=True
)

compiler = brainsmith.HardwareCompiler(config)
result = compiler.compile_from_file("my_model.onnx")
```

### Configuration File Usage

```yaml
# build_config.yaml
blueprint: bert
output_dir: ./build
target_fps: 3000
clk_period_ns: 3.33
board: V80
save_intermediate: true
run_fifo_sizing: false
```

```python
import brainsmith

config = brainsmith.CompilerConfig.from_yaml("build_config.yaml")
compiler = brainsmith.HardwareCompiler(config)
result = compiler.compile_from_file("model.onnx")
```

## Implementation Notes

### Phase 1 (Current): Core Classes
- Implement all core classes in `brainsmith/core/`
- Update `brainsmith/__init__.py` to export new interface
- Keep legacy `forge()` function working
- Add comprehensive tests

### Phase 2: CLI Interface
- Implement CLI in `brainsmith/cli/`
- Add entry point in `setup.py`
- Create configuration file loading
- Add blueprint management commands

### Phase 3: Migration
- Update demos to optionally use new interface
- Add deprecation warnings
- Create migration documentation

This modular design provides:
- Clean separation of concerns
- Easy testing and maintenance
- Flexible configuration options
- Backward compatibility
- Extensibility for future features