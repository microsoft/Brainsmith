# Brainsmith Hardware Compiler Refactor Design

## Current State Analysis

### Problems with Current `hw_compiler.py`:

1. **Monolithic Function**: `forge()` does too many things in one place
2. **Hardcoded Assumptions**: Specific to BERT-like models and arguments structure
3. **Poor Separation of Concerns**: Mixes configuration, preprocessing, build execution, and post-processing
4. **Limited Extensibility**: Hard to add new model types or build configurations
5. **No Library Interface**: Only works with specific argparse structure
6. **Environmental Dependencies**: Relies on `BSMITH_BUILD_DIR` environment variable
7. **Poor Error Handling**: Limited error recovery and reporting
8. **Hardcoded Paths**: Many filesystem operations are not configurable

### Current Usage Pattern:
```python
# In demos/bert/end2end_bert.py
from brainsmith.core.hw_compiler import forge
forge('bert', model, args)  # args is argparse object
```

## Proposed Modular Architecture

### 1. Core Compiler Classes

#### `CompilerConfig` - Configuration Management
```python
@dataclass
class CompilerConfig:
    blueprint: str
    output_dir: str
    build_dir: Optional[str] = None
    
    # Model preprocessing
    save_intermediate: bool = True
    
    # FINN configuration
    target_fps: int = 3000
    clk_period_ns: float = 3.33
    folding_config_file: Optional[str] = None
    stop_step: Optional[str] = None
    
    # Verification
    run_fifo_sizing: bool = False
    fifosim_n_inferences: int = 2
    verification_atol: float = 1e-1
    
    # Hardware target
    board: str = "V80"
    generate_dcp: bool = True
    
    # Advanced options
    standalone_thresholds: bool = True
    split_large_fifos: bool = True
```

#### `HardwareCompiler` - Main Orchestrator
```python
class HardwareCompiler:
    def __init__(self, config: CompilerConfig):
        self.config = config
        self.preprocessor = ModelPreprocessor(config)
        self.builder = DataflowBuilder(config)
        self.postprocessor = OutputProcessor(config)
    
    def compile(self, model: onnx.ModelProto) -> CompilerResult:
        # Orchestrate the full compilation pipeline
        pass
        
    def compile_from_file(self, model_path: str) -> CompilerResult:
        # Load and compile model from file
        pass
```

#### `CompilerResult` - Structured Output
```python
@dataclass
class CompilerResult:
    success: bool
    output_dir: str
    final_model_path: str
    build_artifacts: Dict[str, str]
    metadata: Dict[str, Any]
    logs: List[str]
    errors: List[str]
    build_time: float
```

### 2. Specialized Components

#### `ModelPreprocessor` - Model Preparation
```python
class ModelPreprocessor:
    def preprocess(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # Simplification, cleanup, validation
        pass
        
    def generate_test_data(self, model: onnx.ModelProto) -> Tuple[np.ndarray, np.ndarray]:
        # Generate input/expected output for verification
        pass
```

#### `DataflowBuilder` - FINN Integration
```python
class DataflowBuilder:
    def build(self, model: onnx.ModelProto, input_data: np.ndarray, expected_output: np.ndarray) -> str:
        # Handle FINN dataflow build process
        pass
        
    def get_build_steps(self, blueprint: str) -> List[Callable]:
        # Get steps from blueprint system
        pass
```

#### `OutputProcessor` - Post-build Processing
```python
class OutputProcessor:
    def process_outputs(self, build_dir: str, steps: List[Callable]) -> Dict[str, str]:
        # Copy outputs, generate metadata, create handover files
        pass
        
    def generate_handover_metadata(self, build_dir: str, model_metadata: Dict) -> str:
        # Create shell handover files
        pass
```

### 3. Library Interface Design

#### Simple API for Python Library Use:
```python
import brainsmith

# Simple interface
result = brainsmith.compile_model(
    model=my_onnx_model,
    blueprint="bert",
    output_dir="./build",
    target_fps=3000
)

# Advanced interface
config = brainsmith.CompilerConfig(
    blueprint="bert",
    output_dir="./build",
    target_fps=3000,
    board="V80"
)
compiler = brainsmith.HardwareCompiler(config)
result = compiler.compile(my_onnx_model)
```

#### Configuration from File:
```python
# Load from YAML
config = brainsmith.CompilerConfig.from_yaml("build_config.yaml")
compiler = brainsmith.HardwareCompiler(config)
result = compiler.compile_from_file("model.onnx")

# Load from dict
config_dict = {
    "blueprint": "bert",
    "output_dir": "./build",
    "target_fps": 3000
}
config = brainsmith.CompilerConfig.from_dict(config_dict)
```

### 4. CLI Interface Design

#### Command Structure:
```bash
# Basic compilation
brainsmith compile model.onnx --blueprint bert --output ./build

# With configuration file
brainsmith compile model.onnx --config build_config.yaml

# List available blueprints
brainsmith blueprints list

# Validate blueprint
brainsmith blueprints validate bert

# Show blueprint details
brainsmith blueprints show bert

# Interactive mode
brainsmith interactive
```

#### Configuration File Format (YAML):
```yaml
# build_config.yaml
blueprint: bert
output_dir: ./build
build_dir: /tmp/brainsmith_builds

model:
  save_intermediate: true
  
finn:
  target_fps: 3000
  clk_period_ns: 3.33
  folding_config_file: folding.json
  
verification:
  run_fifo_sizing: false
  fifosim_n_inferences: 2
  verification_atol: 0.1
  
hardware:
  board: V80
  generate_dcp: true
  
advanced:
  standalone_thresholds: true
  split_large_fifos: true
```

### 5. Backward Compatibility

#### Legacy Function Wrapper:
```python
def forge(blueprint: str, model: onnx.ModelProto, args) -> None:
    """Legacy interface for backward compatibility."""
    # Convert args to CompilerConfig
    config = CompilerConfig.from_args(args)
    config.blueprint = blueprint
    
    # Run new compiler
    compiler = HardwareCompiler(config)
    result = compiler.compile(model)
    
    # Handle legacy behavior (no return value, raises on error)
    if not result.success:
        raise RuntimeError(f"Compilation failed: {result.errors}")
```

### 6. Extension Points

#### Custom Preprocessors:
```python
class CustomPreprocessor(ModelPreprocessor):
    def preprocess(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # Custom preprocessing logic
        pass

config.preprocessor = CustomPreprocessor
```

#### Plugin System:
```python
# Register custom blueprint
brainsmith.register_blueprint("my_custom_blueprint", steps=my_steps)

# Register custom post-processor
brainsmith.register_postprocessor("my_metadata_generator", MyPostProcessor)
```

## Implementation Benefits

### For Library Users:
1. **Clean API**: Simple function calls with clear return values
2. **Flexible Configuration**: Multiple ways to configure builds
3. **Better Error Handling**: Structured error reporting
4. **Extensibility**: Easy to customize behavior
5. **Type Safety**: Proper type hints and validation

### For CLI Users:
1. **Intuitive Commands**: Natural command structure
2. **Configuration Files**: Reusable build configurations
3. **Discovery**: Easy to explore available blueprints
4. **Validation**: Pre-flight checks for configurations

### For Developers:
1. **Modularity**: Clear separation of concerns
2. **Testability**: Each component can be tested independently
3. **Extensibility**: Easy to add new features
4. **Maintainability**: Cleaner code organization

## Migration Strategy

### Phase 1: Create New Classes (This PR)
- Implement `CompilerConfig`, `HardwareCompiler`, etc.
- Add new API functions to `brainsmith` package
- Keep existing `forge()` function as wrapper

### Phase 2: CLI Implementation
- Create `brainsmith` CLI command
- Implement configuration file loading
- Add blueprint management commands

### Phase 3: Migration and Deprecation
- Update demos to use new API (optional)
- Add deprecation warnings to `forge()`
- Create migration guide

### Phase 4: Full Transition
- Remove legacy `forge()` function
- Clean up deprecated code
- Update all documentation