# Brainsmith-2 Core APIs Reference

## Hardware Compiler Core

### `brainsmith.core.hw_compiler`

#### `forge(blueprint: str, model: str, args: Dict[str, Any]) -> CompilationResult`

**Primary entry point for hardware compilation.**

**Parameters:**
- `blueprint` (str): Build process identifier
  - `'bert'` - BERT and transformer models
  - `'custom_transformer'` - Custom transformer variants
  - Custom blueprint names registered via `@register_blueprint`
- `model` (str): Path to ONNX model file
- `args` (Dict[str, Any]): Compilation configuration
  - `target_fps` (int): Target inference frames per second
  - `resource_budget` (str): `'conservative'`, `'moderate'`, `'aggressive'`
  - `precision` (str): `'INT8'`, `'INT16'`, `'mixed'`
  - `output_dir` (str): Output directory path (optional)
  - `validate_only` (bool): Validation mode without full compilation

**Returns:** `CompilationResult`
- `success` (bool): Compilation success status
- `output_path` (Path): Generated artifacts directory
- `performance_metrics` (Dict): Performance estimates
  - `estimated_fps` (float): Estimated inference FPS
  - `resource_utilization` (float): Resource usage ratio (0-1)
  - `memory_bandwidth` (float): Required memory bandwidth (GB/s)
- `error_message` (str): Error description if `success=False`
- `warnings` (List[str]): Non-fatal warnings

**Example:**
```python
from brainsmith.core.hw_compiler import forge

result = forge(
    blueprint='bert',
    model='models/bert_base.onnx',
    args={
        'target_fps': 1000,
        'resource_budget': 'moderate',
        'precision': 'INT8',
        'output_dir': './output'
    }
)

if result.success:
    print(f"Compilation successful: {result.output_path}")
    print(f"Estimated FPS: {result.performance_metrics['estimated_fps']}")
else:
    print(f"Compilation failed: {result.error_message}")
```

## Hardware Kernel Generator

### `brainsmith.tools.hw_kernel_gen.hkg`

#### `class HardwareKernelGenerator`

**Automated generation of FINN integration components from RTL specifications.**

##### `__init__(rtl_file_path: str, compiler_data_path: str, output_dir: str = "./generated", config: PipelineConfig = None)`

**Parameters:**
- `rtl_file_path` (str): Path to SystemVerilog RTL file
- `compiler_data_path` (str): Path to metadata/compiler data file
- `output_dir` (str): Output directory for generated components
- `config` (PipelineConfig): Pipeline configuration (optional, uses defaults)

##### `run() -> Dict[str, Path]`

**Execute complete generation pipeline.**

**Returns:** Dict mapping component types to file paths
- `'hwcustomop'` - Generated HWCustomOp implementation
- `'rtlbackend'` - Generated RTL backend  
- `'test_suite'` - Generated test suite
- `'documentation'` - Generated documentation
- `'rtl_template'` - Generated RTL wrapper template

**Example:**
```python
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator

hkg = HardwareKernelGenerator(
    rtl_file_path='custom_op.sv',
    compiler_data_path='op_metadata.py',
    output_dir='./generated'
)

artifacts = hkg.run()

print("Generated components:")
for component_type, file_path in artifacts.items():
    print(f"  {component_type}: {file_path}")
```

##### `generate_hwcustomop() -> Path`

**Generate only HWCustomOp component.**

##### `generate_rtlbackend() -> Path`

**Generate only RTL backend component.**

##### `generate_test_suite() -> Path`

**Generate only test suite component.**

## Dataflow Framework

### `brainsmith.dataflow.core.dataflow_interface`

#### `class DataflowInterface`

**Core interface abstraction with three-tier dimension system.**

##### `__init__(name: str, interface_type: str, qDim: int, tDim: int, sDim: int, dtype: str, **kwargs)`

**Parameters:**
- `name` (str): Interface identifier
- `interface_type` (str): Interface category
  - `'INPUT'` - Data input interface
  - `'OUTPUT'` - Data output interface  
  - `'WEIGHT'` - Weight/parameter interface
  - `'CONFIG'` - Configuration interface
  - `'CONTROL'` - Control signal interface
- `qDim` (int): Query dimension (original tensor dimension)
- `tDim` (int): Tensor dimension (processing granularity)
- `sDim` (int): Stream dimension (hardware parallelism)
- `dtype` (str): Data type specification (`'INT8'`, `'INT16'`, `'FLOAT32'`)

**Constraints:**
- `sDim ≤ tDim ≤ qDim`
- `qDim × tDim = original_tensor_shape` (for tensor operations)

##### `validate_constraints() -> ValidationResult`

**Validate dimensional relationships and constraints.**

##### `calculate_stream_parallelism() -> int`

**Calculate effective streaming parallelism.**

##### `calculate_memory_bandwidth() -> float`

**Calculate required memory bandwidth in GB/s.**

**Example:**
```python
from brainsmith.dataflow import DataflowInterface

# BERT attention input interface
attention_input = DataflowInterface(
    name="attention_input",
    interface_type="INPUT",
    qDim=512,      # Sequence length
    tDim=64,       # Processing chunk size
    sDim=8,        # 8-way parallel processing
    dtype="INT8"
)

# Validate configuration
validation = attention_input.validate_constraints()
if validation.success:
    parallelism = attention_input.calculate_stream_parallelism()
    bandwidth = attention_input.calculate_memory_bandwidth()
    print(f"Parallelism: {parallelism}, Bandwidth: {bandwidth} GB/s")
```

### `brainsmith.dataflow.core.dataflow_model`

#### `class DataflowModel`

**Unified computational model for performance analysis.**

##### `__init__(interfaces: List[DataflowInterface], operation_type: str, **kwargs)`

**Parameters:**
- `interfaces` (List[DataflowInterface]): Operation interfaces
- `operation_type` (str): Operation category (`'attention'`, `'layernorm'`, `'custom'`)
- `**kwargs`: Additional operation-specific parameters

##### `calculate_initiation_intervals(iPar: int, wPar: int) -> Dict[str, int]`

**Calculate performance characteristics for given parallelism.**

**Parameters:**
- `iPar` (int): Input parallelism factor
- `wPar` (int): Weight parallelism factor

**Returns:** Dictionary with timing analysis
- `'compute_ii'` - Computation initiation interval
- `'memory_ii'` - Memory access initiation interval
- `'overall_ii'` - Overall initiation interval

##### `get_parallelism_bounds() -> Dict[str, Tuple[int, int]]`

**Get parallelism bounds for optimization.**

**Returns:** Dictionary mapping interface types to (min, max) parallelism

##### `optimize_parallelism(constraints: Dict) -> Dict[str, int]`

**Find optimal parallelism configuration.**

**Parameters:**
- `constraints` (Dict): Resource and performance constraints
  - `'max_luts'` (int): Maximum LUT budget
  - `'max_dsps'` (int): Maximum DSP budget
  - `'target_frequency'` (int): Target frequency in MHz

**Returns:** Optimal parallelism configuration

**Example:**
```python
from brainsmith.dataflow import DataflowInterface, DataflowModel

# Create interfaces
interfaces = [
    DataflowInterface("input", "INPUT", qDim=768, tDim=64, sDim=8),
    DataflowInterface("output", "OUTPUT", qDim=768, tDim=64, sDim=8)
]

# Create model
model = DataflowModel(
    interfaces=interfaces,
    operation_type="layernorm"
)

# Optimize parallelism
optimal_config = model.optimize_parallelism({
    'max_luts': 50000,
    'max_dsps': 200,
    'target_frequency': 250
})

print(f"Optimal configuration: {optimal_config}")
```

### `brainsmith.dataflow.core.auto_hw_custom_op`

#### `class AutoHWCustomOp`

**Base class for automatically generated HWCustomOp implementations.**

##### `__init__(onnx_node, dataflow_model: DataflowModel, **kwargs)`

**Parameters:**
- `onnx_node`: ONNX node representation
- `dataflow_model` (DataflowModel): Associated dataflow model
- `**kwargs`: Additional configuration parameters

**Automatically Implemented Methods:**
- `get_input_datatype(ind: int = 0) -> DataType`
- `get_output_datatype(ind: int = 0) -> DataType`
- `bram_estimation() -> int`
- `lut_estimation() -> int`
- `dsp_estimation(fpgapart: str) -> int`
- `get_exp_cycles() -> int`
- `verify_node() -> None`

**Example:**
```python
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow import DataflowModel

class CustomLayerNorm(AutoHWCustomOp):
    """Custom layer normalization implementation."""
    
    def __init__(self, onnx_node, **kwargs):
        # Create dataflow model
        dataflow_model = self._create_layernorm_model()
        
        # Initialize with minimal configuration
        super().__init__(onnx_node, dataflow_model, **kwargs)
    
    def _create_layernorm_model(self):
        # Define operation-specific dataflow model
        return DataflowModel(
            interfaces=self._define_interfaces(),
            operation_type="layer_normalization"
        )
    
    # All standard methods inherited automatically from AutoHWCustomOp
```

### `brainsmith.dataflow.core.auto_rtl_backend`

#### `class AutoRTLBackend`

**Base class for automatically generated RTL backend implementations.**

##### `__init__(dataflow_model: DataflowModel = None, **kwargs)`

**Parameters:**
- `dataflow_model` (DataflowModel): Associated dataflow model
- `**kwargs`: Additional configuration parameters

**Automatically Implemented Methods:**
- `get_instream_width() -> int`
- `get_outstream_width() -> int`
- `get_precision_config() -> Dict`
- `estimate_resources() -> Dict`

**Abstract Methods (Must Implement):**
- `generate_params(model, path) -> Dict`
- `code_generation_dict() -> Dict`

**Example:**
```python
from brainsmith.dataflow.core.auto_rtl_backend import AutoRTLBackend

class CustomLayerNormRTLBackend(AutoRTLBackend):
    """RTL backend for custom layer normalization."""
    
    def generate_params(self, model, path):
        """Generate hardware parameters from dataflow model."""
        parallelism = self.dataflow_model.get_optimal_parallelism()
        
        return {
            'HIDDEN_SIZE': model.get_nodeattr("hidden_size"),
            'PARALLELISM': parallelism['data_parallelism'],
            'DATA_WIDTH': self.get_instream_width()
        }
    
    def code_generation_dict(self):
        """Generate RTL instantiation dictionary."""
        code_gen_dict = super().code_generation_dict()
        
        # Add custom RTL generation
        code_gen_dict["$CUSTOM_LOGIC$"] = self._generate_custom_logic()
        
        return code_gen_dict
```

## Configuration System

### `brainsmith.tools.hw_kernel_gen.enhanced_config`

#### `class PipelineConfig`

**Comprehensive pipeline configuration management.**

##### Class Methods

##### `development_defaults() -> PipelineConfig`

**Get development-optimized configuration.**

##### `production_defaults() -> PipelineConfig`

**Get production-optimized configuration.**

##### `from_file(file_path: str) -> PipelineConfig`

**Load configuration from YAML file.**

##### `from_environment(prefix: str = 'BRAINSMITH_') -> PipelineConfig`

**Load configuration from environment variables.**

##### Instance Methods

##### `merge(other: PipelineConfig) -> PipelineConfig`

**Merge with another configuration (other takes precedence).**

##### `to_dict() -> Dict`

**Convert to dictionary representation.**

##### `save(file_path: str) -> None`

**Save configuration to YAML file.**

**Example:**
```python
from brainsmith.tools.hw_kernel_gen.enhanced_config import PipelineConfig

# Load base configuration
config = PipelineConfig.development_defaults()

# Customize for specific use case
config.dataflow.optimization_level = 'aggressive'
config.generation.enabled_generators = {'hwcustomop', 'rtlbackend'}

# Save customized configuration
config.save('custom_config.yaml')

# Load and merge configurations
base_config = PipelineConfig.from_file('base_config.yaml')
custom_config = PipelineConfig.from_file('custom_config.yaml')
final_config = base_config.merge(custom_config)
```

## Blueprint System

### `brainsmith.blueprints`

#### `register_blueprint(name: str)`

**Decorator for registering custom blueprints.**

**Parameters:**
- `name` (str): Blueprint identifier for use with `forge()`

**Example:**
```python
from brainsmith.blueprints import register_blueprint, BuildStep

@register_blueprint("custom_transformer")
def custom_transformer_pipeline(model, args):
    """Custom build process for specialized transformers."""
    return [
        BuildStep("preprocess", custom_preprocessing_step),
        BuildStep("optimize", custom_optimization_step),
        BuildStep("hardware_map", custom_hardware_mapping)
    ]

# Use with forge()
result = forge('custom_transformer', model_path, args)
```

#### `class BuildStep`

**Individual build process step.**

##### `__init__(name: str, function: Callable, **kwargs)`

**Parameters:**
- `name` (str): Step identifier
- `function` (Callable): Step implementation function
- `**kwargs`: Additional step configuration

## Error Handling

### Exception Classes

#### `CompilationError`

**Raised when model compilation fails.**

**Attributes:**
- `message` (str): Error description
- `context` (Dict): Error context information
- `suggestions` (List[str]): Suggested fixes

#### `HardwareKernelGeneratorError`

**Raised when hardware kernel generation fails.**

#### `ValidationError`

**Raised when validation fails.**

## Utility Functions

### `brainsmith.tools.profiling`

#### `class ModelProfiler`

**Model performance profiling utilities.**

##### `__init__(model_path: str)`

##### `analyze_computational_requirements() -> ProfileResult`

**Analyze model computational characteristics.**

#### `class SystemProfiler`

**System performance monitoring.**

##### `__init__(metrics: List[str], export_format: str = 'json')`

##### `monitor(operation_name: str) -> ContextManager`

**Context manager for monitoring operations.**

**Example:**
```python
from brainsmith.tools.profiling import SystemProfiler

profiler = SystemProfiler(
    metrics=['compilation_time', 'memory_usage'],
    export_format='prometheus'
)

with profiler.monitor('bert_compilation'):
    result = forge('bert', model_path, args)

profiler.export_metrics('metrics.json')
```

This API reference provides comprehensive documentation for all public APIs in the Brainsmith-2 platform, enabling developers to effectively integrate and extend the system.