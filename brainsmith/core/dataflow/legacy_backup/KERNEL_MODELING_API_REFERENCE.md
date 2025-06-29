############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# Brainsmith Kernel Modeling API Reference

## Quick Links

- [InterfaceDefinition](#interfacedefinition)
- [InterfaceModel](#interfacemodel)
- [KernelDefinition](#kerneldefinition)
- [KernelModel](#kernelmodel)
- [Tiling Functions](#tiling-functions)
- [TilingConfig](#tilingconfig)
- [Types and Enums](#types-and-enums)
- [Relationships](#relationships)
- [Expression System](#expression-system)

## InterfaceDefinition

Defines the schema for a kernel interface.

### Constructor

```python
InterfaceDefinition(
    name: str,
    direction: InterfaceDirection,
    dtype: DataType,
    block_dims_expr: Optional[Union[List[Union[str, int]], Callable]] = None,
    onnx_layout: Optional[str] = None,
    granularity: Optional[Shape] = None,
    optional: bool = False,
    rate_pattern: Optional[List[int]] = None
)
```

### Parameters

- **name** (str): Unique identifier for the interface
- **direction** (InterfaceDirection): Direction of data flow (INPUT/OUTPUT/WEIGHT)
- **dtype** (DataType): Data type specification
- **block_dims_expr** (optional): Block dimension specification
  - List of expressions: `["1", "C_TILE", "tensor[2]//16"]`
  - Callable function: `parameterized_tiles("M", "K")`
  - None: Use default chunking strategy
- **onnx_layout** (str, optional): Layout hint ("NCHW", "NHWC", "OIHW", etc.)
- **granularity** (Shape, optional): Dimension constraints for hardware
- **optional** (bool): Whether interface is required
- **rate_pattern** (List[int], optional): CSDF rate pattern

### Methods

#### create_model

```python
def create_model(
    self,
    tensor_dims: Shape,
    parameter_binding: Optional[Dict[str, int]] = None,
    config: Optional[Dict[str, Any]] = None
) -> InterfaceModel
```

Creates a runtime model instance.

**Parameters:**
- **tensor_dims**: Full tensor dimensions
- **parameter_binding**: Parameter values for expressions
- **config**: Runtime configuration

**Returns:** InterfaceModel instance

#### derive_block_dims

```python
def derive_block_dims(
    self,
    tensor_dims: Shape,
    parameter_binding: Optional[Dict[str, int]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Shape
```

Derives concrete block dimensions.

### Example Usage

```python
# Simple definition
input_def = InterfaceDefinition(
    name="input",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("UINT8")
)

# With parameterized tiling
matrix_def = InterfaceDefinition(
    name="matrix",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("INT8"),
    block_dims_expr=parameterized_tiles("TILE_M", "TILE_K"),
    granularity=(1, 8)  # K must be multiple of 8
)

# With channel-major tiling
conv_def = InterfaceDefinition(
    name="conv_input",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("UINT8"),
    block_dims_expr=channel_major_tiling(32, "tile"),
    onnx_layout="NCHW"
)
```

## InterfaceModel

Runtime instance with concrete dimensions.

### Constructor

```python
InterfaceModel(
    definition: InterfaceDefinition,
    tensor_dims: Shape,
    block_dims: List[Shape],
    phase_count: int = 1
)
```

### Properties

- **definition**: Link to interface schema
- **tensor_dims**: Full tensor dimensions
- **block_dims**: List of block dimensions (per phase)
- **ipar**: Interface parallelism (getter/setter)
- **stream_dims**: Automatically calculated from iPar (read-only)
- **dtype**: Data type from definition
- **direction**: Interface direction from definition

### Methods

#### calculate_performance_metrics

```python
def calculate_performance_metrics(
    self,
    frequency_mhz: float = 100.0,
    phase: int = 0
) -> Dict[str, Any]
```

Calculates interface performance metrics.

**Returns:**
```python
{
    "interface_parallelism": int,
    "block_volume": int,
    "bandwidth_mbps": float,
    "rate_pattern": int,
    "initiation_interval": int
}
```

### Example Usage

```python
# Create model from definition
model = input_def.create_model(
    tensor_dims=(128, 256, 224, 224),
    parameter_binding={"C_TILE": 32}
)

# Set parallelism
model.ipar = 16

# Access dimensions
print(f"Block: {model.block_dims}")    # [(128, 32, 14, 14)]
print(f"Stream: {model.stream_dims}")  # (128, 2, 1, 1)

# Get performance
metrics = model.calculate_performance_metrics()
```

## KernelDefinition

Defines complete kernel schema with relationships.

### Constructor

```python
KernelDefinition(
    name: str,
    interface_definitions: Optional[List[InterfaceDefinition]] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

### Methods

#### add_interface

```python
def add_interface(self, interface: InterfaceDefinition) -> None
```

Adds an interface definition to the kernel.

#### add_relationship

```python
def add_relationship(
    self,
    source_name: str,
    target_name: str,
    relationship_type: RelationType,
    source_dim: Optional[int] = None,
    target_dim: Optional[int] = None,
    **kwargs
) -> None
```

Adds a relationship between interfaces.

**Parameters:**
- **source_name**: Name of source interface
- **target_name**: Name of target interface
- **relationship_type**: Type of relationship (EQUAL, COUPLED)
- **source_dim**: Dimension index in source (optional)
- **target_dim**: Dimension index in target (optional)

### Example Usage

```python
# Define kernel
kernel_def = KernelDefinition(name="matmul")

# Add interfaces
kernel_def.add_interface(a_def)
kernel_def.add_interface(b_def)
kernel_def.add_interface(c_def)

# Add relationships
kernel_def.add_relationship("A", "B", RelationType.EQUAL, 
                           source_dim=1, target_dim=0)
kernel_def.add_relationship("A", "C", RelationType.EQUAL,
                           source_dim=0, target_dim=0)
kernel_def.add_relationship("B", "C", RelationType.EQUAL,
                           source_dim=1, target_dim=1)
```

## KernelModel

Complete runtime kernel instance.

### Constructor

```python
KernelModel(
    interface_models: List[InterfaceModel],
    definition: KernelDefinition,
    validate: bool = True
)
```

### Methods

#### apply_parallelism

```python
def apply_parallelism(self, ipar_config: Dict[str, int]) -> None
```

Applies and propagates parallelism configuration.

**Parameters:**
- **ipar_config**: Dictionary mapping interface names to iPar values

**Example:**
```python
kernel_model.apply_parallelism({
    "output": 16,
    "input": 8
})
```

#### calculate_performance_metrics

```python
def calculate_performance_metrics(
    self,
    frequency_mhz: float = 100.0
) -> Dict[str, Any]
```

Calculates aggregate kernel performance.

**Returns:**
```python
{
    "total_bandwidth_mbps": float,
    "initiation_interval": int,
    "interface_parallelisms": Dict[str, int],
    "interface_bandwidths": Dict[str, float]
}
```

#### get_interface_model

```python
def get_interface_model(self, name: str) -> InterfaceModel
```

Gets interface model by name.

### Example Usage

```python
# Create interface models
a_model = a_def.create_model((512, 1024), {"TILE_M": 64, "TILE_K": 32})
b_model = b_def.create_model((1024, 256), {"TILE_K": 32, "TILE_N": 128})
c_model = c_def.create_model((512, 256), {"TILE_M": 64, "TILE_N": 128})

# Create kernel model
kernel_model = KernelModel(
    interface_models=[a_model, b_model, c_model],
    definition=kernel_def
)

# Apply parallelism
kernel_model.apply_parallelism({"C": 16})

# Get metrics
metrics = kernel_model.calculate_performance_metrics()
print(f"Bandwidth: {metrics['total_bandwidth_mbps']} MB/s")
print(f"II: {metrics['initiation_interval']} cycles")
```

## Tiling Functions

Pre-defined tiling strategies for common patterns.

### fixed_tiles

```python
def fixed_tiles(*tile_sizes: int) -> Callable
```

Fixed tile sizes.

```python
block_dims_expr = fixed_tiles(64, 32, 3, 3)
```

### parameterized_tiles

```python
def parameterized_tiles(*param_names: str) -> Callable
```

Tiles from parameter binding.

```python
block_dims_expr = parameterized_tiles("TILE_M", "TILE_K")
# Requires parameter_binding={"TILE_M": 64, "TILE_K": 32}
```

### adaptive_tiles

```python
def adaptive_tiles(
    config_key: str,
    default: Optional[List[int]] = None
) -> Callable
```

Configuration-driven tiling.

```python
block_dims_expr = adaptive_tiles("optimization_mode", default=[32, 32])
# Uses config["optimization_mode"] if available
```

### channel_major_tiling

```python
def channel_major_tiling(
    channel_tile: Union[int, str] = "C_TILE",
    spatial_mode: str = "tile"
) -> Callable
```

CNN-optimized tiling.

```python
block_dims_expr = channel_major_tiling(32, "tile")
# NCHW → [1, 32, 14, 14]
# OIHW → [32, 32, 3, 3]
```

### memory_constrained_tiles

```python
def memory_constrained_tiles(
    memory_limit_bytes: int,
    bytes_per_element: int
) -> Callable
```

Memory-aware tiling.

```python
block_dims_expr = memory_constrained_tiles(
    memory_limit_bytes=512*1024,  # 512KB
    bytes_per_element=4           # FP32
)
```

### power_of_two_tiles

```python
def power_of_two_tiles(
    min_size: int = 1,
    max_size: int = 1024
) -> Callable
```

Hardware-friendly power-of-two tiles.

```python
block_dims_expr = power_of_two_tiles(min_size=16, max_size=256)
```

### ratio_based_tiles

```python
def ratio_based_tiles(ratios: List[float]) -> Callable
```

Proportional tiling.

```python
block_dims_expr = ratio_based_tiles([1.0, 0.25, 0.1, 0.1])
# Full batch, 1/4 channels, 1/10 spatial
```

### phase_dependent_tiles

```python
def phase_dependent_tiles(
    phase_tiles: List[List[Union[int, str]]]
) -> Callable
```

CSDF phase-specific tiling.

```python
block_dims_expr = phase_dependent_tiles([
    [128, 128],    # Phase 0
    [64, 64],      # Phase 1
    [32, 32],      # Phase 2
])
```

### composite_tiling

```python
def composite_tiling(*strategies: Callable) -> Callable
```

Fallback strategy chain.

```python
block_dims_expr = composite_tiling(
    adaptive_tiles("custom"),
    parameterized_tiles("TILE"),
    fixed_tiles(32, 32)  # fallback
)
```

### full_tensor

```python
def full_tensor() -> Callable
```

Process entire tensor (no tiling).

```python
block_dims_expr = full_tensor()
# Returns [":"] for all dimensions
```

## TilingConfig

Configuration dataclass for tiling strategies.

### Constructor

```python
@dataclass
class TilingConfig:
    strategy: TilingStrategy
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Strategy-specific fields
    tile_sizes: Optional[List[int]] = None
    param_names: Optional[List[str]] = None
    config_key: Optional[str] = None
    default_tiles: Optional[List[int]] = None
    # ... more fields
```

### Methods

#### to_function

```python
def to_function(self) -> Callable
```

Converts configuration to tiling function.

#### to_dict / from_dict

```python
def to_dict(self) -> Dict[str, Any]
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'TilingConfig'
```

Dictionary serialization.

#### to_json / from_json

```python
def to_json(self) -> str
@classmethod
def from_json(cls, json_str: str) -> 'TilingConfig'
```

JSON serialization.

### Example Usage

```python
# Create configuration
config = TilingConfig(
    strategy=TilingStrategy.CHANNEL_MAJOR,
    channel_tile=32,
    spatial_mode="tile"
)

# Serialize
json_str = config.to_json()

# Load and use
loaded = TilingConfig.from_json(json_str)
interface_def = InterfaceDefinition(
    name="conv",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("UINT8"),
    block_dims_expr=loaded.to_function()
)
```

## Types and Enums

### InterfaceDirection

```python
class InterfaceDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"  # For constants/parameters
```

### DataType

```python
class DataType:
    @staticmethod
    def from_string(dtype_str: str) -> DataType
    
    @property
    def bitwidth(self) -> int
```

Common types: "UINT8", "INT8", "INT16", "INT32", "FP16", "FP32"

### RelationType

```python
class RelationType(Enum):
    EQUAL = "equal"      # Dimensions must match
    COUPLED = "coupled"  # Dimensions scale together
```

### TilingStrategy

```python
class TilingStrategy(Enum):
    FIXED = "fixed"
    PARAMETERIZED = "parameterized"
    ADAPTIVE = "adaptive"
    CHANNEL_MAJOR = "channel_major"
    MEMORY_CONSTRAINED = "memory_constrained"
    POWER_OF_TWO = "power_of_two"
    RATIO_BASED = "ratio_based"
    PHASE_DEPENDENT = "phase_dependent"
    COMPOSITE = "composite"
    FULL_TENSOR = "full_tensor"
```

### Shape

```python
Shape = Tuple[int, ...]  # Dimension tuple
```

## Relationships

### InterfaceRelationship

```python
@dataclass
class InterfaceRelationship:
    source_interface: str
    target_interface: str
    type: RelationType
    source_dim: Optional[int] = None
    target_dim: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Usage Examples

```python
# MatMul relationships
kernel_def.add_relationship("A", "B", RelationType.EQUAL,
                           source_dim=1, target_dim=0)

# Broadcast relationship (future)
kernel_def.add_relationship("bias", "output", RelationType.BROADCAST,
                           source_dim=0, target_dim=1)

# Coupled dimensions (future)
kernel_def.add_relationship("input", "output", RelationType.COUPLED,
                           metadata={"scale_factor": 2})
```

## Expression System

### Supported Expression Syntax

1. **Literals**: `"64"`, `"128"`
2. **Colon notation**: `":"` (full dimension)
3. **Tensor indexing**: `"tensor[0]"`, `"tensor[2]"`
4. **Parameters**: `"TILE_M"`, `"C_TILE"`
5. **Parameter dict**: `"params['tile_size']"`
6. **Config dict**: `"config['mode']"`
7. **Arithmetic**: `"tensor[0]//16"`, `"TILE*2"`
8. **Conditionals**: `"64 if config.get('large') else 32"`

### Expression Evaluation

```python
# Internal evaluation context
context = {
    "tensor": tensor_dims,
    "params": parameter_binding or {},
    "config": config or {},
    **parameter_binding  # Direct parameter access
}

# Safe evaluation
ast.literal_eval() for simple expressions
Custom AST parser for complex expressions
```

### Example Expressions

```python
# Simple literals
block_dims_expr = ["1", "32", "14", "14"]

# With parameters
block_dims_expr = ["1", "C_TILE", "H_TILE", "W_TILE"]

# Mixed expressions
block_dims_expr = [
    "1",                    # Batch = 1
    "params['channels']",   # From parameter dict
    "tensor[2]//16",       # Spatial tiling
    ":"                    # Full width
]

# Conditional
block_dims_expr = [
    "1",
    "128 if config.get('large_model') else 64",
    ":",
    ":"
]
```

## Complete Example

```python
from brainsmith.core.dataflow.core import *

# 1. Define interfaces
input_def = InterfaceDefinition(
    name="input",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("FP16"),
    block_dims_expr=adaptive_tiles("mode", default=[1, 16, 256]),
    onnx_layout="NLC"  # Transformer-style
)

output_def = InterfaceDefinition(
    name="output", 
    direction=InterfaceDirection.OUTPUT,
    dtype=DataType.from_string("FP16"),
    block_dims_expr=adaptive_tiles("mode", default=[1, 16, 256]),
    onnx_layout="NLC"
)

# 2. Create kernel definition
kernel_def = KernelDefinition(
    name="transformer_layer",
    interface_definitions=[input_def, output_def]
)

# 3. Add relationships
kernel_def.add_relationship("input", "output", RelationType.EQUAL)

# 4. Create models with configuration
config = {"mode": [1, 8, 128]}  # Smaller tiles for testing
input_model = input_def.create_model(
    tensor_dims=(4, 512, 768),
    config=config
)
output_model = output_def.create_model(
    tensor_dims=(4, 512, 768),
    config=config
)

# 5. Create kernel model
kernel_model = KernelModel(
    interface_models=[input_model, output_model],
    definition=kernel_def
)

# 6. Apply parallelism
kernel_model.apply_parallelism({"output": 8})

# 7. Get performance metrics
metrics = kernel_model.calculate_performance_metrics()
print(f"Throughput: {metrics['total_bandwidth_mbps']:.1f} MB/s")
print(f"Latency: {metrics['initiation_interval']} cycles")
```