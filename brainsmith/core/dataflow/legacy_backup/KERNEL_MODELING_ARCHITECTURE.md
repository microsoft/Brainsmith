############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

# Brainsmith Kernel Modeling Architecture

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Component Design](#component-design)
4. [Dimension Hierarchy](#dimension-hierarchy)
5. [Tiling System](#tiling-system)
6. [Relationship System](#relationship-system)
7. [Performance Modeling](#performance-modeling)
8. [CSDF Support](#csdf-support)
9. [Design Patterns](#design-patterns)
10. [Integration Guide](#integration-guide)

## Overview

The Brainsmith kernel modeling system is a sophisticated framework for modeling FPGA hardware kernels with built-in support for parallelism, tiling, and dataflow optimization. It implements a two-tier architecture that separates static definitions from runtime models, enabling flexible configuration and efficient design space exploration.

### Key Features

- **Two-tier Architecture**: Definition/Model separation for flexibility
- **Three-level Dimension Hierarchy**: Tensor → Block → Stream dimensions
- **Flexible Tiling**: Expression-based, function-based, and configuration-driven
- **Relationship Constraints**: Automatic validation and propagation
- **Performance Modeling**: Built-in metrics with caching
- **CSDF Support**: Native cyclo-static dataflow patterns

## Core Architecture

### Two-Tier Design

The system implements a clear separation of concerns:

```
┌─────────────────────┐         ┌─────────────────────┐
│ Definition Classes  │         │   Model Classes     │
├─────────────────────┤         ├─────────────────────┤
│ • Static schemas   │  ──►    │ • Runtime instances │
│ • Constraints      │ creates │ • Actual dimensions │
│ • Templates        │         │ • Performance data  │
└─────────────────────┘         └─────────────────────┘
```

**Definition Classes** represent:
- Static schemas and constraints
- Configurable templates
- Validation rules
- Default behaviors

**Model Classes** represent:
- Concrete runtime instances
- Actual dimension values
- Performance characteristics
- Dynamic state

### Component Hierarchy

```
KernelDefinition
    ├── InterfaceDefinition[]
    │   ├── name, direction, dtype
    │   ├── block_dims_expr (expression/function)
    │   └── constraints (granularity, etc.)
    └── Relationships[]
        ├── source/target interfaces
        ├── relationship type
        └── dimension mappings

KernelModel
    ├── InterfaceModel[]
    │   ├── tensor_dims (actual data shape)
    │   ├── block_dims (computed tiling)
    │   ├── stream_dims (parallelism)
    │   └── performance metrics
    └── Validated relationships
```

## Component Design

### InterfaceDefinition

Defines the schema for a kernel interface:

```python
class InterfaceDefinition:
    name: str                    # Unique identifier
    direction: InterfaceDirection # INPUT/OUTPUT/WEIGHT
    dtype: DataType             # Data type specification
    block_dims_expr: Optional[Union[List[Union[str, int]], Callable]]
    onnx_layout: Optional[str]  # Layout hint (NCHW, NHWC, etc.)
    granularity: Optional[Shape]  # Dimension constraints
    optional: bool              # Whether interface is required
```

Key features:
- **Flexible block dimension specification**: expressions, functions, or defaults
- **ONNX layout awareness**: Optimized defaults for common layouts
- **Constraint support**: Granularity requirements for hardware

### InterfaceModel

Runtime instance with concrete dimensions:

```python
class InterfaceModel:
    definition: InterfaceDefinition  # Link to schema
    tensor_dims: Shape              # Actual data dimensions
    block_dims: List[Shape]         # Computed block tiling
    _ipar: Optional[int]            # Interface parallelism
    _stream_dims: Optional[Shape]   # Cached stream dimensions
    
    @property
    def stream_dims(self) -> Shape:
        """Automatically derived from iPar"""
        if self._stream_dims is None:
            self._stream_dims = self._calculate_stream_dims()
        return self._stream_dims
```

Key features:
- **Automatic stream dimension calculation**: From iPar values
- **Performance metric caching**: Lazy evaluation
- **Dynamic recalculation**: Cache invalidation on changes

### KernelDefinition

Defines complete kernel schema:

```python
class KernelDefinition:
    name: str
    interface_definitions: List[InterfaceDefinition]
    relationships: List[InterfaceRelationship]
    
    def add_relationship(self, source: str, target: str, 
                        rel_type: RelationType, **kwargs):
        """Add inter-interface constraint"""
```

### KernelModel

Complete runtime kernel instance:

```python
class KernelModel:
    interface_models: List[InterfaceModel]
    definition: KernelDefinition
    
    def apply_parallelism(self, ipar_config: Dict[str, int]):
        """Apply and propagate parallelism"""
        
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate performance"""
```

## Dimension Hierarchy

### Three-Level System

```
┌─────────────────────────────────────────────────┐
│            Tensor Dimensions                     │
│        Full data shape [N, C, H, W]             │
└─────────────────┬───────────────────────────────┘
                  │ Tiling
┌─────────────────▼───────────────────────────────┐
│            Block Dimensions                      │
│     Tiling for processing [n, c, h, w]          │
└─────────────────┬───────────────────────────────┘
                  │ Parallelism (iPar)
┌─────────────────▼───────────────────────────────┐
│           Stream Dimensions                      │
│    Parallel execution units [n/p, c, h, w]      │
└─────────────────────────────────────────────────┘
```

### Dimension Flow Example

```python
# Definition
interface_def = InterfaceDefinition(
    name="input",
    block_dims_expr=parameterized_tiles("TILE_N", "TILE_C", "TILE_H", "TILE_W")
)

# Model creation
model = interface_def.create_model(
    tensor_dims=(128, 256, 224, 224),  # Full tensor
    parameter_binding={"TILE_N": 1, "TILE_C": 32, "TILE_H": 14, "TILE_W": 14}
)
# → block_dims = [(1, 32, 14, 14)]

# Apply parallelism
model.ipar = 16
# → stream_dims = (1, 16, 1, 1)  # 32/16 = 2 for C dimension
```

## Tiling System

### Specification Methods

1. **Expression-based** (strings with variables):
   ```python
   block_dims_expr=["1", "C_TILE", "tensor[2]//16", "tensor[3]//16"]
   ```

2. **Function-based** (callable strategies):
   ```python
   block_dims_expr=channel_major_tiling(channel_tile=32, spatial_mode="tile")
   ```

3. **Default behavior** (full tensor or layout-aware):
   ```python
   # No block_dims_expr → uses _default_block_chunking()
   ```

### Tiling Function Library

```python
# Fixed tiling
fixed_tiles(64, 32, 3, 3)

# Parameterized (from binding)
parameterized_tiles("TILE_M", "TILE_K", "TILE_N")

# Adaptive (config-driven)
adaptive_tiles("optimization_mode", default=[32, 32])

# Channel-major (CNN-optimized)
channel_major_tiling(channel_tile="C_TILE", spatial_mode="full")

# Memory-constrained
memory_constrained_tiles(memory_limit_bytes=512*1024, bytes_per_element=4)

# Composite strategies
composite_tiling(
    adaptive_tiles("custom"),
    parameterized_tiles("TILE"),
    fixed_tiles(32, 32)  # fallback
)
```

### Tiling Configuration

```python
config = TilingConfig(
    strategy=TilingStrategy.CHANNEL_MAJOR,
    channel_tile=32,
    spatial_mode="tile",
    parameters={"min_tile": 8}
)

# Serialization support
json_config = config.to_json()
loaded = TilingConfig.from_json(json_config)
```

## Relationship System

### Relationship Types

```python
class RelationType(Enum):
    EQUAL = "equal"           # Dimensions must match
    COUPLED = "coupled"       # Dimensions scale together
    # Future: BROADCAST, REDUCTION, etc.
```

### Constraint Propagation

```
┌─────────┐     EQUAL      ┌─────────┐
│ Input A │ ────────────► │ Output C│
│ [M, K]  │ dim[0]→dim[0] │ [M, N] │
└─────────┘                └─────────┘
     │                          ▲
     │ EQUAL                    │ EQUAL
     │ dim[1]→dim[0]           │ dim[1]→dim[1]
     ▼                          │
┌─────────┐                     │
│ Input B │ ────────────────────┘
│ [K, N]  │
└─────────┘
```

### Parallelism Propagation

```python
def _propagate_parallelism(self):
    """Propagate iPar through EQUAL relationships"""
    for relationship in self.definition.relationships:
        if relationship.type == RelationType.EQUAL:
            source_ipar = source_model.ipar
            if source_ipar and not target_model.ipar:
                # Propagate if compatible
                if self._can_propagate_parallelism(source_model, target_model):
                    target_model.ipar = source_ipar
```

## Performance Modeling

### Metrics Calculation

```python
# Interface-level metrics
bandwidth_bits = volume * dtype.bitwidth * rate_multiplier
bandwidth_mbps = (bandwidth_bits / 1e6) / period_seconds

# Kernel-level aggregation
total_bandwidth = sum(interface_bandwidths)
initiation_interval = max(interface_iis)
```

### Performance Caching

```python
class InterfaceModel:
    def _invalidate_performance_cache(self):
        """Clear cached metrics on dimension change"""
        self._cached_bandwidth = None
        self._cached_volume = None
        self._cached_ii = None
```

## CSDF Support

### Phase-Dependent Behavior

```python
# Phase-dependent tiling
phase_tiles = phase_dependent_tiles([
    [128, 128],    # Phase 0: Large blocks
    [64, 64],      # Phase 1: Medium blocks
    [32, 32],      # Phase 2: Small blocks
])

# Usage with phase config
model = interface_def.create_model(
    tensor_dims=(256, 256),
    config={"csdf_phase": 1}  # Use phase 1 tiling
)
```

### Rate Patterns

```python
rate_pattern=[2, 1, 1, 3]  # Variable consumption/production
```

## Design Patterns

### 1. **Builder Pattern** (KernelDefinition)
```python
kernel_def = KernelDefinition(name="matmul")
kernel_def.add_interface(a_def)
kernel_def.add_interface(b_def)
kernel_def.add_relationship("A", "B", RelationType.EQUAL)
```

### 2. **Factory Pattern** (Model Creation)
```python
model = interface_def.create_model(tensor_dims, **kwargs)
```

### 3. **Strategy Pattern** (Tiling Functions)
```python
def custom_tiling(tensor_dims, params, config):
    # Custom tiling logic
    return computed_tiles

interface_def.block_dims_expr = custom_tiling
```

### 4. **Observer Pattern** (Cache Invalidation)
```python
# Setting ipar invalidates dependent caches
model.ipar = 16  # → triggers _invalidate_performance_cache()
```

## Integration Guide

### Basic Usage

```python
# 1. Define kernel schema
kernel_def = KernelDefinition(
    name="conv2d",
    interface_definitions=[
        InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=channel_major_tiling(32)
        ),
        # ... more interfaces
    ]
)

# 2. Create runtime model
input_model = kernel_def.interface_definitions[0].create_model(
    tensor_dims=(1, 128, 224, 224),
    parameter_binding={"C_TILE": 32}
)

# 3. Build kernel model
kernel_model = KernelModel(
    interface_models=[input_model, ...],
    definition=kernel_def
)

# 4. Apply optimizations
kernel_model.apply_parallelism({"output": 16})

# 5. Get performance metrics
metrics = kernel_model.calculate_performance_metrics()
```

### Advanced Integration

```python
# DSE Integration
for config in design_space:
    models = create_models_with_config(kernel_def, config)
    kernel = KernelModel(models, kernel_def)
    kernel.apply_parallelism(config.parallelism)
    score = evaluate_kernel(kernel)

# ADFG Integration
actor = Actor(
    kernel_model=kernel_model,
    execution_time=metrics["initiation_interval"]
)
```

### Extension Points

1. **Custom Tiling Functions**: Implement domain-specific strategies
2. **New Relationship Types**: Extend RelationType enum
3. **Performance Models**: Override calculation methods
4. **Validation Rules**: Add custom validators

## Best Practices

1. **Use appropriate tiling strategies**:
   - Fixed for known-good configurations
   - Parameterized for DSE
   - Adaptive for runtime flexibility

2. **Leverage relationship constraints**:
   - Define all dimensional dependencies
   - Let propagation handle complexity

3. **Consider memory hierarchy**:
   - Use memory-constrained tiling for large tensors
   - Power-of-two tiles for hardware efficiency

4. **Profile and optimize**:
   - Monitor cache hit rates
   - Minimize redundant calculations

5. **Document intent**:
   - Clear interface names
   - Meaningful parameter names
   - Relationship rationale