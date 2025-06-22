# Unified Framework Architecture Design Document

## Executive Summary

The Unified Framework is a clean-slate implementation that integrates Brainsmith's Unified Kernel Modeling Framework with FINN's HWCustomOp infrastructure. It provides a modern, extensible architecture for defining, optimizing, and implementing FPGA hardware operators while maintaining compatibility with the FINN ecosystem.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ONNX Model                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      Unified Framework                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    KernelDefinition                          │   │
│  │  • Interface specifications                                  │   │
│  │  • Constraints and pragmas                                   │   │
│  │  • Performance/Resource models                               │   │
│  │  • RTL parameters                                            │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │                 UnifiedHWCustomOp                            │   │
│  │  • FINN HWCustomOp integration                               │   │
│  │  • Kernel instance management                                │   │
│  │  • Shape/datatype propagation                                │   │
│  │  • Attribute management                                      │   │
│  └──────────┬─────────────────────────────────┬────────────────┘   │
│             │                                 │                      │
│  ┌──────────▼──────────┐          ┌──────────▼────────────────┐   │
│  │   UnifiedDSEMixin   │          │   UnifiedRTLBackend       │   │
│  │  • Auto-optimization │          │  • RTL generation         │   │
│  │  • Pareto analysis   │          │  • Template processing    │   │
│  │  • Constraint solving│          │  • File management        │   │
│  └─────────────────────┘          └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼────────┐     ┌───────▼────────┐
            │ Optimized RTL  │     │  Performance   │
            │     Code       │     │    Reports     │
            └────────────────┘     └────────────────┘
```

## Core Components

### 1. KernelDefinition System

The `KernelDefinition` is the foundation of the unified framework, providing a complete, declarative specification of a hardware kernel.

#### Structure
```python
@dataclass
class KernelDefinition:
    # Identity
    name: str
    interfaces: List[InterfaceDefinition]
    
    # Behavior
    constraints: List[Any]
    rtl_parameters: Dict[str, Any]
    exposed_parameters: Set[str]
    
    # Optimization hints
    performance_model: Optional[PerformanceModel]
    resource_model: Optional[ResourceModel]
```

#### InterfaceDefinition
Each interface is fully specified with:
- **Type**: INPUT, OUTPUT, WEIGHT, CONFIG
- **Protocol**: AXI_STREAM, AXI_LITE, etc.
- **Datatype Constraints**: Supported datatypes and bit widths
- **Parameter Links**: Mapping to RTL parameters (BDIM, SDIM, datatype)

#### Example
```python
InterfaceDefinition(
    name="input",
    type=InterfaceType.INPUT,
    protocol=ProtocolType.AXI_STREAM,
    datatype_constraints=[
        DatatypeConstraint("INT", 4, 16),
        DatatypeConstraint("UINT", 4, 16)
    ],
    parameter_links={
        'BDIM': 'N_INPUTS',
        'SDIM': 'input_parallelism'
    }
)
```

### 2. UnifiedHWCustomOp Base Class

The `UnifiedHWCustomOp` class bridges the gap between FINN's infrastructure and the new kernel modeling framework.

#### Key Responsibilities

1. **Kernel Management**
   - Creates `Kernel` instances from `KernelDefinition`
   - Manages kernel lifecycle and state
   - Handles constraint validation

2. **FINN Integration**
   - Inherits from FINN's `HWCustomOp`
   - Provides required abstract methods
   - Manages ONNX node attributes

3. **Shape Propagation**
   - Maps ONNX tensor shapes to interface dimensions
   - Handles tensor → block → stream hierarchy
   - Supports dynamic shape inference

4. **Type System Bridge**
   - Maps `InterfaceType` (dataflow) ↔ `InterfaceDirection` (kernel)
   - Converts between QONNX and core datatypes
   - Validates datatype constraints

#### Initialization Flow
```python
def __init__(self, onnx_node, **kwargs):
    # 1. Get kernel definition from subclass
    self.kernel_def = self.get_kernel_definition()
    
    # 2. Create kernel instance with interfaces
    self.kernel = self._create_kernel_from_definition()
    
    # 3. Create single-kernel dataflow graph
    self.graph = DataflowGraph()
    self.graph.add_kernel(self.kernel)
    
    # 4. Initialize from ONNX
    self._initialize_from_onnx()
```

### 3. DSE Integration Layer

The `UnifiedDSEMixin` provides automatic optimization capabilities using the Design Space Exploration framework.

#### Features

1. **Target-based Optimization**
   ```python
   optimize_for_target({
       "target_throughput": 200.0,  # MHz
       "optimization_objective": "throughput",
       "max_resources": {"LUT": 5000, "BRAM": 10}
   })
   ```

2. **Multi-objective Support**
   - Throughput maximization
   - Latency minimization
   - Resource minimization
   - Balanced (Pareto-optimal)

3. **Constraint Handling**
   - Resource constraints (LUT, BRAM, DSP, URAM)
   - Performance constraints (throughput, bandwidth)
   - Kernel-specific constraints from metadata

4. **Optimization Workflow**
   ```
   Target Spec → DSEConstraints → DesignSpaceExplorer
        ↓              ↓                    ↓
   Configuration ← Best Config ← Pareto Analysis
   ```

### 4. RTL Generation System

The `UnifiedRTLBackend` provides clean RTL code generation using modern template systems.

#### Architecture
```
KernelDefinition → Template Variables → Jinja2 Templates → RTL Code
       ↓                    ↓                               ↓
   Parameters    Current Configuration             Generated Files
```

#### Key Features

1. **Template Processing**
   - Jinja2 template engine
   - Hierarchical template search
   - Caching for performance

2. **Variable Generation**
   - Converts kernel config to template variables
   - Maintains FINN compatibility (SIMD/PE mapping)
   - Supports custom parameters

3. **File Management**
   - Generates main module
   - Optional wrapper generation
   - Configuration packages
   - Support file copying

### 5. Kernel Factory

The `KernelDefinitionFactory` provides multiple ways to create kernel definitions:

```python
# From RTL file
kernel_def = KernelDefinitionFactory.from_rtl_file("threshold.sv")

# From specification
kernel_def = KernelDefinitionFactory.from_specification({
    'name': 'threshold',
    'interfaces': [...],
    'constraints': [...]
})

# From YAML/JSON
kernel_def = KernelDefinitionFactory.from_yaml_file("threshold.yaml")
```

## Data Flow

### 1. Initialization Flow
```
ONNX Node → UnifiedHWCustomOp.__init__()
    ↓
get_kernel_definition() → KernelDefinition
    ↓
_create_kernel_from_definition() → Kernel + Interfaces
    ↓
_initialize_from_onnx() → Shape/Datatype Setup
    ↓
Ready for Execution/Optimization
```

### 2. Optimization Flow
```
optimize_for_target() → DSEConstraints
    ↓
DesignSpaceExplorer.explore() → List[DSEResult]
    ↓
_select_best_config() → Optimal Configuration
    ↓
_update_attributes_from_config() → Node Attributes Updated
```

### 3. RTL Generation Flow
```
generate_hdl() → prepare_codegen_rtl_values()
    ↓
Template Selection → Variable Substitution
    ↓
File Generation → Output Directory
```

## Key Design Patterns

### 1. Declarative Kernel Definition
Kernels are defined declaratively with all metadata upfront, enabling:
- Static analysis and validation
- Automatic code generation
- Optimization without RTL parsing

### 2. Clean Separation of Concerns
- **Definition**: What the kernel does (KernelDefinition)
- **Instance**: Specific configuration (Kernel)
- **Optimization**: How to configure (DSEMixin)
- **Generation**: How to implement (RTLBackend)

### 3. Type System Bridge
Careful mapping between different type systems:
- InterfaceType (existing) ↔ InterfaceDirection (new)
- QONNX DataType ↔ Core DataType
- Shape (type alias) ↔ tuple (concrete type)

### 4. Composition over Inheritance
Uses mixins for optional features:
- UnifiedDSEMixin for optimization
- UnifiedRTLBackend for code generation
- Easy to extend with new capabilities

## Example: Thresholding Operator

### Definition
```python
class UnifiedThresholding(UnifiedHWCustomOp, UnifiedRTLBackend, UnifiedDSEMixin):
    def get_kernel_definition(self) -> KernelDefinition:
        return KernelDefinition(
            name="thresholding",
            interfaces=[
                InterfaceDefinition(name="input", type=InterfaceType.INPUT, ...),
                InterfaceDefinition(name="threshold", type=InterfaceType.WEIGHT, ...),
                InterfaceDefinition(name="output", type=InterfaceType.OUTPUT, ...)
            ],
            constraints=[
                "output.tensor_shape == input.tensor_shape",
                "threshold.tensor_shape[0] == input.tensor_shape[-1]"
            ],
            performance_model=PerformanceModel(base_latency=3, ...),
            resource_model=ResourceModel(base_luts=100, ...)
        )
```

### Usage
```python
# Create instance
op = UnifiedThresholding(onnx_node)

# Optimize
op.optimize_for_target({
    "target_throughput": 200.0,
    "optimization_objective": "balanced"
})

# Generate RTL
op.generate_hdl(model, fpgapart, clk)
```

## Advantages

1. **Clean Architecture**: No legacy compatibility layers
2. **Optimal Integration**: Full use of new framework capabilities
3. **Extensibility**: Easy to add new operators and features
4. **Performance**: DSE integration enables automatic optimization
5. **Maintainability**: Clear separation of concerns

## Extension Points

### Adding New Operators
1. Create class inheriting from `UnifiedHWCustomOp`
2. Implement `get_kernel_definition()`
3. Optionally add mixins for DSE/RTL
4. Override `execute_node()` for simulation

### Adding New Protocols
1. Extend `ProtocolType` enum
2. Update interface creation logic
3. Add protocol-specific handling in RTL generation

### Custom Optimization Objectives
1. Extend `UnifiedDSEMixin`
2. Add new objective to `_select_best_config()`
3. Implement selection logic

## Conclusion

The Unified Framework provides a modern, clean implementation that bridges Brainsmith's advanced kernel modeling with FINN's proven infrastructure. By avoiding compatibility layers and embracing clean abstractions, it enables rapid development of optimized hardware operators while maintaining the flexibility to evolve with future requirements.