# Brainsmith Core Dataflow Architecture Guide

## Overview

The Brainsmith dataflow module provides a sophisticated abstraction layer for representing hardware accelerator kernels on FPGAs. It bridges high-level PyTorch neural networks to low-level RTL hardware descriptions through a clean, type-safe API implementing the SDIM (Streaming Dimensions) architecture.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Architecture Layers](#architecture-layers)
3. [Data Hierarchy and Tiling](#data-hierarchy-and-tiling)
4. [SDIM System](#sdim-system)
5. [Component Architecture](#component-architecture)
6. [Type System](#type-system)
7. [Relationships and Constraints](#relationships-and-constraints)
8. [Usage Patterns](#usage-patterns)
9. [Performance Considerations](#performance-considerations)

## Core Concepts

### Design Philosophy

The dataflow module follows five key design principles:

```mermaid
graph LR
    subgraph "Design Principles"
        A[Definition/Model<br/>Separation]
        B[Type Safety]
        C[QONNX<br/>Integration]
        D[Constraint-Based<br/>Validation]
        E[SDIM<br/>Architecture]
    end
    
    A --> F[Static Schemas vs<br/>Runtime Instances]
    B --> G[Separate Input/Output<br/>Interface Classes]
    C --> H[Native QONNX<br/>DataType Usage]
    D --> I[Definitions Specify<br/>Constraints]
    E --> J[Multi-dimensional<br/>Streaming]
    
    classDef principle fill:#7c3aed,stroke:#333,stroke-width:2px,color:#fff
    classDef detail fill:#dbeafe,stroke:#60a5fa,stroke-width:1px
    
    class A,B,C,D,E principle
    class F,G,H,I,J detail
```

### Definition vs Model Pattern

The system maintains a clear separation between "what CAN be" (definitions) and "what IS" (models):

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│          DEFINITION                 │     │             MODEL                   │
│  "What CAN be" - Schema & Rules    │     │   "What IS" - Runtime Instance      │
├─────────────────────────────────────┤     ├─────────────────────────────────────┤
│  • Constraints and validation       │     │  • Concrete types                   │
│  • Allowed datatype groups          │     │  • Actual SDIM values               │
│  • Tiling expressions               │     │  • Performance metrics              │
│  • Relationships                    │     │  • Cached calculations              │
│                                     │     │                                     │
│  validate() ─────────┐             │     │  calculate_performance_metrics()    │
│                      ▼             │     │                                     │
│  create_model() ─────┼─────────────┼────►│  configure_sdim()                   │
│                      │             │     │                                     │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
```

## Architecture Layers

The dataflow module implements a layered architecture for clean separation of concerns:

```mermaid
graph TB
    subgraph "Application Layer"
        APP[PyTorch Models]
        ONNX[ONNX Graphs]
        USER[User Code]
    end
    
    subgraph "Definition Layer"
        ID[InputDefinition]
        OD[OutputDefinition]
        KD[KernelDefinition]
    end
    
    subgraph "Model Layer"
        II[InputInterface]
        OI[OutputInterface]
        KM[KernelModel]
    end
    
    subgraph "Support Layer"
        REL[Relationships]
        TILE[Tiling Functions]
        QONNX[QONNX Types]
    end
    
    APP --> KD
    ONNX --> KD
    USER --> KD
    
    KD --> ID
    KD --> OD
    
    ID --> II
    OD --> OI
    KD --> KM
    
    KM --> II
    KM --> OI
    
    II --> REL
    OI --> REL
    II --> TILE
    OI --> TILE
    II --> QONNX
    OI --> QONNX
    
    classDef app fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    classDef def fill:#dbeafe,stroke:#60a5fa,stroke-width:2px
    classDef model fill:#d1fae5,stroke:#10b981,stroke-width:2px
    classDef support fill:#e0e7ff,stroke:#818cf8,stroke-width:2px
    
    class APP,ONNX,USER app
    class ID,OD,KD def
    class II,OI,KM model
    class REL,TILE,QONNX support
```

## Data Hierarchy and Tiling

The system decomposes data into four hierarchical levels, each mapping to specific hardware concepts:

```
┌────────────────────────────────────────────────────────────┐
│                           TENSOR                           │
│                    Full inference data                     │
│                    e.g., 512×256 matrix                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                         BLOCK                       │   │
│  │                 Tile processed by kernel            │   │
│  │                    e.g., 64×32 tile                 │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │                    STREAM                    │   │   │
│  │  │             Data per clock cycle             │   │   │
│  │  │               e.g., 8×16 patch               │   │   │
│  │  │  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐       │   │   │
│  │  │  │E│L│E│M│E│N│T│ │ │ │ │ │ │ │ │ │ │ │       │   │   │
│  │  │  │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │       │   │   │
│  │  │  │Individual data items (e.g., INT8) │       │   │   │
│  │  │  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘       │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### Tiling Expression System

Tiling uses intuitive list-based expressions to specify how tensors decompose:

```mermaid
graph TD
    subgraph "Tiling Expression Types"
        S[Singleton: 1<br/>Fixed size of 1]
        F[Full: Colon<br/>No tiling, full dimension]
        L[Literal: 32<br/>Fixed tile size]
        P[Parameter: CH_TILES<br/>Runtime parameter]
    end
    
    subgraph "Example: Conv2D Input"
        EXPR["Array: 1, CH_TILES, :, :"]
        
        D1[Batch: 1<br/>Always singleton]
        D2[Channels: CH_TILES<br/>Parameterized]
        D3[Height: Colon<br/>Full height]
        D4[Width: Colon<br/>Full width]
        
        EXPR --> D1
        EXPR --> D2
        EXPR --> D3
        EXPR --> D4
    end
    
    classDef type fill:#7c3aed,stroke:#333,stroke-width:2px,color:#fff
    classDef example fill:#fef3c7,stroke:#f59e0b,stroke-width:1px
    
    class S,F,L,P type
    class EXPR,D1,D2,D3,D4 example
```

## SDIM System

SDIM (Streaming Dimensions) replaces ambiguous parallelism parameters with precise multi-dimensional control:

```
                        Traditional Approach
    ┌─────────────────────────────────────────────────────┐
    │  iPar = 8  (What does this mean? Channels? Width?) │
    │  Ambiguous, kernel-specific interpretation          │
    └─────────────────────────────────────────────────────┘
                               ▼
                         SDIM Approach
    ┌─────────────────────────────────────────────────────┐
    │  SDIM = [1, 8, 1, 1]  (Clear: 8 channels/cycle)    │
    │  Precise control over each dimension                 │
    └─────────────────────────────────────────────────────┘

    Example: Matrix-Vector Unit Processing
    ┌────────────────────────────────────────┐
    │  Tensor Shape: [1, 256, 14, 14]        │
    │  Block Shape:  [1, 64, 14, 14]         │
    │  SDIM:         [1, 8, 1, 1]            │
    │                                        │
    │  Result: Process 8 channels per cycle  │
    │  Cycles per block: 64/8 = 8            │
    └────────────────────────────────────────┘
```

### SDIM Configuration Flow

```mermaid
sequenceDiagram
    participant User
    participant KernelModel
    participant InputInterface
    participant Relationships
    participant OutputInterface
    
    User->>KernelModel: configure_sdim({"input": [1,8,1,1]})
    
    activate KernelModel
    KernelModel->>KernelModel: Validate targets are inputs
    
    KernelModel->>InputInterface: Apply user SDIM
    activate InputInterface
    InputInterface->>InputInterface: Validate against block dims
    InputInterface->>InputInterface: Cache invalidation
    InputInterface-->>KernelModel: Success
    deactivate InputInterface
    
    KernelModel->>Relationships: Propagate through relationships
    activate Relationships
    Relationships->>Relationships: Process EQUAL relationships
    Relationships->>Relationships: Process DEPENDENT relationships
    Relationships->>InputInterface: Update linked interfaces
    Relationships-->>KernelModel: Propagation complete
    deactivate Relationships
    
    KernelModel->>KernelModel: Validate all constraints
    
    KernelModel->>OutputInterface: Compute output rates
    activate OutputInterface
    OutputInterface->>OutputInterface: set_streaming_rate()
    OutputInterface-->>KernelModel: Rates computed
    deactivate OutputInterface
    
    KernelModel-->>User: Configuration complete
    deactivate KernelModel
```

## Component Architecture

### Class Hierarchy

```mermaid
classDiagram
    class BaseDefinition {
        <<abstract>>
        +name: str
        +validate()
        +create_model()
    }
    
    class BaseModel {
        <<abstract>>
        +name: str
        +calculate_performance_metrics()
    }
    
    class InputDefinition {
        +datatype_constraints: List[DatatypeConstraintGroup]
        +block_tiling: List[Union[int, str]]
        +stream_tiling: List[Union[int, str]]
        +create_model(datatype, tensor_shape, params)
    }
    
    class OutputDefinition {
        +datatype_constraints: List[DatatypeConstraintGroup]
        +block_tiling: List[Union[int, str]]
        +create_model(datatype, tensor_shape, params)
    }
    
    class InputInterface {
        +datatype: DataType
        +tensor_shape: Shape
        +block_shape: Shape
        +sdim: List[int]
        +configure_sdim(sdim)
        +streaming_bandwidth: int
        +validate_connection(upstream)
    }
    
    class OutputInterface {
        +datatype: DataType
        +tensor_shape: Shape
        +block_shape: Shape
        -streaming_rate: int
        +set_streaming_rate(rate)
        +production_interval: int
    }
    
    class KernelDefinition {
        +input_definitions: List[InputDefinition]
        +output_definitions: List[OutputDefinition]
        +relationships: List[DimensionRelationship]
        +create_model(type_specs, shape_specs, params)
    }
    
    class KernelModel {
        +input_models: List[InputInterface]
        +output_models: List[OutputInterface]
        +configure_sdim(sdim_config)
        +compute_output_rates()
        +get_sdim_parameters()
    }
    
    BaseDefinition <|-- InputDefinition
    BaseDefinition <|-- OutputDefinition
    BaseDefinition <|-- KernelDefinition
    
    BaseModel <|-- InputInterface
    BaseModel <|-- OutputInterface
    BaseModel <|-- KernelModel
    
    InputDefinition ..> InputInterface : creates
    OutputDefinition ..> OutputInterface : creates
    KernelDefinition ..> KernelModel : creates
    
    KernelDefinition o-- InputDefinition
    KernelDefinition o-- OutputDefinition
    KernelModel o-- InputInterface
    KernelModel o-- OutputInterface
```

### Interface Differences

Key architectural difference between input and output interfaces:

```
┌─────────────────────────┐     ┌─────────────────────────┐
│    INPUT INTERFACE      │     │   OUTPUT INTERFACE      │
├─────────────────────────┤     ├─────────────────────────┤
│ • Configurable SDIM     │     │ • NO configurable SDIM  │
│ • User sets streaming   │     │ • Kernel sets rate      │
│ • configure_sdim()      │     │ • set_streaming_rate()  │
│ • Drives data flow      │     │ • Follows data flow     │
└─────────────────────────┘     └─────────────────────────┘
          │                               ▲
          │      Data Flow Direction      │
          └───────────────────────────────┘
```

## Type System

### QONNX Integration

The system uses QONNX's DataType exclusively for hardware-software consistency:

```mermaid
graph TB
    subgraph "QONNX DataTypes"
        subgraph "Integer Types"
            INT["INT<br/>(1-32 bits)"]
            UINT["UINT<br/>(1-32 bits)"]
        end
        
        subgraph "Fixed Point"
            FIXED["FIXED<br/>(total, frac bits)"]
        end
        
        subgraph "Floating Point"
            FLOAT["FLOAT<br/>(16, 32 bits)"]
        end
        
        subgraph "Quantized"
            BINARY["BINARY"]
            BIPOLAR["BIPOLAR<br/>(-1, +1)"]
            TERNARY["TERNARY<br/>(-1, 0, +1)"]
        end
    end
    
    subgraph "Constraint Groups"
        CG1["IntegerGroup<br/>[INT, UINT]"]
        CG2["QuantizedGroup<br/>[BINARY, BIPOLAR, TERNARY]"]
        CG3["FixedGroup<br/>[FIXED]"]
        CG4["FloatGroup<br/>[FLOAT]"]
    end
    
    INT --> CG1
    UINT --> CG1
    BINARY --> CG2
    BIPOLAR --> CG2
    TERNARY --> CG2
    FIXED --> CG3
    FLOAT --> CG4
    
    classDef qonnx fill:#7c3aed,stroke:#333,stroke-width:2px,color:#fff
    classDef constraint fill:#dbeafe,stroke:#60a5fa,stroke-width:1px
    
    class INT,UINT,FIXED,FLOAT,BINARY,BIPOLAR,TERNARY qonnx
    class CG1,CG2,CG3,CG4 constraint
```

### Shape Types

```python
# Regular shape - all dimensions known
shape = Shape([1, 256, 14, 14])

# Ragged shape - CSDF phases with different sizes
ragged = RaggedShape([
    [1, 256, 14, 14],  # Phase 0
    [1, 256, 14, 12],  # Phase 1 (different width)
    [1, 256, 14, 14],  # Phase 2
])

# Shape expressions in tiling
block_tiling = [1, "CH_TILES", ":", ":"]  # Parameterized tiling
stream_tiling = [1, "SIMD", 1, 1]         # Stream subdivision
```

## Relationships and Constraints

### Relationship Types

```mermaid
graph LR
    subgraph "Structural Relationships"
        EQ[EQUAL<br/>Dimensions match]
        DEP[DEPENDENT<br/>Scaled relationship]
        COUP[COUPLED<br/>Co-dependent]
    end
    
    subgraph "Arithmetic Relationships"
        MULT[MULTIPLE<br/>A = n × B]
        DIV[DIVISIBLE<br/>A % B = 0]
    end
    
    subgraph "Comparison Relationships"
        GT[GREATER_THAN]
        LT[LESS_THAN]
        GTE[GREATER_EQUAL]
        LTE[LESS_EQUAL]
    end
    
    EQ --> EX1["input0.shape == input1.shape"]
    DEP --> EX2["output.CH = input.CH × FACTOR"]
    MULT --> EX3["weights.size = 4 × input.CH"]
    
    classDef reltype fill:#7c3aed,stroke:#333,stroke-width:2px,color:#fff
    classDef example fill:#fef3c7,stroke:#f59e0b,stroke-width:1px
    
    class EQ,DEP,COUP,MULT,DIV,GT,LT,GTE,LTE reltype
    class EX1,EX2,EX3 example
```

### Constraint Propagation

```
    Initial Configuration              After Propagation
┌──────────────────────┐         ┌──────────────────────┐
│ input0: SDIM=[1,8,1,1]│         │ input0: SDIM=[1,8,1,1]│
│ input1: SDIM=unset   │   ───►  │ input1: SDIM=[1,8,1,1]│ (EQUAL)
│ input2: SDIM=unset   │         │ input2: SDIM=[1,4,1,1]│ (DEPENDENT/2)
│ output: rate=unset   │         │ output: rate=8       │ (computed)
└──────────────────────┘         └──────────────────────┘
```

## Usage Patterns

### Basic Kernel Definition

```python
# Define a simple element-wise operation kernel
kernel_def = KernelDefinition(
    name="elementwise_add",
    input_definitions=[
        InputDefinition(
            name="input0",
            datatype_constraints=[DatatypeConstraintGroup(["INT", "UINT"], 1, 32)],
            block_tiling=[1, ":", ":", ":"],  # Process full tensor
            stream_tiling=[1, "PE", 1, 1]     # PE parallel elements
        ),
        InputDefinition(
            name="input1",
            datatype_constraints=[DatatypeConstraintGroup(["INT", "UINT"], 1, 32)],
            block_tiling=[1, ":", ":", ":"],
            stream_tiling=[1, "PE", 1, 1]
        )
    ],
    output_definitions=[
        OutputDefinition(
            name="output",
            datatype_constraints=[DatatypeConstraintGroup(["INT", "UINT"], 1, 32)],
            block_tiling=[1, ":", ":", ":"]
        )
    ],
    relationships=[
        DimensionRelationship(
            interfaces=["input0", "input1"],
            relationship_type=RelationType.EQUAL
        ),
        DimensionRelationship(
            interfaces=["input0", "output"],
            relationship_type=RelationType.EQUAL
        )
    ]
)

# Create model with concrete types
model = kernel_def.create_model(
    type_specifications={
        "input0": DataType["UINT8"],
        "input1": DataType["UINT8"],
        "output": DataType["UINT8"]
    },
    shape_specifications={
        "input0": Shape([1, 256, 14, 14]),
        "input1": Shape([1, 256, 14, 14])
    },
    parameter_values={"PE": 8}
)

# Configure streaming
model.configure_sdim({"input0": [1, 8, 1, 1]})  # Process 8 channels/cycle
# input1 automatically gets [1, 8, 1, 1] due to EQUAL relationship
# output streaming rate automatically computed as 8
```

### Matrix Multiplication Kernel

```python
# More complex example with dependent relationships
matmul_def = KernelDefinition(
    name="matmul",
    input_definitions=[
        InputDefinition(
            name="activation",
            datatype_constraints=[DatatypeConstraintGroup(["INT", "UINT"], 1, 8)],
            block_tiling=[1, "TILE_K"],
            stream_tiling=[1, "SIMD"]
        ),
        InputDefinition(
            name="weights",
            datatype_constraints=[DatatypeConstraintGroup(["INT", "UINT"], 1, 8)],
            block_tiling=["TILE_M", "TILE_K"],
            stream_tiling=["PE", "SIMD"]
        )
    ],
    output_definitions=[
        OutputDefinition(
            name="output",
            datatype_constraints=[DatatypeConstraintGroup(["INT", "UINT"], 8, 32)],
            block_tiling=[1, "TILE_M"]
        )
    ],
    relationships=[
        DimensionRelationship(
            interfaces=["activation", "weights"],
            relationship_type=RelationType.DEPENDENT,
            dimension_indices={"activation": 1, "weights": 1},  # K dimensions
            parameters={"factor": 1}  # Must match
        ),
        DimensionRelationship(
            interfaces=["weights", "output"],
            relationship_type=RelationType.DEPENDENT,
            dimension_indices={"weights": 0, "output": 1},  # M dimensions
            parameters={"factor": 1}
        )
    ],
    parameters={
        "TILE_M": Parameter(default=64),
        "TILE_K": Parameter(default=64),
        "PE": Parameter(default=16),
        "SIMD": Parameter(default=8)
    }
)
```

### SDIM Parameter Discovery

```python
# The system can analyze which interfaces need configuration
sdim_params = model.get_sdim_parameters()
# Returns: {
#   "free": ["activation"],        # User must configure
#   "constrained": ["weights"],    # Will be set by relationships
#   "hidden": []                   # Not visible (internal use)
# }
```

## Performance Considerations

### Performance Metrics

```mermaid
graph TD
    subgraph "Key Metrics"
        BW[Streaming Bandwidth<br/>Elements/cycle]
        BWB[Bandwidth Bits<br/>Bits/cycle]
        II[Initiation Interval<br/>Cycles/tensor]
        EBW[Effective Bandwidth<br/>MB/s @ clock freq]
    end
    
    subgraph "Calculation"
        CALC1["bandwidth = Π(sdim)"]
        CALC2["bits = bandwidth × datatype.bitwidth"]
        CALC3["interval = tensor_size / bandwidth"]
        CALC4["eff_bw = bits × freq / 8e6"]
    end
    
    BW --> CALC1
    BWB --> CALC2
    II --> CALC3
    EBW --> CALC4
    
    classDef metric fill:#7c3aed,stroke:#333,stroke-width:2px,color:#fff
    classDef calc fill:#dbeafe,stroke:#60a5fa,stroke-width:1px
    
    class BW,BWB,II,EBW metric
    class CALC1,CALC2,CALC3,CALC4 calc
```

### Optimization Strategies

1. **SDIM Tuning**: Balance parallelism across dimensions
2. **Block Sizing**: Match BRAM/URAM capacities
3. **Stream Width**: Align with AXI bus widths (typically 512/1024 bits)
4. **Pipeline Depth**: Consider initiation interval vs latency trade-offs

### Caching Architecture

```
┌─────────────────────────────────────────────┐
│            InputInterface                   │
├─────────────────────────────────────────────┤
│ _cached_metrics: Dict[str, Any]            │
│                                             │
│ Key Operations:                             │
│ • configure_sdim() → invalidate cache      │
│ • streaming_bandwidth → check cache first   │
│ • Lazy evaluation of expensive metrics      │
└─────────────────────────────────────────────┘
```

## Best Practices

### 1. Start with Definitions
Always design your kernel interfaces as definitions first. This enforces thinking about constraints and relationships upfront.

### 2. Use Type Groups
Prefer constraint groups over specific types in definitions:
```python
# Good: Flexible
DatatypeConstraintGroup(["INT", "UINT"], 1, 32)

# Less flexible: Fixed type
# Only use when algorithm requires specific type
```

### 3. Explicit Relationships
Document all interface dependencies through relationships. This enables automatic validation and configuration.

### 4. Parameter Naming
Use semantic parameter names that indicate their purpose:
- `PE`: Processing elements
- `SIMD`: SIMD lanes  
- `CH_TILES`: Channel tiling factor
- `TILE_M`, `TILE_K`: Matrix dimensions

### 5. Validate Early
Call `validate()` on definitions before creating models to catch errors early in the design process.

## Integration with Brainsmith

The dataflow module serves as the foundation for:

1. **RTL Parser**: Extracts metadata to create KernelDefinitions
2. **FINN Integration**: Maps to HWCustomOp through AutoHWCustomOp base class
3. **Code Generation**: Templates use definitions to generate implementation code
4. **Optimization**: Performance models guide design space exploration

This architecture enables Brainsmith to automatically generate optimized FPGA accelerators while maintaining a clean separation between algorithmic intent and hardware implementation details.