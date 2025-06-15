# Interface-Wise Dataflow Modeling: Mathematical Framework

## Introduction

Interface-Wise Dataflow Modeling is Brainsmith's mathematical framework for accurately and robustly representing computational nodes in dataflow accelerators. This document focuses on the core mathematical principles that enable precise modeling of hardware operations through their interface characteristics.

## Dataflow Node Representation

A dataflow accelerator node is completely characterized by its interfaces and their mathematical relationships:

```mermaid
graph TD
    subgraph "Dataflow Node"
        A[Input Interfaces]
        B[Output Interfaces] 
        C[Weight Interfaces]
        D[Computation Core]
    end
    
    A --> D
    C --> D
    D --> B
    
    subgraph "Mathematical Model"
        E[Tensor Flow Equations]
        F[Timing Relationships]
        G[Resource Requirements]
    end
    
    A -.-> E
    B -.-> E
    C -.-> E
    D -.-> F
    D -.-> G
```

## Mathematical Foundation

### Interface-Wise Decomposition

Every dataflow node is mathematically decomposed into its constituent interfaces, each representing a distinct data flow:

```mermaid
graph LR
    subgraph "Mathematical Decomposition"
        A["Node Operation F"] --> B["Interface Set I"]
        B --> C["I_input union I_output union I_weight"]
    end
    
    subgraph "Interface Properties"
        D["Tensor Flow T_i"]
        E["Timing Constraints Delta_t_i"] 
        F["Resource Usage R_i"]
    end
    
    C --> D
    C --> E
    C --> F
```

**Key Principle**: `Node Behavior = ∑(Interface Behaviors + Interface Interactions)`

### Tensor Flow Mathematics

Each interface represents a structured tensor flow with four mathematical levels:

```mermaid
graph TD
    A[Level 1: Tensor Dimensions<br/>T in R^n dimensions] --> B[Level 2: Block Dimensions<br/>B in R^n blocks]
    B --> C[Level 3: Stream Dimensions<br/>S in R^n streams]
    C --> D[Level 4: Element Type<br/>E in DataType]
    
    A -.-> E1[Mathematical Constraint:<br/>tensor_dims = N × block_dims]
    B -.-> E2[Mathematical Constraint:<br/>block_dims = M × stream_dims]
    C -.-> E3[Hardware Constraint:<br/>stream_width ≤ AXI_width]
```

**Fundamental Axioms:**
1. **Completeness**: `tensor_dims[i] = num_blocks[i] × block_dims[i]`
2. **Streamability**: `block_dims[i] = num_cycles[i] × stream_dims[i]`
3. **Hardware Feasibility**: `stream_width = stream_dims[0] × datatype.bitwidth()`

## Interface Mathematical Model

### DataflowInterface: Complete Mathematical Specification

Each interface is a mathematical object with complete tensor flow specification:

```mermaid
classDiagram
    class DataflowInterface {
        +tensor_dims: List[int]
        +block_dims: List[int]  
        +stream_dims: List[int]
        +dtype: DataType
        
        +invariant_1() tensor_dims = N × block_dims
        +invariant_2() block_dims = M × stream_dims
        +calculate_cII() product(block_dims[i] / stream_dims[i])
        +calculate_flow_rate() stream_dims[0] × dtype.bitwidth()
    }
```

**Mathematical Properties:**
- **Tensor Conservation**: Total elements preserved across all levels
- **Block Alignment**: All dimensions evenly divisible 
- **Stream Feasibility**: Hardware can process stream_dims elements per cycle
- **Timing Predictability**: cII (Calculation Initiation Interval) deterministically calculated

### Interface Timing Mathematics

Each interface has precisely defined timing characteristics:

```mermaid
graph TD
    subgraph "Timing Model"
        A["Block Processing Time<br/>cII = product of bi/si"] 
        B["Memory Access Time<br/>mII = memory_cycles"]
        C["Execution Interval<br/>eII = max of cII and mII"]
    end
    
    subgraph "Flow Equations"
        D["Input Flow Rate<br/>R_in = s_in times f_clk"]
        E["Output Flow Rate<br/>R_out = s_out times f_clk"] 
        F["Processing Rate<br/>R_proc = 1/eII times f_clk"]
    end
    
    A --> C
    B --> C
    C --> F
    D --> F
    E --> F
```

**Key Equations:**
- `cII = ∏ᵢ(block_dims[i] / stream_dims[i])` (cycles per block)
- `throughput = stream_dims[0] / cII` (elements per cycle)
- `bandwidth = throughput × dtype.bitwidth()` (bits per cycle)

## Node-Level Mathematical Model

### DataflowModel: Multi-Interface System

A dataflow node is represented as a system of mathematically related interfaces:

```mermaid
graph TD
    subgraph "Mathematical Node Model"
        A["Input Interface Set<br/>I = set of I1, I2, In"]
        B["Output Interface Set<br/>O = set of O1, O2, Om"]
        C["Weight Interface Set<br/>W = set of W1, W2, Wk"]
        D["Node Function<br/>F maps I and W to O"]
    end
    
    subgraph "System Equations"
        E["Conservation Laws<br/>sum input_elements = sum output_elements"]
        F["Timing Constraints<br/>max cII_i ≤ min cII_o"]
        G["Resource Bounds<br/>sum R_i ≤ R_max"]
    end
    
    A --> E
    B --> E
    A --> F
    B --> F
    A --> G
    C --> G
```

**Mathematical Invariants:**
1. **Data Conservation**: `∑elements_in = ∑elements_out` (for non-reducing operations)
2. **Temporal Consistency**: All interfaces must have compatible timing
3. **Resource Feasibility**: Total resource usage must be implementable

### Parallelism Mathematics

Parallelism parameters define the interface relationships and system performance:

```mermaid
graph LR
    subgraph "Parallelism Parameters"
        A["iPar: Input Parallelism<br/>Elements per cycle per input"]
        B["wPar: Weight Parallelism<br/>Weight accesses per cycle"]
    end
    
    subgraph "Derived Quantities"
        C["Stream Dimensions<br/>s_i = function of iPar and wPar"]
        D["Processing Rate<br/>R = iPar divided by cII"]
        E["Bottleneck Analysis<br/>beta = min of R_i"]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
```

**Parallelism Equations:**
- `stream_dims[0] = iPar` for input interfaces
- `stream_dims[0] = wPar × scaling_factor` for weight interfaces  
- `stream_dims[0] = min(input_parallelism, weight_parallelism)` for outputs
- `node_throughput = bottleneck_interface.throughput`

## Performance Analysis Framework

### Initiation Interval Mathematics

The framework calculates precise performance metrics through mathematical analysis:

```mermaid
graph TD
    subgraph "Interface-Level Metrics"
        A["cII_1 = product of b_1i/s_1i"]
        B["cII_2 = product of b_2i/s_2i"]  
        C["cII_n = product of b_ni/s_ni"]
    end
    
    subgraph "Node-Level Metrics"
        D["Node cII = max of cII_i"]
        E["Node eII = cII times memory_factor"]
        F["Node Latency L = eII times total_blocks"]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
```

**Performance Equations:**
- `cII_interface = ∏ᵢ(block_dims[i] / stream_dims[i])` (cycles per block per interface)
- `cII_node = max(cII_interface for all interfaces)` (node bottleneck)
- `throughput_node = 1 / cII_node` (blocks per cycle)
- `latency_total = cII_node × total_num_blocks` (total inference cycles)

### Mathematical Validation Framework

The system ensures mathematical correctness through comprehensive validation:

```mermaid
graph LR
    subgraph "Validation Levels"
        A[Interface Invariants<br/>Tensor/Block/Stream consistency]
        B[Node Invariants<br/>Multi-interface compatibility]
        C[System Invariants<br/>Resource/timing feasibility]
    end
    
    subgraph "Error Detection"
        D[Dimension Misalignment]
        E[Timing Violations]
        F[Resource Overflow]
    end
    
    A --> D
    B --> E  
    C --> F
```

**Validation Rules:**
1. `∀i: tensor_dims[i] % block_dims[i] == 0` (perfect chunking)
2. `∀i: block_dims[i] % stream_dims[i] == 0` (perfect streaming)  
3. `∀interfaces: compatible_timing(cII_values)` (temporal consistency)
4. `total_resources ≤ available_resources` (resource feasibility)

## Robustness and Accuracy Properties

### Mathematical Soundness

The framework ensures robust node representation through mathematical foundations:

```mermaid
graph TD
    subgraph "Soundness Properties"
        A[Completeness<br/>All tensor flows represented]
        B[Consistency<br/>No contradictory constraints]
        C[Decidability<br/>All properties computable]
    end
    
    subgraph "Accuracy Properties"  
        D[Precise Timing<br/>Exact cycle calculation]
        E[Exact Resources<br/>Deterministic estimation]
        F[Faithful Modeling<br/>Hardware-accurate representation]
    end
    
    A --> D
    B --> E
    C --> F
```

**Robustness Guarantees:**
1. **No Invalid Configurations**: Mathematical constraints prevent impossible tensor chunking
2. **Deterministic Behavior**: Same inputs always produce same outputs
3. **Composability**: Multiple nodes can be reliably composed into graphs
4. **Hardware Fidelity**: Model accurately reflects actual hardware behavior

### Error Prevention Through Mathematics

```mermaid
graph LR
    subgraph "Common Errors Prevented"
        A[Misaligned Tensors]
        B[Incompatible Parallelism]  
        C[Resource Overflow]
        D[Timing Violations]
    end
    
    subgraph "Mathematical Prevention"
        E[Modular Arithmetic Checks]
        F[Constraint Satisfaction]
        G[Bounds Analysis]
        H[Temporal Logic]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
```

## Case Study: Matrix Multiplication Node

### Legacy vs Modern Parallelism Systems

The matrix multiplication node demonstrates the evolution from FINN's legacy PE/SIMD system to Brainsmith's modern iPar/wPar framework:

```mermaid
graph TD
    subgraph "Legacy FINN System"
        A1["SIMD: Input Parallelism<br/>Elements read per cycle"]
        B1["PE: Processing Elements<br/>Output computations per cycle"]
        C1["Hardcoded Mapping<br/>input_width = SIMD × dtype<br/>output_width = PE × dtype"]
    end
    
    subgraph "Modern Brainsmith System"  
        A2["iPar: Input Parallelism<br/>stream_dims[0] for inputs"]
        B2["wPar: Weight Parallelism<br/>stream_dims[0] for weights"]
        C2["Derived Mapping<br/>PE = wPar (weight parallelism)<br/>SIMD = iPar (input parallelism)"]
    end
    
    A1 -.-> A2
    B1 -.-> B2
    C1 -.-> C2
```

### Interface Mathematical Specification

```mermaid
graph TD
    subgraph "Interface Specification"
        A["Input: (N, K) → blocks (1, iPar)"]
        B["Weight: (K, M) → blocks (iPar, wPar)"]  
        C["Output: (N, M) → blocks (1, wPar)"]
    end
    
    subgraph "Legacy Translation"
        D["SIMD = iPar ✓"]
        E["PE = wPar ✓"]
        F["Processing: iPar×wPar MAC ops/cycle"]
    end
    
    subgraph "Mathematical Relationships"
        G["K_input = K_weight (dimension consistency)"]
        H["M_weight = M_output (dimension consistency)"]
        I["iPar ≤ K (parallelism constraint)"]
        J["wPar ≤ M (parallelism constraint)"]
    end
    
    A --> D
    B --> E
    A --> F
    B --> F
    A --> G
    B --> G
    B --> H
    C --> H
    A --> I
    B --> J
```

### Parallelism Mapping Mathematics

**Legacy FINN Attributes → Modern Interface Properties:**

| Legacy Attribute | Modern Equivalent | Mathematical Relationship |
|------------------|-------------------|---------------------------|
| `SIMD` | `iPar` | `input_interface.stream_dims[0] = iPar` |
| `PE` | `wPar` | `weight_interface.stream_dims[0] = wPar` |
| `inputDataType` | `input_interface.dtype` | Direct datatype mapping |
| `weightDataType` | `weight_interface.dtype` | Direct datatype mapping |
| `outputDataType` | `output_interface.dtype` | Direct datatype mapping |

**Performance Model Translation:**
- **Legacy**: `cII = (K/SIMD) × (M/PE)` cycles per output block
- **Modern**: `cII = (K/iPar) × (M/wPar)` cycles per output block
- **Equivalence**: When `SIMD = iPar` and `PE = wPar`, performance is identical

### Concrete Example: BERT Matrix Multiplication

**Problem**: Multiply input `[128, 768]` with weight `[768, 256]` to get output `[128, 256]`

```mermaid
graph LR
    subgraph "Legacy Configuration"
        A1["SIMD = 8<br/>PE = 4<br/>Hardcoded parallelism"]
    end
    
    subgraph "Modern Configuration"
        A2["iPar = 8<br/>wPar = 4<br/>Interface-derived parallelism"]
    end
    
    subgraph "Identical Results"
        B["Input: 8 elements/cycle<br/>Weight: 32 elements/cycle<br/>Output: 4 elements/cycle<br/>Performance: 6144 cycles"]
    end
    
    A1 --> B
    A2 --> B
```

**Mathematical Model:**
- `input_elements_per_cycle = iPar = 8` (replaces SIMD)
- `weight_elements_per_cycle = iPar × wPar = 32` (derived from interface relationships)
- `output_elements_per_cycle = wPar = 4` (replaces PE) 
- `cII = (768/8) × (256/4) = 96 × 64 = 6144` cycles per [128,256] output
- `total_latency = cII × 1 = 6144` cycles for complete matrix (batch=1)

**Key Advantages of Modern System:**
1. **Mathematical Consistency**: iPar/wPar derived from interface mathematics, not hardcoded
2. **Flexible Mapping**: Can handle complex multi-interface operations systematically  
3. **Validation**: Mathematical constraints prevent invalid parallelism configurations
4. **Composability**: Interface-wise modeling enables reliable multi-node composition

### Validation and Accuracy

```mermaid
sequenceDiagram
    participant User
    participant Framework
    participant Validator
    participant Hardware
    
    User->>Framework: Define MatMul(N=128, K=768, M=256)
    Framework->>Validator: Check tensor_dims % block_dims
    Validator->>Framework: ✓ Valid chunking
    Framework->>Validator: Check SIMD=8, PE=4 feasibility  
    Validator->>Framework: ✓ Hardware compatible
    Framework->>Hardware: Generate cII=768/8 × 256/4 = 6144
    Hardware-->>User: Accurate timing prediction
```

## Conclusion

Interface-Wise Dataflow Modeling provides a mathematically rigorous framework for accurately and robustly representing computational nodes in dataflow accelerators. The framework's key contributions are:

### Mathematical Foundation
- **Complete Specification**: Every interface fully characterized by tensor flow mathematics
- **Invariant Enforcement**: Mathematical constraints prevent invalid configurations  
- **Deterministic Calculation**: Precise timing and resource predictions
- **Compositional Properties**: Nodes can be reliably combined into larger systems

### Robustness Properties  
- **Error Prevention**: Mathematical validation catches problems early
- **Hardware Fidelity**: Accurate representation of actual hardware behavior
- **Scalable Validation**: Constraint checking scales to complex operations
- **Predictable Performance**: Timing calculations match hardware reality

### Accuracy Benefits
- **Exact Metrics**: No approximation in core calculations
- **Resource Precision**: Deterministic memory and compute requirements
- **Timing Accuracy**: Cycle-accurate performance prediction
- **Interface Completeness**: All data flows explicitly modeled

This mathematical framework enables reliable, automatic generation of hardware accelerator nodes with guaranteed correctness and predictable performance, forming the foundation for robust dataflow accelerator design and optimization.