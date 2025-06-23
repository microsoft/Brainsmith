# Brainsmith Dataflow Framework Architecture

## Executive Summary

The Brainsmith dataflow framework provides a comprehensive modeling and optimization system for FPGA-based AI accelerators. It represents hardware kernels as streaming dataflow graphs, enabling automated design space exploration, performance analysis, and resource optimization. The framework supports advanced features like cyclo-static dataflow patterns, real-time scheduling analysis, and multi-objective optimization.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dataflow Framework                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │    Core     │  │     ADFG     │  │       DSE           │   │
│  │   Types     │  │  Scheduling  │  │   Exploration       │   │
│  │             │  │              │  │                     │   │
│  │ • DataType  │  │ • Actor      │  │ • ConfigSpace      │   │
│  │ • Shape     │  │ • CSDF       │  │ • Evaluator        │   │
│  │ • Interface │  │ • SRTA       │  │ • Explorer         │   │
│  │ • Kernel    │  │ • Buffer ILP │  │ • Constraints      │   │
│  │ • Graph     │  │              │  │                     │   │
│  │ • Pragma    │  │              │  │                     │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Type System (`core/types.py`)

The foundation of the framework, providing:

- **DataType**: Represents hardware data types with bit-width and signedness
  - Supports FINN-style string parsing ("INT8", "UINT16", "BIPOLAR")
  - Special handling for binary and bipolar types
  - Extensible to floating-point types

- **Shape System**: Flexible shape representation
  - `Shape`: Standard tensor dimensions as tuples
  - `RaggedShape`: Variable shapes for CSDF patterns
  - Utilities for shape manipulation, broadcasting, and tiling

- **InterfaceDirection**: Enumeration for port types
  - INPUT: Streaming input data
  - OUTPUT: Streaming output data
  - WEIGHT: Model parameters
  - CONFIG: Runtime configuration

### 2. Interface Model (`core/interface.py`)

Represents hardware streaming interfaces with hierarchical data organization:

```
Tensor Level    [batch, channels, height, width]
     ↓
Block Level     [c_block, h_block, w_block]  (per kernel computation)
     ↓
Stream Level    [c_stream, pixels]           (per clock cycle)
```

**Key Features:**
- **Hierarchical Dimensions**: Separate tensor, block, and stream levels
- **CSDF Support**: Variable block sizes for cyclo-static patterns
- **Sparsity Modeling**: Skip probabilities for sparse data
- **Validation**: Ensures dimensional compatibility and tiling constraints
- **Bandwidth Calculation**: Automatic computation of interface bandwidth

**Example:**
```python
Interface(
    name="input",
    direction=InterfaceDirection.INPUT,
    dtype=DataType.from_string("INT8"),
    tensor_dims=(1, 32, 224, 224),    # ImageNet-like input
    block_dims=(32, 14, 14),          # Process 14x14 patches
    stream_dims=(4, 2)                # 4 channels × 2 pixels per cycle
)
```

### 3. Kernel Abstraction (`core/kernel.py`)

Represents hardware accelerator modules with:

**Core Properties:**
- **Identity**: Name and hardware module reference
- **Interfaces**: Collection of input/output/weight/config ports
- **Timing**: Latency, initiation interval, pipeline depth
- **Resources**: DSP, BRAM, LUT estimates
- **Constraints**: Pragma-based architectural requirements

**Timing Model:**
```
Latency = priming_cycles + calculation_latency + flush_cycles
Throughput = batch_size / (execution_ii * n_executions)
```

**Key Methods:**
- `validate()`: Comprehensive validation including pragma constraints
- `apply_parallelism()`: Transform kernel with new stream dimensions
- `estimate_resources()`: Predict FPGA resource usage
- `to_adfg_rates()`: Convert to scheduling representation

### 4. Graph Model (`core/graph.py`)

Manages directed acyclic graphs of connected kernels:

**Features:**
- **NetworkX Backend**: Leverages graph algorithms
- **Edge Management**: Tracks connections with buffer depths
- **Validation**: Ensures DAG property and connection validity
- **Analysis**: Critical path, topological sort, subgraph extraction
- **Visualization**: Text and dictionary representations

**Example Graph Construction:**
```python
graph = DataflowGraph()
graph.add_kernel(conv_kernel)
graph.add_kernel(pool_kernel)
graph.add_edge("conv", "output", "pool", "input", buffer_depth=1024)
graph.validate()
```

### 5. Pragma System (`core/pragma.py`)

Constraint specification language for hardware requirements:

**Pragma Types:**
- **TiePragma**: Equality constraints (`TIE mat[1] vec`)
- **ConstrPragma**: Unary constraints (`CONSTR vec % BURST`)

**Expression Language:**
- Interface dimension access: `mat[0]`
- Total size: `vec` (product of dimensions)
- Arithmetic: `mat[0] * SIMD`
- Environment variables: `BURST`, `SIMD`

**Use Cases:**
- Enforce architectural constraints
- Ensure memory alignment
- Couple related parameters

## ADFG Scheduling Framework

### 1. Actor Model (`adfg/actor.py`)

Converts kernels to dataflow actors for scheduling analysis:

**Features:**
- **Rate Patterns**: Production/consumption rates per phase
- **Timing Info**: WCET, pipeline costs
- **Repetition Vector**: Ensures rate consistency
- **Resource Requirements**: For allocation decisions

### 2. CSDF Analysis (`adfg/csdf.py`)

Cyclo-Static Dataflow utilities for complex patterns:

**Algorithms:**
- **Buffer Sizing**: Event-based simulation for minimum buffers
- **Hyperperiod**: LCM computation for cyclic behavior
- **Phase Scheduling**: Topological sort with phase awareness
- **Memory Allocation**: First-fit algorithm for buffer placement

### 3. SRTA Scheduler (`adfg/scheduler.py`)

Real-time scheduling analysis and period assignment:

**Features:**
- **Schedulability Analysis**: Response time analysis
- **Period Assignment**: Binary search for minimal periods
- **Priority Assignment**: Deadline-monotonic ordering
- **Optimization Objectives**: Hyperperiod, utilization, slack

**Algorithm:**
```
For each actor:
  1. Assign priority (deadline-monotonic)
  2. Compute response time (fixed-point iteration)
  3. Check schedulability (response ≤ deadline)
  4. Optimize period via binary search
```

### 4. Buffer ILP (`adfg/buffer_ilp.py`)

Integer Linear Programming for optimal buffer sizing:

**Objectives:**
- Minimize total memory usage
- Support multi-bank architectures
- Ensure deadlock freedom
- Handle initial tokens

## Design Space Exploration

### 1. Configuration Management (`dse/config.py`)

**ParallelismConfig**: Manages interface parallelism settings
- Global and per-interface parallelism
- Intelligent factorization to match tensor structure
- Resource estimation with scaling
- Pragma validation

**ConfigurationSpace**: Defines exploration space
- Coupling constraints for related interfaces
- Pruning based on divisibility
- Efficient enumeration of valid configs

### 2. Performance Evaluation (`dse/evaluator.py`)

Comprehensive performance modeling:

**Metrics Computed:**
- **Throughput**: Based on hyperperiod and batch size
- **Latency**: Pipeline fill + critical path + drain
- **Resources**: Aggregated across kernels with scaling
- **Power**: Static + dynamic + bandwidth components
- **Utilization**: Kernel activity analysis

**Sparsity Modeling:**
```
effective_throughput = base_throughput / (1 - skip_probability)
```

### 3. Exploration Engine (`dse/explorer.py`)

Orchestrates the optimization process:

**Workflow:**
1. Generate configuration space
2. Apply configurations to graph
3. Validate and schedule
4. Evaluate performance
5. Check constraints
6. Find Pareto-optimal solutions

**Optimization Features:**
- Multi-objective (throughput, latency, power)
- Constraint satisfaction (resources, performance)
- Caching for efficiency
- Progress callbacks

## Usage Patterns

### Basic Kernel Definition
```python
kernel = Kernel(
    name="conv2d",
    hw_module="conv2d_v1",
    interfaces=[
        Interface("input", INPUT, "INT8", (1,3,224,224), (3,7,7), (3,1)),
        Interface("weights", WEIGHT, "INT8", (64,3,7,7), (3,7,7), (1,1)),
        Interface("output", OUTPUT, "INT16", (1,64,224,224), (64,1,1), (4,1))
    ],
    latency_cycles=(49, 49),
    calculation_ii=1,
    pragmas=[TiePragma("input[1]", "weights[1]")]
)
```

### Graph Construction
```python
graph = DataflowGraph()
graph.add_kernel(dma_read)
graph.add_kernel(conv)
graph.add_kernel(dma_write)

graph.add_edge("dma_read", "stream_out", "conv", "input")
graph.add_edge("conv", "output", "dma_write", "stream_in")

graph.validate()
```

### Design Space Exploration
```python
constraints = DSEConstraints(
    min_throughput=30.0,  # FPS
    max_resources={"DSP": 2000, "BRAM": 1000}
)

space = ConfigurationSpace(
    graph,
    interface_parallelisms={"conv.input": [1, 4, 8, 16]},
    coupling_groups=[["conv.input", "conv.output"]]
)

explorer = DesignSpaceExplorer(graph)
results = explorer.explore(space, constraints)
pareto = explorer.find_pareto_optimal(results)
```

## Design Principles

1. **Hierarchical Abstraction**: Separate concerns at tensor/block/stream levels
2. **Immutability**: Use frozen dataclasses and functional updates
3. **Validation-First**: Comprehensive checks at every level
4. **Extensibility**: Plugin architecture for new kernel types
5. **Performance Modeling**: Accurate prediction without synthesis
6. **Standards Compliance**: FINN compatibility, SystemVerilog pragmas

## Key Innovations

1. **Unified CSDF Support**: Seamlessly handles static and cyclo-static patterns
2. **Hierarchical Tiling**: Natural representation of hardware parallelism
3. **Pragma Constraints**: Flexible architectural requirement specification
4. **Integrated DSE**: Automated optimization with real-world constraints
5. **Sparsity-Aware**: Models performance benefits of sparse data
6. **Multi-Bank Memory**: Supports complex memory architectures

## Future Extensions

1. **Dynamic Dataflow**: Support for data-dependent rates
2. **Hierarchical Graphs**: Nested kernel compositions
3. **Power Modeling**: More detailed power estimation
4. **Placement Aware**: Consider FPGA placement in optimization
5. **ML-Guided DSE**: Learn from previous explorations
6. **Streaming Debugger**: Trace token flow through graph

## Conclusion

The Brainsmith dataflow framework provides a solid foundation for modeling, analyzing, and optimizing FPGA-based AI accelerators. Its hierarchical abstraction, comprehensive validation, and integrated optimization capabilities enable rapid development of high-performance streaming architectures. The framework successfully bridges the gap between high-level dataflow models and low-level hardware implementation details.