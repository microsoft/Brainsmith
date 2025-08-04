# Unified Kernel Modeling Framework: Introduction and User's Guide

## Welcome to the Unified Kernel Modeling Framework

The Unified Kernel Modeling Framework is Brainsmith's foundational technology for designing, analyzing, and optimizing FPGA accelerators for AI workloads. It provides a mathematically rigorous yet practical approach to modeling hardware kernels through their interfaces, enabling automated design space exploration and guaranteed-correct implementations.

### Why Use This Framework?

Traditional hardware design requires manual specification of parallelism, timing, and resource usage. This framework automates these decisions while ensuring:

- **Correctness by Construction**: Mathematical constraints prevent invalid configurations
- **Optimal Performance**: Automated exploration finds Pareto-optimal designs
- **Hardware Accuracy**: Models precisely match actual FPGA behavior
- **Rapid Development**: Go from algorithm to optimized hardware in hours, not months

## Getting Started

### Installation

The framework is part of the Brainsmith platform:

```bash
# Clone the repository
git clone https://github.com/microsoft/brainsmith-2.git
cd brainsmith-2

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m pytest brainsmith/core/dataflow/tests/
```

### Your First Example

Let's create a simple vector addition accelerator:

```python
from brainsmith.core.dataflow import (
    Interface, InterfaceDirection, INT16,
    Kernel, DataflowGraph
)

# Define interfaces
vec_a = Interface(
    name="vec_a",
    direction=InterfaceDirection.INPUT,
    dtype=INT16,
    tensor_dims=(1024,),    # Full vector size
    block_dims=(64,),       # Process 64 elements at a time
    stream_dims=(8,)        # Stream 8 elements per cycle
)

vec_b = Interface(
    name="vec_b", 
    direction=InterfaceDirection.INPUT,
    dtype=INT16,
    tensor_dims=(1024,),
    block_dims=(64,),
    stream_dims=(8,)
)

vec_out = Interface(
    name="vec_out",
    direction=InterfaceDirection.OUTPUT,
    dtype=INT16,
    tensor_dims=(1024,),
    block_dims=(64,),
    stream_dims=(8,)
)

# Create kernel
vadd = Kernel(
    name="vector_add",
    interfaces=[vec_a, vec_b, vec_out],
    latency_cycles=(10, 8),  # (worst_case, average)
    resources={"DSP": 0, "LUT": 500, "BRAM": 0}
)

# Build graph
graph = DataflowGraph()
graph.add_kernel(vadd)
graph.validate()

print(f"Vector add throughput: {vadd.calculate_throughput():.2f} GOPS")
```

## Core Concepts

### Interface-Wise Dataflow Modeling

The framework's key innovation is modeling hardware kernels through their interfaces rather than their internal implementation. Each kernel is viewed as a "black box" with well-defined data flows:

```
┌─────────────────────────┐
│                         │
│    Hardware Kernel      │
│                         │
├─── Input Interfaces ────┤ ← Streaming data in
│                         │
├─── Weight Interfaces ───┤ ← Parameters/weights
│                         │
├─── Output Interfaces ───┤ → Results out
│                         │
└─────────────────────────┘
```

This approach enables:
- **Composability**: Kernels connect like LEGO blocks
- **Optimization**: Change parallelism without changing functionality
- **Validation**: Mathematically verify correctness

### The Three-Tier Data Hierarchy

Every interface defines how data flows at three levels:

#### 1. Tensor Level
The complete data structure (e.g., entire matrix, full feature map):
```python
tensor_dims=(256, 256)  # 256×256 matrix
```

#### 2. Block Level
The unit of computation - minimum data for one calculation:
```python
block_dims=(16, 16)  # Process 16×16 tiles
```

#### 3. Stream Level
Data transferred per clock cycle:
```python
stream_dims=(4, 4)  # Transfer 4×4 elements/cycle
```

**Key Relationship**: 
```
Tensor = N × Blocks
Block = M × Streams
```

### Interface Types

The framework supports four interface types:

| Type | Purpose | Direction | Example |
|------|---------|-----------|---------|
| **INPUT** | Streaming activation data | Kernel ← Memory | Feature maps |
| **OUTPUT** | Streaming results | Kernel → Memory | Predictions |
| **WEIGHT** | Model parameters | Kernel ← Memory | Conv filters |
| **CONFIG** | Control registers | CPU → Kernel | Hyperparameters |

## Basic Usage

### Creating Interfaces

Interfaces define the contract between kernels and the system:

```python
from brainsmith.core.dataflow import Interface, InterfaceDirection, INT8

# Simple 1D interface
feature = Interface(
    name="features",
    direction=InterfaceDirection.INPUT,
    dtype=INT8,
    tensor_dims=(1, 512),     # Batch=1, Features=512
    block_dims=(1, 64),       # Process 64 features at once
    stream_dims=(1, 8)        # Stream 8 features per cycle
)

# 2D convolution input
image = Interface(
    name="image",
    direction=InterfaceDirection.INPUT,
    dtype=INT8,
    tensor_dims=(224, 224, 3),   # H×W×C format
    block_dims=(8, 8, 3),        # 8×8 patches, all channels
    stream_dims=(1, 1, 3)        # One pixel per cycle
)

# Variable-rate interface (CSDF)
adaptive = Interface(
    name="sparse_data",
    direction=InterfaceDirection.INPUT,
    dtype=INT8,
    tensor_dims=(1024,),
    block_dims=[(32,), (64,), (32,)],  # Variable blocks
    stream_dims=(8,),
    skip_prob=[0.5, 0.0, 0.5]  # 50% sparsity in phases 1&3
)
```

### Defining Kernels with Pragmas

Kernels encapsulate computation with constraints:

```python
from brainsmith.core.dataflow import Kernel, TiePragma, ConstrPragma

# Matrix multiply kernel
matmul = Kernel(
    name="MatMul_256",
    interfaces=[
        Interface("vec_in", InterfaceDirection.INPUT, INT16, 
                 (1, 256), (1, 256), (1, 16)),
        Interface("mat_in", InterfaceDirection.WEIGHT, INT16,
                 (256, 256), (16, 16), (16, 16)),
        Interface("vec_out", InterfaceDirection.OUTPUT, INT16,
                 (1, 256), (1, 16), (1, 16))
    ],
    latency_cycles=(1000, 800),
    pragmas=[
        # Matrix columns must equal vector size
        TiePragma("mat_in[0]", "vec_in[1]"),
        # Vector size must be divisible by SIMD width
        ConstrPragma("vec_in[1]", "%", "SIMD_WIDTH"),
        # At least 64 elements
        ConstrPragma("vec_in[1]", ">=", 64)
    ],
    pragma_env={"SIMD_WIDTH": 16},
    resources={"DSP": 256, "BRAM": 32}
)

# Validate constraints
matmul.validate()  # Throws if constraints violated
```

### Building Dataflow Graphs

Connect kernels to form complete accelerators:

```python
# Create graph
graph = DataflowGraph()

# Add kernels
graph.add_kernel(preprocessor)
graph.add_kernel(conv1)
graph.add_kernel(pool1)
graph.add_kernel(conv2)

# Connect dataflow
graph.add_edge("preprocessor", "output", "conv1", "input")
graph.add_edge("conv1", "output", "pool1", "input")
graph.add_edge("pool1", "output", "conv2", "input")

# Validate connections and constraints
graph.validate()

# Analyze performance
path, latency = graph.get_critical_path()
print(f"Critical path: {' -> '.join(path)}")
print(f"Total latency: {latency} cycles")
```

### Running Scheduling Analysis

Analyze timing and resource usage with ADFG scheduling:

```python
from brainsmith.core.dataflow.adfg import ADFGActor, SRTAScheduler

# Convert to ADFG format
actors = []
for name, kernel in graph.kernels.items():
    actor = ADFGActor.from_kernel(kernel)
    actor.name = name
    actors.append(actor)

# Extract edges
edges = [
    (e.producer_kernel, e.producer_intf,
     e.consumer_kernel, e.consumer_intf)
    for e in graph.edges.values()
]

# Run scheduling analysis
scheduler = SRTAScheduler()
result = scheduler.analyze(actors, edges)

if result.schedulable:
    print(f"Design is schedulable!")
    print(f"Utilization: {result.total_utilization:.1%}")
    print(f"Hyperperiod: {result.hyperperiod} cycles")
else:
    print(f"Not schedulable: {result.failure_reason}")
```

### Performing Design Space Exploration

Find optimal configurations automatically:

```python
from brainsmith.core.dataflow.dse import (
    DSEConstraints, DesignSpaceExplorer,
    ConfigurationSpace
)

# Define constraints
constraints = DSEConstraints(
    max_dsp=2000,           # Available DSP slices
    max_bram=500,           # Available BRAM blocks
    max_bandwidth_gbps=25,  # Memory bandwidth
    min_fps=30,             # Performance requirement
    target_frequency_mhz=250
)

# Create search space
space = ConfigurationSpace()
space.add_interface("conv1", "input", [1, 2, 4, 8])  # Parallelism options
space.add_interface("conv1", "output", [1, 2, 4, 8])

# Explore designs
explorer = DesignSpaceExplorer(graph, constraints)
results = explorer.explore(space)

# Find Pareto optimal
pareto = explorer.find_pareto_optimal(results)
print(f"Found {len(pareto)} Pareto-optimal designs")

# Show best for each metric
for result in pareto[:3]:
    print(f"Config: {result.config}")
    print(f"  FPS: {result.metrics.fps:.1f}")
    print(f"  Power: {result.metrics.power_estimate:.1f}W")
    print(f"  DSP usage: {result.metrics.resource_usage['DSP']}")
```

## Practical Examples

### Example 1: Matrix Multiplication Accelerator

Complete example of a scalable matrix multiply unit:

```python
def create_matmul_accelerator(M, K, N, iPar=8, wPar=4):
    """Create matrix multiply accelerator C[M,N] = A[M,K] × B[K,N]
    
    Args:
        M, K, N: Matrix dimensions
        iPar: Input parallelism (SIMD width)
        wPar: Weight parallelism (PE count)
    """
    # Input matrix A
    mat_a = Interface(
        name="A",
        direction=InterfaceDirection.INPUT,
        dtype=INT16,
        tensor_dims=(M, K),
        block_dims=(1, iPar),    # Row-wise processing
        stream_dims=(1, iPar)    # iPar elements per cycle
    )
    
    # Weight matrix B  
    mat_b = Interface(
        name="B",
        direction=InterfaceDirection.WEIGHT,
        dtype=INT16,
        tensor_dims=(K, N),
        block_dims=(iPar, wPar), # Tile for systolic array
        stream_dims=(iPar, wPar) # Full tile per cycle
    )
    
    # Output matrix C
    mat_c = Interface(
        name="C",
        direction=InterfaceDirection.OUTPUT,
        dtype=INT16,
        tensor_dims=(M, N),
        block_dims=(1, wPar),    # Output wPar results
        stream_dims=(1, wPar)    # wPar elements per cycle
    )
    
    # Create kernel
    kernel = Kernel(
        name=f"MatMul_{M}x{K}x{N}",
        interfaces=[mat_a, mat_b, mat_c],
        latency_cycles=(K // iPar + 10, K // iPar + 5),
        pragmas=[
            TiePragma("A[1]", "B[0]"),  # Dimension match
            ConstrPragma("A[1]", "%", iPar),  # Alignment
        ],
        resources={
            "DSP": iPar * wPar,  # MAC units
            "BRAM": (K * wPar * 2) / 1024  # Weight buffer
        }
    )
    
    return kernel

# Create and analyze
matmul = create_matmul_accelerator(128, 768, 256, iPar=16, wPar=8)
print(f"Throughput: {matmul.calculate_throughput():.2f} GOPS")
print(f"DSP Efficiency: {matmul.resources['DSP'] / 128:.1%}")
```

### Example 2: Convolution Layer

Flexible convolution accelerator with automatic tiling:

```python
def create_conv2d_kernel(H, W, C, K, R, S, stride=1, iPar=4):
    """Create 2D convolution kernel
    
    Args:
        H, W: Input height/width
        C: Input channels  
        K: Output channels
        R, S: Kernel height/width
        stride: Convolution stride
        iPar: Channel parallelism
    """
    # Determine output dimensions
    H_out = (H - R) // stride + 1
    W_out = (W - S) // stride + 1
    
    # Input feature map
    ifmap = Interface(
        name="ifmap",
        direction=InterfaceDirection.INPUT,
        dtype=INT8,
        tensor_dims=(H, W, C),
        block_dims=(R, S, C),      # One output position
        stream_dims=(1, 1, iPar)   # iPar channels/cycle
    )
    
    # Convolution weights
    weights = Interface(
        name="weights",
        direction=InterfaceDirection.WEIGHT,
        dtype=INT8,
        tensor_dims=(K, R, S, C),
        block_dims=(K, R, S, iPar),
        stream_dims=(K, 1, 1, iPar)
    )
    
    # Output feature map
    ofmap = Interface(
        name="ofmap",
        direction=InterfaceDirection.OUTPUT,
        dtype=INT8,
        tensor_dims=(H_out, W_out, K),
        block_dims=(1, 1, K),
        stream_dims=(1, 1, K)
    )
    
    kernel = Kernel(
        name=f"Conv2D_{R}x{S}",
        interfaces=[ifmap, weights, ofmap],
        latency_cycles=(R * S * C // iPar, R * S * C // iPar),
        resources={
            "DSP": K * iPar,  # Multiply-accumulate units
            "BRAM": K * R * S * C * 1 / 1024  # Weight storage
        }
    )
    
    return kernel
```

### Example 3: Complete CNN Pipeline

End-to-end CNN accelerator with automatic optimization:

```python
def build_cnn_accelerator():
    """Build complete CNN accelerator graph"""
    graph = DataflowGraph()
    
    # Layer 1: 3x3 Conv, 3→32 channels
    conv1 = create_conv2d_kernel(
        H=224, W=224, C=3, K=32, R=3, S=3, iPar=3
    )
    graph.add_kernel(conv1)
    
    # Layer 2: 2x2 MaxPool
    pool1 = Kernel(
        name="MaxPool2x2_1",
        interfaces=[
            Interface("in", InterfaceDirection.INPUT, INT8,
                     (112, 112, 32), (2, 2, 32), (1, 1, 32)),
            Interface("out", InterfaceDirection.OUTPUT, INT8,
                     (56, 56, 32), (1, 1, 32), (1, 1, 32))
        ],
        latency_cycles=(4, 4),
        resources={"LUT": 1000}
    )
    graph.add_kernel(pool1)
    
    # Layer 3: 3x3 Conv, 32→64 channels  
    conv2 = create_conv2d_kernel(
        H=56, W=56, C=32, K=64, R=3, S=3, iPar=8
    )
    graph.add_kernel(conv2)
    
    # Connect layers
    graph.add_edge("Conv2D_3x3", "ofmap", "MaxPool2x2_1", "in")
    graph.add_edge("MaxPool2x2_1", "out", "Conv2D_3x3", "ifmap")
    
    # Run DSE to find optimal configuration
    constraints = DSEConstraints(
        max_dsp=1000,
        max_bram=200,
        min_fps=30,
        target_frequency_mhz=200
    )
    
    explorer = DesignSpaceExplorer(graph, constraints)
    results = explorer.explore()
    
    # Get best configuration
    best = max([r for r in results if r.feasible], 
               key=lambda x: x.metrics.fps)
    
    return graph, best

# Build and analyze
graph, best_config = build_cnn_accelerator()
print(f"Best configuration achieves {best_config.metrics.fps:.1f} FPS")
print(f"Using {best_config.metrics.resource_usage['DSP']} DSPs")
```

## Best Practices

### 1. Start with Interfaces
Always design your accelerator by first defining the data flows:
- What are the inputs and outputs?
- What are the natural chunk sizes?
- What parallelism makes sense?

### 2. Use Meaningful Block Dimensions
Block dimensions should represent logical units of computation:
- Matrix multiply: One output vector
- Convolution: One output pixel/position
- Pooling: One pooling window

### 3. Let DSE Find Optimal Parallelism
Don't hardcode parallelism values. Instead:
1. Define interfaces with parameterized stream dimensions
2. Set resource and performance constraints
3. Let DSE explore the space

### 4. Validate Early and Often
Use the validation methods to catch errors early:
```python
kernel.validate()  # Check pragmas
graph.validate()   # Check connections
```

### 5. Profile Before Optimizing
Use the analysis tools to understand bottlenecks:
```python
# Find critical path
path, latency = graph.get_critical_path()

# Check scheduling
result = scheduler.analyze(actors, edges)
print(f"Bottleneck: {result.critical_actors}")
```

## API Quick Reference

### Core Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `Interface` | Define data flow | `validate()`, `repetition_count()` |
| `Kernel` | Hardware operation | `validate()`, `calculate_throughput()` |
| `DataflowGraph` | Connect kernels | `add_kernel()`, `add_edge()`, `validate()` |
| `TiePragma` | Equality constraint | `evaluate()` |
| `ConstrPragma` | Comparison constraint | `evaluate()` |

### ADFG Scheduling

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `ADFGActor` | Scheduling model | `from_kernel()` |
| `SRTAScheduler` | Timing analysis | `analyze()`, `optimize_periods()` |
| `SchedulabilityResult` | Analysis results | Properties: `schedulable`, `hyperperiod` |

### Design Space Exploration

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `DSEConstraints` | Resource limits | `check_resources()`, `check_performance()` |
| `ConfigurationSpace` | Search space | `add_interface()`, `add_coupling()` |
| `DesignSpaceExplorer` | Optimization | `explore()`, `find_pareto_optimal()` |
| `DSEResult` | Configuration result | Properties: `feasible`, `metrics` |

## Next Steps

Now that you understand the basics:

1. **Explore the Examples**: The `examples/` directory contains complete accelerator designs
2. **Read the Math Guide**: Understand the theoretical foundations in the [Mathematical Foundation](mathematical_foundation.md)
3. **Check Implementation Details**: Learn how to extend the framework in the [Technical Implementation Guide](technical_implementation.md)
4. **Join the Community**: Contribute to Brainsmith's open-source development

Happy accelerator design!