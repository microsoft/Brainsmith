# Kernel Architecture

*[Hardware Kernels](index.md) > Architecture*

Deep dive into the architectural patterns that enable efficient design space exploration and schema-driven automation.


## Understanding Schemas

**Schemas separate kernel structure from execution context.**

A schema defines the kernel's **invariant properties**—how it processes data, what parallelization it supports, what constraints it enforces. These properties remain constant whether the kernel processes a `(1, 224, 224, 64)` tensor or a `(1, 56, 56, 128)` tensor.

```python
InputSchema(
    name="input",
    block_tiling=[FULL_DIM],           # Relationship: "process entire dimension"
    stream_tiling=["SIMD"],            # Capability: "parallelize by SIMD"
    datatype="input",                  # Constraint: "match input datatype"
    required_layout="NHWC",            # Requirement: "needs NHWC layout"
)
```

**This separation enables:**

**Reusability** - Same LayerNorm kernel works on any tensor shape
: Schema defines processing rules, not specific dimensions

**Single Source of Truth** - ModelWrapper owns graph state
: Tensor shapes and datatypes queried from ONNX graph, never duplicated

**Two-Phase Construction** - Build structure once, explore configurations many times
: Schema defines what varies (SIMD parameter), builder computes valid ranges (divisors)

**Clear Ownership** - Each component has distinct responsibility
: Schema: structure and constraints | ModelWrapper: current graph state | Builder: runtime models

**Example: A LayerNorm kernel processes `(1, 224, 224, 64)` and `(1, 56, 56, 128)` with the same schema, but different runtime models built from different graph contexts.**


## Two-Phase Construction: Build vs Configure

The dataflow modeling system separates **structure** (what doesn't change) from **configuration** (what varies during DSE).

### Phase 1: Build Design Space (Once per structure)

**When:** After kernel initialization, before DSE
**Input:** KernelSchema + ModelWrapper (ONNX graph context)
**Output:** KernelDesignSpace with valid parameter ranges

**What's Resolved:**
- Tensor shapes from ONNX graph
- Block shapes from schema templates
- Interface datatypes (inputs, outputs, internals)
- Valid parallelization ranges (divisor sets)
- Structural constraints validated

**What's Deferred:**
- Stream shapes (depend on SIMD/PE values)
- Configuration-specific constraints
- Performance metrics

**Trigger:** First call to method requiring design space
```python
op._ensure_ready(model_w)  # Internal - builds design_space
design_space = op.design_space  # Cached after first build
```

### Phase 2: Configure Design Point (Many times for DSE)

**When:** During DSE exploration
**Input:** KernelDesignSpace + parameter configuration
**Output:** KernelDesignPoint with resolved stream shapes

**What's Resolved:**
- Stream shapes for specific SIMD/PE values
- Configuration-specific constraints
- Performance metrics (cycles, bandwidth)

**Trigger:** Explicit configuration
```python
point = design_space.configure({"SIMD": 16, "PE": 4})
```

### Lifecycle Example

```python
# Phase 1: Build (expensive, once)
op = LayerNorm(onnx_node)
op._ensure_ready(model_w)
design_space = op.design_space  # Built from schema + graph

# Phase 2: Configure (cheap, thousands of times)
for simd in [1, 2, 4, 8, 16, 32, 64]:
    point = design_space.configure({"SIMD": simd})
    cycles = point.initiation_interval  # Computed from stream shapes
    area = estimate_area(point)
    pareto.add(cycles, area)

# Apply winner
op.apply_design_point(best_point)
```

### Memory Efficiency: Flyweight Pattern

Design points use the **flyweight pattern** for memory efficiency:

- **DesignSpace:** Heavy - stores tensor shapes, block shapes, datatypes, parameter ranges
- **DesignPoint:** Light - stores only `config` dict, references parent DesignSpace

```python
# DesignSpace: ~10KB (shapes, datatypes, ranges)
design_space = op.design_space

# Each DesignPoint: ~100 bytes (just config dict + reference)
point1 = design_space.configure({"SIMD": 16})
point2 = design_space.configure({"SIMD": 32})
# point1 and point2 share same DesignSpace in memory
```

This enables exploring millions of configurations without memory exhaustion.


## Lazy Initialization and Caching

### Initialization Strategy

KernelOp uses **lazy initialization** for performance:

- `kernel_schema`: Built in `__init__()` from node structure
- `design_space`: Built on first access (cached, invalidated on structural changes)
- `design_point`: Regenerated on each access (guarantees consistency with nodeattrs)

### When Design Space Rebuilds

**Structural Changes** (invalidate cache):
- Datatype changes: `set_nodeattr("input0Datatype", "INT16")`
- Parameters in block_tiling (rare): Changes to block shape computation

**Configuration Changes** (no rebuild):
- Parallelization parameters: `set_nodeattr("SIMD", 32)`
- DSE parameters: `set_nodeattr("ram_style", "block")`

Design points regenerate from current nodeattrs on every access - no cache to invalidate.

### QONNX Execution Compatibility

QONNX's `execute_onnx()` creates fresh KernelOp instances per node, losing cached state. Use this pattern in `execute_node()`:

```python
def execute_node(self, context, graph):
    # Ensure design_space initialized (QONNX executor creates fresh instances)
    self._ensure_initialized_for_execution(graph)

    # Now safe to access design_point (regenerates from design_space)
    dtype = self.design_point.inputs["input"].datatype
    # ... rest of execution logic
```

This reconstructs ModelWrapper from the GraphProto to initialize design_space on demand (~1ms overhead, only when needed).

### Performance: Caching Design Point

Multiple accesses to `self.design_point` trigger regeneration:

```python
# AVOID: Multiple accesses (regenerates 3 times)
simd = self.design_point.inputs["input"].stream_shape[-1]
width = self.design_point.inputs["input"].tensor_shape[-1]
dtype = self.design_point.inputs["input"].datatype

# BETTER: Cache locally (regenerates once)
point = self.design_point
simd = point.inputs["input"].stream_shape[-1]
width = point.inputs["input"].tensor_shape[-1]
dtype = point.inputs["input"].datatype
```

### Quick Reference: DesignSpace vs DesignPoint

| Aspect | KernelDesignSpace | KernelDesignPoint |
|--------|------------------|-------------------|
| **Lifecycle** | Built once per structure | Configured many times |
| **Mutability** | Immutable (frozen dataclass) | Immutable (frozen dataclass) |
| **Caching** | Cached in KernelOp | Regenerated on each access |
| **Contains** | Tensor shapes, block shapes, datatypes, parameter ranges | Stream shapes, config dict, parent reference |
| **Memory** | ~10KB (heavy) | ~100 bytes (flyweight) |
| **Navigation** | `configure(config)` | `with_dimension()`, `sweep_dimension()` |
| **Performance Metrics** | No | Yes (`initiation_interval`, `stream_width_bits`) |
| **Access Pattern** | `op.design_space` | `op.design_point` or `design_space.configure(...)` |
| **Invalidation** | On structural changes | Never (always fresh) |

**Key Principle:** DesignSpace is the **factory**, DesignPoint is the **product**.


## Data Hierarchy Deep Dive

!!! info "Deep Dive Section"
    This section provides theoretical foundation for the TENSOR/BLOCK/STREAM hierarchy.

    **Required reading for:** Complex kernels with reduction operations, custom DSE logic, understanding performance characteristics

    **Safe to skip if:** Using simple element-wise operations with FULL_SHAPE blocks

The stream relationships that define kernels require systematic refinement to connect ONNX's model-level abstractions to RTL's cycle-accurate streaming implementation. ONNX describes computation in terms of complete tensors (e.g., `(1, 224, 224, 64)` activation map), while RTL executes on bit-streams cycling through hardware (16 elements per clock).

Brainsmith bridges this semantic gap through a three-tier hierarchy, each level refining the data representation for hardware realization:

<div align="center" markdown>
![Input Chunking](../../images/input_chunking.png){ width="600" }
</div>

### Tensor / Block / Stream

**TENSOR** - Complete dimensions from ONNX graph (e.g., `(1, 224, 224, 64)`)

- Defines functional correctness and accuracy requirements

**BLOCK** - Kernel's atomic computation unit (e.g., `(1, 7, 7, 64)`)

- Data required for one cycle of the kernel's computation state
- Controls memory footprint, pipeline depth, and latency

**STREAM** - Elements processed per clock cycle (e.g., 16 8-bit Integers)

- Determines throughput and resource usage
- Constrained by BLOCK shape (cannot exceed block dimensions)

### Kernel-Specific Block Semantics

The BLOCK shape adapts to kernel computation characteristics:

**Simple kernels** (elementwise operations like Add, Multiply):

- BLOCK = TENSOR (entire input processed as one unit)
- No intermediate computation state between elements
- STREAM parallelization limited only by resource availability

**Complex kernels** (reduction operations like MatMul, LayerNorm):

- BLOCK = one quantum of the calculation state
- Example: Dot product BLOCK = one input vector
- Kernel must accumulate/reduce across BLOCK before producing output
- STREAM parallelization cannot exceed BLOCK dimensions

This constraint exists because **kernels process one BLOCK at a time**. A MatMul with input vectors of length 64 can parallelize up to STREAM=64, but STREAM=128 is invalid—the kernel's computation state operates on 64-element vectors. Higher throughput requires instantiating multiple parallel kernel instances, each processing independent BLOCKs.

### Lowering from ONNX to RTL

The hierarchy enables automatic derivation of hardware execution characteristics:

```python
# Spatial decomposition
tensor_blocks = ceil(tensor_dim / block_dim)

# Temporal execution
stream_cycles = ceil(block_dim / stream_dim)

# Total latency
total_cycles = prod(tensor_blocks) × prod(stream_cycles)
```

**Example:** For tensor `(100, 64)`, block `(32, 16)`, stream `(8, 4)`:

- Tensor blocks: `(4, 4)` → 16 blocks cover the full tensor
- Stream cycles: `(4, 4)` → 16 cycles stream each block
- Total cycles: 256


## Inter-Kernel Dataflow

!!! info "Deep Dive Section"
    Understanding how kernels connect through streaming interfaces and elastic buffering.

    **Required for:** Custom infrastructure transforms, debugging FIFO depth issues, understanding dataflow composition

    **Safe to skip if:** Standard pipeline usage with automatic FIFO sizing

<div align="center" markdown>
![Dataflow Chunking with FIFO](../../images/dataflow_chunking_fifo.png){ width="700" }
</div>

Kernels communicate via **streaming interfaces**, producing and consuming data cycle-by-cycle. Elastic FIFOs between kernels accumulate these streams as **data blocks** for buffering, then stream them out to downstream consumers. This infrastructure automatically adapts to different kernel semantics through shape-driven buffering.

**👉 The compiler handles this automatically via schemas. You rarely need to think about it.**

### Composing Kernels with Different Block Semantics

Consider a simple pipeline: `Add → LayerNorm → Softmax`

```
Add (elementwise)           LayerNorm (reduction)       Softmax (reduction)
BLOCK = TENSOR (1,224,64)   BLOCK = (1,1,64)           BLOCK = (1,1,N_classes)
STREAM = (1,1,16)           STREAM = (1,1,16)          STREAM = (1,1,8)
```

**What happens at kernel boundaries:**

1. **Add → LayerNorm**: Producer outputs (1,224,64) blocks, consumer expects (1,1,64) blocks

   - FIFO buffers shape transformation
   - Add streams 14 blocks × 16 cycles each = 224 cycles
   - LayerNorm consumes in (1,1,64) chunks, computing normalization per spatial position

2. **LayerNorm → Softmax**: Block shapes may differ based on computation semantics

   - Each kernel's BLOCK reflects its reduction domain
   - FIFOs provide elastic buffering for rate adaptation

### Automatic Infrastructure Derivation

Block and stream shapes drive hardware generation:

**FIFO Depths** - Determined by producer/consumer rate mismatch:
```python
producer_rate = prod(block_shape) / prod(stream_shape)  # cycles per block
consumer_rate = # depends on consumer's internal pipeline depth
fifo_depth = max(producer_burst, consumer_backpressure_tolerance)
```

!!! note "Implementation Reality"
    The formula above describes the theoretical basis for FIFO sizing. In practice, the current implementation uses **iterative cycle-accurate RTL simulation** to converge on efficient FIFO depths. This will be expanded on in future releases.

**Stream Width Matching** - Interface widths must align:

- Producer STREAM=(1,1,16) × INT8 → 128-bit AXI-Stream
- Consumer expects matching width or automatic width conversion
- Datatype changes (INT8 → FLOAT32) insert conversion logic

**Rate Mismatch Handling** - Kernels may have different throughputs:

- Elementwise: 1 output per cycle (after initial latency)
- Reduction: Multiple cycles per output (accumulation phase)
- FIFOs absorb transient rate differences, prevent pipeline stalls

### Schema-Driven Interface Resolution

Schemas declare interfaces using templates that adapt to runtime shapes:

```python
# Producer (Add kernel)
OutputSchema(
    block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM],  # Process entire spatial dims
    stream_tiling=[1, 1, "PE"]                      # Parallelize over channels
)

# Consumer (LayerNorm kernel)
InputSchema(
    block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM],  # Same spatial processing
    stream_tiling=[1, 1, "SIMD"]                    # Match channel parallelism
)
```

At compile time:

1. ONNX tensor shapes resolve `FULL_DIM` → actual dimensions
2. DSE parameters (PE=16, SIMD=16) resolve stream tiling
3. Infrastructure generates FIFOs matching computed shapes
4. Validation ensures producer/consumer compatibility

This declarative approach separates *what* (tensor semantics) from *how* (hardware implementation), enabling design space exploration while maintaining correctness by construction. The compiler automatically inserts width converters, reshaping logic, and elastic buffering as needed.


## See Also

- **[Kernel Schema Reference](kernel-schema-reference.md)** - Complete API reference for schema components
- **[Kernel Tutorial](kernel-tutorial.md)** - Hands-on examples demonstrating these concepts
- **[Dataflow API Reference](../api/dataflow.md)** - Complete API documentation
