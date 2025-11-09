# Fundamental Concepts

This chapter introduces the core mental models you need to work effectively with the kernel op system.

## The Three-Level Shape Hierarchy

Hardware doesn't process "tensors" in the mathematical sense. It processes **streams** of data flowing through **processing elements**. The three-level hierarchy bridges this gap:

### TENSOR Level: The Mathematical Problem

```python
tensor_shape = (1, 224, 224, 64)  # Batch, Height, Width, Channels
# "Process this entire 4D tensor"
# Total: 1 × 224 × 224 × 64 = 3,211,264 elements
```

This is what your algorithm operates on - the **logical view**.

**Questions answered:**
- What is the mathematical operation?
- What are the input/output dimensions?
- What datatypes are involved?

### BLOCK Level: The Spatial Tiling

```python
block_shape = (1, 224, 224, 64)  # Same as tensor in this case
# "Process the entire tensor as one block"
# OR:
block_shape = (1, 32, 32, 64)
# "Process 32×32 spatial tiles at a time"
# Number of blocks: (224/32) × (224/32) = 49 blocks
```

This is how you **decompose the problem** into chunks that fit in on-chip memory.

**Questions answered:**
- Can this fit in BRAM? (Block size vs available memory)
- Do I need to tile spatially? (Large images → multiple blocks)
- How much data reuse within a block? (Affects buffering strategy)

**Mental model:** Think of baking cookies. Block size is your baking sheet capacity. Large tensor = multiple batches.

### STREAM Level: The Temporal Parallelism

```python
stream_shape = (1, 1, 1, 32)
# "Process 32 channels in parallel per cycle"
# Cycles per block: 64/32 = 2 cycles (for channel dimension)
```

This is **how many elements you process per clock cycle** - the hardware parallelism.

**Questions answered:**
- How many processing elements (PEs)?
- How fast can I process each block?
- What's my resource usage (LUTs, DSPs)?

**Mental model:** Think of factory workers. Stream size is how many workers on the assembly line working in parallel.

### The Complete Picture

```
Tensor (1, 224, 224, 64)    →    Block (1, 32, 32, 64)    →    Stream (1, 1, 1, 32)
   ↓                                    ↓                              ↓
What to compute                  How to tile it               How fast to compute

3.2M elements total         →    32×32×64 = 65,536         →    32 elements/cycle
                                 per block
                                 49 blocks total                 2,048 cycles/block
                                                                99,456 total cycles
```

### Why Three Levels?

Each level addresses different hardware constraints:

```
Level      | Constraint              | Design Question
-----------|------------------------|----------------------------------
TENSOR     | Problem definition     | "What am I computing?"
BLOCK      | On-chip memory (BRAM)  | "Can I fit this in memory?"
STREAM     | Compute resources      | "How much parallelism can I afford?"
```

## Folding: Trading Time for Space

Folding is the fundamental hardware tradeoff. You **fold** the computation across time to reduce spatial resources.

**Folding factor** = dimension_size / parallelism. Example: Process 768 elements with 64 parallel units by reusing them 12 times.

### Two Types of Folding

#### 1. Tensor Folding (Spatial Decomposition)

How many blocks to process the entire tensor:

```python
tensor_shape = (1, 224, 224, 64)
block_shape = (1, 32, 32, 64)

# Compute blocks per dimension
tensor_folding = (
    ceil(1 / 1),      # = 1  (batch dimension)
    ceil(224 / 32),   # = 7  (height)
    ceil(224 / 32),   # = 7  (width)
    ceil(64 / 64)     # = 1  (channels)
)
# Total blocks: 1 × 7 × 7 × 1 = 49 blocks

# Accessed via:
interface.tensor_folding_factor  # = 49
```

**Physical interpretation:** Number of times you "load the assembly line" with new data.

#### 2. Block Folding (Temporal Decomposition)

How many cycles to process one block:

```python
block_shape = (1, 32, 32, 64)
stream_shape = (1, 1, 1, 32)

# Compute cycles per dimension
block_folding = (
    ceil(1 / 1),     # = 1  (batch)
    ceil(32 / 1),    # = 32 (height, no parallelism)
    ceil(32 / 1),    # = 32 (width, no parallelism)
    ceil(64 / 32)    # = 2  (channels, PE=32)
)
# Total cycles: 1 × 32 × 32 × 2 = 2,048 cycles per block

# Accessed via:
interface.block_folding_factor  # = 2,048
```

**Physical interpretation:** Number of "assembly line rotations" per batch.

### The Folding Tradeoff

```python
# Configuration 1: High parallelism
stream_shape = (1, 1, 1, 64)
block_folding = 64 / 64 = 1 cycle (dimension 3 only)
# Fast! But uses 64 PEs (expensive)

# Configuration 2: Low parallelism
stream_shape = (1, 1, 1, 8)
block_folding = 64 / 8 = 8 cycles (dimension 3 only)
# Slower! But uses only 8 PEs (cheap)

Tradeoff: Speed ↔ Hardware Cost
```

### Initiation Interval: Total Latency

The complete latency is the product of folding factors:

```python
# Formula:
initiation_interval = tensor_folding_factor × block_folding_factor

# Example:
tensor_folding = 49 blocks
block_folding = 2,048 cycles/block
initiation_interval = 49 × 2,048 = 100,352 cycles

# Accessed via:
design_point.initiation_interval  # = 100,352
```

## Design Space vs Design Point

Understanding this distinction is crucial for effective DSE.

### Design Space: The Manifold of Possibilities

A **design space** defines all valid configurations. Built once (~1-10ms), it contains tensor/block shapes, datatypes, valid dimension ranges, and optimization constraints. Invalidated only when structural parameters change.

```python
# Example: 68 SIMD values × 2 ram_style options = 136 configurations
design_space.dim_min("SIMD")           # Query ranges
design_space.get_dimension("SIMD")     # Iterate options
design_space.configure({"SIMD": 32})   # Create point
```

### Design Point: A Location in the Space

A **design point** is one specific configuration. Created quickly (~100μs), it's immutable and uses the flyweight pattern (stores only config, references parent space).

```python
point = design_space.configure({"SIMD": 64, "ram_style": "distributed"})
point.initiation_interval              # Estimate performance
point.with_step_up("SIMD")             # Navigate to nearby point
```

One design space → many design points (all sharing space structure, each with unique config). Flyweight pattern provides ~100x memory savings.

## Parallelization as Divisor Factorization

You can't pick arbitrary parallelism - it must **divide evenly**.

### The Divisibility Requirement

```python
block_shape = (768,)  # 768 elements to process

# Valid SIMD values (divide 768):
divisors(768) = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768}

# Why? Because:
768 / 64 = 12 cycles  ✓ (integer, perfect)
768 / 50 = 15.36 cycles  ✗ (fractional, impossible in hardware!)
```

**Physical interpretation:** You can't process "0.36 of an element" in a clock cycle.

### Automatic Range Computation

The system automatically computes valid ranges:

```python
# You define:
InputSchema(
    name="input",
    block_tiling=FULL_SHAPE,     # (768,)
    stream_tiling=["SIMD"]        # Last dim parallelized
)

# Builder automatically computes:
dimensions = {
    "SIMD": OrderedDimension("SIMD", divisors(768))
    # = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768)
}
```

### Multi-Dimensional Parallelization

When a parameter appears in multiple dimensions:

```python
block_shape = (32, 768)
stream_tiling = [1, "SIMD"]

# SIMD must divide block_shape[-1] = 768
# Valid: divisors(768)
```

If it spans multiple interfaces, use GCD:

```python
input_block = (768,)
output_block = (512,)
both_use_simd = True

# SIMD must divide gcd(768, 512) = 256
# Valid: divisors(256) = {1, 2, 4, 8, 16, 32, 64, 128, 256}
```

### Navigating Divisor Space

```python
design_space = builder.build(...)
base_point = design_space.configure({"SIMD": 1})

# Navigate through divisors
base_point.with_min("SIMD")           # → 1
base_point.with_max("SIMD")           # → 768
base_point.with_percentage("SIMD", 0.5)  # → ~32 (middle divisor)
base_point.with_step_up("SIMD", 3)    # → 4th divisor in sequence
```

## Structural vs Optimization Parameters

Not all parameters are equal. Understanding the distinction is critical.

### Structural Parameters

**Change WHAT you can build.** Affect backend compatibility, internal datatypes, and block shapes. Validated during `build()`. Changes invalidate entire design space (~1-10ms rebuild).

**Examples:** Input/output datatypes, tensor shapes, kernel-specific parameters (epsilon, axis, algorithm).

### Optimization Parameters

**Change HOW WELL it performs.** Determine performance/area tradeoff. Validated during `configure()`. Changes invalidate only design point (~100μs reconfigure).

**Examples:** Parallelization (SIMD, PE, MW, MH), resource allocation (ram_style, res_type), implementation strategy (mem_mode).

### Why It Matters

```python
# BAD: Treating everything as structural
for simd in [1, 2, 4, 8, 16, 32, 64]:
    op.set_nodeattr("SIMD", simd)
    design_space = builder.build(...)  # REBUILDS every time! (slow)
    # Total: 7 × 5ms = 35ms

# GOOD: Build once, configure many times
design_space = builder.build(...)  # Build once: 5ms
for simd in [1, 2, 4, 8, 16, 32, 64]:
    point = design_space.configure({"SIMD": simd})  # Configure: 100μs each
    # Total: 5ms + 7 × 0.1ms = 5.7ms (6x faster!)
```

### Automatic Invalidation

The system tracks this automatically:

```python
class KernelOp:
    def set_nodeattr(self, name, value):
        super().set_nodeattr(name, value)

        if name in self.kernel_schema.get_structural_nodeattrs():
            # Structural change → invalidate everything
            self._design_space = None
            self._design_point = None
        elif name in self._design_space.dimensions:
            # Optimization change → invalidate point only
            self._design_point = None
        # else: Runtime parameter, no invalidation
```

## Immutability and Functional Navigation

Design points are **immutable**. All navigation returns **new instances**.

### Why Immutability?

**Without immutability (mutable):**
```python
point = design_space.configure({"SIMD": 32})
point.SIMD = 64  # Mutates in place
# Lost the original! Can't compare, can't backtrack
```

**With immutability (functional):**
```python
point1 = design_space.configure({"SIMD": 32})
point2 = point1.with_dimension("SIMD", 64)  # Returns new point

# Both still available
print(point1.config)  # {"SIMD": 32}
print(point2.config)  # {"SIMD": 64}

# Can compare
if point2.initiation_interval < point1.initiation_interval:
    print(f"point2 is {point1.cycles / point2.cycles}x faster!")
```

### Navigation Patterns

**Direct assignment:**
```python
point = base.with_dimension("SIMD", 128)
```

**Relative movement:**
```python
faster = base.with_step_up("SIMD", 3)    # 3 steps higher
slower = base.with_step_down("SIMD", 2)  # 2 steps lower
```

**Boundary access:**
```python
fastest = base.with_max("SIMD")      # Maximum parallelism
smallest = base.with_min("SIMD")     # Minimum resources
```

**Percentage-based:**
```python
balanced = base.with_percentage("SIMD", 0.5)  # Middle of range
aggressive = base.with_percentage("SIMD", 0.75)
```

**Sweeping:**
```python
# Explore all values
for point in base.sweep_dimension("SIMD"):
    evaluate(point)

# Explore percentage points
for point in base.sweep_percentage("SIMD", [0.0, 0.25, 0.5, 0.75, 1.0]):
    evaluate(point)
```

### Comparison and Backtracking

```python
# Try different strategies
current = design_space.configure({"SIMD": 32})

strategies = {
    "speed": current.with_max("SIMD"),
    "area": current.with_min("SIMD"),
    "balanced": current.with_percentage("SIMD", 0.5)
}

# Compare all strategies
results = {}
for name, point in strategies.items():
    results[name] = {
        "cycles": point.initiation_interval,
        "area": estimate_area(point),
        "config": point.config
    }

# Pick winner
best = min(results.items(), key=lambda x: x[1]["cycles"])
winner_point = strategies[best[0]]

# Can still access all alternatives!
```

## Next Steps

You now understand the fundamental concepts:

✓ Three-level shape hierarchy (tensor → block → stream)
✓ Folding tradeoffs (time vs space)
✓ Design space vs design point
✓ Parallelization as factorization
✓ Structural vs optimization parameters
✓ Immutable functional navigation

**Next chapter:** Build your first kernel and see these concepts in practice.
