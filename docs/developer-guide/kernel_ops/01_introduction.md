# Introduction to the Kernel Op System

## What is the Kernel Op System?

The kernel op system is Brainsmith's framework for defining, validating, and optimizing FPGA hardware accelerators. It bridges the gap between high-level neural network operations (ONNX nodes) and low-level hardware implementations (HLS/RTL).

**Core Purpose:** Enable systematic design space exploration (DSE) of hardware implementations while maintaining correctness guarantees through declarative constraints.

## Why Does This System Exist?

### The Hardware Design Problem

When implementing a neural network operation in hardware, you face an explosion of design choices:

```
Operation: Matrix-Vector Multiply (768 × 768)

Questions:
- How many multipliers in parallel? (1, 2, 4, 8, ..., 768?)
- How to tile the problem? (All at once? 32×32 chunks? 64×64?)
- What datatype precision? (INT8? INT16? Mixed precision?)
- What memory architecture? (Distributed RAM? Block RAM?)
- What implementation style? (HLS? RTL? Pre-built IP?)

Combinations: Tens of thousands of valid configurations!
```

### Traditional Approaches Fall Short

**Manual exploration** requires generating and synthesizing RTL for each configuration (infeasible for thousands of options). **Trial-and-error** catches invalid configurations late during synthesis. **Ad-hoc validation** scatters checks across kernels with inconsistent error messages.

## The Kernel Op Solution

The kernel op system provides three key innovations:

### 1. Declarative Structure via Schemas

Define **what** your kernel needs, not **how** to validate it:

```python
LAYERNORM_SCHEMA = KernelSchema(
    name="LayerNorm",
    inputs=[
        InputSchema(
            name="input",
            block_tiling=FULL_SHAPE,     # Copy tensor dimensions
            stream_tiling=["SIMD"],       # Parallelizable by SIMD
            datatype=None,                # Use from ONNX graph
            required_layout="NHWC"        # Expect NHWC layout
        )
    ],
    outputs=[
        OutputSchema(
            name="output",
            block_tiling=FULL_SHAPE,
            stream_tiling=["SIMD"],
            datatype="input"              # Match input datatype
        )
    ],
    constraints=[
        DatatypeFloat(("input", "output")),
        ShapesEqual(("input", "output"))
    ]
)
```

**Benefits:**
- Self-documenting (schema IS the specification)
- Centralized validation (no scattered checks)
- Reusable (same schema for all LayerNorm instances)

### 2. Two-Phase Design Space Exploration

Separate "what's possible" from "what's configured":

```python
# Phase 1: Build design space (expensive, once per kernel instance)
design_space = builder.build(context)
# → KernelDesignSpace with valid ranges:
#    SIMD ∈ {1, 2, 3, 4, 6, 8, 12, ..., 768}  (divisors of 768)
#    (Computed automatically from tensor shapes!)

# Phase 2: Explore configurations (cheap, thousands per second)
point1 = design_space.configure({"SIMD": 64})   # Low latency
point2 = design_space.configure({"SIMD": 8})    # Low area
point3 = design_space.configure({"SIMD": 32})   # Balanced

# Compare performance without regenerating RTL
print(f"Cycles: {point1.initiation_interval}")  # 12 cycles
print(f"Cycles: {point2.initiation_interval}")  # 96 cycles
```

**Benefits:**
- Fast exploration (no RTL generation needed for comparison)
- Guaranteed validity (invalid configs rejected at configure time)
- Systematic search (iterate through all valid combinations)

### 3. Immutable Functional Navigation

Explore design space like navigating a map:

```python
# Start at minimum parallelism
base = design_space.configure({"SIMD": 1})

# Navigate to better designs
faster = base.with_max("SIMD")                    # Maximum parallelism
balanced = base.with_percentage("SIMD", 0.5)      # Middle ground
incremental = base.with_step_up("SIMD", 3)        # 3 steps higher

# Sweep through options
for point in base.sweep_dimension("SIMD"):
    cycles = point.initiation_interval
    area = estimate_area(point)
    # Plot Pareto frontier...
```

**Benefits:**
- Safe exploration (can't corrupt current state)
- Composable operations (chain navigation methods)
- Parallel-friendly (multiple threads exploring independently)

## Core Philosophy

The kernel op system embodies several design principles:

### Separation of Concerns

```
┌─────────────────────────────────────────────────┐
│ WHAT (Schema)                                   │
│ - Declares kernel structure                     │
│ - Specifies constraints                         │
│ - Independent of ONNX graph                     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ HOW (Builder)                                   │
│ - Constructs design space from schema + ONNX   │
│ - Resolves templates to concrete values        │
│ - Validates structural constraints              │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ WHICH (Design Point)                            │
│ - Specific configuration choice                 │
│ - Navigable, comparable                         │
│ - Validates optimization constraints            │
└─────────────────────────────────────────────────┘
```

### Schemas Define Structure, Not Storage

**Anti-pattern:**
```python
# DON'T think of schemas as storing shapes
schema.block_shape = (1, 784)  # WRONG - schemas don't store!
```

**Correct understanding:**
```python
# Schemas are TEMPLATES that get RESOLVED at build time
schema = InputSchema(
    block_tiling=FULL_SHAPE  # Template: "copy from tensor shape"
)
# At build time: reads tensor_shape from ONNX → resolves to (1, 784)
# Storage: ModelWrapper (ONNX) holds tensor, DesignSpace holds resolved block
```

### Immutability Enables Exploration

Design points are **immutable snapshots**, enabling:

- **Comparison:** Keep multiple configurations to compare
- **Backtracking:** Return to previous configurations
- **Parallel search:** Multiple threads exploring independently
- **Reproducibility:** Same inputs always produce same outputs

```python
# Chess-style exploration
current = design_space.configure({"SIMD": 32})

# Branch 1: Optimize for speed
speed_focused = current.with_max("SIMD")

# Branch 2: Optimize for area
area_focused = current.with_min("SIMD")

# Compare without affecting original
print(f"Current: {current.config}")        # {"SIMD": 32}
print(f"Speed: {speed_focused.config}")    # {"SIMD": 768}
print(f"Area: {area_focused.config}")      # {"SIMD": 1}
```

## When to Use This System

### Perfect For:

✅ **Systematic DSE:** Need to explore thousands of configurations

✅ **Multi-objective optimization:** Trading off latency vs area vs power

✅ **Parameterized kernels:** Same kernel, different sizes (SIMD, PE, etc.)

✅ **Constraint-heavy operations:** Complex requirements (datatype ranges, shape relationships)

✅ **Reusable components:** Kernel used in multiple networks/contexts

### Not Necessary For:

❌ **Fixed implementations:** Single configuration, no exploration needed

❌ **Trivial operations:** Simple pass-through or reshape nodes

❌ **Legacy kernels:** Existing FINN kernels (unless refactoring)

## Architecture Overview

The system has four layers:

1. **User Level:** Define schemas, navigate design space, apply configurations
2. **KernelOp Layer:** FINN integration with lazy initialization and caching
3. **Builder Layer:** Resolves templates, validates constraints, computes valid ranges
4. **Model Layer:** Immutable design spaces and points with navigation methods

## Next Steps

Now that you understand the "why" and "what" of the kernel op system, the following chapters will teach you **how** to use it:

- **Chapter 2: Fundamental Concepts** - The mental models you need
- **Chapter 3: Building Your First Kernel** - Step-by-step tutorial
- **Chapter 4: Schema Design** - Crafting effective kernel definitions
- **Chapter 5: Design Space Exploration** - Navigating and optimizing
- **Chapter 6: Advanced Topics** - Broadcasting, static optimization, custom derivation
- **Chapter 7: Best Practices** - Patterns, anti-patterns, troubleshooting

Let's begin with the fundamental concepts that underpin the entire system.
