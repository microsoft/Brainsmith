# Understanding Hardware Kernels

This guide explains the core concepts behind Brainsmith's kernel architecture. If you're new to FPGA accelerators or wondering why Brainsmith models hardware the way it does, start here. For implementation details and code examples, see [Kernels](../3-reference/kernels.md) and [Kernel Modeling](../2-core-systems/kernel-modeling.md).

---

## Prerequisites

You should understand:
- Basic neural network concepts (layers, tensors, operations)
- What FPGAs are at a high level
- The difference between compilation and execution

No prior hardware design experience required.

---

## Part 1: The Hardware Context

### Why Hardware Accelerators?

Modern neural networks spend most of their time doing the same operations repeatedly: matrix multiplies, convolutions, activations. A CPU executes these operations by fetching instructions and data from memory, computing results, then writing back—millions of times per inference.

This **von Neumann bottleneck** (separating memory from processing) limits performance. Hardware accelerators solve this by embedding computation directly in silicon. Instead of fetching instructions, the hardware *is* the computation.

**When hardware wins:**
- Operations are predictable and repetitive (neural networks)
- Parallelism is abundant (matrix operations)
- Data access patterns are regular (streaming)
- Energy efficiency matters (embedded systems)

**When hardware loses:**
- Algorithms change frequently (research)
- Control flow is complex (branching logic)
- Flexibility outweighs performance (general computing)

### FPGAs: Programmable Silicon

An FPGA is a reconfigurable fabric of logic blocks. Think of it as programmable silicon—you describe the computation you want, and the FPGA configures its internal circuitry to implement exactly that computation.

**The key tradeoff:**
- **ASICs** (Application-Specific Integrated Circuits): Fixed function, maximum performance, expensive to design
- **CPUs/GPUs**: Completely flexible, good performance, must follow their execution model
- **FPGAs**: Configurable after manufacturing, high performance for specific workloads, moderate design cost

Configuration happens at compile time (like C → binary), not runtime. Once configured, an FPGA runs your design at hardware speed.

### Spatial vs Temporal Computing

This distinction is fundamental to understanding hardware acceleration:

**Temporal computing** (CPUs):
- Reuse the same ALU/multiplier over time
- Execute instruction sequence: `LOAD, MUL, ADD, STORE, repeat`
- Hardware is general-purpose, software provides specificity

**Spatial computing** (FPGAs):
- Dedicate different regions of silicon to different operations
- All operations execute simultaneously in their respective regions
- Hardware is problem-specific, configured for your exact computation

Example: Computing `(a × b) + (c × d)` repeatedly:
- **CPU**: Fetch-execute 4 instructions per iteration, reusing the multiplier twice
- **FPGA**: Place two multipliers and one adder in silicon, stream data through continuously

Spatial computing trades flexibility for throughput.

---

## Part 2: Kernels as Building Blocks

### What is a Kernel?

A **kernel** is a parameterized hardware implementation of a neural network operation. Think of kernels as LEGO bricks: standardized interfaces, combinable in various ways, but different internal complexity.

Examples:
- **MVAU**: Matrix-Vector-Activation Unit (fully connected layer)
- **Convolutional Window**: Extracts sliding windows from images
- **Thresholding**: Multi-level activation function
- **LayerNorm**: Layer normalization

**Why layer-level operations?** Why not primitive operations like ADD, MUL, SHIFT?

Three reasons:

1. **Preserve semantic meaning**: Neural networks are designed in terms of layers. Decomposing to primitives loses this structure, making optimization harder.

2. **Prevent combinatorial explosion**: A ResNet layer might become thousands of primitive operations. Exploring configurations for thousands of tiny blocks is computationally intractable.

3. **Enable hand optimization**: Hardware engineers can hand-tune a matrix multiply kernel without understanding the full network. This

 balances automation with expert optimization.

### Standardized Interfaces: The LEGO Principle

All kernels communicate via **AXI-Stream**, a hardware handshaking protocol. Like LEGO bricks with standardized connectors, kernels with matching interfaces can connect in any order.

**AXI-Stream basics:**
- Data flows on `TDATA` signal
- Producer signals `TVALID` when data is ready
- Consumer signals `TREADY` when it can accept data
- Transfer occurs when both are high

This handshaking enables **backpressure**: if a downstream kernel is busy, upstream kernels automatically pause. The system self-throttles without central control.

**Why streams matter:**
- Continuous data flow (no start/stop overhead)
- Elastic composition (kernels don't need to know about each other's timing)
- Memory efficiency (no large intermediate buffers between kernels)

### Parameterization: Hardware as Configuration

Unlike software functions with runtime parameters, hardware kernels have **compile-time parameters** that determine their silicon structure:

- **PE** (Processing Elements): Number of parallel computation units
- **SIMD** (Single Instruction Multiple Data): Operations per unit
- **Bit-widths**: Precision of computations (2-bit to 32-bit)
- **Memory organization**: Block RAM vs distributed, depth, banking

Changing these parameters changes the hardware—different silicon layout, different resource usage, different performance. This is why we can't just "try different settings" at runtime; we must explore the space at design time.

---

## Part 3: Parallelism and Resource Tradeoffs

### Clock Cycles as Currency

In hardware, time is measured in **clock cycles**. At 200 MHz, one cycle is 5 nanoseconds. Every operation costs cycles.

Processing a 256-element vector with different configurations:
- **SIMD=1**: 256 cycles (process 1 element per cycle)
- **SIMD=64**: 4 cycles (process 64 elements per cycle)
- **SIMD=256**: 1 cycle (process all 256 elements in parallel)

More SIMD = fewer cycles = higher throughput. But there's a catch.

### The Resource-Throughput Tradeoff

FPGAs have finite resources:
- **LUTs** (Look-Up Tables): Basic logic elements
- **DSPs** (Digital Signal Processors): Hardened multipliers
- **BRAMs** (Block RAMs): On-chip memory blocks
- **FFs** (Flip-Flops): Registers for state

Higher parallelism consumes more resources:
- SIMD=256 requires 256 multipliers
- SIMD=1 requires 1 multiplier (reused 256 times)

This is the fundamental tradeoff: **throughput vs resources**. Your FPGA has a fixed silicon budget. Spend it on more parallelism in one kernel, and you have less for others.

### PE and SIMD: Two Dimensions of Parallelism

Think of a factory assembly line:

**SIMD**: How many operations happen per workstation
- SIMD=1: One worker per station
- SIMD=16: Sixteen workers per station, processing 16 items simultaneously

**PE**: How many workstations operate in parallel
- PE=1: One assembly line (sequential processing)
- PE=8: Eight assembly lines (eight products manufactured simultaneously)

For a matrix-vector multiply (output = matrix × input):
- **SIMD** parallelizes across input channels (dot product computation)
- **PE** parallelizes across output channels (multiple outputs computed together)

Both reduce cycles, both consume resources.

### Folding: Time-Sharing Hardware

**Folding** is the inverse of parallelization. It's how many times we reuse the same hardware to complete the operation.

```
Folding = Problem_Size / Parallelism
```

Example: 256 output channels with PE=32
```
Folding = 256 / 32 = 8
```

The hardware computes 32 outputs per cycle, then reuses those same 32 computational units 8 times to produce all 256 outputs.

Folding is deterministic time-sharing—not like CPU context switching (unpredictable), but like a precise clockwork mechanism that processes chunks in sequence.

---

## Part 4: Shape Hierarchy—Three Views of Data

This is where hardware modeling gets interesting. The same data exists in three different contexts:

### TENSOR Shape: The Complete Problem

The **tensor shape** is what you see in PyTorch or ONNX—the logical dimensions of your data.

```
Input tensor: [1, 224, 224, 64]
  - 1 image
  - 224×224 pixels
  - 64 channels
```

This is the problem you're solving. It doesn't change based on hardware decisions.

### BLOCK Shape: Spatial Tiling

The **block shape** is how you decompose the problem spatially. Instead of processing the entire 224×224 image at once, you tile it into smaller blocks.

```
Block shape: [1, 7, 7, 64]
  - Process 7×7 pixel patches
  - All 64 channels per patch
  - Requires 32×32 = 1024 blocks to cover full image
```

**Why tile?**
- Reduces on-chip memory requirements (7×7×64 fits in BRAM, 224×224×64 doesn't)
- Enables reuse of the same hardware for different spatial regions
- Matches hardware memory hierarchy

### STREAM Shape: Cycle-Level Processing

The **stream shape** is how many elements you process per clock cycle—the hardware parallelism.

```
Stream shape: [1, 1, 1, 16]
  - 1 pixel at a time (no spatial parallelism)
  - 16 channels in parallel (PE=16)
```

**Why all three?**

They answer different questions:
- **TENSOR**: What's the accuracy/functionality requirement? (algorithm level)
- **BLOCK**: How much memory do we need? What's the latency? (architecture level)
- **STREAM**: What's the throughput? Resource usage? (implementation level)

You can't make good hardware decisions without understanding all three perspectives.

---

## Part 5: The Design Space

### Not All Configurations Are Valid

When you parameterize a kernel, you can't just pick arbitrary numbers. Hardware has constraints:

**Divisibility constraints:**
- If input has 64 channels and SIMD=7, what happens on the last iteration? The hardware would compute 63 elements (9×7) and leave 1 element unprocessed.
- Solution: SIMD must divide channel count evenly

**Interface matching constraints:**
- If kernel A outputs 16 elements/cycle but kernel B expects 32 elements/cycle, the interface doesn't match
- Solution: Stream shapes must align at kernel boundaries

**Resource constraints:**
- FPGA has 2600 DSPs total
- Your configuration requires 3000 DSPs
- Solution: Not all high-parallelism configurations fit

The **design space** is the set of all valid configurations. It's smaller than you might think.

### The Combinatorial Explosion Problem

For a simple 3-layer network:
- 3 kernels × 2 backends (HLS/RTL) = 8 kernel combinations
- Each kernel has 5-10 valid SIMD/PE configurations
- Result: Hundreds to thousands of configurations

Evaluating each configuration requires synthesis—compiling the hardware design to see actual resource usage and timing. This takes hours per configuration.

Exploring thousands of configurations naively would take weeks. We need a smarter approach.

---

## Part 6: Modeling for Exploration

### Schemas: Structure vs Storage

A **schema** is like a blueprint. It defines what a kernel needs (inputs, outputs, constraints), not where that information is stored.

**Why separate structure from storage?**

The same kernel structure might store its state differently in different contexts:
- ONNX node attributes (for FINN compatibility)
- Python objects (for DSE algorithms)
- Hardware configuration registers (for runtime reconfiguration)

The schema remains constant; the storage mechanism adapts to context. This is separation of concerns.

### Templates: From Structure to Configuration

Schemas use **templates** that resolve to concrete values given context:

Template: `stream_shape = [1, 1, 1, "PE"]`
- "PE" is a parameter to be filled in later

Resolution with PE=16: `stream_shape = [1, 1, 1, 16]`

This delayed resolution enables:
- Defining structure before knowing parameter values
- Reusing the same schema across configurations
- Validating structure independent of specific values

Think of templates as functions: structure is the function definition, configuration is calling it with specific arguments.

### Constraints: Validation Boundaries

**Constraints** define what makes a configuration legal. They're predicates that evaluate to true/false.

Examples:
- "Both inputs must have the same shape"
- "Input must be an integer datatype"
- "SIMD must evenly divide the last dimension"

Constraints catch errors early:
- Before synthesis (hours saved)
- At design space construction (with meaningful error messages)
- At specific points in the workflow (structural vs optimization constraints)

They're the type system for hardware configurations.

---

## Part 7: Two-Phase Construction

Here's where the modeling system's efficiency comes from. Understanding this is key to understanding why Brainsmith can explore thousands of configurations quickly.

### The Cost Model: Expensive Once, Cheap Many Times

Building a kernel model has two distinct costs:

**Phase 1 - Build Design Space** (Expensive):
- Extract tensor shapes from ONNX graph (file I/O)
- Resolve block shapes from templates (computation)
- Compute valid parallelization ranges (factorization, divisor sets)
- Validate structural constraints (graph analysis)

Cost: Milliseconds to seconds (depends on model complexity)

**Phase 2 - Configure Design Point** (Cheap):
- Resolve stream shapes from templates + parameters (substitution)
- Validate optimization constraints (arithmetic checks)
- Create immutable snapshot

Cost: Microseconds

**Why this split matters:**

During design space exploration, Phase 1 properties are constant—the tensor shape doesn't change, the valid SIMD ranges don't change. But you evaluate thousands of configurations (different PE/SIMD combinations).

Naive approach: Rebuild everything for each configuration
- Cost: 1000 configurations × 10ms = 10 seconds (just for modeling!)

Two-phase approach: Build once, configure 1000 times
- Cost: 10ms + (1000 × 0.01ms) = 20ms

This 500× speedup makes extensive exploration tractable.

### Map Once, Navigate Many Times

Analogy: Planning a road trip.

**Phase 1 (expensive)**: Build the map
- Download road network data
- Compute all valid routes
- Identify constraints (one-way streets, construction)
- Cache everything

**Phase 2 (cheap)**: Query the map
- "Show me the route via Highway 101"
- "What if I avoid tolls?"
- "How about the scenic route?"

You don't rebuild the map for each query. You build it once, then navigate efficiently.

Design space construction is building the map. Configuration is querying it.

---

## Part 8: Immutability and Navigation

### Design Points as Snapshots

A **design point** is an immutable configuration snapshot. Once created, it never changes.

**Why immutability?**

When exploring configurations in parallel:
```
Thread 1: point.set_PE(16)  # Mutating
Thread 2: point.set_SIMD(32)  # Mutating same object
Thread 3: evaluate(point)     # Which configuration am I evaluating?
```

With mutation, you get race conditions and ambiguity.

With immutability:
```
Thread 1: point2 = point.with_PE(16)      # New point
Thread 2: point3 = point.with_SIMD(32)    # Different new point
Thread 3: evaluate(point)                 # Clear: evaluating original
```

Each design point is a photograph—a frozen moment in configuration space. You can hold many photographs simultaneously without them interfering.

### Navigation: Moving Through Design Space

Since design points are immutable, "changing" a parameter means creating a new point:

```python
base_point = design_space.configure({"PE": 1, "SIMD": 64})
higher_pe = base_point.with_PE(16)  # New point: PE=16, SIMD=64
higher_simd = base_point.with_SIMD(128)  # New point: PE=1, SIMD=128
```

This is **navigation**—moving through the design space by creating new points.

**Navigation operations:**
- `with_dimension("PE", 16)`: Jump to specific value
- `with_min("PE")`, `with_max("PE")`: Jump to extremes
- `with_step_up("SIMD", n=2)`: Move two steps higher
- `sweep_dimension("PE")`: Iterate through all valid values

Navigation is exploration without mutation.

---

## Part 9: Performance Without Synthesis

### The Synthesis Problem

Full hardware synthesis (converting your design to a bitstream) is expensive:
- Elaboration: Parse design, resolve parameters
- Synthesis: Convert RTL to netlist
- Optimization: Logic minimization, resource mapping
- Place & Route: Assign logic to physical locations
- Timing analysis: Verify clock constraints
- Bitstream generation: Create configuration file

Total time: **30 minutes to 8 hours** per design, depending on complexity.

If you need to evaluate 1000 configurations, you can't wait days or weeks for results.

### Estimation: Fast Approximation

Instead of synthesizing, we **estimate** performance using analytical models:

**Cycle estimation** (latency/throughput):
```
Cycles = (Input_Size / SIMD) × (Output_Size / PE) × Batch_Size
```

Example:
- Input: 256 elements, SIMD=64 → 4 iterations
- Output: 512 elements, PE=16 → 32 iterations
- Total: 4 × 32 = 128 cycles

At 200 MHz: 128 cycles = 0.64 μs

**Resource estimation** (utilization):
```
DSPs = PE × SIMD (for multiply-accumulate)
BRAMs = (Weight_Size × Bit_Width) / (36 Kb per BRAM)
LUTs = ~500 × PE (empirical approximation)
```

**Accuracy:**
- Cycle estimates: Very accurate (deterministic hardware behavior)
- Resource estimates: Approximate (tools optimize differently)
- Post-synthesis reports provide ground truth

**Why estimates matter:**
- Explore 1000 configurations in minutes instead of weeks
- Narrow down to top 10 candidates
- Synthesize only the promising designs

Fast iteration enables optimization. Slow iteration forces guesswork.

---

## Part 10: HLS vs RTL—The Control-Productivity Tradeoff

### Two Paths to Hardware

Hardware kernels can be implemented in two languages:

**RTL (Register-Transfer Level)**: SystemVerilog/Verilog
- Hand-coded hardware description
- Explicit control over every register, every wire
- Think: Assembly language for hardware

**HLS (High-Level Synthesis)**: C++
- Algorithmic description
- Compiler generates RTL automatically
- Think: C for hardware

### The Tradeoff

**RTL wins on:**
- Resource efficiency (hand-optimized, no compiler overhead)
- Predictability (you control exactly what gets built)
- Performance-critical paths (tight timing requirements)
- Production designs (validated IP)

**HLS wins on:**
- Development speed (10× faster to write)
- Simulation speed (C++ sim is 100× faster than RTL sim)
- Algorithm iteration (easy to modify)
- Prototyping (quick validation)

**When to use which:**

Start with HLS for rapid prototyping and algorithm validation. Once the design stabilizes and you hit resource constraints, consider hand-coding critical kernels in RTL.

Example: LayerNorm kernel
- HLS version: 2 days development, 500 LUTs
- RTL version: 2 weeks development, 300 LUTs (40% savings)

Is 40% resource savings worth 5× more development time? Depends on your constraints.

---

## Part 11: Putting It Together

### From Kernels to Accelerators

Individual kernels compose into complete dataflow accelerators:

```
Input → Conv → Pool → FC → LayerNorm → Softmax → Output
```

Each arrow is an AXI-Stream connection. Each box is a kernel instance with specific PE/SIMD/bit-width configuration.

The accelerator's performance is the product of all kernel decisions:
- Bottleneck kernel determines throughput
- Sum of all latencies determines end-to-end latency
- Sum of all resources determines FPGA utilization

Optimizing one kernel in isolation doesn't guarantee better overall performance. You need system-level optimization—hence design space exploration.

### Configuration Enables Optimization

Without parameterization, you'd build one fixed design. You might hit resource limits, miss throughput targets, or waste silicon on over-provisioned kernels.

With parameterization, you can:
- Trade off different kernels (more resources to bottleneck, fewer to non-critical)
- Meet specific targets (exactly 60 FPS, exactly 75% FPGA utilization)
- Explore Pareto frontiers (latency vs resources)

Configuration converts a single design into a design space—from one option to thousands of options, from "take it or leave it" to "optimize for your constraints."

### Fast Exploration Enables Good Designs

The quality of your final design depends on how many configurations you can evaluate:

**Slow exploration** (days per configuration):
- Evaluate 5-10 configurations
- Rely heavily on intuition and guesswork
- Miss better solutions

**Fast exploration** (minutes per configuration):
- Evaluate hundreds to thousands
- Use automated search algorithms
- Find near-optimal solutions

Brainsmith's modeling system enables fast exploration through:
- Two-phase construction (build once, configure many)
- Estimation instead of synthesis
- Immutable navigation (parallel exploration without conflicts)
- Constraint-based pruning (eliminate invalid configurations early)

The faster you can iterate, the better your final design.

---

## Summary: Key Takeaways

1. **Hardware accelerators win on predictable, parallel workloads** by embedding computation in silicon (spatial computing vs temporal computing).

2. **Kernels are parameterized building blocks** with standardized interfaces (AXI-Stream) that compose into complete accelerators.

3. **The fundamental tradeoff is throughput vs resources**. More parallelism (PE/SIMD) means fewer clock cycles but more silicon area.

4. **Three shape hierarchies** (TENSOR → BLOCK → STREAM) reflect different perspectives: algorithm, architecture, implementation.

5. **The design space is constrained** by hardware realities (divisibility, resources, interfaces). Not all parameter combinations are valid.

6. **Schemas separate structure from storage**, enabling reuse across contexts and generating persistence layers automatically.

7. **Two-phase construction** (expensive build, cheap configure) makes exploring thousands of configurations tractable.

8. **Immutability enables safe parallel exploration**. Design points are snapshots, navigation creates new points.

9. **Estimation enables fast iteration**. Cycle/resource models approximate performance without hours of synthesis.

10. **Configuration converts constraints into optimization opportunities**. The better you can explore, the better your final design.

---

## Next Steps

Now that you understand the concepts, dive into the technical details:

- **[Kernels](../3-reference/kernels.md)**: The five-component architecture, parallelization parameters, performance modeling, creating custom kernels

- **[Kernel Modeling](../2-core-systems/kernel-modeling.md)**: Schema syntax, union type system, builder architecture, navigation API, complete code examples

- **[Design Space Exploration](../2-core-systems/design-space-exploration.md)**: Segment-based exploration, execution trees, blueprint configuration, caching strategies

- **[Dataflow Accelerators](dataflow-accelerators.md)**: AXI-Stream protocol, memory hierarchy, pipeline architecture, hardware composition
