# FINN and Brainsmith Interfacing

## Interface Definitions

Both FINN and Brainsmith accept a QONNX network and configuration details, then create a custom Dataflow Core implementing that model for FPGA. However, they offer interfaces to optimize with different levels of granularity.

**FINN Builder:** Runs FINN flow to optimize the within the *search space*.
- [DataflowBuildConfig](DataflowBuildConfig) – Defines the local search space and the DSE strategies used to optimize within that space.
- [build_dataflow](build_dataflow) – Runs the FINN flow, configured by *DataflowBuildConfig*.

**Brainsmith Core:** Runs FINN iteratively to optimize the within the *design space*.
- *Blueprint* – Defines a range of parameters that define various search spaces, the set of which constitute the global design space, and the DSE strategies to optimize within that space.
- *forge* – Iteratively runs the FINN flow, configured by *Blueprint*.

## Search/Design Spaces

**Search Space** – a set of potential implementations of an FPGA accelerator architecture discoverable by a DSE strategy.

**Design Space** – a set of architectures and DSE strategies whose permutations construct potential search spaces.

## Optimization Points

| Local Search Space (FINN) | Global Design Space (Brainsmith) |
|---------------------------|-----------------------------------|
| Network optimizations | Platform (board, fpga_part) |
| FIFO sizing | Kernel Implementations |
| Kernel parallelism (PE, SIMD, Tiling) | DSE model transforms (streamlining) |
| Kernel variations (RTL vs HLS, LUT vs DSP) | DSE HW transforms (auto-folding) |
| | HW targets (target clk, mvau_wwidth_max) |