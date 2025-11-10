---
hide:
  - toc
---

# Brainsmith

## Compile Neural Networks to FPGA Accelerators

Brainsmith is an end-to-end compiler to transform ONNX models into dataflow accelerators for FPGAs. Through design space exploration, it evaluates hardware configurations to find the optimal configuration for your use case.

<div class="comparison-grid" markdown>

<div class="comparison-item" markdown>

<div class="comparison-title">ONNX Model</div>

<img src="images/mha_onnx.png" alt="ONNX Graph Structure">

</div>

<div class="comparison-arrow">
→
</div>

<div class="comparison-item" markdown>

<div class="comparison-title">Dataflow Core</div>

<img src="images/bert_dfc.png" alt="Generated Dataflow Accelerator">

</div>

</div>

**Automated RTL generation from ONNX models. Design space exploration to identify optimal configurations.**

[Get Started](getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/microsoft/brainsmith){ .md-button }


## Key Features

<div class="grid cards" markdown>

-   :material-chart-timeline-variant:{ .lg .middle } **Automatic Design Space Exploration**

    ---

    Navigate parallelization factors, resource allocation, and architectural choices. Explore multiple configurations to identify promising designs.

-   :material-code-braces:{ .lg .middle } **Schema-Driven Kernel Development**

    ---

    Define hardware semantics declaratively. Validation, design space construction, and interface generation are derived from schema definitions.

-   :material-chip:{ .lg .middle } **Synthesizable RTL Generation**

    ---

    Generate Verilog/VHDL with standard AXI-Stream interfaces. Compatible with Vivado IP Integrator workflows.

-   :material-puzzle:{ .lg .middle } **Growing Kernel Library**

    ---

    Built-in support for MVAU, LayerNorm, Softmax, and other common operations. Extensible architecture for adding custom kernels.

-   :material-speedometer:{ .lg .middle } **Performance Estimation**

    ---

    Resource estimation, cycle-accurate simulation support, and throughput analysis. Evaluate design tradeoffs before synthesis.

-   :material-layers:{ .lg .middle } **Multi-Layer Offload**

    ---

    Scale to large models with constant FPGA resources. Stream weights from external memory to process arbitrarily deep networks without increasing hardware footprint.

</div>


## Basic Usage

Generate an accelerator with a single command:

```bash
# Design space exploration and RTL generation
smith model.onnx blueprint.yaml

# Output: RTL + performance estimates + resource reports
```


## Example: BERT Accelerator

```yaml
# blueprint.yaml - Define your design space
name: "BERT Accelerator"
clock_ns: 5.0  # 200MHz target

design_space:
  kernels:
    - MVAU           # Matrix-vector operations
    - LayerNorm      # Layer normalization
    - Softmax        # Attention softmax

  steps:
    - "streamline"           # Graph optimization
    - "infer_kernels"        # Hardware kernel mapping
    - "specialize_layers"    # Backend selection
    - "dataflow_partition"   # Multi-layer offload
```

Run design space exploration:

```bash
smith bert.onnx blueprint.yaml --output-dir ./results
```

Results include:
- Synthesizable RTL in `results/stitched_ip/`
- Performance estimates in `results/report/estimate_reports.json`
- Detailed build logs for debugging

The example targets V80 platform using Vivado 2024.2 and is compatible with Xilinx Zynq/Ultrascale+ platforms.

*See examples/bert for full implementation*


## Open Source & Collaborative

Brainsmith is MIT-licensed and builds upon a foundation of proven open-source tools:

- [FINN](https://github.com/Xilinx/finn) - Dataflow compiler for quantized neural networks
- [QONNX](https://github.com/fastmachinelearning/qonnx) - Quantized ONNX representation
- [Brevitas](https://github.com/Xilinx/brevitas) - PyTorch quantization library

Brainsmith extends FINN with automated design space exploration, blueprint inheritance, and a schema-driven kernel system. FINN provides the low-level RTL generation and QONNX transformations.

Developed through collaboration between **Microsoft** and **AMD**.

**License**: MIT - see [LICENSE](https://github.com/microsoft/brainsmith/blob/main/LICENSE)


## Community & Support

- [Feature Roadmap](https://github.com/orgs/microsoft/projects/2017) - See what's planned and in progress
- [GitHub Issues](https://github.com/microsoft/brainsmith/issues) - Report bugs or request features
- [GitHub Discussions](https://github.com/microsoft/brainsmith/discussions) - Ask questions and share experiences
- [Contributing Guide](https://github.com/microsoft/brainsmith/blob/main/CONTRIBUTING.md) - Learn how to contribute

**New to Brainsmith?** [Get started with the quickstart guide →](getting-started.md)
