# Brainsmith

**Compile PyTorch models into FPGA accelerators with automatic design space exploration**

Brainsmith takes quantized neural networks and generates optimized dataflow accelerators for FPGAs. Instead of manually configuring hundreds of hardware parameters, define your design space once and let Brainsmith explore, compare, and generate implementations automatically.

!!! info "Launch Date: November 2025"
    This repository is in pre-release and under active co-development by **Microsoft** and **AMD**. Stay tuned for our official release!

---

## Why Brainsmith?

**From Hours to Minutes**
Traditional FPGA development requires iterative manual configuration. Brainsmith's segment-based design space exploration (DSE) reuses computation across similar configurations, letting you explore hundreds of design points in the time it used to take to build one.

**Extensible by Design**
Add custom hardware kernels, optimization transforms, or build steps through a component registry. Your accelerator isn't limited to built-in operations.

**Production-Ready Workflow**
Built on industry-proven tools (FINN, QONNX, Brevitas) with support for major FPGA platforms. Goes from PyTorch to synthesizable RTL with resource estimates, performance metrics, and bitstream generation.

---

## See It In Action

Define your design space in YAML:

```yaml
name: "BERT Accelerator"
fpga_part: "xczu3eg-sbva484-1-e"
clock_ns: 5.0  # 200MHz
output: "estimates"

design_space:
  kernels:
    - MVAU
    - LayerNorm: [LayerNorm_hls, LayerNorm_rtl]
```

Run design space exploration:

```bash
smith dfc bert_quantized.onnx blueprint.yaml
```

Get performance estimates across all configurations in 30-60 minutes. Choose your design point, generate RTL, synthesize to bitstream.

---

## Get Started

**New to FPGA acceleration?**
→ [Quick Start](getting-started/quickstart.md) - Run the BERT example and explore results

**Ready to build your own?**
→ [Installation](getting-started/installation.md) - Set up your development environment
→ [Blueprints](user-guide/blueprints.md) - Learn the configuration format

**Extending Brainsmith?**
→ [Component Registry](architecture/registry.md) - Add custom kernels and transforms *(Phase 2)*
→ [Kernel Development](kernel-development/index.md) - Create hardware accelerators *(Phase 2)*

---

## Built With

Brainsmith builds upon:

- [FINN](https://github.com/Xilinx/finn) - Dataflow compiler for quantized neural networks
- [QONNX](https://github.com/fastmachinelearning/qonnx) - Quantized ONNX representation
- [Brevitas](https://github.com/Xilinx/brevitas) - PyTorch quantization library

Developed through collaboration between **Microsoft** and **AMD**.

**License**: MIT - see [LICENSE](https://github.com/microsoft/brainsmith/blob/main/LICENSE)