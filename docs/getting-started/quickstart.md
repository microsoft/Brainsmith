# Quick Start

Get started with Brainsmith in 5 minutes by running the BERT example.

## Prerequisites

Make sure you've completed the [installation](installation.md) and activated your environment:

```bash
# Option 1: direnv users
cd /path/to/brainsmith

# Option 2: Manual activation
source .venv/bin/activate && source .brainsmith/env.sh
```

## Run Your First DSE

### 1. Navigate to the BERT Example

```bash
cd examples/bert
```

### 2. Run the Quick Test

```bash
./quicktest.sh
```

This automated script will:

1. **Generate a quantized ONNX model** - Single-layer BERT for rapid testing
2. **Generate a folding configuration** - Minimal resource usage
3. **Build the dataflow accelerator** - RTL generation with FINN
4. **Run RTL simulation** - Verify correctness

!!! info "Build Time"
    The quicktest takes approximately **30-60 minutes**, depending on your system. The build process involves:

    - ONNX model transformations
    - Kernel inference and specialization
    - RTL code generation
    - IP packaging with Vivado
    - RTL simulation

### 3. Monitor Progress

The script outputs progress to the console. Key stages include:

```
[INFO] Generating BERT ONNX model...
[INFO] Creating folding configuration...
[INFO] Running dataflow build...
  ├─ Streamlining
  ├─ QONNX to FINN conversion
  ├─ Dataflow partition creation
  ├─ Kernel inference
  ├─ RTL code generation
  ├─ IP packaging
  └─ RTL simulation
[SUCCESS] Build complete!
```

---

## Explore Results

Results are saved in `examples/bert/quicktest/`:

```
quicktest/
├── model.onnx              # Quantized ONNX model
├── final_output/           # Generated RTL and reports
│   ├── stitched_ip/       # Synthesizable RTL
│   │   ├── finn_design_wrapper.v
│   │   └── *.v
│   └── report/            # Performance estimates
│       └── estimate_reports.json
└── build_dataflow.log     # Detailed build log
```

### Understanding the Output

#### Performance Report

Check `final_output/report/estimate_reports.json`:

```json
{
  "critical_path_cycles": 123,
  "max_cycles": 456,
  "estimated_throughput_fps": 1234.5,
  "resources": {
    "LUT": 12345,
    "FF": 23456,
    "BRAM_18K": 34,
    "DSP48": 56
  }
}
```

#### RTL Output

The generated RTL is in `final_output/stitched_ip/`:

- `finn_design_wrapper.v` - Top-level wrapper with AXI stream interfaces
- Individual kernel implementations (e.g., `MVAU_hls_*.v`, `Thresholding_rtl_*.v`)
- Stream infrastructure (FIFOs, width converters, etc.)

---

## Customize the Design

Now that you've run the basic example, try customizing it:

### Adjust Target Performance

Edit `bert_quicktest.yaml`:

```yaml
design_space:
  steps:
    - step_target_fps_parallelization:
        target_fps: 100  # Increase target throughput
```

Higher target FPS will:
- Increase parallelization factors
- Use more FPGA resources
- Reduce latency

### Change Backend

Try different hardware implementations:

```yaml
design_space:
  kernels:
    - MVAU: [mvau_hls, mvau_rtl]  # Try both HLS and RTL backends
```

### Run Full Example

The full BERT demo (not just quicktest) processes actual text:

```bash
python bert_demo.py --config bert_demo.yaml
```

This uses a larger model and generates a complete accelerator.

---

## Understanding Blueprints

The `bert_quicktest.yaml` file is a **Blueprint** - a YAML configuration that defines your design space:

```yaml
name: "BERT Quicktest"
clock_ns: 5.0           # 200MHz target clock
output: "rtlsim"        # Generate RTL and run simulation

design_space:
  kernels:              # Hardware kernels to use
    - MVAU
    - Thresholding
    - StreamingFCLayer

  steps:                # Build pipeline steps
    - streamline
    - qonnx_to_finn
    - step_create_dataflow_partition
    - step_specialize_layers
    - step_target_fps_parallelization:
        target_fps: 50
    - step_minimize_bit_width
    - step_generate_estimate_reports
    - step_hw_codegen
    - step_hw_ipgen
    - step_set_fifo_depths
    - step_create_stitched_ip
    - step_synthesize_bitfile
    - step_make_pynq_driver
    - step_deployment_package
```

Learn more in the [Blueprint Reference](../developer-guide/3-reference/blueprints.md).

---

## Next Steps

### Explore Design Space

Run multiple configurations to compare trade-offs:

```bash
# Low resource usage
smith dfc model.onnx blueprint_small.yaml --output-dir ./results_small

# High performance
smith dfc model.onnx blueprint_fast.yaml --output-dir ./results_fast
```

### Try Custom Models

1. Quantize your PyTorch model with Brevitas
2. Export to ONNX
3. Create a blueprint
4. Run DSE:
   ```bash
   smith dfc my_model.onnx my_blueprint.yaml
   ```

### Learn the Tools

- [Blueprint Reference](../developer-guide/3-reference/blueprints.md) - Master the YAML format
- [CLI Reference](../developer-guide/3-reference/cli.md) - Explore all commands
- [Design Space Exploration](../developer-guide/2-core-systems/design-space-exploration.md) - Understand DSE concepts

---

## Troubleshooting

### Build Fails During Kernel Inference

**Error:** `Could not find suitable kernel for operation X`

**Solution:** Add the missing kernel to your blueprint's `kernels` list.

### RTL Simulation Fails

**Error:** `Simulation mismatch detected`

**Solution:** This usually indicates a configuration issue. Check:
- Folding factors are valid for your model dimensions
- FIFO depths are sufficient
- Input/output data types match model requirements

### Out of Memory

**Error:** `MemoryError` during build

**Solution:**
- Close other applications
- Reduce model size for testing
- Use `--stop-step` to build incrementally

### Vivado Not Found

**Error:** `vivado: command not found`

**Solution:**
1. Check config: `brainsmith project info`
2. Re-activate environment: `source .brainsmith/env.sh`
3. Verify Vivado path: `ls $XILINX_PATH`

---

## Getting Help

- Check the [build log](../../examples/bert/quicktest/build_dataflow.log) for detailed error messages
- Search existing [GitHub Issues](https://github.com/microsoft/brainsmith/issues)
- Ask on [GitHub Discussions](https://github.com/microsoft/brainsmith/discussions)
