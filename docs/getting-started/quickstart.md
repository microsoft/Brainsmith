# Quick Start

Get started with Brainsmith in 5 minutes.

## Prerequisites

Make sure you've completed the [installation](installation.md).

## Run Your First DSE

### 1. Prepare Your Model

For this quickstart, we'll use the included BERT example:

```bash
cd examples/bert
```

### 2. Run the Quick Test

```bash
./quicktest.sh
```

This will:

1. Generate a folding configuration for minimal resources
2. Build a single-layer BERT accelerator
3. Run RTL simulation to verify correctness

!!! info "Build Time"
    The quicktest takes approximately 30-60 minutes, depending on your system.

### 3. Explore Results

Results are saved in `examples/bert/quicktest/`:

```
quicktest/
├── model.onnx              # Quantized ONNX model
├── final_output/           # Generated RTL and reports
│   ├── stitched_ip/       # Synthesizable RTL
│   └── report/            # Performance estimates
└── build_dataflow.log     # Build log
```

## Understanding the Output

### Performance Report

Check `final_output/report/estimate_reports.json`:

```json
{
  "throughput": "1234.5 fps",
  "latency": "0.81 ms",
  "resources": {
    "LUT": 12345,
    "FF": 23456,
    "BRAM": 34,
    "DSP": 56
  }
}
```

### RTL Output

The generated RTL is in `final_output/stitched_ip/`:

- `finn_design_wrapper.v` - Top-level module
- `*.v` - Individual kernel implementations

## Next Steps

### Customize the Design

Edit the blueprint to explore different configurations:

```yaml
# bert_quicktest.yaml
design_space:
  kernels:
    - name: MatMul
      backends:
        - matmul_rtl  # Try different backends
  steps:
    - step_target_fps_parallelization:
        target_fps: 100  # Adjust target performance
```

### Run Full DSE

```bash
smith model.onnx blueprint.yaml --output-dir ./results
```

## Learn More

- [Architecture Overview](../architecture/overview.md) - Understand how Brainsmith works
- [CLI Reference](../api-reference/core.md) - Explore CLI commands
