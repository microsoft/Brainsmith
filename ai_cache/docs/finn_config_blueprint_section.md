# FINN Configuration in BrainSmith Blueprints

The `finn_config` section in BrainSmith blueprints provides direct control over FINN's DataflowBuildConfig parameters, allowing users to override any default behavior set by BrainSmith.

## Overview

While BrainSmith provides sensible defaults for FINN compilation, advanced users may need fine-grained control over the build process. The `finn_config` section allows you to specify any DataflowBuildConfig parameter directly.

## Basic Usage

Add a `finn_config` section to your blueprint YAML:

```yaml
finn_config:
  synth_clk_period_ns: 4.0  # 250MHz
  target_fps: 1000
  verbose: true
```

## Available Parameters

### Build Control
- `steps`: Custom step list (advanced users only)
- `start_step`: Start from specific step (e.g., "step_hw_codegen")
- `stop_step`: Stop at specific step (e.g., "step_measure_rtlsim_performance")
- `output_dir`: Build output directory
- `save_intermediate_models`: Save intermediate ONNX files (default: true)

### Performance & Optimization
- `synth_clk_period_ns`: Target clock period for synthesis (default: 5.0 = 200MHz)
- `hls_clk_period_ns`: HLS synthesis clock (defaults to synth_clk_period_ns)
- `target_fps`: Target inference performance
- `folding_config_file`: Path to pre-computed folding configuration JSON file
- `folding_two_pass_relaxation`: Two-pass folding optimization (default: true)
- `mvau_wwidth_max`: Max PE stream width (default: 36)

### Verification
- `verify_steps`: List of verification points:
  - `"QONNX_TO_FINN_PYTHON"`: After QONNX to FINN conversion
  - `"TIDY_UP_PYTHON"`: After tidy up
  - `"STREAMLINED_PYTHON"`: After streamlining
  - `"FOLDED_HLS_CPPSIM"`: After folding (C++ simulation)
  - `"NODE_BY_NODE_RTLSIM"`: After IP generation
  - `"STITCHED_IP_RTLSIM"`: After stitching
- `verify_input_npy`: Test input file
- `verify_expected_output_npy`: Expected output file
- `verify_save_full_context`: Save full execution context (default: false)
- `verify_save_rtlsim_waveforms`: Save simulation waveforms (default: false)
- `verification_atol`: Verification tolerance (default: 0.1)

### Hardware Generation
- `generate_outputs`: List of outputs to generate:
  - `"STITCHED_IP"`: Stitched IP core
  - `"ESTIMATE_REPORTS"`: Resource estimates
  - `"OOC_SYNTH"`: Out-of-context synthesis
  - `"RTLSIM_PERFORMANCE"`: RTL simulation performance
  - `"BITFILE"`: Full bitfile
  - `"PYNQ_DRIVER"`: PYNQ Python driver
  - `"DEPLOYMENT_PACKAGE"`: Deployment package
- `stitched_ip_gen_dcp`: Generate DCP for stitched IP (default: false)
- `board`: Target board (e.g., "U250", "Pynq-Z1")
- `fpga_part`: FPGA part number
- `shell_flow_type`: "VIVADO_ZYNQ" or "VITIS_ALVEO"
- `enable_hw_debug`: Insert debug cores (default: false)

### FIFO Configuration
- `auto_fifo_depths`: Auto-size FIFOs (default: true)
- `auto_fifo_strategy`: "CHARACTERIZE" or "LARGEFIFO_RTLSIM"
- `large_fifo_mem_style`: "AUTO", "BRAM", "LUTRAM", or "URAM"
- `split_large_fifos`: Split FIFOs > 32768 (default: false)
- `fifosim_n_inferences`: Inferences for FIFO sizing
- `fifosim_input_throttle`: Throttle simulation input (default: true)
- `fifosim_save_waveform`: Save FIFO sizing waveforms (default: false)

### Other Options
- `standalone_thresholds`: Separate threshold layers (default: false)
- `minimize_bit_width`: Optimize bit widths (default: true)
- `max_multithreshold_bit_width`: Quant to MultiThreshold conversion (default: 8)
- `rtlsim_batch_size`: RTL simulation batch size (default: 1)
- `verbose`: Detailed output (default: false)
- `enable_build_pdb_debug`: PDB on failure (default: true)

## Precedence Rules

1. `finn_config` parameters have highest priority
2. `build_configuration` parameters (legacy) are applied first
3. Top-level blueprint parameters (e.g., `stop_step` at root) for backward compatibility
4. BrainSmith defaults are used for anything not specified

## Examples

### Performance-Optimized Build
```yaml
finn_config:
  synth_clk_period_ns: 3.33  # 300MHz
  target_fps: 2000
  mvau_wwidth_max: 128
  folding_two_pass_relaxation: false  # Single pass for lower latency
  generate_outputs: ["STITCHED_IP", "RTLSIM_PERFORMANCE"]
```

### Debug-Focused Build
```yaml
finn_config:
  verbose: true
  save_intermediate_models: true
  verify_steps: ["FOLDED_HLS_CPPSIM", "NODE_BY_NODE_RTLSIM", "STITCHED_IP_RTLSIM"]
  verify_save_rtlsim_waveforms: true
  enable_hw_debug: true
  stop_step: "step_hw_ipgen"  # Stop before FIFO sizing
```

### Resource-Optimized Build
```yaml
finn_config:
  minimize_bit_width: true
  standalone_thresholds: false
  large_fifo_mem_style: "LUTRAM"  # Use distributed RAM
  auto_fifo_strategy: "CHARACTERIZE"
  vitis_opt_strategy: "SIZE"
```

### Using Pre-Computed Folding Configuration
```yaml
finn_config:
  folding_config_file: "./configs/l1_simd12_pe8.json"  # Pre-computed parallelization
  target_fps: null  # Disable auto-folding when using folding_config_file
  synth_clk_period_ns: 5.0
  generate_outputs: ["STITCHED_IP", "RTLSIM_PERFORMANCE"]
```

## Implementation Notes

- Enum values (like `generate_outputs` items) should be provided as strings
- Lists are supported for appropriate parameters
- Invalid enum values will generate warnings but not fail the build
- Parameters are type-checked by FINN's DataflowBuildConfig
- When using `folding_config_file`, set `target_fps` to null to avoid conflicts
- The `folding_config_file` overrides any auto-folding from `target_fps`

## Migration from Legacy Format

If you have parameters in `build_configuration`, they still work but consider moving them to `finn_config`:

```yaml
# Old way (still works)
build_configuration:
  stop_step: "step_hw_ipgen"
  
# New way (preferred)
finn_config:
  stop_step: "step_hw_ipgen"
```