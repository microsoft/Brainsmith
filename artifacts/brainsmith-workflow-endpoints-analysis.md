# BrainSmith Workflow Endpoints Analysis

## Current End Point: Yes, It Runs Real FINN Builds

The BrainSmith workflow **does execute actual FINN builds**. Here's exactly what happens:

### 1. Execution Flow

```python
# In FINNAdapter.build():
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig

# Convert dict to FINN config
config = DataflowBuildConfig(**config_dict)

# Execute FINN build (this is the real FINN)
exit_code = build_dataflow_cfg(str(input_model), config)
```

The workflow:
1. **Explorer** orchestrates execution
2. **Executor** traverses the tree, calling FINNAdapter for each segment
3. **FINNAdapter** runs actual FINN builds via `build_dataflow_cfg()`
4. **FINN** processes the model through transformation steps

### 2. Output Organization

Outputs follow a hierarchical structure mirroring the execution tree:

```
output_dir/
├── tree.json                           # Execution tree structure
├── summary.json                        # Execution results
│
├── root/                              # Root segment
│   ├── input.onnx                     # Input model copy
│   ├── root_output.onnx               # Output from segment
│   └── intermediate_models/           # FINN working directory
│       ├── step_1_cleanup.onnx
│       ├── step_2_streamline.onnx
│       └── ...
│
├── cleanup_quantize_int8/             # Branch path segment
│   ├── input.onnx
│   ├── cleanup_quantize_int8_output.onnx
│   ├── report/
│   │   └── estimate_layer_resources.json
│   └── intermediate_models/
│       └── ...
│
└── cleanup_quantize_int4/             # Alternative branch
    ├── input.onnx
    ├── cleanup_quantize_int4_output.onnx
    └── ...
```

**Key Points:**
- Each segment gets its own directory
- Segment IDs become directory paths (e.g., "cleanup/quantize" → "cleanup_quantize/")
- Artifacts are copied from parent to child segments at branch points
- FINN creates `intermediate_models/` with step-by-step transformations

### 3. Output Products & Artifacts

Based on `output_stage` configuration:

#### **"generate_reports"** (Analysis Only)
```
segment_dir/
├── report/
│   ├── estimate_layer_resources.json    # Per-layer resource estimates
│   ├── estimate_layer_cycles.json       # Cycle counts
│   └── estimate_total_resources.json    # Total FPGA resources
└── intermediate_models/*.onnx           # Transformed models
```

#### **"compile_and_package"** (RTL Generation)
```
segment_dir/
├── report/                              # All analysis reports
├── rtlsim_performance/                  # RTL simulation results
├── stitched_ip/                         # Vivado IP blocks
│   ├── ip/                              # Individual IP cores
│   ├── vivado_project/                  # Vivado project files
│   └── stitched_design.v                # Top-level Verilog
└── intermediate_models/*.onnx
```

#### **"synthesize_bitstream"** (Full Hardware)
```
segment_dir/
├── report/                              # All analysis reports
├── rtlsim_performance/                  # RTL simulation results
├── stitched_ip/                         # Vivado IP blocks
├── bitfile/
│   ├── design.bit                       # FPGA bitstream
│   ├── design.hwh                       # Hardware handoff file
│   └── timing_report.txt                # Timing analysis
├── deploy/                              # Deployment package
│   ├── driver/                          # PYNQ drivers
│   ├── notebooks/                       # Jupyter examples
│   └── runtime/                         # Runtime libraries
└── intermediate_models/*.onnx
```

### 4. Result Collection

The `TreeExecutionResult` collects:

```python
@dataclass
class SegmentResult:
    success: bool
    segment_id: str
    output_model: Optional[Path]      # Path to output ONNX
    output_dir: Optional[Path]        # Segment directory
    error: Optional[str]              # Error message if failed
    execution_time: float             # Time in seconds
    cached: bool                      # Was result cached?

@dataclass
class TreeExecutionResult:
    segment_results: Dict[str, SegmentResult]  # All segments
    total_time: float
    stats: Dict[str, int]  # succeeded, failed, cached, skipped
```

### 5. Caching Mechanism

Simple file-based caching:
```python
# Check if output exists
output_onnx = segment_dir / f"{safe_name}_output.onnx"
if output_onnx.exists():
    # Mark as cached, skip execution
    return SegmentResult(cached=True, ...)
```

### 6. Error Handling

- **Fail-fast mode**: Stop on first error
- **Continue mode**: Failed segments mark descendants as skipped
- All errors captured in `SegmentResult.error`

### 7. FINN Integration Details

**Necessary Workarounds in FINNAdapter:**

1. **Working Directory Change**: FINN requires os.chdir() to output directory
2. **Output Discovery**: FINN doesn't return output path, must search `intermediate_models/`
3. **Model Copying**: FINN modifies models in-place, must copy inputs

### Example Output Summary

After running a design space exploration:

```json
{
  "stats": {
    "total": 4,
    "succeeded": 3,
    "failed": 1,
    "cached": 0,
    "skipped": 0
  },
  "total_time": 245.3,
  "segments": {
    "root": {
      "success": true,
      "execution_time": 45.2,
      "output_model": "output/root/root_output.onnx"
    },
    "cleanup_quantize_int8": {
      "success": true,
      "execution_time": 89.1,
      "output_model": "output/cleanup_quantize_int8/cleanup_quantize_int8_output.onnx"
    },
    "cleanup_quantize_int4": {
      "success": false,
      "error": "Quantization failed: bit width not supported",
      "execution_time": 12.3
    }
  }
}
```

## Key Insights

1. **Yes, it runs real FINN** - Not mocked, actual hardware synthesis
2. **Organized outputs** - Hierarchical structure matching execution tree
3. **Complete artifacts** - From estimates to bitstreams
4. **Smart caching** - Reuses completed segments
5. **Error isolation** - Failed branches don't affect others

The workflow endpoint is **fully functional hardware generation** via FINN, with systematic organization of all artifacts for design space exploration.