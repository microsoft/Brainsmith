# Kernel Workflows

End-to-end workflows for deploying kernels, design space exploration, troubleshooting, and debugging.


## Complete Workflow: ONNX to Bitstream

This section shows the complete journey from ONNX model to FPGA bitstream.

### 1. Prepare ONNX Model

```python
# create_model.py
import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantLayerNorm
from qonnx.core.modelwrapper import ModelWrapper

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = QuantLayerNorm(64, weight_bit_width=8)
        self.fc = QuantLinear(64, 10, weight_bit_width=8)

    def forward(self, x):
        x = self.norm(x)
        return self.fc(x)

model = SimpleModel()
torch.onnx.export(model, torch.randn(1, 64), "model.onnx")
```

### 2. Create Blueprint

```yaml
# blueprint.yaml
design_space:
  kernels:
    - LayerNorm
    - MatrixVectorActivation

  steps:
    # Preprocessing
    - "streamline"
    - "infer_shapes"

    # Kernel inference (pattern matching)
    - "infer_hardware_kernels"

    # DSE and folding
    - "target_fps_parallelization"
    - "apply_folding_config"

    # HLS synthesis
    - "prepare_hls"
    - "hls_synthesis"

    # Validation
    - "folded_hls_cppsim"
```

### 3. Run Compilation Pipeline

```bash
# Full pipeline to bitstream
smith model.onnx blueprint.yaml

# Stop at specific step for debugging
smith model.onnx blueprint.yaml --stop-step folded_hls_cppsim

# Resume from checkpoint
smith model.onnx blueprint.yaml --start-step hls_synthesis
```

### 4. Kernel Pattern Matching

During `infer_hardware_kernels` step:

```python
# Transform iterates over all ONNX nodes
for node in model.graph.node:
    # Try each registered kernel
    for KernelClass in registered_kernels:
        if KernelClass.can_infer_from(node, model):
            # Transform ONNX → HW
            result = KernelClass.infer_from(node, model, insert_index)
            apply_transformation(result)
            break
```

Your `can_infer_from()` determines if kernel matches the node.

### 5. Design Space Exploration

During `target_fps_parallelization` step:

```python
# For each kernel in graph
for node in hw_nodes:
    op = getCustomOp(node)
    design_space = op.get_valid_ranges(model)

    # Explore configurations
    for point in explore_pareto_optimal(design_space):
        cycles = point.initiation_interval
        resources = estimate_resources(point)

        if meets_constraints(cycles, resources, target_fps):
            op.apply_design_point(point)
            break
```

### 6. Code Generation

During `prepare_hls` step:

```python
# For each hardware kernel
for node in hw_nodes:
    op = getCustomOp(node)
    backend = get_backend(op)  # Finds MyKernel_hls

    # Generate HLS C++
    backend.prepare_codegen(model, fpgapart)
    backend.generate_params(model, ".")

    # Creates:
    # - top_function.cpp (with $DEFINES$, $DOCOMPUTE$ resolved)
    # - Makefile
    # - Tcl scripts
```

### 7. Validation

```bash
# CPPSim: Functional validation (fast)
smith model.onnx blueprint.yaml --stop-step folded_hls_cppsim

# RTLSim: Cycle-accurate validation (slow, accurate)
smith model.onnx blueprint.yaml --stop-step stitched_ip_rtlsim

# Compare with reference
python validate_output.py
```

### 8. FPGA Deployment

```bash
# Generate bitstream
smith model.onnx blueprint.yaml --stop-step bitfile

# Deploy to target
vivado -mode batch -source deploy.tcl

# Test on hardware
python test_fpga.py --bitfile output/design.bit
```


## Design Space Exploration

Schemas automatically generate design spaces from interface specifications:

```python
# Initialize kernel with model context
op._ensure_ready(model)

# Access design space
design_space = op.design_space
print(f"Parameters: {design_space.parameters.keys()}")
# Output: Parameters: dict_keys(['SIMD', 'PE', 'ram_style'])

# Navigate configurations
# Configure initial point, then sweep
base_point = design_space.configure({"SIMD": 1})
for point in base_point.sweep_dimension("SIMD"):
    cycles = point.initiation_interval
    resources = estimate_resources(point)
    # Evaluate performance vs resource tradeoff

# Apply chosen configuration
op.apply_design_point(best_point)
```

### Design Point Navigation

Design points are immutable snapshots with fluent navigation APIs:

**Interface-based API** (for stream parameters):
```python
point = point.with_input_stream(0, 32)   # Set first input PE=32
point = point.with_output_stream(0, 16)  # Set first output PE=16
```

**Dimension-based API** (for generic DSE parameters):
```python
point = point.with_dimension("SIMD", 64)
point = point.with_dimension("ram_style", "distributed")
```

**Accessing configuration:**
```python
simd = point.config["SIMD"]
input_shape = point.inputs["input"].stream_shape
tensor_shape = point.inputs["input"].tensor_shape
```

### Integration with Blueprint DSE

Brainsmith's segment-based DSE automatically explores kernel design spaces:

1. **Schema generates design space** - Valid parameter ranges from stream_tiling
2. **Blueprint specifies steps** - Transformation pipeline including DSE
3. **DSE explores configurations** - Evaluates performance vs resources
4. **Best points applied** - Optimal configurations used for RTL generation

**Blueprint example:**
```yaml
design_space:
  kernels:
    - LayerNorm
  steps:
    - "infer_kernels"
    - "target_fps_parallelization"  # DSE step
    - "apply_folding_config"
```

The DSE step uses kernel schemas to:
- Determine valid parallelization factors
- Estimate resource usage
- Calculate throughput
- Find Pareto-optimal configurations

See [Design Space Exploration API](../api/dse.md) for complete DSE workflow and [Dataflow API Reference](../api/dataflow.md) for design space navigation details.


## Troubleshooting Guide

### Common Errors and Solutions

#### Error: "Design space not initialized"

```
RuntimeError: MyKernel_node42: Not initialized. Call a method with model_w parameter first.
```

**Cause:** Accessing `design_space` or `design_point` before initialization

**Solution:** Most KernelOp methods trigger initialization automatically. If calling from custom code:
```python
op._ensure_ready(model_w)
design_space = op.design_space  # Now safe
```

#### Error: "Parameter not found in dimensions dict"

```
ValueError: Interface 'output' references stream param 'PE' not found in dimensions dict
```

**Cause:** Stream parameter referenced in `stream_tiling` but not defined

**Solution:** Parameters in `stream_tiling` are auto-created. This usually means:
1. Typo in parameter name
2. Parameter appears in output but not input (no dimension to divide)

```python
# Check parameter appears in at least one block_shape
inputs=[df.InputSchema(
    block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM, "K"],  # K must exist here
    stream_tiling=[1, 1, 1, "SIMD"]                      # For SIMD to divide it
)]
```

#### Error: "Template length exceeds reference rank"

```
ValueError: template length 5 exceeds reference rank 4
```

**Cause:** Tiling template has more dimensions than tensor

**Solution:** Use rank-agnostic pattern:
```python
# Wrong
block_tiling=[FULL_DIM, FULL_DIM, FULL_DIM, FULL_DIM, "K"]  # Assumes 5D

# Right
block_tiling=FULL_SHAPE  # Adapts to any rank
```

#### Error: "Stream width mismatch"

```
AssertionError: Stream width mismatch: producer=128 bits, consumer=256 bits
```

**Cause:** Adjacent kernels have incompatible stream parallelization

**Solution:** Use dimension derivation to match widths:
```python
outputs=[df.OutputSchema(
    stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)]
)]
```

Or add explicit constraint:
```python
constraints=[
    df.DimensionEquals("input", "output", -1, ShapeHierarchy.STREAM)
]
```

#### Error: "Invalid design point configuration"

```
ValueError: Invalid SIMD=128. Valid range: [1, 64], values: (1, 2, 4, 8, 16, 32, 64)
```

**Cause:** Configuration value not in valid range

**Solution:** Check valid ranges before configuring:
```python
simd_param = design_space.get_ordered_parameter("SIMD")
print(f"Valid SIMD: {simd_param.min()} to {simd_param.max()}")
print(f"All values: {simd_param.values}")

# Use valid value
point = design_space.configure({"SIMD": simd_param.max()})
```

### Debug Workflow

1. **Check schema builds:** `SCHEMA.validate()` in Python REPL
2. **Verify registration:** `brainsmith registry | grep MyKernel`
3. **Test pattern matching:** Add logging to `can_infer_from()`
4. **Validate transformation:** Check ONNX graph before/after transform
5. **Test reference impl:** Run `execute_node()` with sample data
6. **CPPSim validation:** `--stop-step folded_hls_cppsim`
7. **RTLSim validation:** `--stop-step stitched_ip_rtlsim`

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# See design space construction
logger = logging.getLogger("brainsmith.dataflow.builder")
logger.setLevel(logging.DEBUG)

# See parameter computation
logger = logging.getLogger("brainsmith.dataflow.template_resolution")
logger.setLevel(logging.DEBUG)
```


## Performance Optimization

### Profiling Kernels

**Identify bottlenecks:**

1. **Cycles per inference** - From RTLSim output:
   ```
   RTLSim: 12,345 cycles for 1 inference
   Throughput: 81 FPS @ 100 MHz
   ```

2. **Resource usage** - From HLS synthesis reports:
   ```
   DSP: 240 / 2520 (9%)
   BRAM: 150 / 912 (16%)
   LUT: 45000 / 274080 (16%)
   ```

3. **Critical paths** - From timing reports:
   ```
   Worst slack: -0.523 ns
   Critical path: LayerNorm_0 → FIFO_12 → MatMul_1
   ```

### Optimization Strategies

**Increase parallelization:**
```python
# Before: SIMD=8, 500 cycles
# After: SIMD=16, 250 cycles (if resources allow)
point = design_space.configure({"SIMD": 16})
```

**Pipeline depth tuning:**
```python
# Add pipeline stages in HLS
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY max=10
```

**Memory optimization:**
```python
# Switch RAM style for better timing
dse_parameters={
    "ram_style": df.ParameterSpec("ram_style", {"distributed", "block", "ultra"})
}
```

**FIFO depth optimization:**
```bash
# Enable automatic FIFO depth tuning
smith model.onnx blueprint.yaml --finn-config split_large_fifos=true
```

### Performance Checklist

- [ ] Parallelization factors maximize throughput without exceeding resources
- [ ] Critical paths meet timing (positive slack)
- [ ] FIFO depths prevent stalls (check RTLSim for backpressure)
- [ ] Pipeline initiation interval (II) is 1 for throughput-critical kernels
- [ ] Memory accesses are pipelined and don't create bottlenecks
- [ ] DSP utilization is balanced (not over/under-subscribed)


## Advanced Debugging

### Inspecting Intermediate Models

Save models at each step:

```yaml
finn_config:
  save_intermediate_models: true
```

Outputs appear in build directory:
```
build/
├── step_000_streamline.onnx
├── step_001_infer_shapes.onnx
├── step_002_infer_hardware_kernels.onnx
└── ...
```

View in Netron to verify transformations.

### Custom Validation

Compare kernel output with reference:

```python
import onnxruntime as ort
import numpy as np

# Reference ONNX execution
sess = ort.InferenceSession("model.onnx")
ref_out = sess.run(None, {"input": test_data})[0]

# Hardware simulation
hw_sess = ort.InferenceSession("folded_model.onnx")
hw_out = hw_sess.run(None, {"input": test_data})[0]

# Compare
diff = np.abs(ref_out - hw_out)
print(f"Max difference: {diff.max()}")
print(f"Mean difference: {diff.mean()}")
```

### Waveform Analysis

For detailed timing analysis:

```bash
# Enable waveform generation
smith model.onnx blueprint.yaml \
    --stop-step stitched_ip_rtlsim \
    --finn-config generate_waveforms=true

# View in simulator
vivado -mode gui build/rtlsim/MyKernel/sim.wdb
```


## See Also

- **[Hardware Kernels](hardware-kernels.md)** - High-level overview and complete kernel examples
- **[Dataflow Modeling](dataflow-modeling.md)** - Theoretical foundations for dataflow composition
- **[Design Space Exploration API](../api/dse.md)** - Programmatic DSE control
- **[CLI Reference](../api/cli.md)** - Command-line options and flags
