# Zero Dimension Root Cause Analysis

## Problem Summary
Zero-dimension shapes are introduced during the `infer_hardware_step`, causing 10 nodes to have output shape `(0, 128, 384, 1)` instead of `(1, 128, 384, 1)`.

## Root Cause
The issue occurs when hardware inference transforms abstract ONNX operations into concrete FPGA dataflow nodes. The key problem is:

1. **Dynamic Batch Dimension**: After head/tail removal, the model has input shape `['unk__0', 128, 384]` where the first dimension is a dynamic/unknown parameter
2. **Hardware Inference**: When creating hardware nodes, this dynamic dimension gets interpreted as 0
3. **Propagation**: The zero dimension propagates through all connected nodes

## Detailed Timeline

### Before infer_hardware_step
- Input shape: `['unk__0', 128, 384]` (dynamic batch)
- MultiThreshold nodes don't have explicit shape attributes
- No zero dimensions present

### During infer_hardware_step
- Creates DuplicateStreams_global_in with `numInputVectors: [0, 128]`
- The first element (0) comes from the unknown batch dimension
- All downstream nodes inherit this zero dimension

### Affected Nodes
- DuplicateStreams_global_in
- Thresholding_MultiThreshold_0, 1, 2, 9
- ElementwiseAdd (Add_0, Add_1)
- LayerNorm_FuncLayerNorm_0, 1

## Why This Happens
The hardware inference step needs concrete dimensions to:
- Allocate FIFO buffers
- Set parallelization parameters
- Generate RTL code

When it encounters a dynamic dimension, it defaults to 0, which breaks the dataflow.

## Solution
The model needs a concrete batch dimension before hardware inference. Options:

1. **Set Batch Size**: Replace dynamic dimension with concrete value (e.g., 1)
2. **Fix During Cleanup**: Modify cleanup_step to set batch dimension
3. **Fix After Head Removal**: Update the input shape after removing head/tail

The old demo likely avoids this by either:
- Having a concrete batch dimension from the start
- Fixing it in one of the custom steps
- Using a different model export process