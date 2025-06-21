# BERT Demo Step Ordering Issue

## Problem
The `generate_reference_io_step` is failing because it's trying to execute a model that has already had its head/tail removed but hasn't been properly cleaned up yet.

## Error Details
- Error occurs in multithreshold operation trying to reshape array of size 0
- Input shape shows dimension 0, meaning empty tensor
- This happens in step 5/20 after head/tail removal

## Old Demo Step Order (from bert.py line 380):
1. custom_step_cleanup
2. custom_step_remove_head
3. custom_step_remove_tail
4. custom_step_qonnx2finn
5. custom_step_generate_reference_io  # <-- This works in old demo
6. custom_streamlining_step
7. custom_step_infer_hardware

## Current New Demo Order:
1. cleanup_step
2. remove_head_step
3. remove_tail_step
4. qonnx_to_finn_step
5. generate_reference_io_step  # <-- This fails

## Key Insight
The old demo's `custom_step_qonnx2finn` does important transformations (ExpandNorms, FoldConstants, ConvertDivToMul, ConvertQONNXtoFINN) that prepare the model for execution. Our `qonnx_to_finn_step` might not be doing enough preparation.

## Solution
We need to ensure the model is in a valid state before trying to generate reference IO. This might mean:
1. Moving generate_reference_io_step later in the pipeline
2. Or ensuring qonnx_to_finn_step does proper cleanup/validation