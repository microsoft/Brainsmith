# Deep Analysis: FIFO Shape Mismatch Root Cause

## Current Status
- bert_direct: ✅ Works perfectly (no FIFO mismatches)
- bert_new: ❌ Fails at step_set_fifo_depths with FIFO shape mismatch
- Both use identical step sequences (19 steps)
- Both have identical parameters (including standalone_thresholds=true)

## Key Differences Found

### 1. Model Export Parameters
The most significant difference is in the ONNX export:

**bert_new (FAILS):**
```python
bo.export_qonnx(
    quant_model,
    (input_ids),
    model_path,
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['global_out_0', 'global_out_1'],  # ← KEY DIFFERENCE
    opset_version=17,
)
```

**bert_direct (WORKS):**
```python
bo.export_qonnx(
    quant_model,
    (input_ids),
    model_path,
    do_constant_folding=True,
    input_names=['input_ids'],
    # NO output_names parameter!
    opset_version=17,
)
```

### 2. Export Output Comparison
- bert_new exports: `['global_out_0', 'global_out_1']` (forced names)
- bert_direct exports: `['onnx::Gather_236', '246']` (auto-generated names)

### 3. Reference IO Generation
- bert_new: Uses full `generate_reference_io_step` (6+ minutes)
- bert_direct: Uses cached `generate_reference_io_cached_step` (instant)

## Hypothesis
The `output_names` parameter in the ONNX export is causing the model structure to be different, leading to incompatible folded shapes between nodes during FIFO insertion.

## Recommended Fix
Remove the `output_names` parameter from the ONNX export in bert_new to match bert_direct exactly.

## Additional Observations
1. The error occurs consistently at the same point (step 17/19)
2. All configuration parameters are correctly propagated
3. The 6-entrypoint system itself is working correctly
4. The issue is in the model preparation, not the infrastructure