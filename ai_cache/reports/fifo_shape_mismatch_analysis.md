# FIFO Shape Mismatch Analysis Report

## Executive Summary

Analysis of the FINN intermediate models reveals **14 FIFO shape mismatches** across **5 distinct patterns**. These mismatches occur when a producer node's folded output shape doesn't match the consumer node's expected folded input shape, causing pipeline integration issues.

## Key Findings

### 1. Most Critical Issue: Thresholding_rtl → MVAU_rtl (6 occurrences)
- **Problem**: Thresholding outputs shape `(0, 128, 384, 1)` with 0 total elements
- **Expected**: MVAU expects `(1, 128, 96, 4)` with 49,152 elements
- **Impact**: Complete data flow failure due to 0-element output
- **Root Cause**: Incorrect folding configuration in Thresholding layers

### 2. Shape Transformation Issues (8 occurrences)
All other mismatches involve shape transformations where total elements match but chunking differs:
- MVAU_rtl → Shuffle_hls: `(1, 128, 128, 3)` vs `(1, 128, 12, 32, 1)`
- Thresholding_rtl → DynMVU_rtl: `(1, 12, 128, 32, 1)` vs `(1, 12, 128, 8, 4)`
- MVAU_rtl → ElementwiseMul_hls: `(1, 128, 128, 3)` vs `(1, 128, 384, 1)`
- MVAU_rtl → Thresholding_rtl: `(1, 128, 128, 12)` vs `(1, 128, 1536, 1)`

## Detailed Mismatch Patterns

### Pattern 1: Thresholding_rtl → MVAU_rtl
```
Affected connections:
- Thresholding_rtl_0 → MVAU_rtl_0 (tensor: CRLuA4)
- Thresholding_rtl_1 → MVAU_rtl_1 (tensor: FkKbcU)
- Thresholding_rtl_2 → MVAU_rtl_2 (tensor: fcwrDZ)
- Thresholding_rtl_8 → MVAU_rtl_3 (tensor: BRbSm6)
- Thresholding_rtl_9 → MVAU_rtl_4 (tensor: vA9Y0p)
- Thresholding_rtl_10 → MVAU_rtl_5 (tensor: iX7IfU)

Issue: Zero-element output from Thresholding
```

### Pattern 2: MVAU_rtl → Shuffle_hls
```
Affected connections:
- MVAU_rtl_0 → Shuffle_hls_1 (query path)
- MVAU_rtl_1 → Shuffle_hls_2 (key path)
- MVAU_rtl_2 → Shuffle_hls_0 (value path)

Issue: Different chunking but same total elements (49,152)
```

### Pattern 3: Thresholding_rtl → DynMVU_rtl
```
Affected connections:
- Thresholding_rtl_4 → DynMVU_rtl_0 (tensor: JqK8vP)
- Thresholding_rtl_7 → DynMVU_rtl_1 (tensor: QK8fRO)

Issue: Different packing (32,1) vs (8,4) but same total elements
```

### Pattern 4: MVAU_rtl → ElementwiseMul_hls
```
Affected connections:
- MVAU_rtl_3 → ElementwiseMul_hls_1 (dense path)
- MVAU_rtl_5 → ElementwiseMul_hls_3 (dense path)

Issue: Different chunking (128,3) vs (384,1) but same total elements
```

### Pattern 5: MVAU_rtl → Thresholding_rtl
```
Affected connection:
- MVAU_rtl_4 → Thresholding_rtl_10 (dense path)

Issue: Different chunking (128,12) vs (1536,1) but same total elements
```

## Recommended Solutions

### Immediate Fix for Critical Issue
1. **Fix Thresholding_rtl folding configuration**
   - The 0-dimension output indicates incorrect PE/SIMD settings
   - Ensure Thresholding nodes have valid folding parameters
   - Check that PE × SIMD divides evenly into the channel dimension

### Shape Transformation Fixes
2. **Insert StreamingDataWidthConverter nodes**
   - Add between all mismatched connections
   - Configure to transform from producer's output shape to consumer's input shape
   - FINN should handle this automatically but may need manual intervention

3. **Adjust Folding Configuration**
   - Ensure consistent folding factors across connected layers
   - Use folding configurations that maintain compatible shapes
   - Consider using the same SIMD factor for connected layers

### Verification Steps
1. Re-run constrain_folding with corrected parameters
2. Verify all Thresholding nodes produce non-zero output shapes
3. Check that StreamingDataWidthConverter nodes are inserted where needed
4. Run shape inference to validate the complete dataflow

## Model Locations Analyzed
- `/home/tafk/dev/brainsmith-1/demos/bert_new/finn_output/intermediate_models/constrain_folding_and_set_pumped_compute_step.onnx`
- `/home/tafk/dev/brainsmith-1/demos/bert_new/finn_output/intermediate_models/step_apply_folding_config.onnx`
- `/home/tafk/dev/brainsmith-1/demos/bert_new/finn_output/intermediate_models/step_hw_codegen.onnx`

All three checkpoints show identical mismatches, indicating the issue originates early in the pipeline and persists throughout.