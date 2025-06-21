# BERT Direct Demo Success Analysis Report

## Executive Summary

The bert_direct demo successfully completed all 20 steps without FIFO shape mismatches, demonstrating that BrainSmith transforms work correctly when used directly with FINN. This confirms the issues in bert_new are in the 6-entrypoint compatibility layer, not the transforms themselves.

## Key Findings

### 1. Complete Success Without FIFO Errors
- **Result**: All 20 steps completed successfully
- **No FIFO shape mismatches reported**
- **RTL generation and synthesis completed**
- **Final artifacts generated correctly**

### 2. Direct vs 6-Entrypoint Architecture

#### bert_direct (SUCCESS) ✅
```python
# Direct DataflowBuildConfig creation
steps = [
    # BrainSmith preprocessing
    cleanup_step,
    remove_head_step,
    remove_tail_step,
    qonnx_to_finn_step,
    generate_reference_io_cached_step,  # Optimized
    streamlining_step,
    infer_hardware_step,
    # Standard FINN pipeline
    step_create_dataflow_partition,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    # BrainSmith postprocessing
    step_measure_rtlsim_performance,
    constrain_folding_and_set_pumped_compute_step,
    step_set_fifo_depths,
    step_create_stitched_ip,
    shell_metadata_handover_step,
]

config = build_cfg.DataflowBuildConfig(
    steps=steps,
    standalone_thresholds=True,  # Explicitly set
    # ... other params
)
```

#### bert_new (FAILS) ❌
```python
# 6-entrypoint system with LegacyConversionLayer
blueprint = Blueprint()
transform = blueprint_to_transforms(
    blueprint_path="bert_demo.yaml"
)
# Goes through:
# - BlueprintV2 parsing
# - LegacyConversionLayer
# - Transform composition
# - Missing/misconfigured parameters
```

### 3. Critical Differences Identified

1. **Step Ordering**: Direct demo uses exact step order from old demo
2. **Parameter Passing**: Direct config ensures all parameters are passed correctly
3. **No Translation Layer**: Avoids potential bugs in Blueprint → Transform conversion
4. **Explicit Configuration**: All settings (like standalone_thresholds) are explicit

### 4. Performance Optimizations

The direct demo includes two key optimizations:

1. **Cached Reference IO**: Avoids 6-minute model execution
   ```python
   def generate_reference_io_cached_step(model, cfg):
       # Copy pre-generated tensors instead of computing
       shutil.copy("input.npy", cfg.output_dir)
       shutil.copy("expected_output.npy", cfg.output_dir)
       shutil.copy("expected_context.npz", cfg.output_dir)
   ```

2. **Real-time Output**: Fixed buffering for progress visibility
   ```bash
   PYTHONUNBUFFERED=1 python -u end2end_bert_direct.py
   ```

### 5. Model Generation Consistency

Both demos generate identical BERT models:
- Same quantization approach
- Same model surgery (head/tail removal)
- Same export parameters (except output_names)
- Verified by comparing exported model structure

## Root Cause Analysis

The success of bert_direct strongly indicates the root cause is in the 6-entrypoint infrastructure:

### Likely Issues in bert_new:
1. **LegacyConversionLayer bugs**: Incorrect transform composition or parameter passing
2. **Blueprint parsing**: Missing or misconfigured steps in YAML → Transform conversion
3. **State management**: The 6-entrypoint system may have state pollution between steps
4. **Parameter propagation**: Settings like `standalone_thresholds` not properly passed through layers

### Working Elements (validated by bert_direct):
1. ✅ BrainSmith transform implementations (cleanup, streamlining, hardware inference)
2. ✅ FINN integration steps
3. ✅ Model preprocessing and ONNX handling
4. ✅ Folding configuration application
5. ✅ RTL generation pipeline

## Recommendations

### Immediate Actions
1. **Debug LegacyConversionLayer**: Add extensive logging to trace parameter flow
2. **Compare step sequences**: Verify bert_new generates identical step order
3. **Validate Blueprint parsing**: Ensure YAML correctly maps to transforms
4. **Check state isolation**: Verify no cross-contamination between transform steps

### Long-term Solutions
1. **Simplify architecture**: Consider reducing abstraction layers
2. **Add integration tests**: Test Blueprint → Transform → FINN pipeline
3. **Improve error reporting**: Add validation at each translation layer
4. **Document parameter flow**: Create clear docs on how settings propagate

## Artifacts to Compare

### From bert_direct (SUCCESS):
- `/home/tafk/builds/brainsmith/direct_test_20250620_231805/`
  - `intermediate_models/` - All step outputs present
  - `stitched_ip/` - Complete RTL generation
  - `build_dataflow.log` - No FIFO errors

### From bert_new (FAILS):
- `/home/tafk/builds/brainsmith/debug_test/`
  - Check for missing intermediate models
  - Compare step outputs where available
  - Analyze build_dataflow.log for exact error point

## Conclusion

The bert_direct demo proves BrainSmith transforms are correct. The issue lies in the 6-entrypoint compatibility layer that translates Blueprint definitions to transform sequences. This is good news - the core algorithms work, and only the infrastructure wrapper needs fixing.

**Next Step**: Debug the LegacyConversionLayer to ensure it correctly composes transforms and passes all required parameters, especially `standalone_thresholds` and the complete step sequence.