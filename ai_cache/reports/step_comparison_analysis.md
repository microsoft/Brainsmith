# BERT Demo Step Implementation Comparison Report

## Executive Summary
After detailed analysis of each custom step in the old BERT demo (bert.py) versus the corresponding BrainSmith library steps, I found that **all implementations are functionally identical**. The FIFO sizing error is not caused by differences in step implementations.

## Detailed Step-by-Step Comparison

### 1. cleanup_step
- **Old Demo**: `custom_step_cleanup` 
- **BrainSmith**: `cleanup_step`
- **Transformations Applied**:
  - SortCommutativeInputsInitializerLast()
  - RemoveIdentityOps()
- **Verdict**: IDENTICAL

### 2. remove_head_step
- **Old Demo**: `custom_step_remove_head`
- **BrainSmith**: `remove_head_step`
- **Logic**: Removes nodes up to first LayerNormalization, rewires input
- **Verdict**: IDENTICAL

### 3. remove_tail_step
- **Old Demo**: `custom_step_remove_tail`
- **BrainSmith**: `remove_tail_step`
- **Logic**: Removes from global_out_1 back to last LayerNorm
- **Verdict**: IDENTICAL

### 4. qonnx_to_finn_step
- **Old Demo**: `custom_step_qonnx2finn`
- **BrainSmith**: `qonnx_to_finn_step`
- **Transformations Applied**:
  - ExpandNorms()
  - FoldConstants()
  - ConvertDivToMul()
  - ConvertQONNXtoFINN()
- **Differences**: BrainSmith adds dependency checking and better error handling
- **Verdict**: FUNCTIONALLY IDENTICAL

### 5. streamlining_step
- **Old Demo**: `custom_streamlining_step`
- **BrainSmith**: `streamlining_step`
- **Transformations Applied** (in order):
  - AbsorbSignBiasIntoMultiThreshold()
  - AbsorbAddIntoMultiThreshold()
  - AbsorbMulIntoMultiThreshold()
  - RoundAndClipThresholds()
  - MoveOpPastFork(["Mul"])
  - MoveScalarMulPastMatMul()
  - MoveScalarLinearPastInvariants()
  - AbsorbMulIntoMultiThreshold()
  - AbsorbAddIntoMultiThreshold()
  - InferDataTypes(allow_scaledint_dtypes=False)
  - GiveUniqueNodeNames()
- **Verdict**: IDENTICAL

### 6. infer_hardware_step
- **Old Demo**: `custom_step_infer_hardware`
- **BrainSmith**: `infer_hardware_step`
- **Transformations Applied**:
  - InferLayerNorm()
  - InferDuplicateStreamsLayer()
  - InferElementwiseBinaryOperation()
  - InferShuffle()
  - InferHWSoftmax()
  - InferThresholdingLayer()
  - InferQuantizedMatrixVectorActivation()
- **Verdict**: IDENTICAL

### 7. constrain_folding_and_set_pumped_compute_step
- **Old Demo**: `custom_step_constrain_folding_and_set_pumped_compute`
- **BrainSmith**: `constrain_folding_and_set_pumped_compute_step`
- **Transformations Applied**:
  - TempShuffleFixer()
  - SetPumpedCompute()
- **Verdict**: IDENTICAL

### 8. shell_metadata_handover_step
- **Old Demo**: `custom_step_shell_metadata_handover`
- **BrainSmith**: `shell_metadata_handover_step`
- **Logic**: Extracts metadata for shell integration
- **Minor Difference**: BrainSmith adds `return model` when condition not met
- **Verdict**: FUNCTIONALLY IDENTICAL

## Non-Implementation Differences

### Error Handling
- BrainSmith versions include better error handling
- Dependency checking for required packages
- More informative error messages

### Code Organization
- BrainSmith uses proper module imports
- Better separation of concerns
- Cleaner dependency management

### Documentation
- BrainSmith versions have better docstrings
- Category and dependency metadata for step discovery

## Root Cause Analysis

Since all step implementations are identical, the FIFO sizing error must be caused by:

1. **Different Step Ordering**: The sequence of steps might differ
2. **Configuration Differences**: 
   - Different folding configurations
   - Different build parameters (target_fps, clock period, etc.)
   - standalone_thresholds setting
3. **Model Input Differences**: The initial model might be different
4. **Missing Steps**: Some steps might not be called in the new flow
5. **Runtime Context**: Different DataflowBuildConfig parameters

## Key Finding

The FIFO sizing issue is **NOT** due to differences in transformation logic. The problem lies in the orchestration, configuration, or context rather than the implementation of individual steps.

## Recommendations

1. **Verify Step Order**: Ensure the exact same sequence is followed
2. **Check Configuration**: Compare all DataflowBuildConfig parameters
3. **Validate Folding**: Ensure folding configuration is compatible
4. **Debug Model State**: Save intermediate models to compare states
5. **Check for Missing Steps**: Verify all 20 steps are executed