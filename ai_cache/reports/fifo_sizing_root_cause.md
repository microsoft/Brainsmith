# FIFO Sizing Error Root Cause Analysis

## Problem Statement
The new BERT demo encounters FIFO shape mismatches during `step_set_fifo_depths`, even though all transformation steps are identical to the old demo.

## Key Observations

### 1. Step Implementations Are Identical
- All 8 custom steps have identical transformation logic
- Only differences are in error handling and code organization
- This rules out transformation logic as the cause

### 2. Error Pattern
```
The folded output shape of the first node is not the same as the folded output shape of the second node
```
This occurs when nodes have incompatible folding configurations that create mismatched tensor dimensions.

### 3. Working vs Failing Scenarios
- **Old Demo**: Works with hardcoded settings
- **New Demo**: Fails with auto-calculated folding
- **Key Difference**: How folding configuration is generated

## Likely Root Causes

### 1. **Folding Configuration Generation**
The new demo uses auto-calculated folding based on target_fps, while the old demo may use specific folding configurations. The auto-folding might create incompatible PE/SIMD values.

### 2. **Missing Step Execution**
Although `constrain_folding_and_set_pumped_compute_step` is in the pipeline, it's executed AFTER the standard FINN steps that do folding. In the old demo, this constraint step might need to run earlier.

### 3. **Model State Differences**
The `generate_reference_io_step` had to be patched to handle invalid models, suggesting the model state differs from the old demo at that point.

### 4. **Configuration Parameter Differences**
Critical parameters that affect folding:
- `target_fps`: Different values lead to different folding
- `mvau_wwidth_max`: Affects PE stream width constraints
- `standalone_thresholds`: Already fixed to `true`

## Specific Issue: Step Ordering

Looking at the old demo's BUILD_STEPS (line 380-402):
1. The `custom_step_constrain_folding_and_set_pumped_compute` is listed AFTER `step_measure_rtlsim_performance`
2. But in our blueprint, it's at the beginning of `legacy_postproc`

This means the constraint step runs at different times, potentially after folding has already been applied incorrectly.

## Recommended Fix

### Option 1: Adjust Step Order
Move `constrain_folding_and_set_pumped_compute_step` to run between standard FINN steps:
- After `step_apply_folding_config` 
- Before `step_minimize_bit_width`

### Option 2: Pre-compute Compatible Folding
Generate a folding configuration that ensures compatible shapes:
- Use conservative SIMD/PE values
- Ensure all connected nodes have matching dimensions

### Option 3: Add Shape Validation
Add a new step that validates and fixes shape compatibility after folding but before FIFO sizing.

## Immediate Action
The quickest fix is to reorder the steps to match the old demo exactly. The constraint step should run as part of the standard pipeline, not as a post-processing step.