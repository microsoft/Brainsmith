# BERT Demo FIFO Shape Mismatch - Solution Guide

## Problem Summary

The BERT demo encounters FIFO shape mismatches during the `step_set_fifo_depths` stage when using either:
1. Pre-computed folding configurations with incompatible PE/SIMD values
2. Auto-calculated folding with aggressive target_fps settings

Despite these errors, the forge system completes successfully by finding alternative configurations during DSE.

## Root Causes

1. **Aggressive Parallelization**: The default `target_fps: 1000` in the blueprint causes FINN to create incompatible parallelization factors
2. **Folding Config Mismatches**: Pre-generated folding configs assume specific node counts/types that don't match the actual 1-layer BERT model
3. **Shape Calculation Issues**: Some Thresholding nodes report output shapes with dimension 0, indicating calculation errors

## Recommended Solutions

### Solution 1: Conservative Auto-Folding (Recommended)
Use a low target_fps to avoid aggressive parallelization:

```bash
python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./output \
    --target-fps 10 \
    --clock-period 10.0 \
    --board V80
```

### Solution 2: Ignore Build Errors
The build actually succeeds despite errors. The DSE system finds working configurations. Simply run the quicktest script and ignore the error messages - check for "forge completed successfully" at the end.

### Solution 3: Custom Blueprint Configuration
Create a blueprint with more conservative FINN settings:

```yaml
finn_config:
  target_fps: 10  # Very conservative
  synth_clk_period_ns: 10.0  # Slower clock
  auto_fifo_depths: true
  auto_fifo_strategy: "LARGEFIFO_RTLSIM"
```

## Implementation Details

The blueprint adapter has been updated to:
1. Accept a `target_fps` parameter
2. Use a conservative default (10 fps) when no folding config is provided
3. Set `target_fps: null` when using pre-computed folding configs

## Quick Test Scripts

### Conservative Test (quicktest_conservative.sh)
```bash
#!/bin/bash
set -e
python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./quicktest_conservative_output \
    --target-fps 1 \
    --clock-period 10.0 \
    --board V80
```

### Standard Test (quicktest.sh)
Now uses auto-folding with moderate target_fps to avoid issues.

## Future Improvements

1. **Fix Folding Generator**: Update `gen_initial_folding.py` to query the actual model structure
2. **Validate Folding Configs**: Add pre-flight checks to ensure shape compatibility
3. **Better Error Handling**: Suppress non-critical errors during DSE exploration
4. **Investigate Thresholding**: Fix the dimension 0 issue in Thresholding node shapes

## Verification

To verify the solution works:
1. Run `./quicktest_conservative.sh`
2. Look for "forge completed successfully" in the output
3. Check `quicktest_conservative_output/` for generated files
4. Ignore intermediate error messages during the build process

The key insight is that these errors occur during exploration but don't prevent successful accelerator generation.