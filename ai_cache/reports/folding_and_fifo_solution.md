# BERT Demo Folding and FIFO Solution Summary

## Problem Analysis

### 1. FIFO Shape Mismatches
- **Root Cause**: Dynamic batch dimension `'unk__0'` interpreted as 0 during hardware inference
- **Not caused by**: PE=1 values (old demo also uses PE=1 for Thresholding)
- **Solution Applied**: Fixed `remove_head_step` to set concrete batch dimension = 1

### 2. Long RTL Simulation Time
- **Issue**: Auto-folding with target_fps triggers 12-hour RTL simulation for FIFO sizing
- **Solution**: Use pre-computed folding configuration to skip RTL simulation

## Solutions Implemented

### 1. Fixed Zero-Dimension Issue
Modified `brainsmith/libraries/transforms/steps/bert.py`:
```python
# Fix dynamic batch dimension to concrete value
if model.graph.input[0].type.tensor_type.shape.dim[0].HasField('dim_param'):
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
```

### 2. Created Safe Folding Generator
`gen_safe_folding.py` generates conservative folding configs that:
- Match the pattern from working old demo (SIMD=12, PE=8 for MVAUs)
- Use PE=1 for Thresholding (same as old demo)
- Handle all node types appropriately

### 3. New Quick Test Scripts
- `quicktest_folding.sh`: Uses pre-computed folding to avoid 12-hour simulation
- Generates folding config on-the-fly for the specific model configuration

## Usage Recommendations

### For Quick Testing (Avoid 12-hour wait):
```bash
./quicktest_folding.sh
```
This uses pre-computed folding configuration.

### For Auto-Folding (12+ hours):
```bash
./quicktest.sh
```
This lets FINN calculate optimal folding but requires long RTL simulation.

### For Custom Folding:
```bash
# Generate folding for specific parameters
python gen_initial_folding.py -s 12 -p 8 -t 1 -n 1 -o custom.json

# Use it
python end2end_bert.py --param custom.json ...
```

## Key Insights

1. **PE=1 is valid**: The old demo successfully uses PE=1 for Thresholding
2. **Batch dimension critical**: Must be concrete (not dynamic) for hardware inference
3. **Folding config essential**: Pre-computed folding avoids expensive RTL simulation
4. **Shape compatibility**: With fixed batch dimension, shape mismatches should be resolved

## Next Steps

With these fixes:
1. The zero-dimension issue should be resolved
2. FIFO shape mismatches should be eliminated
3. Quick testing can use pre-computed folding to avoid 12-hour waits
4. The demo should work end-to-end like the old version