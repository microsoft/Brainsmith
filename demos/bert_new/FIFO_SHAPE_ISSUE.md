# FIFO Shape Mismatch Issue

## Problem Description

When using pre-computed folding configurations with the BERT demo, FINN may encounter FIFO shape mismatches between connected nodes. This occurs when the folding parameters (PE/SIMD) create incompatible tensor shapes at node boundaries.

### Example Error
```
AssertionError: The folded output shape of the first node is not the same as the folded output shape of the second node. A streaming fifo can't be implemented in between these nodes.
```

### Root Cause

The issue stems from:
1. **Incompatible parallelization factors**: When Thresholding nodes have PE=1 but feed into MVAU nodes expecting SIMD=12
2. **Node count mismatches**: The folding config generator assumes a certain number of nodes that may not match the actual model
3. **Shape calculation errors**: Some nodes report incorrect dimensions (e.g., dimension 0 being 0)

## Workarounds

### Option 1: Auto-calculated Folding (Recommended)
Let FINN automatically determine compatible folding parameters:

```bash
python end2end_bert.py \
    --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 \
    --output-dir ./output \
    --target-fps 100 \
    --board V80
```

### Option 2: Minimal Folding
Use PE=1, SIMD=1 for all nodes (slow but guaranteed to work):

```bash
python test_minimal_folding.py  # Generate minimal config
python end2end_bert.py -p ./minimal_folding.json ...
```

### Option 3: Custom Compatible Folding
Generate a folding config that maintains shape compatibility:

```bash
python gen_compatible_folding.py -s 12 -p 8 -o compatible_folding.json
python end2end_bert.py -p ./compatible_folding.json ...
```

## Long-term Solution

The folding configuration generator needs to be updated to:
1. Query the actual model structure before generating configs
2. Ensure compatible shapes between all connected nodes
3. Validate the configuration before applying it

## Debug Tools

Use `debug_fifo_shapes.py` to analyze shape mismatches:

```bash
python debug_fifo_shapes.py
```

This will show all shape mismatches in the model after folding is applied.