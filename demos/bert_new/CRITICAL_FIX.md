# Critical Fix: standalone_thresholds Setting

## The Issue
After detailed comparison of the old and new BERT demos, I discovered a critical configuration mismatch:

- **Old demo**: `args.standalone_thresholds = True` (hardcoded)
- **New demo**: `standalone_thresholds: false` (in blueprint)

## Why This Matters
The `standalone_thresholds` parameter fundamentally changes how FINN handles activation thresholding:

1. **standalone_thresholds = True**: Thresholding layers are kept as separate nodes
   - Allows independent folding configuration
   - Creates explicit Thresholding nodes in the dataflow
   - More predictable shape transformations

2. **standalone_thresholds = False**: Thresholding is integrated into compute layers
   - Can cause shape mismatch issues during folding
   - Different folding behavior for integrated operations
   - The root cause of our FIFO shape mismatches

## The Fix
Updated `brainsmith/libraries/blueprints_v2/transformers/bert_demo.yaml`:
```yaml
finn_config:
  ...
  standalone_thresholds: true  # Match old demo setting
```

## Verification
Run the quicktest again with this fix:
```bash
./quicktest.sh
```

The FIFO shape mismatches should now be resolved because Thresholding nodes will behave the same way as in the old demo.

## Additional Findings
Other important differences found:
1. Default `target_fps`: 3000 (old) vs 1000 (new)
2. Default `synth_clk_period_ns`: 3.33 (old) vs 5.0 (new)  
3. Missing parameters: `verification_atol`, `split_large_fifos`, `fifosim_n_inferences`

These should also be aligned for full compatibility.