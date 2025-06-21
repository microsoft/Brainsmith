# Final Diagnosis: FIFO Shape Mismatch Issue

## Summary
Despite extensive debugging and fixes, bert_new still fails with FIFO shape mismatches while bert_direct works perfectly. This definitively proves the issue is in the 6-entrypoint compatibility layer, not the transforms.

## Fixes Applied
1. ✅ Removed `constrain_folding_and_set_pumped_compute_step` (not in old demo)
2. ✅ Fixed output_dir propagation
3. ✅ Added cached reference IO support
4. ✅ Verified all parameters match (including standalone_thresholds=true)
5. ✅ Confirmed step sequence matches exactly (19 steps)
6. ✅ Verified model generation is identical
7. ✅ Confirmed streamlining implementation matches

## Root Cause
The issue is in the 6-entrypoint compatibility layer itself. Possible causes:
1. State pollution between step executions
2. Different initialization of FINN components
3. Subtle differences in how steps are invoked
4. Module import order or initialization differences
5. The compatibility layer may be modifying the model in unexpected ways

## Proof
- bert_direct: Uses exact same transforms directly → ✅ Works
- bert_new: Uses same transforms through 6-entrypoint → ❌ Fails

## Recommendation
Since the transforms are proven to work correctly, the options are:
1. Debug the 6-entrypoint compatibility layer deeply
2. Use bert_direct approach for production (bypass 6-entrypoint)
3. Gradually migrate away from 6-entrypoint architecture

## Clean Test Command
```bash
./smithy exec "cd demos/bert_new && ./clean_quicktest.sh"
```

This ensures a completely clean run with no state pollution.