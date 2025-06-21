# Critical Analysis: Old vs New BERT Demo

## Overview
This document provides a critical comparison of the old (`demos/bert/`) and new (`demos/bert_new/`) BERT demos, identifying key differences, gaps, and potential issues.

## 1. Structural Differences

### Old Demo Structure
```
demos/bert/
├── end2end_bert.py      # Main entry point
├── bert.py              # Model generation (if different from end2end)
├── hw_compiler.py       # Direct FINN interface
├── gen_initial_folding.py
├── quicktest.sh
└── configs/             # Pre-generated folding configs
```

### New Demo Structure
```
demos/bert_new/
├── end2end_bert.py      # Main entry point with blueprint adapter
├── blueprint_adapter.py # Runtime blueprint configuration
├── gen_initial_folding.py
├── quicktest.sh
└── Various debug/fix scripts (added during troubleshooting)
```

## 2. Key Workflow Differences

### Old Demo Workflow
1. **Direct FINN Integration**: Uses `brainsmith.blueprints.REGISTRY` to get steps
2. **Direct DataflowBuildConfig**: Creates FINN config directly in `hw_compiler.py`
3. **Simple Parameter Passing**: Command-line args map directly to FINN params
4. **Fixed Blueprint**: Uses a pre-defined "bert" blueprint from registry

### New Demo Workflow
1. **Blueprint-Based**: Uses adaptive blueprint YAML files
2. **Indirect FINN Config**: Goes through blueprint system and legacy conversion
3. **Complex Parameter Mapping**: Args → Blueprint → Legacy conversion → FINN
4. **Dynamic Blueprint**: Creates runtime-adapted blueprint files

## 3. Critical Gaps and Mismatches

### Gap 1: FINN Configuration Mapping
**Old Demo:**
```python
df_cfg = build_cfg.DataflowBuildConfig(
    target_fps=args.fps,              # Direct from args
    synth_clk_period_ns=args.clk,     # Direct from args
    folding_config_file=args.param,   # Direct from args
    auto_fifo_depths=args.run_fifo_sizing,
    ...
)
```

**New Demo:**
- `target_fps` comes from blueprint YAML (default 1000)
- `synth_clk_period_ns` is converted from `clock_period` (5.0 vs 3.33)
- Parameters go through multiple layers of transformation

### Gap 2: Default Values Mismatch
| Parameter | Old Demo Default | New Demo Default |
|-----------|-----------------|------------------|
| target_fps | 3000 | 1000 (in blueprint) |
| clock_period | 3.33ns | 5.0ns |
| auto_fifo_depths | False (explicit flag) | True (always) |
| board | (from env?) | V80 |

### Gap 3: Missing Functionality
**Old Demo Features Not in New:**
- `--run-fifo-sizing` flag (new always runs it)
- `--stop-step` parameter
- `--dcp` flag for DCP generation
- Direct verification step configuration
- `--standalone-thresholds` option

### Gap 4: Model Generation Differences
**Old Demo:**
- Saves model to temp file, loads it, then deletes
- Calls `forge('bert', model, args)` with ONNX model object

**New Demo:**
- Saves model to output directory
- Calls `forge(model_path=..., blueprint_path=...)` with file paths
- Different forge API signature

## 4. Root Cause of FIFO Issues

The FIFO shape mismatch likely stems from:

1. **Different Default Parameters**: 
   - Old: `target_fps=3000` with `clock=3.33`
   - New: `target_fps=1000` with `clock=5.0`
   - But blueprint overrides to 1000 regardless of CLI

2. **Always-On FIFO Sizing**: 
   - Old: Only with `--run-fifo-sizing`
   - New: Always enabled via blueprint

3. **Missing Folding Logic**:
   - The folding config generator creates same config for both
   - But the aggressive auto-folding in new demo conflicts

## 5. Recommended Fixes

### Fix 1: Align Default Parameters
```python
# In blueprint_adapter.py
if 'finn_config' not in adapted:
    adapted['finn_config'] = {}

# Match old demo defaults
adapted['finn_config']['target_fps'] = args.target_fps  # Use CLI value
adapted['finn_config']['synth_clk_period_ns'] = args.clock_period
adapted['finn_config']['auto_fifo_depths'] = True  # Or make it configurable
```

### Fix 2: Add Missing CLI Arguments
```python
# In end2end_bert.py
parser.add_argument('--run-fifo-sizing', action='store_true', 
                   help='Enable FIFO sizing (default: enabled)')
parser.add_argument('--stop-step', type=str, default=None,
                   help='Stop at specific step')
parser.add_argument('--dcp', type=bool, default=True,
                   help='Generate DCP')
```

### Fix 3: Fix Command Compatibility
The old quicktest uses:
```bash
python end2end_bert.py -o quicktest -n 12 -l 1 -z 384 -i 1536 --run-fifo-sizing -p ./configs/l1_simd12_pe8.json -d False
```

The new uses different argument names:
```bash
python end2end_bert.py --output-dir ./quicktest_output --num-heads 12 --num-layers 1 --hidden-size 384 --intermediate-size 1536 -p ./l1_simd12_pe8.json
```

## 6. Conclusion

The new demo adds complexity through the blueprint system but loses direct control over FINN parameters. The FIFO issues arise from:
1. Different default parameters (especially target_fps)
2. Always-on FIFO sizing without proper folding config
3. Multiple layers of parameter transformation obscuring the actual FINN config

The most straightforward fix is to make the new demo respect command-line parameters more directly, especially for target_fps, rather than having them overridden by blueprint defaults.