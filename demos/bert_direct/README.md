# BERT Direct Demo

This demo bypasses the Blueprint V2 6-entrypoint system and LegacyConversionLayer to test BrainSmith transforms directly with FINN's DataflowBuildConfig.

## Purpose

Isolate whether FIFO shape mismatch issues are in:
- **BrainSmith transforms themselves** (if this demo fails)
- **6-entrypoint compatibility layer** (if this demo works)

## Key Differences from Other Demos

| Demo | Architecture | Step Source | API |
|------|-------------|-------------|-----|
| `bert` (old) | Direct DataflowBuildConfig | Custom step functions in bert.py | FINN builder directly |
| `bert_new` | 6-entrypoint → LegacyConversionLayer → DataflowBuildConfig | BrainSmith transforms via compatibility | BrainSmith API |
| `bert_direct` | **Direct DataflowBuildConfig** | **BrainSmith transforms directly** | **FINN builder directly** |

## Usage

```bash
cd demos/bert_direct
./quicktest.sh
```

## Expected Outcomes

- **Success**: BrainSmith transforms work correctly, issue is in compatibility layer
- **Failure**: Issue is in BrainSmith transform implementations
- **Either way**: Direct validation of transforms independent of infrastructure

## Technical Details

- Uses identical ONNX generation as `bert_new` (without output_names parameter)
- Builds step sequence directly using `brainsmith.libraries.transforms.steps`
- Calls `finn.builder.build_dataflow_cfg()` directly
- Uses proven folding configuration (PE=8, SIMD=12)
- Skips DCP generation for fast testing
- **Uses pre-generated reference IO tensors** to avoid 6-minute computation delay

## Reference IO Tensors

This demo includes pre-generated reference tensors to speed up testing:
- `input.npy` - Input tensor (1, 128, 384)
- `expected_output.npy` - Expected output tensor (1, 128, 384)
- `expected_context.npz` - Full execution context

These were generated from a BERT model after head/tail removal and are reused in each run via `generate_reference_io_cached_step()`.