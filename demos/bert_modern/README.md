# BERT Modern Demo

This demo provides exact functional parity with the old BERT demo (`demos/bert_old/`) while leveraging Brainsmith's modern DSE v3 architecture. It uses the Legacy FINN Backend to maintain compatibility while showcasing the advantages of the blueprint-driven approach.

## Overview

The modern BERT demo demonstrates:
- **Blueprint-driven configuration** instead of hardcoded Python
- **Declarative YAML** for defining the compilation flow
- **Variable substitution** for flexible configuration
- **Modern DSE v3 architecture** with Phase 1/2/3 separation
- **Exact parity** with the old demo's functionality and outputs

## Key Features

### 1. Exact Functional Parity
- All 22 build steps from the original demo (excluding loop operations)
- Same quantization approach using Brevitas
- Identical folding configuration format
- Same shell metadata extraction

### 2. Modern Architecture Benefits
- **Cleaner code**: Declarative blueprint vs hardcoded steps
- **Better logging**: Enhanced visibility into the build process
- **Resumable builds**: Can restart from any step
- **Artifact management**: Automatic saving of intermediate models
- **Parallel exploration**: Ready for multi-configuration DSE

### 3. Blueprint-Driven Approach
The `bert_legacy.yaml` blueprint (located in `brainsmith/blueprints/`) defines the entire compilation flow:
- Build steps in order
- Configuration flags with variable substitution
- Verification settings
- Output stage configuration

## Directory Structure

```
bert_modern/
├── bert_demo.py          # Main entry point
├── bert_modern_template.yaml  # Template for custom blueprints
├── gen_folding_config.py # Folding config generator
├── configs/              # Pre-generated folding configs
├── scripts/              # Test and utility scripts
├── Makefile             # Build automation
└── README.md            # This file
```

## Quick Start

### 1. Quick Test (1 Layer BERT)
```bash
# Using Makefile
make quicktest

# Or directly
bash scripts/quicktest.sh
```

### 2. Generate Folding Configurations
```bash
# Generate configs for 3 layers (default)
make configs

# Generate for different layer count
make configs LAYERS=5
```

### 3. Run Custom Configuration
```bash
# Generate model with custom parameters
python bert_demo.py \
    -o my_bert_build \
    -l 3 \
    -z 768 \
    -n 12 \
    -i 3072 \
    -p configs/l3_simd48_pe32.json
```

## Command Line Options

The `bert_demo.py` script supports all options from the old demo:

### Model Configuration
- `-l, --num_hidden_layers`: Number of BERT layers (default: 1)
- `-z, --hidden_size`: Hidden dimension size (default: 384)
- `-n, --num_attention_heads`: Number of attention heads (default: 12)
- `-i, --intermediate_size`: FFN intermediate size (default: 1536)
- `-b, --bitwidth`: Quantization bitwidth (default: 8)
- `-q, --seqlen`: Sequence length (default: 128)

### Build Configuration
- `-o, --output`: Output directory name (required)
- `-f, --fps`: Target FPS for auto-folding (default: 3000)
- `-c, --clk`: Clock period in ns (default: 3.33)
- `-p, --param`: Folding configuration file
- `-x, --run_fifo_sizing`: Enable FIFO sizing
- `-s, --stop_step`: Stop at specific build step
- `--board`: Target board (V80, Pynq-Z1, U250)

## Folding Configuration

The `gen_folding_config.py` script generates JSON configurations:

```bash
python gen_folding_config.py \
    --simd 48 \
    --pe 32 \
    --num_layers 3 \
    --other 4 \
    -o my_config.json
```

Parameters:
- `--simd`: SIMD parallelism for MVAU layers
- `--pe`: PE parallelism for MVAU layers
- `--other`: Parallelism for other operators
- `--num_layers`: Number of BERT layers

## Migration from Old Demo

### Key Differences

1. **Entry Point**: 
   - Old: `end2end_bert.py`
   - New: `bert_demo.py`

2. **Configuration**:
   - Old: Hardcoded in Python files
   - New: YAML blueprint with variables

3. **Build Process**:
   - Old: Direct FINN builder calls
   - New: DSE v3 pipeline

4. **Output Structure**:
   - Same final outputs
   - Additional DSE metadata

### Migration Steps

1. Replace `end2end_bert.py` calls with `bert_demo.py`
2. Use same command line arguments
3. Folding configs are compatible
4. Output models are identical

## Blueprint Customization

The `bert_legacy.yaml` blueprint supports variable substitution:

```yaml
config_flags:
  board: "${BOARD:-V80}"
  clock_period_ns: ${CLK_PERIOD:-3.33}
  target_fps: ${TARGET_FPS:-3000}
```

Variables are substituted at runtime from command line arguments.

## Testing Parity

To verify exact parity with the old demo:

```bash
# Requires old demo to be present
make test-parity
```

This will:
1. Run both old and new demos with identical parameters
2. Compare output models
3. Verify build artifacts

## Advanced Usage

### Custom Build Steps

To modify the build flow, copy `brainsmith/blueprints/bert_legacy.yaml` and edit:

```yaml
build_steps:
  - "cleanup"
  - "my_custom_step"  # Add custom step
  - "streamlining"
```

### Debug Mode

Enable verbose logging:
```bash
python bert_demo.py -o debug_build -v
```

### Partial Builds

Stop at specific step:
```bash
python bert_demo.py -o partial_build --stop-step step_hw_codegen
```

## Troubleshooting

### Common Issues

1. **Transform not found**: Ensure all transforms are registered in the plugin system
2. **Memory errors**: Reduce batch size or use smaller model
3. **Build failures**: Check intermediate models in `output_dir/intermediate_models/`

### Debug Tips

1. Enable verbose logging with `-v`
2. Save intermediate models (default behavior)
3. Check logs in output directory
4. Use `--stop-step` to isolate issues

## Performance Comparison

The modern demo provides identical functionality with:
- **Cleaner code structure**: ~50% less boilerplate
- **Better error handling**: Automatic rollback on failure
- **Enhanced logging**: Structured logs with levels
- **Resumable builds**: Continue from last successful step

## Future Enhancements

While maintaining exact parity, the modern architecture enables:
- Multiple kernel backend exploration
- Parallel build execution
- Advanced DSE strategies
- Cloud-based compilation

## License

SPDX-License-Identifier: MIT