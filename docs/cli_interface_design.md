# Brainsmith CLI Interface Design

## Overview

This document specifies the design for the Brainsmith command-line interface (CLI) that will provide an intuitive way to use the hardware compiler from the command line.

## Command Structure

### Main Entry Point

```bash
brainsmith [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Global Options

```bash
-v, --verbose          Increase verbosity (can be used multiple times)
-q, --quiet           Suppress output
--config FILE         Use configuration file
--build-dir DIR       Override build directory
--help               Show help
--version            Show version
```

## Commands

### 1. Compile Command

Compile an ONNX model to hardware.

```bash
brainsmith compile MODEL_FILE [OPTIONS]
```

#### Required Arguments
- `MODEL_FILE`: Path to ONNX model file

#### Options
```bash
-b, --blueprint NAME      Blueprint to use (required)
-o, --output DIR          Output directory (required)
-c, --config FILE         Configuration file
--target-fps FPS          Target FPS (default: 3000)
--clk-period PERIOD       Clock period in ns (default: 3.33)
--board BOARD             Target board (default: V80)
--folding-config FILE     Folding configuration file
--stop-step STEP          Stop at specific step
--save-intermediate       Save intermediate models
--run-fifo-sizing         Run FIFO sizing optimization
--no-dcp                  Don't generate DCP files
--verification-atol TOL   Verification tolerance (default: 0.1)
```

#### Examples
```bash
# Basic compilation
brainsmith compile model.onnx --blueprint bert --output ./build

# With specific configuration
brainsmith compile model.onnx --blueprint bert --output ./build \
  --target-fps 5000 --board ZCU104 --save-intermediate

# Using configuration file
brainsmith compile model.onnx --config build_config.yaml

# Stop at specific step for debugging
brainsmith compile model.onnx --blueprint bert --output ./build \
  --stop-step step_hw_codegen
```

### 2. Blueprint Management Commands

#### List Blueprints
```bash
brainsmith blueprints list [OPTIONS]
```

Options:
```bash
--detailed, -d        Show detailed information
--format FORMAT       Output format (table, json, yaml)
```

Examples:
```bash
# List all blueprints
brainsmith blueprints list

# Detailed view
brainsmith blueprints list --detailed

# JSON output
brainsmith blueprints list --format json
```

#### Show Blueprint Details
```bash
brainsmith blueprints show BLUEPRINT_NAME [OPTIONS]
```

Options:
```bash
--format FORMAT       Output format (yaml, json, table)
--steps              Show step details
```

Examples:
```bash
# Show BERT blueprint
brainsmith blueprints show bert

# Show with step details
brainsmith blueprints show bert --steps

# YAML format
brainsmith blueprints show bert --format yaml
```

#### Validate Blueprint
```bash
brainsmith blueprints validate BLUEPRINT_NAME [OPTIONS]
```

Options:
```bash
--strict             Strict validation mode
--fix                Attempt to fix issues
```

Examples:
```bash
# Validate BERT blueprint
brainsmith blueprints validate bert

# Strict validation
brainsmith blueprints validate bert --strict
```

### 3. Configuration Commands

#### Generate Configuration Template
```bash
brainsmith config generate [OPTIONS]
```

Options:
```bash
-o, --output FILE     Output file (default: stdout)
--blueprint NAME      Generate for specific blueprint
--format FORMAT       Format (yaml, json)
```

Examples:
```bash
# Generate basic config
brainsmith config generate > build_config.yaml

# Generate for BERT
brainsmith config generate --blueprint bert --output bert_config.yaml
```

#### Validate Configuration
```bash
brainsmith config validate CONFIG_FILE [OPTIONS]
```

Options:
```bash
--strict             Strict validation mode
```

Examples:
```bash
# Validate configuration
brainsmith config validate build_config.yaml
```

### 4. Interactive Mode

Start an interactive session for model compilation.

```bash
brainsmith interactive [OPTIONS]
```

Options:
```bash
--config FILE         Load configuration file
```

Interactive session provides:
- Model loading and inspection
- Blueprint selection
- Configuration editing
- Step-by-step compilation
- Real-time feedback

### 5. Utility Commands

#### Step Information
```bash
brainsmith steps list [OPTIONS]
brainsmith steps show STEP_NAME [OPTIONS]
```

Options:
```bash
--category CATEGORY   Filter by category
--format FORMAT       Output format
```

Examples:
```bash
# List all steps
brainsmith steps list

# Show transformer steps only
brainsmith steps list --category transformer

# Show specific step
brainsmith steps show transformer.remove_head
```

#### Build Information
```bash
brainsmith build info BUILD_DIR [OPTIONS]
```

Options:
```bash
--format FORMAT       Output format (table, json, yaml)
--artifacts          Show build artifacts
```

## Configuration File Format

### YAML Configuration
```yaml
# brainsmith_config.yaml
blueprint: bert
output_dir: ./build
build_dir: /tmp/brainsmith_builds

# Model preprocessing
model:
  save_intermediate: true

# FINN configuration  
finn:
  target_fps: 3000
  clk_period_ns: 3.33
  folding_config_file: folding.json
  stop_step: null

# Verification
verification:
  run_fifo_sizing: false
  fifosim_n_inferences: 2
  verification_atol: 0.1

# Hardware target
hardware:
  board: V80
  generate_dcp: true

# Advanced options
advanced:
  standalone_thresholds: true
  split_large_fifos: true

# Model-specific metadata
metadata:
  num_hidden_layers: 3
  hidden_size: 384
```

### JSON Configuration
```json
{
  "blueprint": "bert",
  "output_dir": "./build",
  "build_dir": "/tmp/brainsmith_builds",
  "model": {
    "save_intermediate": true
  },
  "finn": {
    "target_fps": 3000,
    "clk_period_ns": 3.33,
    "folding_config_file": "folding.json",
    "stop_step": null
  },
  "verification": {
    "run_fifo_sizing": false,
    "fifosim_n_inferences": 2,
    "verification_atol": 0.1
  },
  "hardware": {
    "board": "V80",
    "generate_dcp": true
  },
  "advanced": {
    "standalone_thresholds": true,
    "split_large_fifos": true
  },
  "metadata": {
    "num_hidden_layers": 3,
    "hidden_size": 384
  }
}
```

## CLI Implementation Structure

### File Organization
```
brainsmith/
├── cli/
│   ├── __init__.py
│   ├── main.py                  # Main entry point
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── compile.py           # Compile command
│   │   ├── blueprints.py        # Blueprint management
│   │   ├── config.py            # Configuration commands
│   │   ├── interactive.py       # Interactive mode
│   │   ├── steps.py             # Step information
│   │   └── build.py             # Build information
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py            # Configuration loading
│   │   └── templates.py         # Configuration templates
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── output.py            # Output formatting
│   │   ├── validation.py        # Input validation
│   │   └── progress.py          # Progress indication
│   └── interactive/
│       ├── __init__.py
│       ├── session.py           # Interactive session
│       └── ui.py                # User interface helpers
```

### CLI Entry Point (setup.py)
```python
setup(
    name="brainsmith",
    # ... other setup parameters
    entry_points={
        'console_scripts': [
            'brainsmith=brainsmith.cli.main:main',
        ],
    },
)
```

### Main CLI Structure
```python
# brainsmith/cli/main.py
import click
from .commands import compile, blueprints, config, interactive, steps, build

@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity')
@click.option('-q', '--quiet', is_flag=True, help='Suppress output')
@click.option('--config', type=click.Path(), help='Configuration file')
@click.option('--build-dir', type=click.Path(), help='Build directory')
@click.version_option()
@click.pass_context
def cli(ctx, verbose, quiet, config, build_dir):
    """Brainsmith Neural Network to FPGA Compiler."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    ctx.obj['config'] = config
    ctx.obj['build_dir'] = build_dir

cli.add_command(compile.compile_cmd)
cli.add_command(blueprints.blueprints)
cli.add_command(config.config)
cli.add_command(interactive.interactive)
cli.add_command(steps.steps)
cli.add_command(build.build)

def main():
    cli()

if __name__ == '__main__':
    main()
```

## Output Formatting

### Progress Indication
```bash
$ brainsmith compile model.onnx --blueprint bert --output ./build
Brainsmith Hardware Compiler v1.0.0

Loading model: model.onnx ✓
Loading blueprint: bert ✓
Validating configuration ✓

Compilation Progress:
[████████████████████████████████████████] 100% | Step 19/19: shell_metadata_handover

Results:
  Status: SUCCESS
  Output: ./build
  Final model: ./build/output.onnx
  Build time: 1m 23s
  
Artifacts:
  ✓ Stitched IP: ./build/stitched_ip/
  ✓ Reports: ./build/reports/
  ✓ DCP: ./build/stitched_ip/finn_design.dcp
  ✓ Handover: ./build/stitched_ip/shell_handover.json
```

### Error Handling
```bash
$ brainsmith compile missing.onnx --blueprint bert --output ./build
Brainsmith Hardware Compiler v1.0.0

ERROR: Model file not found: missing.onnx

$ brainsmith compile model.onnx --blueprint invalid --output ./build
Brainsmith Hardware Compiler v1.0.0

Loading model: model.onnx ✓
Loading blueprint: invalid ✗

ERROR: Blueprint 'invalid' not found
Available blueprints: bert

Use 'brainsmith blueprints list' to see all available blueprints.
```

### Verbose Output
```bash
$ brainsmith -vv compile model.onnx --blueprint bert --output ./build
Brainsmith Hardware Compiler v1.0.0

[DEBUG] Loading configuration from defaults
[DEBUG] Model file: model.onnx (2.3 MB)
[DEBUG] Blueprint: bert (19 steps)
[DEBUG] Output directory: ./build
[DEBUG] Build directory: /tmp/brainsmith/build_20250606_171530

[INFO] Starting model preprocessing
[DEBUG] Simplifying model... done (1.2s)
[DEBUG] Cleaning model... done (0.8s)
[INFO] Model preprocessing completed

[INFO] Starting dataflow build
[DEBUG] Step 1/19: common.cleanup... done (2.1s)
[DEBUG] Step 2/19: transformer.remove_head... done (1.5s)
...
```

## Benefits of CLI Design

### User Experience
1. **Intuitive Commands**: Natural language structure
2. **Helpful Output**: Clear progress and error messages
3. **Flexible Configuration**: Multiple ways to specify options
4. **Discovery**: Easy exploration of blueprints and steps
5. **Validation**: Pre-flight checks and helpful suggestions

### Developer Experience  
1. **Modular Structure**: Easy to add new commands
2. **Consistent Interface**: Uniform option handling
3. **Extensible**: Plugin-like command structure
4. **Testable**: Each command can be tested independently
5. **Maintainable**: Clear separation of concerns

### Integration
1. **CI/CD Friendly**: Scriptable with good exit codes
2. **Configuration Management**: File-based configuration
3. **Build Systems**: Easy integration with make, scripts
4. **Monitoring**: Structured output for log parsing
5. **Automation**: Batch processing capabilities

This CLI design provides a professional, user-friendly interface while maintaining the power and flexibility of the underlying compiler system.