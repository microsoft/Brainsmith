# Configuration

Brainsmith uses a Pydantic-based configuration system with multiple sources for maximum flexibility.

## Configuration Sources

Configurations are loaded in priority order (highest to lowest):

1. **CLI arguments / environment variables** - Override all other sources
2. **Explicit config file** - Via `--config` or `BRAINSMITH_CONFIG`
3. **User config** - `~/.brainsmith/config.yaml`
4. **Project config** - `.brainsmith/config.yaml` (in repo)
5. **Built-in defaults** - Hardcoded fallbacks

## Configuration File Format

```yaml
# Example: ~/.brainsmith/config.yaml

# Xilinx tool paths
xilinx_path: /opt/Xilinx/Vivado/2024.2
xilinx_version: 2024.2

# Build directory
build_dir: /tmp/finn_dev_${USER}

# Optional: Override specific settings
log_level: INFO
parallel_builds: 4
```

## Key Configuration Options

### Xilinx Tools

```yaml
xilinx_path: /opt/Xilinx/Vivado/2024.2
xilinx_version: 2024.2
```

Required for synthesis and implementation. The path should point to your Vivado installation directory.

### Build Directory

```yaml
build_dir: /tmp/finn_dev_${USER}
```

Where temporary build artifacts are stored. Can use environment variables like `${USER}`.

### Logging

```yaml
log_level: DEBUG  # DEBUG, INFO, WARNING, ERROR
```

Controls verbosity of logging output.

## CLI Configuration Commands

### Initialize Config

```bash
# Create default config file
brainsmith config init
```

Creates `~/.brainsmith/config.yaml` with default values.

### View Current Config

```bash
# Show effective configuration (all sources merged)
brainsmith config show
```

### Export Environment

```bash
# Export Xilinx environment variables
eval $(brainsmith config export)
```

This sets up `XILINX_ROOT`, `XILINX_VERSION`, and other environment variables needed by FINN and Vivado.

## CLI Overrides

Override configuration via command-line:

```bash
# Override build directory
smith dse model.onnx blueprint.yaml --build-dir /custom/path

# Use specific config file
smith --config /path/to/config.yaml dse model.onnx blueprint.yaml

# Enable debug mode
smith --debug dse model.onnx blueprint.yaml
```

## Environment Variables

You can also use environment variables:

```bash
export BRAINSMITH_CONFIG=/path/to/config.yaml
export BRAINSMITH_BUILD_DIR=/custom/build/dir
```

## Project-Specific Configuration

For project-specific settings, create `.brainsmith/config.yaml` in your project root:

```yaml
# .brainsmith/config.yaml (in git repo)
build_dir: ./build
log_level: DEBUG
```

This allows different projects to have different defaults while keeping user settings separate.

## Troubleshooting

### Config not found

```bash
# Verify config file exists
ls -la ~/.brainsmith/config.yaml

# Re-initialize if needed
brainsmith config init
```

### Wrong Xilinx path

```bash
# Check current setting
brainsmith config show | grep xilinx

# Update in config file
nano ~/.brainsmith/config.yaml
```

### Environment not set

```bash
# Make sure to run export command
eval $(brainsmith config export)

# Verify environment
echo $XILINX_ROOT
```

## Next Steps

- [CLI Reference](../api-reference/core.md) - Full command documentation
- [Architecture](../architecture/overview.md) - How configuration flows through the system
