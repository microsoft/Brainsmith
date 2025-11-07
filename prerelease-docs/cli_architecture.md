# Brainsmith CLI Architecture

## Overview

Brainsmith provides a dual CLI system designed to separate administrative tasks from operational workflows:

- **`brainsmith`**: Full administrative CLI with all commands
- **`smith`**: Streamlined operational CLI for hardware design generation

## CLI Design Philosophy

### Dual Entry Points

The system uses two distinct entry points to provide different user experiences:

1. **Administrative CLI (`brainsmith`)**: 
   - Includes configuration management (`config`)
   - Provides setup utilities (`setup`)
   - Can invoke operational commands via subcommands
   - Intended for system administrators and initial setup

2. **Operational CLI (`smith`)**:
   - Focused on core workflows (dataflow core creation, kernel generation)
   - Simplified interface for daily use
   - Inherits configuration from brainsmith setup
   - Intended for design engineers and regular users

### Command Organization

Commands are organized into two categories:

```python
OPERATIONAL_COMMANDS = {
    'dfc': dfc,      # Dataflow Core creation
    'kernel': kernel  # Hardware kernel generation
}

ADMIN_COMMANDS = {
    'config': config,  # Configuration management
    'setup': setup     # Dependency installation
}
```

## Configuration Hierarchy

Brainsmith follows a clear precedence order for configuration:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (`BSMITH_*` prefix)
3. **Project configuration** (`./brainsmith_config.yaml`)
4. **User configuration** (`~/brainsmith.yaml`)
5. **Built-in defaults** (lowest priority)

### Configuration Example

```yaml
# brainsmith_config.yaml
build_dir: ${HOME}/.brainsmith/builds
xilinx_path: /tools/Xilinx
xilinx_version: "2024.2"
plugins_strict: true
debug: false
```

## Usage Examples

### Administrative Tasks

```bash
# Initial setup
brainsmith setup all

# Configure project
brainsmith config init
brainsmith config show --verbose

# Export environment
eval $(brainsmith config export)
```

### Operational Workflows

```bash
# Create dataflow core
smith dfc model.onnx blueprint.yaml

# Generate hardware kernel
smith kernel design.sv --output-dir ./output

# Via brainsmith (inherits global options)
brainsmith --debug smith dfc model.onnx blueprint.yaml
```

## Module Structure

```
brainsmith/interface/
├── __init__.py         # Package exports
├── cli.py              # CLI factory and entry points
├── context.py          # Application context management
├── exceptions.py       # Custom exceptions
├── utils.py            # CLI utilities (console, progress)
├── formatters.py       # Output formatters (tables, etc.)
└── commands/
    ├── __init__.py     # Command registry
    ├── config.py       # Configuration management
    ├── dfc.py          # Dataflow core creation
    ├── kernel.py       # Kernel generation
    └── setup.py        # Setup utilities
```

## Key Design Patterns

### Factory Pattern for CLI Creation

The `create_cli()` function in `cli.py` acts as a factory to create appropriately configured Click groups based on the CLI name:

```python
def create_cli(name: str, include_admin: bool = True) -> click.Group:
    """Factory to create CLI with appropriate commands."""
    # ... creates and configures CLI based on name
```

### Context Management

The `ApplicationContext` class manages settings across the entire CLI session:

- Loads configuration from multiple sources
- Applies command-line overrides
- Provides unified access to effective configuration

### Command Registration

Commands are registered dynamically based on CLI type, allowing the same codebase to serve both administrative and operational interfaces.

## Extension Points

### Adding New Commands

1. Create a new command module in `brainsmith/interface/commands/`
2. Implement the command using Click decorators
3. Add to appropriate registry in `commands/__init__.py`:
   - `OPERATIONAL_COMMANDS` for smith commands
   - `ADMIN_COMMANDS` for brainsmith-only commands

### Custom Formatters

Output formatters are centralized in `formatters.py`. To add new output formats:

1. Create a new formatter class
2. Follow the pattern of `ConfigFormatter`
3. Use Rich library for enhanced terminal output

## Best Practices

1. **Keep Commands Focused**: Each command should do one thing well
2. **Use Context**: Pass configuration via `ApplicationContext`
3. **Provide Feedback**: Use progress spinners and clear success/error messages
4. **Handle Errors Gracefully**: Provide helpful error messages with actionable suggestions
5. **Document Commands**: Use Click's help strings and docstrings extensively