# Configuration

Brainsmith uses a project-based configuration system that allows you to manage settings across multiple projects while maintaining clean isolation.

## Configuration Overview

Brainsmith loads configuration from multiple sources with a clear precedence order:

```mermaid
graph LR
    A[CLI Arguments] --> B[Environment Variables]
    B --> C[Project Config]
    C --> D[User Config]
    D --> E[Built-in Defaults]
    E --> F[Effective Config]
```

### Precedence Order (Highest to Lowest)

1. **Command-line arguments** - Flags passed directly to commands
2. **Environment variables** - Variables with `BSMITH_` prefix
3. **Project configuration** - `.brainsmith/config.yaml` in current directory
4. **User configuration** - `~/.brainsmith/config.yaml`
5. **Built-in defaults** - Hardcoded fallbacks in the code

## Project Configuration

Each project directory can have its own configuration in `.brainsmith/config.yaml`.

### Creating a Project

Initialize a new project directory:

```bash
# Activate brainsmith venv first
source /path/to/brainsmith/.venv/bin/activate

# Create and initialize project
brainsmith project init ~/my-fpga-project
cd ~/my-fpga-project
```

This creates:
- `.brainsmith/` directory
- `.brainsmith/config.yaml` - Project-specific configuration
- `.brainsmith/env.sh` - Environment activation script
- `.envrc` - direnv configuration (if direnv installed)

### Project Configuration File

Edit `.brainsmith/config.yaml`:

```yaml
# Required: Xilinx Vivado configuration
xilinx_path: /opt/Xilinx/Vivado/2024.2
xilinx_version: 2024.2

# Optional: Build directory
build_dir: /tmp/finn_dev_${USER}

# Optional: Advanced settings
plugins_strict: false
debug: false
```

### Activating Project Environment

Two options for loading project configuration:

**Option 1: direnv (automatic)**

If direnv is installed and allowed:

```bash
cd ~/my-fpga-project  # Auto-loads .brainsmith/env.sh
```

Enable direnv for a project:

```bash
brainsmith project allow-direnv
```

**Option 2: Manual activation**

Source the environment script:

```bash
source .brainsmith/env.sh
```

!!! tip "Re-activation After Config Changes"
    If you edit `config.yaml`, you must re-activate:
    ```bash
    source .brainsmith/env.sh  # Manual
    # OR
    cd .  # direnv users
    ```

## User Configuration

Global user configuration in `~/.brainsmith/config.yaml` applies to all projects that don't override specific settings.

### Initialize User Config

```bash
brainsmith config init
```

This creates `~/.brainsmith/config.yaml` with defaults:

```yaml
xilinx_path: /opt/Xilinx/Vivado/2024.2
xilinx_version: 2024.2
build_dir: /tmp/finn_dev_${USER}
```

### View Current Configuration

```bash
# Show effective configuration
brainsmith project info

# More detailed view
brainsmith config show --verbose
```

## Environment Variables

Configuration can be overridden with environment variables using the `BSMITH_` prefix.

### Variable Naming

Convert config keys to environment variables:

- `xilinx_path` → `BSMITH_XILINX_PATH`
- `xilinx_version` → `BSMITH_XILINX_VERSION`
- `build_dir` → `BSMITH_BUILD_DIR`

### Example Usage

```bash
# Override Vivado path for one session
export BSMITH_XILINX_PATH=/opt/Xilinx/Vivado/2025.1
export BSMITH_XILINX_VERSION=2025.1

# Run command with overrides
smith dfc model.onnx blueprint.yaml
```

### Export from Config

Generate environment variables from config:

```bash
# Export to shell
eval $(brainsmith config export)

# View export commands
brainsmith config export
```

Output:
```bash
export XILINX_PATH=/opt/Xilinx/Vivado/2024.2
export XILINX_VERSION=2024.2
export BUILD_DIR=/tmp/finn_dev_user
# ... etc
```

## Key Configuration Settings

### Required Settings

#### xilinx_path
Path to Vivado installation directory.

```yaml
xilinx_path: /opt/Xilinx/Vivado/2024.2
```

Environment variable: `BSMITH_XILINX_PATH`

#### xilinx_version
Vivado version number.

```yaml
xilinx_version: 2024.2
```

Environment variable: `BSMITH_XILINX_VERSION`

### Optional Settings

#### build_dir
Directory for build artifacts. Supports variable expansion.

```yaml
build_dir: /tmp/finn_dev_${USER}
```

Default: `/tmp/finn_dev_${USER}`

Environment variable: `BSMITH_BUILD_DIR`

#### plugins_strict
Enable strict plugin loading (fail on missing plugins).

```yaml
plugins_strict: false
```

Default: `false`

Environment variable: `BSMITH_PLUGINS_STRICT`

#### debug
Enable debug output and logging.

```yaml
debug: true
```

Default: `false`

Environment variable: `BSMITH_DEBUG`

## Working with Multiple Projects

### Scenario: Multiple Projects with Different Vivado Versions

**Project A** (Vivado 2024.2):
```bash
cd ~/project-a
cat .brainsmith/config.yaml
```
```yaml
xilinx_path: /opt/Xilinx/Vivado/2024.2
xilinx_version: 2024.2
```

**Project B** (Vivado 2025.1):
```bash
cd ~/project-b
cat .brainsmith/config.yaml
```
```yaml
xilinx_path: /opt/Xilinx/Vivado/2025.1
xilinx_version: 2025.1
```

Each project automatically uses its own Vivado version when you activate the environment.

### Sharing Configuration

You can check project configs into version control:

```bash
# Safe to commit
git add .brainsmith/config.yaml
git commit -m "Add project configuration"
```

Other developers clone and activate:

```bash
git clone <repo>
cd <repo>
source .brainsmith/env.sh
```

## Troubleshooting

### Configuration Not Loading

**Symptom:** `brainsmith project info` displays empty or default values

**Solutions:**

1. Check you're in the correct directory:
   ```bash
   ls .brainsmith/config.yaml
   ```

2. Ensure environment is activated:
   ```bash
   source .brainsmith/env.sh
   ```

3. Verify config syntax is valid YAML:
   ```bash
   cat .brainsmith/config.yaml
   ```

### Vivado Not Found

**Symptom:** `vivado: command not found` during builds

**Solutions:**

1. Verify Vivado path in config:
   ```bash
   brainsmith project info | grep xilinx_path
   ```

2. Check Vivado is installed at specified path:
   ```bash
   ls /opt/Xilinx/Vivado/2024.2
   ```

3. Re-activate environment:
   ```bash
   source .brainsmith/env.sh
   echo $XILINX_PATH
   ```

### Environment Variables Not Set

**Symptom:** `echo $XILINX_PATH` shows nothing

**Solutions:**

1. Make sure you sourced (not ran) the env script:
   ```bash
   source .brainsmith/env.sh  # Correct
   # NOT: ./.brainsmith/env.sh  # Wrong
   ```

2. Check the env script exists:
   ```bash
   cat .brainsmith/env.sh
   ```

3. Verify project was initialized:
   ```bash
   brainsmith project init .  # Re-initialize if needed
   ```

### Config Changes Not Taking Effect

**Symptom:** Modified config values aren't being used

**Solutions:**

1. Re-activate the environment after editing config:
   ```bash
   source .brainsmith/env.sh
   ```

2. For direnv users, trigger reload:
   ```bash
   cd .
   ```

3. Verify effective config:
   ```bash
   brainsmith config show
   ```

## Advanced: Docker Configuration

When using Docker, configuration is passed via environment variables:

```bash
# In ctl-docker.sh
export BSMITH_XILINX_PATH=/opt/Xilinx/Vivado/2024.2
export BSMITH_XILINX_VERSION=2024.2
export XILINXD_LICENSE_FILE=/path/to/license.lic

# Start container
./ctl-docker.sh start
```

Inside the container, these variables are automatically available.

## Next Steps

- [Quick Start](quickstart.md) - Run your first DSE with configured environment
- [CLI Reference](../developer-guide/3-reference/cli.md) - Learn the `brainsmith` and `smith` commands
- [Blueprints](../developer-guide/3-reference/blueprints.md) - Understand project-specific design configuration
