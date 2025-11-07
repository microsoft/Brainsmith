# Installation

!!! note "Prerequisites"
    - **Ubuntu 22.04+** (primary development/testing platform)
    - **[Vivado Design Suite](https://www.xilinx.com/support/download.html) 2024.2** (migration to 2025.1 in process)
    - **[Optional]** Cmake for BERT example V80 shell integration

---

## Installation Options

```bash
git clone https://github.com/microsoft/brainsmith.git ./brainsmith
cd brainsmith
```

### (Option A): Local Development with Poetry

!!! note "Prerequisites"
    - **Python 3.11+** and **[Poetry](https://python-poetry.org/docs/#installation)**
    - [Optional] **[direnv](https://direnv.net/)** for automatic environment activation

Run automated setup script

```bash
./setup-venv.sh
```

Edit the project configuration file to customize project settings (see
[Configuration](./configuration.md) for details)

```bash
vim .brainsmith/config.yaml
```

Activate environment manually or with direnv

```bash
source .venv/bin/activate && source .brainsmith/env.sh
# If using direnv, just reload directory
cd .
```

Query project settings to confirm your configuration is loaded correctly

```bash
brainsmith project info
```

### (Option B): Docker-based Development

Edit `ctl-docker.sh` or set environment variables to directly set brainsmith project settings

```bash
export BSMITH_XILINX_PATH=/tools/Xilinx/
export BSMITH_XILINX_VERSION=2024.2
```

Start container

```bash
./ctl-docker.sh start
```

Open an interactive shell to check your configuration

```bash
./ctl-docker.sh shell
brainsmith project info
```

Or send one-off commands to the container

```bash
./ctl-docker.sh "brainsmith project info"
```

---

## Working with Multiple Projects

While brainsmith creates a single poetry `venv`, but you can isolate multiple
workspaces using *projects*.

### Create a New Project

```bash
# Activate brainsmith venv if not in an active project
source /path/to/brainsmith/.venv/bin/activate

# Create and initialize new project directory
brainsmith project init ~/my-fpga-project
cd ~/my-fpga-project
```

Each project has its own configuration:

```bash
vim .brainsmith/config.yaml
```

[Optional] Enable auto-activation if using direnv

```bash
brainsmith project allow-direnv
cd .  # Triggers direnv
```

Otherwise, refresh env after any config changes:

```bash
source .brainsmith/env.sh
```

---

## Validate Installation

Run the quicktest to verify everything is working:

```bash
./examples/bert/quicktest.sh
```

This runs a minimal BERT example (single layer) to verify:
- Environment is configured correctly
- Vivado is accessible
- Dependencies are installed
- Basic DSE pipeline works

!!! info "Build Time"
    The quicktest can take upwards of an hour, depending on your system due to
    RTL simulation based fifo sizing.

---

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first DSE
- [Configuration Guide](configuration.md) - Learn about configuration options
- [Blueprint Reference](../developer-guide/3-reference/blueprints.md) - Understand YAML format
