# Installation

!!! note "Prerequisites"
    - **Ubuntu 22.04+** (primary development/testing platform)
    - **[Vivado Design Suite](https://www.xilinx.com/support/download.html) 2024.2** (migration to 2025.1 in process)
    - **[Optional]** [direnv](https://direnv.net/) for automatic environment activation

## Installation Options

### 1. Clone the Repository

```bash
git clone https://github.com/microsoft/brainsmith.git ./brainsmith
cd brainsmith
```

### 2 (Option A): Local Development with Poetry

!!! note "Prerequisites"
    Python 3.11+ and [Poetry](https://python-poetry.org/docs/#installation) installed

---

#### i. Run Automated Setup

```bash
./setup-venv.sh
```

This script will:
```
- Create `.venv` virtual environment
- Install pip dependencies via Poetry
- Install Git & C++ dependencies into `deps/`
- Initialize project configuration in `.brainsmith/`
- Set up direnv integration (if direnv is installed)
```

#### ii. Configure Project Settings

Edit the project configuration file to customize project settings.

```bash
vim .brainsmith/config.yaml
```

Example configuration (see all settings [here](../api/settings.md)):

```yaml
xilinx_path: /<path-to-xilinx-tools>/Xilinx/Vivado
xilinx_version: 2024.2
```

#### iii. Activate Environment

You have two options for activating the environment:

**Option 1: direnv (automatic)**

```bash
cd .  # Triggers direnv to load .brainsmith/env.sh
```

**Option 2: Manual activation**

- Requires Docker with [non-root permissions](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)


Must be run from the brainsmith root directory:

```bash
source .venv/bin/activate && source .brainsmith/env.sh
```

#### 5. Verify Setup

```bash
brainsmith project show
```

You should see output confirming your configuration is loaded correctly.

---

### Option B: Docker-based Development

#### 1. Configure Environment Variables

Edit `ctl-docker.sh` or set environment variables:

```bash
export BSMITH_XILINX_PATH=/opt/Xilinx/Vivado/2024.2
export BSMITH_XILINX_VERSION=2024.2
export XILINXD_LICENSE_FILE=/path/to/your/license.lic
```

#### 2. Start Container

```bash
./ctl-docker.sh start
```

#### 3. Open Interactive Shell

```bash
./ctl-docker.sh shell
brainsmith project show
```

Or run one-off commands:

```bash
./ctl-docker.sh "brainsmith project show"
```

---

## Working with Multiple Projects

You can create separate project directories to isolate work from the brainsmith repository.

### Create a New Project

```bash
# Activate brainsmith venv (always required first)
source /path/to/brainsmith/.venv/bin/activate

# Create and initialize new project directory
brainsmith project init ~/my-fpga-project
cd ~/my-fpga-project
```

### Configure Project

Each project has its own configuration:

```bash
vim .brainsmith/config.yaml
```

### Enable Auto-Activation

**Option 1: direnv (recommended)**

```bash
brainsmith project allow-direnv
cd .  # Triggers direnv
```

**Option 2: Manual activation**

Re-run after any config changes:

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
    The quicktest takes approximately 30-60 minutes, depending on your system.

---

## Troubleshooting

### Poetry Installation Issues

If `poetry install` fails:

```bash
# Clear cache and retry
poetry cache clear pypi --all
poetry install
```

### Vivado Not Found

If you see "Vivado not found" errors:

1. Verify `xilinx_path` in `.brainsmith/config.yaml`
2. Check Vivado is installed: `ls /opt/Xilinx/Vivado/`
3. Re-activate environment: `source .brainsmith/env.sh`

### Environment Variables Not Set

If `brainsmith project show` shows empty values:

1. Ensure you've sourced the environment:
   ```bash
   source .brainsmith/env.sh
   ```
2. Check config file exists: `cat .brainsmith/config.yaml`
3. Verify you're in the correct directory

### Docker Permission Errors

If Docker commands fail:

1. Add user to docker group:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```
2. Restart Docker service:
   ```bash
   sudo systemctl restart docker
   ```

---

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first DSE
- [Configuration Guide](configuration.md) - Learn about configuration options
- [Blueprint Reference](../developer-guide/3-reference/blueprints.md) - Understand YAML format
