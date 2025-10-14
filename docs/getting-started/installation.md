# Installation

This guide will help you set up Brainsmith for development.

## Prerequisites

- Ubuntu 22.04+
- Python 3.10+
- Vivado Design Suite 2024.2
- [Poetry](https://python-poetry.org/docs/#installation)

## Poetry Environment Setup

### Automated Setup

```bash
# Run automated setup script
./setup-venv.sh

# Activate virtual environment
source .venv/bin/activate
```

### Manual Setup

If you prefer manual setup:

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Configuration

### Initialize Configuration

```bash
# Create config file
brainsmith config init
```

This creates `~/.brainsmith/config.yaml`.

### Edit Configuration

Edit `~/.brainsmith/config.yaml` to set your Xilinx paths:

```yaml
xilinx_path: /opt/Xilinx/Vivado/2024.2
xilinx_version: 2024.2
build_dir: /tmp/finn_dev_${USER}
```

### Verify Configuration

```bash
# View current configuration
brainsmith config show

# Export environment variables
eval $(brainsmith config export)
```

## Validate Installation

Run the quick validation test:

```bash
./examples/bert/quicktest.sh
```

This runs a minimal BERT example (single layer) to verify everything works.

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first DSE
- [Configuration Guide](configuration.md) - Learn about configuration options
