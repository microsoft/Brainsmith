# Docker-Free Setup Guide

This guide explains how to set up Brainsmith for development without Docker.

## Requirements

- Ubuntu 22.04 or newer
- Python 3.10 or 3.11
- Basic development tools

## Quick Start

### 1. Validate Your Environment

Check if your system meets the requirements:

```bash
./docker/manual/validate_env.sh
# or
make validate
```

If you see errors, install the missing packages:

```bash
sudo apt-get update
sudo apt-get install build-essential git wget zip unzip python3-pip python3-venv
```

For optional features:
```bash
# For finnxsi acceleration (if you have Vivado)
sudo apt-get install pybind11-dev

# For matplotlib GUI support
sudo apt-get install libglib2.0-0 libsm6 libxext6

# For Xilinx tool compatibility
sudo apt-get install libtinfo5 libc6-dev-i386
```

### 2. Set Up Python Environment

Once validation passes:

```bash
./docker/manual/setup_python_env.sh
# or
make setup-python
```

This will:
- Create a virtual environment
- Install all Python dependencies
- Clone and set up Git repositories (FINN, QONNX, etc.)
- Build finnxsi if Vivado is available

### 3. Activate the Environment

```bash
source venv/bin/activate
# or
source activate_brainsmith.sh
```

## Using with Existing Python Environment

If you prefer conda or poetry:

```bash
# With conda
conda create -n brainsmith python=3.10
conda activate brainsmith
./docker/manual/setup_python_env.sh --skip-venv

# With poetry
poetry shell
./docker/manual/setup_python_env.sh --skip-venv
```

## FPGA Development (Optional)

For FPGA synthesis, you need Xilinx tools installed separately:

1. Install Vivado/Vitis from Xilinx
2. Set environment variable:
   ```bash
   export XILINX_VIVADO=/opt/Xilinx/Vivado/2024.2
   ```
3. Source the Xilinx environment:
   ```bash
   source ./docker/manual/setup_xilinx_env.sh
   ```

## Common Commands

```bash
# Clean everything
rm -rf venv/ deps/

# Reinstall from scratch  
./docker/manual/setup_python_env.sh

# Update dependencies (re-run with existing venv)
./docker/manual/setup_python_env.sh

# Verify installation
python -c "import finn, qonnx, brevitas; print('All packages verified')"
```

## Managing Git Dependencies

All Git dependencies are configured in `docker/deps-config.sh`. To use a fork or different branch:

1. Edit the configuration:
   ```bash
   nano docker/deps-config.sh
   ```

2. Update the dependency variables:
   ```bash
   # Example: Using a fork of QONNX
   QONNX_URL="https://github.com/YOUR-FORK/qonnx.git"
   QONNX_COMMIT="your-branch"
   ```

3. Re-run setup to apply changes:
   ```bash
   ./docker/manual/setup_python_env.sh
   ```

### Dependency Groups

- **core**: Always fetched (ML frameworks, Python packages)
- **boards**: Board definition files (set `BSMITH_FETCH_BOARDS=true` to enable)
- **experimental**: Experimental features (set `BSMITH_FETCH_EXPERIMENTAL=true` to enable)

## Troubleshooting

### "Python X.Y found (recommended: 3.10 or 3.11)"
Your Python version should work but may have compatibility issues. Consider using pyenv or deadsnakes PPA to install Python 3.10.

### "pybind11-dev not found"
This is only needed if you want finnxsi acceleration. You can proceed without it, but hardware simulation will be slower.

### "GUI libraries missing"
Only needed if you want to display matplotlib plots. For headless/server usage, you can ignore this warning.

### Git dependencies installation fails
Make sure you have a stable internet connection. You can retry with:
```bash
./docker/manual/setup_python_env.sh --skip-fetch  # Skips re-cloning, just reinstalls
```

## Environment Variables

Brainsmith uses these environment variables (all optional - sensible defaults are provided):

### Xilinx Tools
```bash
# Set these before running setup
export XILINX_VIVADO=/opt/Xilinx/Vivado/2024.2
export XILINX_VITIS=/opt/Xilinx/Vitis/2024.2
export XILINX_HLS=/opt/Xilinx/Vitis_HLS/2024.2
```

### Build Configuration
```bash
export BSMITH_BUILD_DIR=/path/to/build/dir  # Default: /tmp/${USER}_brainsmith
export PYTHON=python3.11                     # Default: python3.10
```

### Dependency Control
```bash
export BSMITH_FETCH_BOARDS=false        # Skip board files
export BSMITH_FETCH_EXPERIMENTAL=true   # Include experimental repos
```

All configuration is centralized in `docker/env-config.sh` which is automatically sourced by all scripts.

## When to Use Docker

You should still use Docker (`./smithy`) for:
- CI/CD pipelines
- Guaranteed reproducible builds
- When you don't have Ubuntu 22.04+
- If you encounter system-specific issues

## Comparison: Docker vs Docker-Free

| Feature | Docker | Docker-Free |
|---------|---------|-------------|
| Setup time | ~30 minutes | ~5 minutes |
| System requirements | Any Linux/macOS | Ubuntu 22.04+ |
| Reproducibility | Perfect | Good |
| IDE integration | Limited | Native |
| Performance | Container overhead | Native speed |
| Xilinx tools | Mounted from host | Used directly |