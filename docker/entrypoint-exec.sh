#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Execution entrypoint for Brainsmith development
# Used for running commands in existing containers

set -e

# Source common environment setup
source /usr/local/bin/setup-env.sh

# Move to project directory
cd "${BSMITH_DIR:-/workspace}"

# 1. Activate project-local virtual environment
activate_venv

# 2. Set up Xilinx tool paths if BSMITH_XILINX_PATH is provided
if [ -n "$BSMITH_XILINX_PATH" ] && [ -d "$BSMITH_XILINX_PATH" ]; then
    # Derive tool paths from base path and version
    XILINX_VERSION="${BSMITH_XILINX_VERSION:-2024.2}"
    
    # Set paths without logging (silent for exec)
    [ -d "$BSMITH_XILINX_PATH/Vivado/$XILINX_VERSION" ] && export XILINX_VIVADO="$BSMITH_XILINX_PATH/Vivado/$XILINX_VERSION"
    [ -d "$BSMITH_XILINX_PATH/Vitis/$XILINX_VERSION" ] && export XILINX_VITIS="$BSMITH_XILINX_PATH/Vitis/$XILINX_VERSION"
    [ -d "$BSMITH_XILINX_PATH/Vitis_HLS/$XILINX_VERSION" ] && export XILINX_HLS="$BSMITH_XILINX_PATH/Vitis_HLS/$XILINX_VERSION"
fi

# 3. Source Xilinx tools (silent mode)
source_xilinx true

# 4. Ensure python symlink exists
ensure_python_symlink

# 5. Execute command or start interactive shell
if [ $# -gt 0 ]; then
    exec "$@"
else
    # Interactive shell with welcome message
    echo ""
    echo "🧠 Brainsmith Development Environment"
    echo "===================================="
    
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "✓ Virtual env: .venv"
    else
        echo "⚠️  Virtual env: Not found (run './setup-dev.sh' on host)"
    fi
    
    if [ -n "$XILINX_VIVADO" ] || [ -n "$XILINX_VITIS" ]; then
        echo "✓ Xilinx: Tools available"
    else
        echo "○ Xilinx: Not configured"
    fi
    
    echo ""
    echo "Quick start:"
    echo "  smith --help          # Brainsmith CLI"
    echo "  smith setup check     # Check setup status"
    echo "  smith setup all       # Complete setup"
    echo ""
    
    exec bash
fi