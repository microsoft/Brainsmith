#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Simplified execution entrypoint for Brainsmith development
# Sources setup-shell.sh then executes commands

set -e

# Verify required environment variables
if [ -z "$BSMITH_DIR" ]; then
    echo "ERROR: BSMITH_DIR not set. This container must be started via ctl-docker.sh" >&2
    exit 1
fi

# Move to project directory
cd "$BSMITH_DIR"

# Source runtime environment
source /usr/local/bin/setup-shell.sh

# Execute command or start interactive shell
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
        echo "⚠️  Virtual env: Not found (run './setup-venv.sh' on host)"
    fi
    
    if [ -n "$XILINX_VIVADO" ] || [ -n "$XILINX_VITIS" ]; then
        echo "✓ Xilinx: Tools available"
    else
        echo "○ Xilinx: Not configured"
    fi
    
    echo ""
    echo "Quick start:"
    echo "  smith --help              # Operational CLI (DSE, kernels)"
    echo "  brainsmith setup check    # Check setup status"
    echo "  brainsmith setup all      # Complete setup"
    echo ""
    
    exec bash
fi