#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Execution entrypoint for Brainsmith development
# Used for running commands in existing containers

set -e

# Move to project directory
cd "${BSMITH_DIR:-/workspace}"

# 1. Activate project-local virtual environment if available
if [ -d ".venv" ]; then
    export VIRTUAL_ENV="$PWD/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    source .venv/bin/activate 2>/dev/null || true
fi

# 2. Source Xilinx tools if available (silent)
if [ -f "${VITIS_PATH}/settings64.sh" ]; then
    source "${VITIS_PATH}/settings64.sh" 2>/dev/null
elif [ -f "${VIVADO_PATH}/settings64.sh" ]; then
    source "${VIVADO_PATH}/settings64.sh" 2>/dev/null
fi

if [ -f "${HLS_PATH}/settings64.sh" ]; then
    source "${HLS_PATH}/settings64.sh" 2>/dev/null
fi

# 3. Set up basic environment
export PYTHONUNBUFFERED=1

# Ensure python symlink exists
if [ ! -e /usr/bin/python ] && [ -e /usr/bin/python3 ]; then
    mkdir -p "$HOME/.local/bin" 2>/dev/null
    ln -sf /usr/bin/python3 "$HOME/.local/bin/python" 2>/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi

# 4. Execute command or start interactive shell
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