#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Streamlined entrypoint for Brainsmith development
# Handles all setup automatically for Docker-only workflows

set -e

# Simple logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 3. Set up basic environment
export PYTHONUNBUFFERED=1

# Ensure python symlink exists (for compatibility)
if [ ! -e /usr/bin/python ] && [ -e /usr/bin/python3 ]; then
    if [ -w /usr/bin ]; then
        ln -sf /usr/bin/python3 /usr/bin/python
    else
        mkdir -p "$HOME/.local/bin"
        ln -sf /usr/bin/python3 "$HOME/.local/bin/python"
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# 1. Fetch Git repositories if needed
if [ ! -d "$BSMITH_DEPS_DIR" ] || [ -z "$(ls -A $BSMITH_DEPS_DIR 2>/dev/null)" ]; then
    log "Fetching Git dependencies..."
    if [ -x "./fetch-repos.sh" ]; then
        ./fetch-repos.sh
    else
        log "⚠️  fetch-repos.sh not found or not executable"
    fi
fi

# 2. Install Poetry dependencies if needed
if [ -f "pyproject.toml" ] && command -v poetry >/dev/null 2>&1; then
    # Configure Poetry for project-local virtual environment
    poetry config virtualenvs.in-project true
    
    # Check if .venv exists and dependencies are installed
    if [ ! -d ".venv" ] || ! .venv/bin/python -c "import torch" 2>/dev/null; then
        log "Installing Poetry dependencies..."
        poetry install --no-interaction
        if [ $? -eq 0 ]; then
            log "✓ Poetry dependencies installed successfully"
        else
            log "⚠️  Poetry install failed"
        fi
    fi
    
    # Activate project-local virtual environment
    if [ -d ".venv" ]; then
        log "Activating virtual environment..."
        export VIRTUAL_ENV="$PWD/.venv"
        export PATH="$VIRTUAL_ENV/bin:$PATH"
        source .venv/bin/activate
    else
        log "⚠️  No .venv directory found"
    fi
fi

# 3. Source Xilinx tools if available
source_xilinx() {
    local sourced=false
    
    # Try Vitis first (includes Vivado)
    if [ -f "${VITIS_PATH}/settings64.sh" ]; then
        source "${VITIS_PATH}/settings64.sh"
        log "✓ Vitis sourced from ${VITIS_PATH}"
        sourced=true
    elif [ -f "${VIVADO_PATH}/settings64.sh" ]; then
        source "${VIVADO_PATH}/settings64.sh"
        log "✓ Vivado sourced from ${VIVADO_PATH}"
        sourced=true
    fi
    
    # Source HLS if available
    if [ -f "${HLS_PATH}/settings64.sh" ]; then
        source "${HLS_PATH}/settings64.sh"
        log "✓ Vitis HLS sourced from ${HLS_PATH}"
        sourced=true
    fi
    
    if [ "$sourced" = "false" ] && [ -n "${VIVADO_PATH}${VITIS_PATH}${HLS_PATH}" ]; then
        log "⚠️  Xilinx paths configured but settings64.sh not found"
    fi
}

if [ -n "${VIVADO_PATH}${VITIS_PATH}${HLS_PATH}" ]; then
    source_xilinx
fi

# 4. Ensure smith is accessible in PATH
# With .venv activated, smith should already be in PATH
if command -v smith >/dev/null 2>&1; then
    log "✓ smith command available"
else
    log "⚠️  smith command not found in PATH"
fi

# 5. Keep container alive in daemon mode
log "Container ready (daemon mode)"
exec tail -f /dev/null