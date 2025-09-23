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

# Source common environment setup
source /usr/local/bin/setup-env.sh

# Log HOME change if it happened
[ "$HOME" = "$BSMITH_CONTAINER_DIR" ] && log "Set HOME=$HOME for container isolation"

# Ensure python symlink exists
ensure_python_symlink

# 1. Fetch Git repositories if needed
if [ "$BSMITH_SKIP_DEP_REPOS" != "1" ]; then
    if [ ! -d "$BSMITH_DEPS_DIR" ] || [ -z "$(ls -A $BSMITH_DEPS_DIR 2>/dev/null)" ]; then
        log "Fetching Git dependencies..."
        if [ -x "./fetch-repos.sh" ]; then
            ./fetch-repos.sh
        else
            log "⚠️  fetch-repos.sh not found or not executable"
        fi
    fi
else
    log "Skipping dependency repository fetch (BSMITH_SKIP_DEP_REPOS=1)"
fi

# 2. Install Poetry dependencies if needed
if [ -f "pyproject.toml" ] && command -v poetry >/dev/null 2>&1; then
    # Set up Poetry environment
    setup_poetry_env
    
    # Poetry will use POETRY_VIRTUALENVS_IN_PROJECT from environment (set to true by ctl-docker.sh)
    
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
    if activate_venv; then
        log "✓ Virtual environment activated"
    else
        log "⚠️  No .venv directory found"
    fi
fi

# 3. Set up Xilinx tool paths if BSMITH_XILINX_PATH is provided
if [ -n "$BSMITH_XILINX_PATH" ] && [ -d "$BSMITH_XILINX_PATH" ]; then
    # Derive tool paths from base path and version
    XILINX_VERSION="${BSMITH_XILINX_VERSION:-2024.2}"
    
    # Check for and set Vivado path
    if [ -d "$BSMITH_XILINX_PATH/Vivado/$XILINX_VERSION" ]; then
        export XILINX_VIVADO="$BSMITH_XILINX_PATH/Vivado/$XILINX_VERSION"
        log "Detected Vivado at $XILINX_VIVADO"
    fi
    
    # Check for and set Vitis path
    if [ -d "$BSMITH_XILINX_PATH/Vitis/$XILINX_VERSION" ]; then
        export XILINX_VITIS="$BSMITH_XILINX_PATH/Vitis/$XILINX_VERSION"
        log "Detected Vitis at $XILINX_VITIS"
    fi
    
    # Check for and set Vitis HLS path
    if [ -d "$BSMITH_XILINX_PATH/Vitis_HLS/$XILINX_VERSION" ]; then
        export XILINX_HLS="$BSMITH_XILINX_PATH/Vitis_HLS/$XILINX_VERSION"
        log "Detected Vitis HLS at $XILINX_HLS"
    fi
fi

# 4. Source Xilinx tools if available
if [ -n "${XILINX_VIVADO}${XILINX_VITIS}${XILINX_HLS}" ]; then
    # Use our common source_xilinx function with logging
    OLD_IFS=$IFS
    IFS=$'\n'
    while read -r line; do
        [ -n "$line" ] && log "$line"
    done < <(source_xilinx false 2>&1)
    IFS=$OLD_IFS
fi

# 5. Ensure smith is accessible in PATH
# With .venv activated, smith should already be in PATH
if command -v smith >/dev/null 2>&1; then
    log "✓ smith command available"
else
    log "⚠️  smith command not found in PATH"
fi

# 6. Keep container alive in daemon mode
log "Container ready (daemon mode)"
exec tail -f /dev/null