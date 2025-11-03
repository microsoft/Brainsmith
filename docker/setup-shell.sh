#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Runtime environment activation for Brainsmith
# This script must be SOURCED to propagate environment variables
# Used by both entrypoint.sh and entrypoint-exec.sh
#
# Sources .brainsmith/env.sh for unified environment (Docker + local development)
# Falls back to legacy BSMITH_XILINX_PATH setup if env.sh doesn't exist

# Python environment
export PYTHONUNBUFFERED=1

# Setup container directories (centralized location)
setup_container_directories() {
    # Create all required directories in HOME
    mkdir -p "$HOME"/{.Xilinx,.cache,.local/bin} 2>/dev/null || true
    
    # Set Xilinx environment to avoid user-specific data
    export XILINX_LOCAL_USER_DATA=no
    export XILINX_USER_DATA_DIR="$HOME/.Xilinx"
}

# Call directory setup function
setup_container_directories

# Ensure python symlink exists (for compatibility)
if [ ! -e /usr/bin/python ] && [ -e /usr/bin/python3 ]; then
    if [ -w /usr/bin ]; then
        ln -sf /usr/bin/python3 /usr/bin/python 2>/dev/null || true
    else
        ln -sf /usr/bin/python3 "$HOME/.local/bin/python" 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# Activate virtual environment if present
if [ -f "${BSMITH_DIR}/.venv/bin/activate" ]; then
    source "${BSMITH_DIR}/.venv/bin/activate"
fi

# Source project environment if available (unified approach)
# This sources .brainsmith/env.sh which sets all variables and sources Xilinx settings64.sh
if [ -f "${BSMITH_DIR}/.brainsmith/env.sh" ]; then
    source "${BSMITH_DIR}/.brainsmith/env.sh"
else
    # Fallback: Legacy manual Xilinx setup for backward compatibility
    # This runs if .brainsmith/env.sh doesn't exist yet (e.g., before project init)
    if [ -n "$BSMITH_XILINX_PATH" ] && [ -d "$BSMITH_XILINX_PATH" ]; then
        XILINX_VERSION="${BSMITH_XILINX_VERSION:-2024.2}"

        # Auto-detect tool installations
        [ -d "$BSMITH_XILINX_PATH/Vivado/$XILINX_VERSION" ] && \
            export XILINX_VIVADO="$BSMITH_XILINX_PATH/Vivado/$XILINX_VERSION"
        [ -d "$BSMITH_XILINX_PATH/Vitis/$XILINX_VERSION" ] && \
            export XILINX_VITIS="$BSMITH_XILINX_PATH/Vitis/$XILINX_VERSION"
        [ -d "$BSMITH_XILINX_PATH/Vitis_HLS/$XILINX_VERSION" ] && \
            export XILINX_HLS="$BSMITH_XILINX_PATH/Vitis_HLS/$XILINX_VERSION"

        # Source Xilinx settings - Vitis includes Vivado, so check it first
        if [ -f "${XILINX_VITIS}/settings64.sh" ]; then
            source "${XILINX_VITIS}/settings64.sh" 2>/dev/null
        elif [ -f "${XILINX_VIVADO}/settings64.sh" ]; then
            source "${XILINX_VIVADO}/settings64.sh" 2>/dev/null
        fi

        # Source HLS if available (separately as it's not included in Vitis)
        if [ -f "${XILINX_HLS}/settings64.sh" ]; then
            source "${XILINX_HLS}/settings64.sh" 2>/dev/null
        fi
    fi
fi