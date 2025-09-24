#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Common environment setup for Brainsmith containers
# This script is sourced by both entrypoint.sh and entrypoint-exec.sh
# to ensure consistent environment configuration

# Override HOME to container-specific directory to avoid permission issues
# This ensures tools like Vivado can write to ~/.Xilinx without errors
if [ -n "$BSMITH_CONTAINER_DIR" ] && [ -d "$BSMITH_CONTAINER_DIR" ]; then
    export HOME="$BSMITH_CONTAINER_DIR"
fi

# Set up basic environment
export PYTHONUNBUFFERED=1

# Ensure python symlink exists (for compatibility)
ensure_python_symlink() {
    if [ ! -e /usr/bin/python ] && [ -e /usr/bin/python3 ]; then
        if [ -w /usr/bin ]; then
            ln -sf /usr/bin/python3 /usr/bin/python
        else
            mkdir -p "$HOME/.local/bin" 2>/dev/null
            ln -sf /usr/bin/python3 "$HOME/.local/bin/python" 2>/dev/null
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
}

# Activate project-local virtual environment if available
activate_venv() {
    if [ -d ".venv" ]; then
        export VIRTUAL_ENV="$PWD/.venv"
        export PATH="$VIRTUAL_ENV/bin:$PATH"
        source .venv/bin/activate 2>/dev/null || true
        return 0
    fi
    return 1
}

# Source Xilinx tools if available
source_xilinx() {
    local silent=${1:-false}
    local sourced=false
    
    # Redirect output if silent mode
    local redirect=""
    [ "$silent" = "true" ] && redirect="2>/dev/null"
    
    # Try Vitis first (includes Vivado)
    if [ -f "${XILINX_VITIS}/settings64.sh" ]; then
        eval "source '${XILINX_VITIS}/settings64.sh' $redirect"
        [ "$silent" = "false" ] && echo "✓ Vitis sourced from ${XILINX_VITIS}"
        sourced=true
    elif [ -f "${XILINX_VIVADO}/settings64.sh" ]; then
        eval "source '${XILINX_VIVADO}/settings64.sh' $redirect"
        [ "$silent" = "false" ] && echo "✓ Vivado sourced from ${XILINX_VIVADO}"
        sourced=true
    fi
    
    # Source HLS if available
    if [ -f "${XILINX_HLS}/settings64.sh" ]; then
        eval "source '${XILINX_HLS}/settings64.sh' $redirect"
        [ "$silent" = "false" ] && echo "✓ Vitis HLS sourced from ${XILINX_HLS}"
        sourced=true
    fi
    
    if [ "$sourced" = "false" ] && [ -n "${XILINX_VIVADO}${XILINX_VITIS}${XILINX_HLS}" ]; then
        [ "$silent" = "false" ] && echo "⚠️  Xilinx paths configured but settings64.sh not found"
    fi
    
    # Set up finn_xsi Python path if available
    if [ -n "${XILINX_VIVADO}" ] && [ -n "${BSMITH_DEPS_DIR}" ]; then
        local finn_xsi_path="${BSMITH_DEPS_DIR}/finn/finn_xsi"
        if [ -d "$finn_xsi_path" ] && [ -f "$finn_xsi_path/xsi.so" ]; then
            export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${finn_xsi_path}"
            [ "$silent" = "false" ] && echo "✓ finn_xsi added to PYTHONPATH"
        fi
    fi
    
    return 0
}

# Poetry environment setup (for container-specific directories)
setup_poetry_env() {
    if [ -n "$BSMITH_CONTAINER_DIR" ]; then
        export POETRY_CONFIG_DIR="${BSMITH_CONTAINER_DIR}/.poetry-config"
        export POETRY_CACHE_DIR="${BSMITH_CONTAINER_DIR}/.poetry-cache"
        mkdir -p "$POETRY_CONFIG_DIR" "$POETRY_CACHE_DIR" 2>/dev/null
    fi
}