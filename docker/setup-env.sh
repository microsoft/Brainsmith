#!/bin/bash
# Minimal environment setup for exec commands
# Duplicates entrypoint-exec.sh logic but for sourcing

# Move to project directory
cd "${BSMITH_DIR:-/workspace}" 2>/dev/null

# Activate Poetry environment if available
if [ -f "pyproject.toml" ] && command -v poetry >/dev/null 2>&1; then
    VENV_PATH=$(poetry env info --path 2>/dev/null || echo "")
    if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        export VIRTUAL_ENV="$VENV_PATH"
        export PATH="$VENV_PATH/bin:$PATH"
    fi
fi

# Source Xilinx tools if available (silent)
if [ -f "${VITIS_PATH}/settings64.sh" ]; then
    source "${VITIS_PATH}/settings64.sh" 2>/dev/null
elif [ -f "${VIVADO_PATH}/settings64.sh" ]; then
    source "${VIVADO_PATH}/settings64.sh" 2>/dev/null
fi

if [ -f "${HLS_PATH}/settings64.sh" ]; then
    source "${HLS_PATH}/settings64.sh" 2>/dev/null
fi

# Basic environment
export PYTHONUNBUFFERED=1