#!/bin/bash
# Minimal environment setup for exec commands
# Duplicates entrypoint-exec.sh logic but for sourcing

# Move to project directory
cd "${BSMITH_DIR:-/workspace}" 2>/dev/null

# Activate project-local virtual environment if available
if [ -d ".venv" ]; then
    export VIRTUAL_ENV="$PWD/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    source .venv/bin/activate 2>/dev/null || true
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