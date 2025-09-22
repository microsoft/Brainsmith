#!/bin/bash
# Brainsmith Environment Configuration
# Source this file to set up environment variables for Brainsmith
#
# Users can override any of these by setting them before sourcing this file

# Auto-detect Brainsmith root directory
if [ -z "$BSMITH_DIR" ]; then
    # Try to detect from script location
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    if [ -n "$_SCRIPT_DIR" ]; then
        export BSMITH_DIR="$(cd "$_SCRIPT_DIR/.." && pwd)"
    fi
    unset _SCRIPT_DIR
fi

# Verify BSMITH_DIR is set
if [ -z "$BSMITH_DIR" ]; then
    echo "ERROR: BSMITH_DIR not set" >&2
    return 1
fi

# Build directory for temporary files
export BSMITH_BUILD_DIR="${BSMITH_BUILD_DIR:-/tmp/${USER}_brainsmith}"

# Dependencies directory
export BSMITH_DEPS_DIR="${BSMITH_DEPS_DIR:-$BSMITH_DIR/deps}"

# FINN compatibility variables
export FINN_ROOT="$BSMITH_DEPS_DIR/finn"
export FINN_BUILD_DIR="$BSMITH_BUILD_DIR"
export FINN_DEPS_DIR="$BSMITH_DEPS_DIR"

# Python environment
export PYTHON="${PYTHON:-python3.10}"
export PYTHONUNBUFFERED=1
export BSMITH_PLUGINS_STRICT="${BSMITH_PLUGINS_STRICT:-true}"

# Poetry-specific environment variables
export POETRY_CACHE_DIR="${POETRY_CACHE_DIR:-$HOME/.cache/pypoetry}"
export POETRY_VIRTUALENVS_IN_PROJECT="${POETRY_VIRTUALENVS_IN_PROJECT:-false}"
export POETRY_VIRTUALENVS_PATH="${POETRY_VIRTUALENVS_PATH:-/tmp/poetry_venvs}"

# Xilinx/AMD tools configuration
# Users should set XILINX_VIVADO, XILINX_VITIS, etc. in their environment
# We'll set derived variables if the main tools are available
if [ -n "$XILINX_VIVADO" ] && [ -d "$XILINX_VIVADO" ]; then
    export VIVADO_PATH="$XILINX_VIVADO"
    export VIVADO_IP_CACHE="${VIVADO_IP_CACHE:-$BSMITH_BUILD_DIR/vivado_ip_cache}"
    export XILINX_LOCAL_USER_DATA="${XILINX_LOCAL_USER_DATA:-no}"
    
    # Set LD_PRELOAD for Xilinx compatibility (if not already set)
    if [ -z "$LD_PRELOAD" ] && [ -f "/lib/x86_64-linux-gnu/libudev.so.1" ]; then
        export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1"
    fi
fi

if [ -n "$XILINX_VITIS" ] && [ -d "$XILINX_VITIS" ]; then
    export VITIS_PATH="$XILINX_VITIS"
fi

if [ -n "$XILINX_HLS" ] && [ -d "$XILINX_HLS" ]; then
    export HLS_PATH="$XILINX_HLS"
elif [ -n "$XILINX_VITIS_HLS" ] && [ -d "$XILINX_VITIS_HLS" ]; then
    export HLS_PATH="$XILINX_VITIS_HLS"
fi

# Legacy Xilinx path support (for backward compatibility)
if [ -n "$BSMITH_XILINX_PATH" ] && [ -z "$XILINX_VIVADO" ]; then
    BSMITH_XILINX_VERSION="${BSMITH_XILINX_VERSION:-2024.2}"
    if [ -d "$BSMITH_XILINX_PATH/Vivado/$BSMITH_XILINX_VERSION" ]; then
        export XILINX_VIVADO="$BSMITH_XILINX_PATH/Vivado/$BSMITH_XILINX_VERSION"
        export VIVADO_PATH="$XILINX_VIVADO"
    fi
    if [ -d "$BSMITH_XILINX_PATH/Vitis/$BSMITH_XILINX_VERSION" ]; then
        export XILINX_VITIS="$BSMITH_XILINX_PATH/Vitis/$BSMITH_XILINX_VERSION"
        export VITIS_PATH="$XILINX_VITIS"
    fi
    if [ -d "$BSMITH_XILINX_PATH/Vitis_HLS/$BSMITH_XILINX_VERSION" ]; then
        export XILINX_HLS="$BSMITH_XILINX_PATH/Vitis_HLS/$BSMITH_XILINX_VERSION"
        export HLS_PATH="$XILINX_HLS"
    fi
fi

# Dependency fetching control
export BSMITH_FETCH_BOARDS="${BSMITH_FETCH_BOARDS:-true}"
export BSMITH_FETCH_EXPERIMENTAL="${BSMITH_FETCH_EXPERIMENTAL:-false}"
# Legacy FINN variable support
if [ "$FINN_SKIP_BOARD_FILES" = "1" ]; then
    export BSMITH_FETCH_BOARDS="false"
fi

# Other tool paths
export OHMYXILINX="${OHMYXILINX:-$BSMITH_DIR/deps/oh-my-xilinx}"
export PLATFORM_REPO_PATHS="${PLATFORM_REPO_PATHS:-/opt/xilinx/platforms}"

# Hardware compiler backend
export BSMITH_HW_COMPILER="${BSMITH_HW_COMPILER:-finn}"

# Network ports
export NETRON_PORT="${NETRON_PORT:-8080}"

# Docker-specific variables
export BSMITH_SSH_KEY_DIR="${BSMITH_SSH_KEY_DIR:-$BSMITH_DIR/ssh_keys}"
export BSMITH_SKIP_DEP_REPOS="${BSMITH_SKIP_DEP_REPOS:-0}"
export BSMITH_DOCKER_RUN_AS_ROOT="${BSMITH_DOCKER_RUN_AS_ROOT:-0}"
export BSMITH_DOCKER_PREBUILT="${BSMITH_DOCKER_PREBUILT:-0}"
export BSMITH_DOCKER_NO_CACHE="${BSMITH_DOCKER_NO_CACHE:-0}"

# Poetry setup flag
export BSMITH_POETRY_SETUP="${BSMITH_POETRY_SETUP:-true}"

# Dependency update policy for Docker
# Options: prompt, skip-modified, stash-always, fail-on-modified
# Default: skip-modified (safe for Docker to prevent data loss)
export BSMITH_DEPS_POLICY="${BSMITH_DEPS_POLICY:-skip-modified}"

# Debug settings
export BSMITH_DEBUG="${BSMITH_DEBUG:-0}"

# Show configuration in debug mode
if [ "$BSMITH_DEBUG" = "1" ]; then
    echo "[DEBUG] Brainsmith environment configured:"
    echo "[DEBUG]   BSMITH_DIR=$BSMITH_DIR"
    echo "[DEBUG]   BSMITH_BUILD_DIR=$BSMITH_BUILD_DIR"
    echo "[DEBUG]   PYTHON=$PYTHON"
    echo "[DEBUG]   POETRY_CACHE_DIR=$POETRY_CACHE_DIR"
    echo "[DEBUG]   BSMITH_POETRY_SETUP=$BSMITH_POETRY_SETUP"
    [ -n "$XILINX_VIVADO" ] && echo "[DEBUG]   XILINX_VIVADO=$XILINX_VIVADO"
    [ -n "$XILINX_VITIS" ] && echo "[DEBUG]   XILINX_VITIS=$XILINX_VITIS"
    [ -n "$XILINX_HLS" ] && echo "[DEBUG]   XILINX_HLS=$XILINX_HLS"
fi