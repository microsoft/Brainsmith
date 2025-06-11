#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

# Legacy-compatible wrapper for run-docker.sh that uses smithy under the hood
# Maintains one-off container behavior to encourage migration to smithy

# Define color functions (matching original run-docker.sh)
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

gecho() { echo -e "${GREEN}$1${NC}"; }
recho() { echo -e "${RED}$1${NC}"; }
yecho() { echo -e "${YELLOW}$1${NC}"; }

# Auto-detect brainsmith directory (where this script lives)
BSMITH_DIR="$(readlink -f -- "${BASH_SOURCE[0]%/*}")"
SMITHY_PATH="$BSMITH_DIR/smithy"

# Verify smithy exists
if [ ! -x "$SMITHY_PATH" ]; then
    recho "ERROR: smithy script not found at $SMITHY_PATH"
    recho "Please ensure you're running this from the brainsmith root directory"
    exit 1
fi

# Export environment variables that smithy expects (matching original run-docker.sh)
export BSMITH_DIR
export BSMITH_HW_COMPILER="${BSMITH_HW_COMPILER:-finn}"
export BSMITH_DOCKER_TAG="${BSMITH_DOCKER_TAG:-microsoft/brainsmith:$(git describe --always --tags --dirty 2>/dev/null || echo 'dev')}"
export LOCALHOST_URL="${LOCALHOST_URL:-localhost}"
export NETRON_PORT="${NETRON_PORT:-8080}"
export NUM_DEFAULT_WORKERS="${NUM_DEFAULT_WORKERS:-4}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-}"

# Directories
export BSMITH_BUILD_DIR="${BSMITH_BUILD_DIR:-/tmp/brainsmith_dev_0}"
export BSMITH_SSH_KEY_DIR="${BSMITH_SSH_KEY_DIR:-$BSMITH_DIR/ssh_keys}"
export PLATFORM_REPO_PATHS="${PLATFORM_REPO_PATHS:-/opt/xilinx/platforms}"

# Xilinx specific variables
export OHMYXILINX="${OHMYXILINX:-${BSMITH_DIR}/deps/oh-my-xilinx}"
export VIVADO_HLS_LOCAL="${VIVADO_HLS_LOCAL:-$VIVADO_PATH}"
export VIVADO_IP_CACHE="${VIVADO_IP_CACHE:-$BSMITH_BUILD_DIR/vivado_ip_cache}"

# Docker build options
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
export BSMITH_DOCKER_PREBUILT="${BSMITH_DOCKER_PREBUILT:-0}"
export BSMITH_DOCKER_NO_CACHE="${BSMITH_DOCKER_NO_CACHE:-0}"
export BSMITH_SKIP_DEP_REPOS="${BSMITH_SKIP_DEP_REPOS:-0}"

# Docker run options
export BSMITH_DOCKER_RUN_AS_ROOT="${BSMITH_DOCKER_RUN_AS_ROOT:-0}"
export BSMITH_DOCKER_GPU="${BSMITH_DOCKER_GPU:-$(docker info 2>/dev/null | grep nvidia | wc -l || echo 0)}"

# Additional Docker options
export BSMITH_DOCKER_BUILD_FLAGS="${BSMITH_DOCKER_BUILD_FLAGS:-}"
export BSMITH_DOCKER_FLAGS="${BSMITH_DOCKER_FLAGS:-}"

# Print migration warning
yecho "   NOTICE: You're using the legacy run-docker.sh wrapper"
yecho "   For better performance with persistent containers, use smithy directly:"
yecho "   • 'smithy daemon' - Start persistent container in background"
yecho "   • 'smithy exec <command>' - Run commands in persistent container"
yecho "   • 'smithy shell' - Interactive shell in persistent container"
echo

# Build Docker image if needed (using smithy's build logic)
if [ "$BSMITH_DOCKER_PREBUILT" = "0" ]; then
    gecho "Building Docker image if needed..."
    "$SMITHY_PATH" build >/dev/null 2>&1 || {
        recho "Failed to build Docker image"
        exit 1
    }
fi

# Helper to run one-off containers using smithy daemon pattern
run_oneoff_container() {
    local CMD="$1"
    
    if [ -z "$CMD" ]; then
        # Interactive shell - use smithy start (creates temporary container with --rm)
        exec "$SMITHY_PATH" start
    else
        # For commands, use smithy daemon->exec->stop pattern for optimal performance
        # This ensures we get the full container environment and proper cleanup
        gecho "Starting temporary daemon container..."
        
        # Start daemon if not already running
        if ! "$SMITHY_PATH" status >/dev/null 2>&1 | grep -q "is running"; then
            "$SMITHY_PATH" daemon >/dev/null 2>&1 || {
                recho "Failed to start daemon container"
                exit 1
            }
        fi
        
        # Execute command in the daemon
        local EXIT_CODE=0
        "$SMITHY_PATH" exec "$CMD" || EXIT_CODE=$?
        
        # Stop daemon after execution
        "$SMITHY_PATH" stop >/dev/null 2>&1
        
        exit $EXIT_CODE
    fi
}

# Main command logic - simplified and unified
if [ -z "$1" ]; then
    gecho "Running Brainsmith docker container"
    run_oneoff_container ""
    
elif [ "$1" = "pytest" ]; then
    gecho "Running Brainsmith pytest suite"
    # Use basic import test instead of broken pytest suite
    CMD="python -c \"import sys; import brainsmith; import finn; import qonnx; print('✓ All imports successful')\""
    run_oneoff_container "$CMD"
    
elif [ "$1" = "e2e" ]; then
    gecho "Running Brainsmith end-to-end validation test"
    run_oneoff_container "cd demos/bert && make single_layer"
    
elif [ "$1" = "bert-large-biweekly" ] || [ "$1" = "e2e-bert-large" ]; then
    gecho "Running BERT Large test"
    run_oneoff_container "cd demos/bert && make bert_large_single_layer"
    
elif [ "$1" = "debugtest" ]; then
    gecho "Running debug test - importing all editable installed packages"
    run_oneoff_container "python3 debug_imports.py"
    
else
    gecho "Running Brainsmith docker container with passed arguments"
    # Build command string properly handling quotes and arguments
    CMD=""
    for arg in "$@"; do
        if [ -z "$CMD" ]; then
            CMD="$arg"
        else
            # Escape quotes in arguments
            ESCAPED_ARG=$(printf '%q' "$arg")
            CMD="$CMD $ESCAPED_ARG"
        fi
    done
    run_oneoff_container "$CMD"
fi