#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

# Fast exec entrypoint for Brainsmith development environment
# This script is optimized for quick command execution in persistent containers

# Enhanced logging for debugging
log_debug() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: $1" >&2
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_debug "entrypoint_exec.sh starting with args: $*"
log_debug "Current working directory: $(pwd)"
log_debug "Environment check - BSMITH_DIR: $BSMITH_DIR"

cd $BSMITH_DIR

# Wait for dependencies to be ready (with timeout)
# This handles the case where exec is called before daemon setup is complete
READINESS_MARKER="/tmp/.brainsmith_deps_ready"
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ]; then
    log_debug "Waiting for dependency readiness marker..."
    wait_count=0
    max_wait=30  # 60 seconds total
    
    while [ $wait_count -lt $max_wait ]; do
        if [ -f "$READINESS_MARKER" ]; then
            log_debug "Dependencies are ready"
            break
        fi
        log_debug "Waiting for dependencies... ($wait_count/$max_wait)"
        sleep 2
        wait_count=$((wait_count + 1))
    done
    
    if [ $wait_count -ge $max_wait ]; then
        log_error "Timeout waiting for dependencies to be ready"
        log_error "Expected marker file: $READINESS_MARKER"
        log_error "This suggests the daemon container is still initializing"
        exit 1
    fi
fi

# Load environment setup (dependencies should already be fetched by main container)
log_debug "Loading environment setup"
source /usr/local/bin/setup_env.sh

# Execute the provided command
if [ $# -gt 0 ]; then
    log_debug "Executing command: $*"
    exec "$@"
else
    log_debug "No command provided, starting bash"
    exec bash
fi
