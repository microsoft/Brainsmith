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


cd $BSMITH_DIR

# Ensure Python output is unbuffered for real-time output
export PYTHONUNBUFFERED=1

# Quick check for dependency readiness
# The new log monitoring system should ensure container is ready before exec is called
READINESS_MARKER="/tmp/.brainsmith_deps_ready"
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ]; then
    # First check if marker exists (fast path)
    if [ ! -f "$READINESS_MARKER" ]; then
        log_error "Container dependencies not ready. The daemon container may still be initializing."
        log_error "Expected marker file: $READINESS_MARKER"
        log_error "Check container logs with: docker logs <container-name>"
        log_error "Or wait for initialization to complete and try again."
        log_info "Debug: BSMITH_SKIP_DEP_REPOS=$BSMITH_SKIP_DEP_REPOS, READINESS_MARKER=$READINESS_MARKER"
        exit 1
    fi
    
    # Since container is ready, dependencies should exist - check for finnxsi directory
    if [ ! -d "${BSMITH_DIR}/deps/finn/finn_xsi" ]; then
        log_error "finnxsi directory not found at ${BSMITH_DIR}/deps/finn/finn_xsi"
        log_error "This suggests dependencies were not fetched properly in daemon mode"
        exit 1
    fi
fi

# Load environment setup (dependencies should already be fetched by main container)
# Set quiet mode for exec to suppress environment messages
export BSMITH_EXEC_QUIET=1
source /usr/local/bin/setup_env.sh

# Execute the provided command
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi
