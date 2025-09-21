#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Fast entrypoint for Brainsmith development environment
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
