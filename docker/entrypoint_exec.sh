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

log_debug "entrypoint_exec.sh starting with args: $*"
log_debug "Current working directory: $(pwd)"
log_debug "Environment check - BSMITH_DIR: $BSMITH_DIR"

cd $BSMITH_DIR

# Since this is exec mode, the main container should already have everything set up
# We just need to load the environment and execute the command

# Load environment setup (dependencies should already be fetched by main container)
source /usr/local/bin/setup_env.sh

# Execute the provided command
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi
