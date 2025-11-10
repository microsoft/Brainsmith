#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Simplified execution entrypoint for Brainsmith development
# Sources setup-shell.sh then executes commands

set -e

# Verify required environment variables
if [ -z "$BSMITH_DIR" ]; then
    echo "ERROR: BSMITH_DIR not set. This container must be started via ctl-docker.sh" >&2
    exit 1
fi

# Move to project directory
cd "$BSMITH_DIR"

# Source runtime environment
source /usr/local/bin/setup-shell.sh

# Execute command or start interactive shell
if [ $# -gt 0 ]; then
    exec "$@"
else
    # Interactive shell without welcome message
    exec bash
fi