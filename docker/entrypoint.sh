#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Simplified entrypoint for Brainsmith development
# Runs setup-venv.sh then sources setup-shell.sh

set -e

# Simple logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Verify required environment variables
if [ -z "$BSMITH_DIR" ]; then
    log "ERROR: BSMITH_DIR not set. This container must be started via ctl-docker.sh"
    exit 1
fi

# Move to project directory
cd "$BSMITH_DIR"

# Ensure home directory exists and is writable for current user
if [ ! -d "$HOME" ]; then
    mkdir -p "$HOME"
fi

# Run one-stop setup (unless explicitly skipped)
if [ "$BSMITH_SKIP_SETUP" != "1" ]; then
    log "Running setup..."

    # Build setup flags
    SETUP_FLAGS="--docker"  # Always use docker mode
    [ -n "$BSMITH_FORCE_SETUP" ] && SETUP_FLAGS="$SETUP_FLAGS --force"
    [ "$BSMITH_QUIET" == "1" ] && SETUP_FLAGS="$SETUP_FLAGS --quiet"

    # Parse skip components from environment
    if [ -n "$BSMITH_SKIP_COMPONENTS" ]; then
        SETUP_FLAGS="$SETUP_FLAGS --skip $BSMITH_SKIP_COMPONENTS"
    fi

    # Run setup
    if ./setup-venv.sh $SETUP_FLAGS; then
        log "✓ Setup completed successfully"
    else
        log "⚠️  Setup completed with warnings"
    fi
else
    log "Skipping setup (BSMITH_SKIP_SETUP=1)"
fi

# Source runtime environment
log "Activating runtime environment..."
source /usr/local/bin/setup-shell.sh

# Keep container alive in daemon mode
log "Container ready"
exec tail -f /dev/null
