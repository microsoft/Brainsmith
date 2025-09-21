#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Main entrypoint for Brainsmith development environment with Poetry
# Handles full setup including dependency fetching and installation

# Enhanced logging for debugging
log_debug() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: $1" >&2
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Status emission for container synchronization
BRAINSMITH_STATUS_PREFIX="BRAINSMITH_STATUS:"
emit_status() {
    local status="$1"
    local detail="${2:-}"
    echo "${BRAINSMITH_STATUS_PREFIX}${status}${detail:+:$detail}"
    log_info "Status: $status${detail:+ - $detail}"
}

log_info "Starting Brainsmith entrypoint"
emit_status "INITIALIZING"

cd $BSMITH_DIR

# Install dependencies using Poetry
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ]; then
    emit_status "UPDATING_DEPENDENCIES"
    
    # First run poetry install to set up base dependencies
    log_info "Running poetry install to set up base dependencies..."
    cd "$BSMITH_DIR" && poetry install --no-interaction --no-ansi
    
    # Always run the developer setup
    if [ -x "$BSMITH_DIR/setup-dev.sh" ]; then
        log_info "Running setup-dev.sh to set up editable dependencies..."
        cd "$BSMITH_DIR" && ./setup-dev.sh
        
        # Get the virtualenv path that Poetry created
        VENV_PATH=$(cd "$BSMITH_DIR" && poetry env info --path)
        
        # Save venv info for later sourcing
        echo "export VIRTUAL_ENV=\"$VENV_PATH\"" > "$BSMITH_BUILD_DIR/.brainsmith_venv"
        echo "export PATH=\"$VENV_PATH/bin:\$PATH\"" >> "$BSMITH_BUILD_DIR/.brainsmith_venv"
        # Set FINN_ROOT to deps location
        echo "export FINN_ROOT=\"$BSMITH_DIR/deps/finn\"" >> "$BSMITH_BUILD_DIR/.brainsmith_venv"
        
        # Activate the venv now for the rest of this script
        source "$VENV_PATH/bin/activate"
    else
        log_error "setup-dev.sh not found or not executable"
        exit 1
    fi
    
    # Optional: Setup C++ simulation dependencies
    # Use the venv's python if available
    PYTHON_CMD="python"
    if [ -d "$VENV_DIR/bin" ]; then
        PYTHON_CMD="$VENV_DIR/bin/python"
    fi
    if $PYTHON_CMD -c "from brainsmith.core.plugins.simulation import SimulationSetup; s = SimulationSetup(); s.setup_cppsim()" 2>/dev/null; then
        log_info "C++ simulation dependencies ready"
    else
        log_info "C++ simulation setup skipped (optional)"
    fi
    
    log_info "Dependencies ready at $(date)"
    
    # For daemon mode, emit READY status after dependencies are installed
    if [ "$BSMITH_CONTAINER_MODE" = "daemon" ]; then
        # Create readiness marker after dependencies are installed
        log_info "Creating dependency readiness marker"
        touch /tmp/.brainsmith_deps_ready
        
        # Emit final ready status for log monitoring
        emit_status "READY"
        log_info "All setup complete - container is now fully ready for exec commands"
    fi
else
    log_info "Skipping dependency installation (BSMITH_SKIP_DEP_REPOS=1)"
fi

# Second: Load environment setup (now that dependencies exist)
source /usr/local/bin/setup_env.sh

# FINN is now installed via Poetry as a Git dependency - no need to check deps/ directory
# Poetry handles all Python dependencies including FINN, QONNX, Brevitas etc.

# Function to build finnxsi if needed
build_finnxsi_if_needed() {
    # First try to find finn installation via Poetry/pip
    FINN_LOCATION=$(python -c "import finn; import os; print(os.path.dirname(finn.__file__))" 2>/dev/null || echo "")
    
    if [ ! -z "${XILINX_VIVADO}" ] && [ ! -z "${FINN_LOCATION}" ] && [ -d "${FINN_LOCATION}/finn_xsi" ]; then
        # Check if xsi.so already exists in the finn package directory
        if [ ! -f "${FINN_LOCATION}/../xsi.so" ]; then
            emit_status "BUILDING_FINNXSI"
            log_info "Building finnxsi (Vivado available and finnxsi source exists)"
            OLDPWD=$(pwd)
            cd ${FINN_LOCATION}/finn_xsi || {
                emit_status "ERROR" "Failed to enter finnxsi directory"
                log_error "Failed to enter finnxsi directory at ${FINN_LOCATION}/finn_xsi"
                exit 1
            }
            if make; then
                log_info "finnxsi built successfully"
                # Copy xsi.so to the expected location
                cp xsi.so ${FINN_LOCATION}/../ || log_info "Could not copy xsi.so to package root"
            else
                emit_status "ERROR" "Failed to build finnxsi"
                log_error "Failed to build finnxsi"
                exit 1
            fi
            cd $OLDPWD
        else
            log_info "finnxsi already built - skipping"
        fi
    elif [ -z "${XILINX_VIVADO}" ]; then
        log_info "Skipping finnxsi build - Vivado not available"
    elif [ -z "${FINN_LOCATION}" ]; then
        log_info "Skipping finnxsi build - FINN package not found via Python"
    elif [ ! -d "${FINN_LOCATION}/finn_xsi" ]; then
        log_info "Skipping finnxsi build - finnxsi source not available"
    fi
}

# Third: Build finnxsi if needed (both daemon and one-shot mode)
build_finnxsi_if_needed

# Install brainsmith package
install_brainsmith_package() {
    # Already handled by the Poetry installation above
    log_info "Brainsmith package already installed"
    return 0
}

# For daemon mode, keep container alive
if [ "$BSMITH_CONTAINER_MODE" = "daemon" ]; then
    log_info "Daemon mode: container setup complete"
    # Common approach: use tail -f /dev/null to keep container alive
    exec tail -f /dev/null
fi

# execute the provided command(s)
if [ $# -gt 0 ] && [ "$1" != "" ]; then
    # For direct commands, install brainsmith package
    install_brainsmith_package
    exec bash -c "$*"
else
    # For interactive mode, install brainsmith package
    install_brainsmith_package
    gecho "Brainsmith development environment ready!"
    exec bash
fi