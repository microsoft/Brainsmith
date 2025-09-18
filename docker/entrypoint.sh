#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Main entrypoint for Brainsmith development environment
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

# First: Install Python dependencies using Poetry
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ]; then
    emit_status "UPDATING_DEPENDENCIES"
    log_info "Installing Python dependencies using Poetry"
    
    # Ensure we have pyproject.toml
    if [ ! -f "$BSMITH_DIR/pyproject.toml" ]; then
        emit_status "ERROR" "pyproject.toml not found"
        log_error "pyproject.toml not found in $BSMITH_DIR"
        exit 1
    fi
    
    # Install Python dependencies with Poetry
    log_info "Running poetry install..."
    if poetry install; then
        log_info "Python dependencies installed successfully"
    else
        emit_status "ERROR" "Failed to install Python dependencies"
        log_error "Poetry install failed"
        exit 1
    fi
    
    # Setup non-Python dependencies for simulation (optional)
    log_info "Setting up optional simulation dependencies..."
    if python -c "from brainsmith.core.plugins.simulation import SimulationSetup; s = SimulationSetup(); s.setup_cppsim()"; then
        log_info "C++ simulation dependencies ready"
    else
        log_info "C++ simulation setup skipped (optional dependency)"
    fi
    
    log_info "Dependencies ready at $(date)"
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

# Install brainsmith in development mode using Poetry
install_brainsmith_package() {
    emit_status "INSTALLING_BRAINSMITH"
    log_info "Installing brainsmith in development mode"
    
    cd "$BSMITH_DIR"
    
    # Use Poetry to install only the root package (deps already installed)
    if poetry install --only-root; then
        log_info "Brainsmith installed successfully"
        return 0
    else
        emit_status "ERROR" "Failed to install brainsmith"
        log_error "Failed to install brainsmith package"
        return 1
    fi
}

# For daemon mode, complete ALL setup before going into background
if [ "$BSMITH_CONTAINER_MODE" = "daemon" ]; then
    log_info "Daemon mode: ensuring brainsmith is installed before going into background"
    
    # Install brainsmith package
    install_brainsmith_package
    
    # Create readiness marker ONLY after everything is truly ready
    log_info "Creating dependency readiness marker"
    touch /tmp/.brainsmith_deps_ready
    
    # Emit final ready status for log monitoring
    emit_status "READY"
    log_info "All setup complete - container is now fully ready for exec commands"
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
