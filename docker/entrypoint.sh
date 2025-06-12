#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

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

log_info "Starting BrainSmith entrypoint"
emit_status "INITIALIZING"

cd $BSMITH_DIR

# First: Fetch dependencies if they don't exist (before environment setup)
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ] && ([ ! -d "$BSMITH_DIR/deps/qonnx" ] || [ ! -d "$BSMITH_DIR/deps/finn" ]); then
    emit_status "FETCHING_DEPENDENCIES"
    log_info "Fetching dependencies to $BSMITH_DIR/deps/ (required before environment setup)"
    
    if source docker/fetch-repos.sh; then
        log_info "Dependencies fetched successfully"
    else
        emit_status "ERROR" "Failed to fetch dependencies"
        log_error "Failed to fetch dependencies"
        exit 1
    fi
    
    log_info "Dependencies ready at $(date)"
else
    log_info "Dependencies already exist, ready at $(date)"
fi

# Second: Load environment setup (now that dependencies exist)
source /usr/local/bin/setup_env.sh

# Check FINN dependency after environment is loaded (so recho function is available)
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ] && [ ! -f "$BSMITH_DIR/deps/finn/setup.py" ]; then
    recho "FINN dependency not found or not fetched"
    recho "Dependencies should be automatically fetched during container initialization"
    exit 1
fi

# Third: For daemon mode, ensure pyxsi is built during initialization
if [ "$BSMITH_CONTAINER_MODE" = "daemon" ] && [ ! -z "${XILINX_VIVADO}" ]; then
    if [ ! -f "${BSMITH_DIR}/deps/pyxsi/pyxsi.so" ] && [ -d "${BSMITH_DIR}/deps/pyxsi" ]; then
        emit_status "BUILDING_PYXSI"
        log_info "Building pyxsi during daemon initialization"
        OLDPWD=$(pwd)
        cd ${BSMITH_DIR}/deps/pyxsi || {
            emit_status "ERROR" "Failed to enter pyxsi directory"
            log_error "Failed to enter pyxsi directory"
            exit 1
        }
        if make; then
            log_info "pyxsi built successfully"
        else
            emit_status "ERROR" "Failed to build pyxsi during daemon initialization"
            log_error "Failed to build pyxsi during daemon initialization"
            exit 1
        fi
        cd $OLDPWD
    fi
fi

# Smart package management with persistent state
CACHE_FILE="/tmp/.brainsmith_packages_installed"

# Function to check if packages are already installed and working
packages_already_installed() {
    if [ -f "$CACHE_FILE" ]; then
        # Quick check if all key packages can be imported
        python -c "
try:
    import qonnx, finnexperimental, brevitas, finn, brainsmith
    exit(0)
except ImportError as e:
    exit(1)
" 2>/dev/null
        return $?
    fi
    return 1
}

# Function to install packages with proper error handling and progress
install_packages_with_progress() {
    log_info "Starting package installation process"
    emit_status "INSTALLING_PACKAGES" "starting"
    
    gecho "Installing development packages (this may take a moment)..."
    
    # Ensure deps directory exists
    mkdir -p "$BSMITH_DIR/deps"
    
    local install_success=true
    local failed_packages=""
    
    # qonnx (using workaround for https://github.com/pypa/pip/issues/7953)
    if [ -d "${BSMITH_DIR}/deps/qonnx" ]; then
        emit_status "INSTALLING_PACKAGES" "qonnx"
        gecho "Installing qonnx..."
        mv ${BSMITH_DIR}/deps/qonnx/pyproject.toml ${BSMITH_DIR}/deps/qonnx/pyproject.tmp 2>/dev/null || true
        if ! pip install --user -e ${BSMITH_DIR}/deps/qonnx; then
            install_success=false
            failed_packages+="qonnx "
        fi
        mv ${BSMITH_DIR}/deps/qonnx/pyproject.tmp ${BSMITH_DIR}/deps/qonnx/pyproject.toml 2>/dev/null || true
    fi

    # finn-experimental
    if [ -d "${BSMITH_DIR}/deps/finn-experimental" ]; then
        emit_status "INSTALLING_PACKAGES" "finn-experimental"
        gecho "Installing finn-experimental..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn-experimental; then
            install_success=false
            failed_packages+="finn-experimental "
        fi
    fi

    # brevitas
    if [ -d "${BSMITH_DIR}/deps/brevitas" ]; then
        emit_status "INSTALLING_PACKAGES" "brevitas"
        gecho "Installing brevitas..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/brevitas; then
            install_success=false
            failed_packages+="brevitas "
        fi
    fi

    # finn
    if [ -d "${BSMITH_DIR}/deps/finn" ]; then
        emit_status "INSTALLING_PACKAGES" "finn"
        gecho "Installing finn..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn; then
            install_success=false
            failed_packages+="finn "
        fi
    fi

    # brainsmith
    if [ -f "${BSMITH_DIR}/setup.py" ]; then
        emit_status "INSTALLING_PACKAGES" "brainsmith"
        gecho "Installing brainsmith..."
        if ! pip install --user -e ${BSMITH_DIR}; then
            install_success=false
            failed_packages+="brainsmith "
        fi
    else
        emit_status "ERROR" "Unable to find Brainsmith source code"
        recho "Unable to find Brainsmith source code in ${BSMITH_DIR}"
        recho "Ensure you have passed -v <path-to-brainsmith-repo>:<path-to-brainsmith-repo> to the docker run command"
        exit 1
    fi
    
    if [ "$install_success" = true ]; then
        # Mark packages as successfully installed
        touch "$CACHE_FILE"
        gecho "Development packages installed and cached successfully!"
        return 0
    else
        emit_status "ERROR" "Package installation failed: $failed_packages"
        recho "Failed to install packages: $failed_packages"
        recho "Some functionality may not work properly."
        return 1
    fi
}

# Install packages only if not already cached and working
if ! packages_already_installed; then
    install_packages_with_progress
else
    gecho "Development packages already installed - using cached setup"
fi

# For daemon mode, complete ALL setup before going into background
# For direct command execution, only install packages if needed for that command
if [ "$BSMITH_CONTAINER_MODE" = "daemon" ]; then
    log_info "Daemon mode: ensuring all packages are installed before going into background"
    # Force package installation/verification in daemon mode
    if ! packages_already_installed; then
        install_packages_with_progress
    else
        gecho "Development packages already installed - using cached setup"
    fi
    
    # Create readiness marker ONLY after everything is truly ready
    log_info "Creating dependency readiness marker"
    touch /tmp/.brainsmith_deps_ready
    
    # Emit final ready status for log monitoring
    emit_status "READY"
    log_info "All setup complete - container is now fully ready for exec commands"
    # Industry standard: use tail -f /dev/null to keep container alive
    exec tail -f /dev/null
fi

# execute the provided command(s)
if [ $# -gt 0 ] && [ "$1" != "" ]; then
    # For direct commands, install packages only if needed
    if ! packages_already_installed; then
        install_packages_with_progress
    fi
    exec bash -c "$*"
else
    # For interactive mode, install packages
    if ! packages_already_installed; then
        install_packages_with_progress
    fi
    exec bash
fi
