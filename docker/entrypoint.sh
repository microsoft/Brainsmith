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

log_info "Starting BrainSmith entrypoint"
log_debug "Environment: BSMITH_DIR=$BSMITH_DIR, BSMITH_BUILD_DIR=$BSMITH_BUILD_DIR"
log_debug "Skip deps: BSMITH_SKIP_DEP_REPOS=$BSMITH_SKIP_DEP_REPOS"
log_debug "Container mode: BSMITH_CONTAINER_MODE=$BSMITH_CONTAINER_MODE"
log_debug "Arguments: $# args: $*"
log_debug "Working directory before cd: $(pwd)"

cd $BSMITH_DIR
log_debug "Changed to directory: $(pwd)"

# First: Fetch dependencies if they don't exist (before environment setup)
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ] && [ ! -d "$BSMITH_DIR/deps/finn" ]; then
    log_info "Fetching dependencies to $BSMITH_DIR/deps/ (required before environment setup)"
    
    if source docker/fetch-repos.sh; then
        log_info "Dependencies fetched successfully"
    else
        log_error "Failed to fetch dependencies"
        exit 1
    fi
    
    log_info "Dependencies ready at $(date)"
    # Create readiness marker for container management
    touch /tmp/.brainsmith_deps_ready
else
    log_info "Dependencies already exist, ready at $(date)"
    # Create readiness marker for container management
    touch /tmp/.brainsmith_deps_ready
fi

# Second: Load environment setup (now that dependencies exist)
log_debug "Loading environment setup"
source /usr/local/bin/setup_env.sh

# Third: For daemon mode, ensure pyxsi is built during initialization
if [ "$BSMITH_CONTAINER_MODE" = "daemon" ] && [ ! -z "${XILINX_VIVADO}" ]; then
    if [ ! -f "${BSMITH_DIR}/deps/pyxsi/pyxsi.so" ] && [ -d "${BSMITH_DIR}/deps/pyxsi" ]; then
        log_info "Building pyxsi during daemon initialization"
        OLDPWD=$(pwd)
        cd ${BSMITH_DIR}/deps/pyxsi || {
            log_error "Failed to enter pyxsi directory"
            exit 1
        }
        if make; then
            log_info "pyxsi built successfully"
        else
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
    
    gecho "Installing development packages (this may take a moment)..."
    
    # Ensure deps directory exists
    mkdir -p "$BSMITH_DIR/deps"
    
    local install_success=true
    local failed_packages=""
    
    # qonnx (using workaround for https://github.com/pypa/pip/issues/7953)
    if [ -d "${BSMITH_DIR}/deps/qonnx" ]; then
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
        gecho "Installing finn-experimental..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn-experimental; then
            install_success=false
            failed_packages+="finn-experimental "
        fi
    fi

    # brevitas
    if [ -d "${BSMITH_DIR}/deps/brevitas" ]; then
        gecho "Installing brevitas..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/brevitas; then
            install_success=false
            failed_packages+="brevitas "
        fi
    fi

    # finn
    if [ -d "${BSMITH_DIR}/deps/finn" ]; then
        gecho "Installing finn..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn; then
            install_success=false
            failed_packages+="finn "
        fi
    fi

    # brainsmith
    if [ -f "${BSMITH_DIR}/setup.py" ]; then
        gecho "Installing brainsmith..."
        if ! pip install --user -e ${BSMITH_DIR}; then
            install_success=false
            failed_packages+="brainsmith "
        fi
    else
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
    log_info "All setup complete - container is now fully ready for exec commands"
    log_debug "Starting daemon mode with tail -f /dev/null"
    # Industry standard: use tail -f /dev/null to keep container alive
    exec tail -f /dev/null
fi

# execute the provided command(s)
log_debug "Command execution logic: args=$#, first_arg='$1'"
if [ $# -gt 0 ] && [ "$1" != "" ]; then
    log_debug "Taking command execution path: $*"
    # For direct commands, install packages only if needed
    if ! packages_already_installed; then
        install_packages_with_progress
    fi
    exec bash -c "$*"
else
    log_debug "No command provided and not daemon mode, starting bash"
    # For interactive mode, install packages
    if ! packages_already_installed; then
        install_packages_with_progress
    fi
    exec bash
fi

