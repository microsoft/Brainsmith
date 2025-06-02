#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT

# Main entrypoint for Brainsmith development environment
# Handles full setup including dependency fetching and installation

cd $BSMITH_DIR

# Load environment setup
source /usr/local/bin/setup_env.sh

# Smart package management with persistent state
CACHE_FILE="$BSMITH_DIR/deps/.brainsmith_packages_installed"
LOCK_FILE="$BSMITH_DIR/deps/.brainsmith_install_lock"

# Fetch dependencies if they don't exist (first time setup)
if [ "$BSMITH_SKIP_DEP_REPOS" = "0" ] && [ ! -d "$BSMITH_DIR/deps/finn" ]; then
    gecho "Fetching dependencies to $BSMITH_DIR/deps/..."
    source docker/fetch-repos.sh
fi

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
    # Prevent concurrent installations
    if [ -f "$LOCK_FILE" ]; then
        gecho "Another installation is in progress, waiting..."
        local timeout=300  # 5 minutes max wait
        local elapsed=0
        
        while [ -f "$LOCK_FILE" ] && [ $elapsed -lt $timeout ]; do
            sleep 2
            elapsed=$((elapsed + 2))
        done
        
        if [ -f "$LOCK_FILE" ]; then
            recho "Installation appears stuck. Removing lock and proceeding..."
            rm -f "$LOCK_FILE"
        else
            packages_already_installed && return 0
        fi
    fi
    
    # Create lock file
    touch "$LOCK_FILE"
    
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
        rm -f "$LOCK_FILE"
        exit 1
    fi
    
    # Remove lock file
    rm -f "$LOCK_FILE"
    
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

# execute the provided command(s)
if [ $# -gt 0 ]; then
    exec bash -c "$*"
else
    exec bash
fi
