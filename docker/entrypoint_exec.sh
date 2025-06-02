#!/bin/bash
# Fast execution entrypoint for Brainsmith
# Optimized for quick command execution with smart package management

cd $BSMITH_DIR

# Setup environment variables (this is fast and necessary)
source /usr/local/bin/setup_env.sh

# Smart package management with persistent state
CACHE_FILE="/tmp/.brainsmith_packages_installed"
LOCK_FILE="/tmp/.brainsmith_install_lock"

# Function to wait for parallel installation to complete
wait_for_installation() {
    local timeout=300  # 5 minutes max wait
    local elapsed=0
    
    while [ -f "$LOCK_FILE" ] && [ $elapsed -lt $timeout ]; do
        echo "Waiting for package installation to complete..."
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    # Check if installation completed successfully
    if [ -f "$LOCK_FILE" ]; then
        echo "Warning: Installation appears to be stuck. Removing lock file."
        rm -f "$LOCK_FILE"
        return 1
    fi
    
    return 0
}

# Function to install packages (fallback only)
install_packages() {
    # Prevent concurrent installations
    if [ -f "$LOCK_FILE" ]; then
        wait_for_installation
        [ -f "$CACHE_FILE" ] && return 0
    fi
    
    # Create lock file
    touch "$LOCK_FILE"
    
    echo "Installing development packages (exec fallback)..."
    
    # Ensure deps directory exists
    mkdir -p "$BSMITH_DIR/deps"
    
    # Install packages with error handling
    local install_success=true
    
    # qonnx (using workaround for pyproject.toml issue)
    if [ -d "${BSMITH_DIR}/deps/qonnx" ]; then
        echo "Installing qonnx..."
        mv ${BSMITH_DIR}/deps/qonnx/pyproject.toml ${BSMITH_DIR}/deps/qonnx/pyproject.tmp 2>/dev/null || true
        if ! pip install --user -e ${BSMITH_DIR}/deps/qonnx >/dev/null 2>&1; then
            install_success=false
        fi
        mv ${BSMITH_DIR}/deps/qonnx/pyproject.tmp ${BSMITH_DIR}/deps/qonnx/pyproject.toml 2>/dev/null || true
    fi
    
    # finn-experimental
    if [ -d "${BSMITH_DIR}/deps/finn-experimental" ] && [ "$install_success" = true ]; then
        echo "Installing finn-experimental..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn-experimental >/dev/null 2>&1; then
            install_success=false
        fi
    fi
    
    # brevitas
    if [ -d "${BSMITH_DIR}/deps/brevitas" ] && [ "$install_success" = true ]; then
        echo "Installing brevitas..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/brevitas >/dev/null 2>&1; then
            install_success=false
        fi
    fi
    
    # finn
    if [ -d "${BSMITH_DIR}/deps/finn" ] && [ "$install_success" = true ]; then
        echo "Installing finn..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn >/dev/null 2>&1; then
            install_success=false
        fi
    fi
    
    # brainsmith
    if [ -f "${BSMITH_DIR}/setup.py" ] && [ "$install_success" = true ]; then
        echo "Installing brainsmith..."
        if ! pip install --user -e ${BSMITH_DIR} >/dev/null 2>&1; then
            install_success=false
        fi
    fi
    
    # Remove lock file
    rm -f "$LOCK_FILE"
    
    if [ "$install_success" = true ]; then
        # Mark packages as successfully installed
        touch "$CACHE_FILE"
        echo "Packages installed successfully!"
        return 0
    else
        echo "Warning: Some packages failed to install"
        return 1
    fi
}

# Main logic: Check packages and install if necessary
if [ ! -f "$CACHE_FILE" ]; then
    # Only show message if this is not a very quick command
    if [[ "$*" != *"echo"* ]] && [[ "$*" != *"ls"* ]] && [[ "$*" != *"pwd"* ]]; then
        echo "Setting up Python packages..."
    fi
    
    # Try to install packages
    if ! install_packages; then
        echo "Warning: Package installation had issues, but continuing..."
    fi
fi

# Execute the command
exec "$@"
