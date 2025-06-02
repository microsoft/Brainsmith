#!/bin/bash
# CI-optimized entrypoint for Brainsmith development environment
# Optimized for GitHub Actions with better error handling and logging

cd $BSMITH_DIR

# Load environment setup
source /usr/local/bin/setup_env.sh

# CI-specific settings
CACHE_FILE="/tmp/.brainsmith_packages_installed"
LOCK_FILE="/tmp/.brainsmith_install_lock"
CI_LOG_FILE="/tmp/brainsmith_ci.log"

# Enhanced logging for CI
ci_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$CI_LOG_FILE"
}

ci_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$CI_LOG_FILE" >&2
}

ci_log "Starting CI entrypoint in $BSMITH_DIR"
ci_log "Build directory: $BSMITH_BUILD_DIR"
ci_log "Skip deps: $BSMITH_SKIP_DEP_REPOS"

# Function to check if packages are already installed
packages_already_installed() {
    if [ -f "$CACHE_FILE" ]; then
        ci_log "Checking if packages are already installed..."
        python -c "
try:
    import qonnx, finnexperimental, brevitas, finn, brainsmith
    print('All packages imported successfully')
    exit(0)
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
" 2>&1 | tee -a "$CI_LOG_FILE"
        return $?
    fi
    return 1
}

# Fetch dependencies with enhanced error handling
fetch_dependencies() {
    ci_log "Checking if dependencies need to be fetched..."
    
    if [ "$BSMITH_SKIP_DEP_REPOS" = "1" ]; then
        ci_log "Skipping dependency fetching (BSMITH_SKIP_DEP_REPOS=1)"
        return 0
    fi
    
    if [ -d "$BSMITH_DIR/deps/finn" ]; then
        ci_log "Dependencies already exist"
        return 0
    fi
    
    ci_log "Fetching dependencies to $BSMITH_DIR/deps/..."
    
    # Create deps directory
    mkdir -p "$BSMITH_DIR/deps"
    
    # Source fetch script with error handling
    if ! source docker/fetch-repos.sh; then
        ci_error "Failed to fetch dependencies"
        return 1
    fi
    
    ci_log "Dependencies fetched successfully"
    return 0
}

# Install packages with CI-optimized approach
install_packages_ci() {
    ci_log "Installing Python packages..."
    
    local failed_packages=""
    local install_success=true
    
    # Ensure deps directory exists
    mkdir -p "$BSMITH_DIR/deps"
    
    # qonnx (using workaround for pyproject.toml issue)
    if [ -d "${BSMITH_DIR}/deps/qonnx" ]; then
        ci_log "Installing qonnx..."
        mv ${BSMITH_DIR}/deps/qonnx/pyproject.toml ${BSMITH_DIR}/deps/qonnx/pyproject.tmp 2>/dev/null || true
        if ! pip install --user -e ${BSMITH_DIR}/deps/qonnx; then
            install_success=false
            failed_packages+="qonnx "
            ci_log "WARNING: qonnx installation failed"
        fi
        mv ${BSMITH_DIR}/deps/qonnx/pyproject.tmp ${BSMITH_DIR}/deps/qonnx/pyproject.toml 2>/dev/null || true
    else
        ci_log "WARNING: qonnx directory not found"
    fi

    # finn-experimental
    if [ -d "${BSMITH_DIR}/deps/finn-experimental" ]; then
        ci_log "Installing finn-experimental..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn-experimental; then
            install_success=false
            failed_packages+="finn-experimental "
            ci_log "WARNING: finn-experimental installation failed"
        fi
    else
        ci_log "WARNING: finn-experimental directory not found"
    fi

    # brevitas
    if [ -d "${BSMITH_DIR}/deps/brevitas" ]; then
        ci_log "Installing brevitas..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/brevitas; then
            install_success=false
            failed_packages+="brevitas "
            ci_log "WARNING: brevitas installation failed"
        fi
    else
        ci_log "WARNING: brevitas directory not found"
    fi

    # finn
    if [ -d "${BSMITH_DIR}/deps/finn" ]; then
        ci_log "Installing finn..."
        if ! pip install --user -e ${BSMITH_DIR}/deps/finn; then
            install_success=false
            failed_packages+="finn "
            ci_log "WARNING: finn installation failed"
        fi
    else
        ci_log "WARNING: finn directory not found"
    fi

    # brainsmith
    if [ -f "${BSMITH_DIR}/setup.py" ]; then
        ci_log "Installing brainsmith..."
        if ! pip install --user -e ${BSMITH_DIR}; then
            install_success=false
            failed_packages+="brainsmith "
            ci_error "brainsmith installation failed"
        fi
    else
        ci_error "Unable to find Brainsmith source code in ${BSMITH_DIR}"
        ci_error "Ensure you have passed -v <path-to-brainsmith-repo>:<path-to-brainsmith-repo> to the docker run command"
        exit 1
    fi
    
    if [ "$install_success" = true ]; then
        ci_log "All packages installed successfully!"
        touch "$CACHE_FILE"
        return 0
    else
        ci_log "WARNING: Some packages failed to install: $failed_packages"
        ci_log "Some functionality may not work properly."
        # Don't exit on package failures in CI - let the tests determine if it's critical
        return 0
    fi
}

# Main execution flow
if ! packages_already_installed; then
    ci_log "Packages not cached, proceeding with setup..."
    
    # Fetch dependencies first
    if ! fetch_dependencies; then
        ci_error "Dependency fetching failed"
        exit 1
    fi
    
    # Install packages
    install_packages_ci
else
    ci_log "Packages already installed and cached"
fi

# Verify critical imports
ci_log "Verifying package installations..."
python -c "
import sys
errors = []
try:
    import brainsmith
    print('[CI] brainsmith import: OK')
except ImportError as e:
    errors.append(f'brainsmith: {e}')

try:
    import finn
    print('[CI] finn import: OK') 
except ImportError as e:
    errors.append(f'finn: {e}')

try:
    import qonnx
    print('[CI] qonnx import: OK')
except ImportError as e:
    errors.append(f'qonnx: {e}')

if errors:
    print('[CI] Import errors (may be non-critical):')
    for error in errors:
        print(f'[CI]   {error}')
else:
    print('[CI] All critical packages imported successfully')
" 2>&1 | tee -a "$CI_LOG_FILE"

ci_log "CI entrypoint setup complete"

# Execute the provided command(s)
if [ $# -gt 0 ]; then
    ci_log "Executing command: $*"
    exec bash -c "$*"
else
    ci_log "No command provided, starting bash shell"
    exec bash
fi
