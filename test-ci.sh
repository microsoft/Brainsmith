#!/bin/bash
# Test script for debugging CI issues locally and in CI
# Usage: ./test-ci.sh [quick|full]

set -e  # Exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] INFO: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"
}

log_debug() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] DEBUG: $1${NC}"
}

# Detect if we're in CI
if [ -n "$GITHUB_WORKSPACE" ]; then
    log_info "Running in GitHub Actions CI environment"
    CI_MODE=true
else
    log_info "Running in local development environment"
    CI_MODE=false
fi

TEST_MODE=${1:-quick}
log_info "Test mode: $TEST_MODE"

# Set up environment
if [ "$CI_MODE" = true ]; then
    export BSMITH_DOCKER_TAG="microsoft/brainsmith:ci-test-$(date +%s)"
else
    export BSMITH_DOCKER_TAG="microsoft/brainsmith:local-test"
fi

log_info "Using Docker tag: $BSMITH_DOCKER_TAG"

# Test 1: Environment check
log_info "=== Test 1: Environment Check ==="
echo "Working directory: $(pwd)"
echo "BSMITH_DIR: ${BSMITH_DIR:-$(pwd)}"
echo "Git status:"
git status --porcelain || log_warn "Git status failed"
echo "Git describe:"
git describe --always --tags --dirty 2>/dev/null || log_warn "Git describe failed"
echo "Git rev-parse:"
git rev-parse --short HEAD 2>/dev/null || log_error "Git rev-parse failed"

# Test 2: Docker availability
log_info "=== Test 2: Docker Check ==="
docker --version || { log_error "Docker not available"; exit 1; }
docker info | head -10

# Test 3: Container script functionality
log_info "=== Test 3: Container Script Check ==="
chmod +x brainsmith-container
./brainsmith-container help

# Test 4: Docker build
log_info "=== Test 4: Docker Build ==="
log_info "Building Docker image..."
if ./brainsmith-container build; then
    log_info "Docker build successful"
else
    log_error "Docker build failed"
    exit 1
fi

# Test 5: Container startup
log_info "=== Test 5: Container Startup ==="
log_info "Starting container in daemon mode..."
if ./brainsmith-container start daemon; then
    log_info "Container started successfully"
else
    log_error "Container startup failed"
    exit 1
fi

# Test 6: Basic container functionality
log_info "=== Test 6: Basic Container Tests ==="
log_info "Testing container status..."
./brainsmith-container status

log_info "Testing basic commands..."
./brainsmith-container exec "echo 'Hello from container'"
./brainsmith-container exec "python --version"
./brainsmith-container exec "pwd"

# Test 7: Environment variables in container
log_info "=== Test 7: Container Environment ==="
./brainsmith-container exec "echo 'BSMITH_DIR='\$BSMITH_DIR"
./brainsmith-container exec "echo 'BSMITH_BUILD_DIR='\$BSMITH_BUILD_DIR"
./brainsmith-container exec "echo 'BSMITH_SKIP_DEP_REPOS='\$BSMITH_SKIP_DEP_REPOS"
./brainsmith-container exec "ls -la \$BSMITH_DIR | head -10"

# Test 8: Dependency directory check
log_info "=== Test 8: Dependency Check ==="
./brainsmith-container exec "ls -la \$BSMITH_DIR/deps/ 2>/dev/null || echo 'Dependencies not yet fetched'"

if [ "$TEST_MODE" = "full" ]; then
    # Test 9: Dependency fetching (full test only)
    log_info "=== Test 9: Dependency Fetching ==="
    log_info "Testing dependency fetch (this may take several minutes)..."
    if timeout 600 ./brainsmith-container exec "cd \$BSMITH_DIR && source docker/fetch-repos.sh"; then
        log_info "Dependency fetching completed"
        ./brainsmith-container exec "find \$BSMITH_DIR/deps -maxdepth 2 -type d | head -20"
    else
        log_warn "Dependency fetching failed or timed out (10 minutes)"
    fi

    # Test 10: Python package installation
    log_info "=== Test 10: Python Package Check ==="
    ./brainsmith-container exec "python -c 'import sys; print(\"Python path:\", sys.path[:3])'"
    ./brainsmith-container exec "pip list | grep -E '(qonnx|finn|brevitas|brainsmith)' || echo 'Packages not yet installed'"

    # Test 11: Simple BERT test
    log_info "=== Test 11: BERT Demo Check ==="
    ./brainsmith-container exec "ls -la demos/bert/ || echo 'BERT demo not found'"
    if ./brainsmith-container exec "cd demos/bert && ls -la Makefile"; then
        log_info "BERT demo found, testing basic make..."
        timeout 120 ./brainsmith-container exec "cd demos/bert && make clean || true" || log_warn "Make clean timed out"
    fi
fi

# Test 12: Container logs
log_info "=== Test 12: Container Logs ==="
log_info "Recent container logs:"
./brainsmith-container logs --tail 20

# Test 13: Resource usage
log_info "=== Test 13: Resource Usage ==="
./brainsmith-container exec "df -h | head -10"
./brainsmith-container exec "free -h"

# Cleanup
log_info "=== Cleanup ==="
./brainsmith-container stop
./brainsmith-container cleanup

if [ "$CI_MODE" = true ]; then
    log_info "Cleaning up Docker images..."
    docker rmi "$BSMITH_DOCKER_TAG" 2>/dev/null || log_warn "Failed to remove Docker image"
fi

log_info "=== Test Summary ==="
log_info "All basic tests completed successfully!"
if [ "$TEST_MODE" = "quick" ]; then
    log_info "Run './test-ci.sh full' for comprehensive testing including dependency fetching"
fi
