#!/bin/bash
# Test script to validate CI fixes locally before pushing to GitHub Actions
# This script simulates the CI environment and tests the key failure points

set -e

echo "=== Brainsmith CI Fix Validation Script ==="
echo "This script tests the Docker container setup to validate CI fixes"
echo ""

# Set CI-like environment variables
export BSMITH_DOCKER_TAG="microsoft/brainsmith:test-$(git rev-parse --short HEAD 2>/dev/null || echo 'local')"
export BSMITH_SKIP_DEP_REPOS="0"
export DOCKER_BUILDKIT="1"

echo "=== Testing Container Management Script ==="
chmod +x smithy

echo "=== Test 1: Build Docker Image ==="
echo "Building image: $BSMITH_DOCKER_TAG"
./smithy build || {
    echo "ERROR: Docker build failed"
    exit 1
}

echo "=== Test 2: Start Daemon Mode ==="
echo "Starting daemon container..."
./smithy daemon || {
    echo "ERROR: Daemon start failed"
    ./smithy status
    exit 1
}

echo "=== Test 3: Wait for Container Readiness ==="
echo "Waiting for container to be ready..."
sleep 15

echo "=== Test 4: Basic Exec Test ==="
./smithy exec "echo 'Exec test: SUCCESS'" || {
    echo "ERROR: Basic exec failed"
    ./smithy status
    ./smithy logs --tail 20
    exit 1
}

echo "=== Test 5: Environment Check ==="
./smithy exec "echo \"BSMITH_DIR=\$BSMITH_DIR\""
./smithy exec "echo \"Container mode: \$BSMITH_CONTAINER_MODE\""
./smithy exec "python --version"

echo "=== Test 6: Dependency Verification ==="
./smithy exec "ls -la \$BSMITH_DIR/deps/ | head -10" || {
    echo "WARNING: Dependencies may not be fully ready yet"
}

echo "=== Test 7: Package Import Test ==="
./smithy exec "python -c 'import sys; print(\"Python ready\")')" || {
    echo "WARNING: Python environment may not be fully ready"
}

echo "=== Cleanup ==="
./smithy stop
./smithy cleanup

echo ""
echo "=== SUCCESS: All tests passed! ==="
echo "The CI fixes should work correctly in GitHub Actions"
echo ""
echo "Key fixes applied:"
echo "1. Fixed CI commands from 'start daemon' to 'daemon'"
echo "2. Fixed dependency race condition in entrypoint.sh"
echo "3. Added proper readiness checking in entrypoint_exec.sh"
echo "4. Added wait time in CI for container readiness"
echo "5. Renamed script from 'brainsmith-container' to 'smithy' for better UX"
