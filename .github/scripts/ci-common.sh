#!/bin/bash
set -euo pipefail

# Common CI operations
case "$1" in
    "check-disk")
        # Disk space check (20GB default)
        REQUIRED="${2:-20}"
        if ! AVAILABLE=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//'); then
            echo "ERROR: Failed to check disk space"
            exit 1
        fi
        echo "Disk space: ${AVAILABLE}GB available (need ${REQUIRED}GB)"
        if [ "$AVAILABLE" -lt "$REQUIRED" ]; then
            echo "ERROR: Insufficient disk space"
            exit 1
        fi
        ;;
        
    "ghcr-pull")
        # Pull and tag image from GHCR with digest verification
        echo "=== Pulling pre-built image from GitHub Container Registry ==="
        echo "GHCR image: $GHCR_IMAGE"
        echo "Local tag: $BSMITH_DOCKER_TAG"
        
        # Secure login with error suppression
        echo "Attempting GHCR login..."
        if ! echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin 2>/dev/null; then
            echo "ERROR: Failed to authenticate with GitHub Container Registry"
            echo "Please check GITHUB_TOKEN permissions and network connectivity"
            exit 1
        fi
        echo "✓ GHCR login successful"
        
        docker pull "$GHCR_IMAGE"
        
        # Verify digest if file exists
        if [ -f /tmp/image-digest.txt ]; then
            EXPECTED_DIGEST=$(cat /tmp/image-digest.txt)
            ACTUAL_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$GHCR_IMAGE" | grep -o 'sha256:[a-f0-9]*')
            
            if [ "$EXPECTED_DIGEST" != "$ACTUAL_DIGEST" ]; then
                echo "ERROR: Image digest mismatch!"
                echo "Expected: $EXPECTED_DIGEST"
                echo "Actual: $ACTUAL_DIGEST"
                exit 1
            fi
            echo "✓ Image digest verified: $ACTUAL_DIGEST"
        else
            echo "⚠️  No digest file found, skipping verification"
        fi
        
        docker tag "$GHCR_IMAGE" "$BSMITH_DOCKER_TAG"
        echo "=== Image ready for use ==="
        docker images | grep "microsoft/brainsmith"
        ;;
        
    "smithy-test")
        # Validate inputs
        if [ $# -lt 3 ]; then
            echo "ERROR: Insufficient arguments for smithy-test"
            exit 1
        fi
        
        TEST_NAME="$2"
        TEST_CMD="$3"
        TIMEOUT="${4:-60}"
        
        # Validate test name (alphanumeric, spaces, hyphens only)
        if ! [[ "$TEST_NAME" =~ ^[a-zA-Z0-9\ \-]+$ ]]; then
            echo "ERROR: Invalid test name format"
            exit 1
        fi
        
        # Validate timeout is a positive number
        if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || [ "$TIMEOUT" -le 0 ]; then
            echo "ERROR: Invalid timeout value"
            exit 1
        fi
        
        # Validate command doesn't contain dangerous patterns
        if [[ "$TEST_CMD" =~ [;&|`$\(\)] ]]; then
            echo "ERROR: Test command contains potentially unsafe characters"
            echo "Blocked characters: ; & | \` \$ ( )"
            exit 1
        fi
        
        chmod +x smithy
        # Environment variables are inherited, including BSMITH_DOCKER_FLAGS
        ./smithy daemon
        sleep 5
        
        # Check if timeout command is available
        if command -v timeout >/dev/null 2>&1; then
            if timeout "${TIMEOUT}m" ./smithy exec "$TEST_CMD"; then
                echo "✓ $TEST_NAME passed"
            else
                echo "✗ $TEST_NAME failed"
                ./smithy logs --tail 50
                exit 1
            fi
        else
            echo "WARNING: timeout command not available, running without timeout"
            if ./smithy exec "$TEST_CMD"; then
                echo "✓ $TEST_NAME passed"
            else
                echo "✗ $TEST_NAME failed"
                ./smithy logs --tail 50
                exit 1
            fi
        fi
        
        ./smithy stop || true
        ;;
        
    "docker-cleanup")
        # Clean Docker resources
        echo "=== Docker cleanup ==="
        docker container prune -f || true
        docker image prune -f || true
        docker volume prune -f || true
        echo "Available space after cleanup: $(df -h / | tail -1 | awk '{print $4}')"
        ;;

    "collect-artifacts")
        # Collect standard CI artifacts
        ARTIFACT_DIR="${2:-artifacts}"
        
        # Validate artifact directory path
        if [[ "$ARTIFACT_DIR" =~ \.\./|^/ ]]; then
            echo "ERROR: Invalid artifact directory path (path traversal detected)"
            exit 1
        fi
        
        if [[ ! "$ARTIFACT_DIR" =~ ^[a-zA-Z0-9_\-]+$ ]]; then
            echo "ERROR: Artifact directory contains invalid characters"
            exit 1
        fi
        
        mkdir -p "$ARTIFACT_DIR"
        
        echo "=== Collecting system info ==="
        df -h > "$ARTIFACT_DIR/disk_usage.txt" 2>/dev/null || true
        free -h > "$ARTIFACT_DIR/memory_usage.txt" 2>/dev/null || true
        
        echo "=== Collecting container info ==="
        if [ -x ./smithy ]; then
            ./smithy status > "$ARTIFACT_DIR/container_status.txt" 2>&1 || echo "Status failed" > "$ARTIFACT_DIR/container_status.txt"
            ./smithy logs > "$ARTIFACT_DIR/container.log" 2>&1 || echo "No logs" > "$ARTIFACT_DIR/container.log"
        fi
        ;;

    "build-verify")
        # Build and verify Docker image
        chmod +x smithy
        echo "=== Building Docker image ==="
        ./smithy build
        
        echo "=== Verifying image was built ==="
        docker images | grep "microsoft/brainsmith" || {
            echo "ERROR: Docker image not found after build"
            exit 1
        }
        ;;

    "push-ghcr")
        # Push to GHCR and save digest
        echo "=== Pushing to GHCR ==="
        docker tag "$BSMITH_DOCKER_TAG" "$GHCR_IMAGE"
        docker push "$GHCR_IMAGE"
        
        DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$GHCR_IMAGE" | grep -o 'sha256:[a-f0-9]*')
        echo "Image digest: $DIGEST"
        
        mkdir -p /tmp
        echo "$DIGEST" > /tmp/image-digest.txt
        ;;
        
    *)
        echo "Usage: $0 {check-disk|ghcr-pull|smithy-test|docker-cleanup|collect-artifacts|build-verify|push-ghcr}"
        exit 1
        ;;
esac