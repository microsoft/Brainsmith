# CI Workflow Implementation - Completion Plan

## Current Status Analysis

### ✅ Components Implemented
1. **`.github/scripts/ci-common.sh`** - Basic operations: check-disk, ghcr-pull, smithy-test
2. **`.github/actions/setup-and-test/action.yml`** - Composite action for common setup
3. **`.github/workflows/ci.yml`** - Partially refactored (301 lines, 48% reduction)

### ⚠️ Critical Issues Identified

#### 1. **Environment Variable Passing Bug** (BREAKING)
The `ghcr-pull` function expects `$GHCR_IMAGE` and `$BSMITH_DOCKER_TAG` but the composite action doesn't pass them.

#### 2. **BERT Large Environment Bug** (BREAKING)
The `BSMITH_DOCKER_FLAGS` in bert-large-biweekly job won't be passed to smithy daemon.

#### 3. **Missing Functionality**
- No Docker cleanup operation
- No artifact collection abstraction
- docker-build-and-test job barely uses new components

## Implementation Plan

### Phase 1: Fix Breaking Issues (Immediate)

#### 1.1 Fix Environment Variable Passing
```yaml
# Update .github/actions/setup-and-test/action.yml
inputs:
  checkout:
    default: 'true'
  check-disk:
    default: 'true'
  pull-image:
    default: 'true'
  ghcr-image:
    default: ''  # Will use env.GHCR_IMAGE if not provided
  docker-tag:
    default: ''  # Will use env.BSMITH_DOCKER_TAG if not provided

# In the pull image step:
env:
  GITHUB_TOKEN: ${{ github.token }}
  GITHUB_ACTOR: ${{ github.actor }}
  GHCR_IMAGE: ${{ inputs.ghcr-image || env.GHCR_IMAGE }}
  BSMITH_DOCKER_TAG: ${{ inputs.docker-tag || env.BSMITH_DOCKER_TAG }}
```

#### 1.2 Fix BERT Large Environment Issue
```bash
# Update ci-common.sh smithy-test function
"smithy-test")
    TEST_NAME="$2"
    TEST_CMD="$3"
    TIMEOUT="${4:-60}"
    
    chmod +x smithy
    # Environment variables are inherited, including BSMITH_DOCKER_FLAGS
    ./smithy daemon
    sleep 5
    
    if timeout "${TIMEOUT}m" ./smithy exec "$TEST_CMD"; then
        echo "✓ $TEST_NAME passed"
    else
        echo "✗ $TEST_NAME failed"
        ./smithy logs --tail 50
        exit 1
    fi
    
    ./smithy stop || true
    ;;
```

### Phase 2: Add Missing Operations

#### 2.1 Add to ci-common.sh
```bash
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
```

### Phase 3: Complete Workflow Refactoring

#### 3.1 Update docker-build-and-test job
```yaml
docker-build-and-test:
  steps:
    - uses: ./.github/actions/setup-and-test
      with:
        pull-image: 'false'
        
    - name: Build and verify image
      run: .github/scripts/ci-common.sh build-verify
      
    - name: Test container functionality  
      run: |
        .github/scripts/ci-common.sh smithy-test \
          "Basic Functionality" \
          "echo 'Container test: SUCCESS' && python --version" \
          5
          
    - name: Login to GHCR
      run: |
        echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
        
    - name: Push to GHCR
      run: .github/scripts/ci-common.sh push-ghcr
      
    - name: Upload digest
      uses: actions/upload-artifact@v4
      with:
        name: image-digest
        path: /tmp/image-digest.txt
        retention-days: 1
```

#### 3.2 Update e2e-test to use artifact collection
```yaml
- name: Collect artifacts
  if: always()
  run: .github/scripts/ci-common.sh collect-artifacts artifacts

- name: Upload artifacts
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: e2e-test-artifacts-${{ github.run_id }}
    path: artifacts/
    retention-days: 7
```

#### 3.3 Add docker cleanup to composite action
```yaml
# In setup-and-test/action.yml, add new input:
inputs:
  docker-cleanup:
    default: 'false'
    
# Add step:
- name: Docker cleanup
  if: inputs.docker-cleanup == 'true'
  shell: bash
  run: .github/scripts/ci-common.sh docker-cleanup
```

## Expected Results

### Metrics After Completion
- **Total lines**: ~220-240 (down from current 301)
- **ci-common.sh**: ~120 lines (up from 69)
- **Duplication eliminated**: 95%+ 
- **New test addition**: 5-10 lines

### Benefits
1. **All breaking issues fixed** - Environment variables flow correctly
2. **Complete abstraction** - All common operations centralized
3. **Consistent patterns** - Every job follows same structure
4. **Easy maintenance** - Single point of change for each operation

## Implementation Order

1. **Fix breaking issues first** (Phase 1) - Critical for CI to work
2. **Add missing operations** (Phase 2) - Enable full refactoring  
3. **Complete refactoring** (Phase 3) - Achieve target metrics

## Time Estimate
- Phase 1: 30 minutes
- Phase 2: 1 hour
- Phase 3: 1-2 hours
- Testing: 1 hour

Total: ~4 hours to completion