# CI Workflow Refactoring - Simplified Plan

## Goal
Eliminate code duplication in `.github/workflows/ci.yml` with minimal complexity.

## Approach: One Script, One Action

### Step 1: Create a Single Reusable Script (Day 1)

Create `.github/scripts/ci-common.sh`:
```bash
#!/bin/bash
set -euo pipefail

# Common CI operations
case "$1" in
    "check-disk")
        # Disk space check (20GB default)
        REQUIRED="${2:-20}"
        AVAILABLE=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
        echo "Disk space: ${AVAILABLE}GB available (need ${REQUIRED}GB)"
        [ "$AVAILABLE" -lt "$REQUIRED" ] && exit 1
        ;;
        
    "ghcr-pull")
        # Pull and tag image from GHCR
        echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin
        docker pull "$GHCR_IMAGE"
        docker tag "$GHCR_IMAGE" "$BSMITH_DOCKER_TAG"
        docker images | grep "microsoft/brainsmith"
        ;;
        
    "smithy-test")
        # Run a test with smithy
        TEST_NAME="$2"
        TEST_CMD="$3"
        TIMEOUT="${4:-60}"
        
        chmod +x smithy
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
        
    *)
        echo "Usage: $0 {check-disk|ghcr-pull|smithy-test}"
        exit 1
        ;;
esac
```

### Step 2: Create One Composite Action (Day 1)

Create `.github/actions/setup-and-test/action.yml`:
```yaml
name: 'Setup and Test'
description: 'Common setup for all test jobs'
inputs:
  checkout:
    default: 'true'
  check-disk:
    default: 'true'
  pull-image:
    default: 'true'

runs:
  using: 'composite'
  steps:
    - name: Checkout
      if: inputs.checkout == 'true'
      uses: actions/checkout@v4
      with:
        submodules: recursive
        fetch-depth: 0
        
    - name: Setup
      shell: bash
      run: |
        chmod +x .github/scripts/ci-common.sh
        
    - name: Check disk
      if: inputs.check-disk == 'true'
      shell: bash
      run: .github/scripts/ci-common.sh check-disk
        
    - name: Pull image
      if: inputs.pull-image == 'true'
      shell: bash
      env:
        GITHUB_TOKEN: ${{ github.token }}
        GITHUB_ACTOR: ${{ github.actor }}
      run: .github/scripts/ci-common.sh ghcr-pull
```

### Step 3: Refactor Jobs (Day 2)

#### Before (e2e-test job): ~90 lines
#### After: ~15 lines

```yaml
e2e-test:
  if: github.event_name == 'schedule' || github.event_name == 'pull_request'
  runs-on: pre-release
  timeout-minutes: 120
  needs: [validate-environment, docker-build-and-test]
  
  steps:
    - uses: ./.github/actions/setup-and-test
    
    - name: Download digest
      uses: actions/download-artifact@v4
      with:
        name: image-digest
        path: /tmp/
    
    - name: Run E2E test
      run: |
        .github/scripts/ci-common.sh smithy-test \
          "E2E BERT" \
          "cd demos/bert && make clean && make" \
          30
```

### Step 4: Apply Pattern to All Jobs (Day 2-3)

1. `docker-build-and-test` - Special case, keep mostly as-is (builds image)
2. `e2e-test` - Use pattern above
3. `full-test-suite` - Same pattern, different test command
4. `bert-large-biweekly` - Same pattern, different test command

## Result

### Before
- 580+ lines of YAML
- Massive duplication
- Hard to maintain

### After  
- ~250 lines of YAML
- One script file (50 lines)
- One action file (30 lines)
- Easy to add new tests

### To Add a New Test
```yaml
my-new-test:
  runs-on: pre-release
  timeout-minutes: 60
  needs: [validate-environment, docker-build-and-test]
  steps:
    - uses: ./.github/actions/setup-and-test
    - run: .github/scripts/ci-common.sh smithy-test "My Test" "make test" 60
```

## Implementation Steps

1. **Day 1**: 
   - Create `ci-common.sh`
   - Create `setup-and-test` action
   - Test with one job

2. **Day 2-3**:
   - Refactor remaining jobs
   - Test in feature branch

3. **Day 4**:
   - Merge to main branch
   - Archive old workflow

## That's it. No phases, no unit tests for CI scripts, no complex shell function libraries. Just practical code reuse.