# Action Runner Dependency Analysis - Same Runner vs Different Runners

## Executive Summary

**No, we do NOT need to worry about composite actions executing on different runners.** Composite actions are fundamentally different from reusable workflows in how they execute. This analysis clarifies the execution model and why the proposed alternative is safe.

## 🔍 EXECUTION MODEL COMPARISON

### Reusable Workflows (Current Problematic Approach)
```yaml
# Main workflow calls reusable workflow
jobs:
  setup:
    uses: ./.github/workflows/setup-test-env.yml  # Runs on Runner A
  test:
    needs: setup
    uses: ./.github/workflows/run-test.yml        # Runs on Runner B
```

**Problem**: Each reusable workflow call creates a **separate job** that runs on a **potentially different runner**. State (files, Docker images, environment) is NOT shared between jobs.

### Composite Actions (Proposed Solution)
```yaml
# Single job using multiple composite actions
jobs:
  test:
    runs-on: pre-release                          # Single runner
    steps:
      - uses: ./.github/actions/check-disk       # Step 1 on same runner
      - uses: ./.github/actions/docker-pull      # Step 2 on same runner  
      - uses: ./.github/actions/smithy-test      # Step 3 on same runner
      - uses: ./.github/actions/collect-artifacts # Step 4 on same runner
```

**Solution**: All composite actions execute as **steps within the same job** on the **same runner**. All state is preserved between actions.

## 📋 KEY DIFFERENCES

### Reusable Workflows
- **Execution**: Each `uses` creates a new job
- **Runner**: Each job can run on different runners
- **State Sharing**: No shared state between jobs
- **Artifacts**: Must explicitly upload/download between jobs
- **Environment**: Environment variables don't carry over
- **File System**: Each job starts with clean workspace

### Composite Actions  
- **Execution**: Each `uses` creates steps within same job
- **Runner**: All steps run on same runner
- **State Sharing**: Full state sharing (files, env vars, Docker)
- **Artifacts**: All files remain on same filesystem
- **Environment**: Environment variables persist
- **File System**: All files persist between actions

## 🔧 PRACTICAL EXAMPLE

### Current Problematic Flow (Reusable Workflows)
```yaml
# .github/workflows/ci.yml
jobs:
  build:
    uses: ./.github/workflows/build-and-push.yml  # Job 1, Runner A
  test:
    needs: build
    uses: ./.github/workflows/run-smithy-test.yml # Job 2, Runner B
```

**Issues**:
- Docker image built on Runner A is NOT available on Runner B
- Files created during build are NOT available for testing
- Environment setup on Runner A doesn't apply to Runner B

### Proposed Solution (Composite Actions)
```yaml
# .github/workflows/ci.yml
jobs:
  build-and-test:
    runs-on: pre-release                    # Single runner for entire job
    steps:
      - uses: ./.github/actions/checkout
      - uses: ./.github/actions/docker-build
      - uses: ./.github/actions/smithy-test  # Can access built image
      - uses: ./.github/actions/collect-artifacts
```

**Benefits**:
- Docker image built in step 2 is available in step 3
- All files persist throughout the job
- Environment setup carries through all steps

## 🛡️ STATE PRESERVATION EXAMPLES

### File System State
```yaml
jobs:
  test:
    runs-on: pre-release
    steps:
      - uses: ./.github/actions/checkout        # Creates files
      - uses: ./.github/actions/docker-build    # Uses checked out files
      - uses: ./.github/actions/smithy-test     # Uses built artifacts
      # All files from previous steps are available
```

### Docker State
```yaml
jobs:
  test:
    runs-on: pre-release
    steps:
      - uses: ./.github/actions/docker-build    # Builds image "my-app:latest"
      - uses: ./.github/actions/smithy-test     # Can use "my-app:latest" image
      # Docker images persist between composite actions
```

### Environment Variables
```yaml
jobs:
  test:
    runs-on: pre-release
    env:
      GHCR_IMAGE: "ghcr.io/my-org/my-app:tag"
    steps:
      - uses: ./.github/actions/docker-pull     # Can access $GHCR_IMAGE
      - uses: ./.github/actions/smithy-test     # Can access $GHCR_IMAGE
      # Environment variables available to all composite actions
```

## 📊 COMPARISON TABLE

| Aspect | Reusable Workflows | Composite Actions |
|--------|-------------------|-------------------|
| **Execution Unit** | Job (separate runners) | Step (same runner) |
| **State Sharing** | ❌ No | ✅ Yes |
| **File Persistence** | ❌ No | ✅ Yes |
| **Docker Images** | ❌ Not shared | ✅ Shared |
| **Environment Vars** | ❌ Job-scoped | ✅ Shared |
| **Performance** | ❌ Slower (multiple runners) | ✅ Faster (single runner) |
| **Complexity** | ❌ Higher (artifact management) | ✅ Lower (direct sharing) |

## ⚠️ WHEN TO WORRY ABOUT RUNNERS

### Scenarios Where Runner Isolation Matters
1. **Job-to-Job Dependencies**: When using `needs:` between different jobs
2. **Reusable Workflow Calls**: When using `uses:` at the job level
3. **Matrix Builds**: When running parallel jobs with different configurations
4. **Self-hosted Runners**: When runners have different capabilities

### Scenarios Where Runner Isolation Doesn't Matter (Our Case)
1. **Composite Actions**: All execute within same job
2. **Step-level Operations**: All steps run sequentially on same runner
3. **Single Job Workflows**: No cross-job dependencies

## 🎯 MIGRATION SAFETY

### Safe Migration Pattern
```yaml
# BEFORE (Problematic - multiple jobs/runners)
jobs:
  setup:
    uses: ./.github/workflows/setup-test-env.yml
  test:
    needs: setup
    uses: ./.github/workflows/run-test.yml

# AFTER (Safe - single job/runner)  
jobs:
  test:
    runs-on: pre-release
    steps:
      - uses: ./.github/actions/setup-test-env
      - uses: ./.github/actions/run-test
```

### State Preservation Verification
```yaml
# Example verification that state persists
jobs:
  test:
    runs-on: pre-release
    steps:
      - uses: ./.github/actions/docker-build
      - name: Verify image exists
        run: docker images | grep my-app  # ✅ Will find image
      
      - uses: ./.github/actions/smithy-test
      - name: Verify test artifacts
        run: ls artifacts/                # ✅ Will find artifacts
```

## 📝 CONCLUSION

### Key Takeaway
**Composite actions execute as steps within the same job on the same runner.** There is **no runner isolation concern** when replacing `ci-common.sh` with individual composite actions.

### Why This Solves Our Problems
1. **Security**: Eliminates command injection while preserving functionality
2. **State Sharing**: All actions share same filesystem, Docker daemon, environment
3. **Performance**: Faster than current approach (no inter-job artifact transfers)
4. **Simplicity**: No complex state management between jobs

### Migration Confidence
The proposed migration from `ci-common.sh` to composite actions is **architecturally sound** and **operationally safe**. All the state sharing that currently works within the shell script will continue to work across composite actions because they execute in the same job context.

**Recommendation**: Proceed with confidence in migrating to individual composite actions. The runner isolation issues that plagued the reusable workflow approach do not apply to composite actions.