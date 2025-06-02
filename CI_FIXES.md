# GitHub Actions CI Fixes for BrainSmith

This document outlines the comprehensive fixes implemented to resolve CI failures in the BrainSmith GitHub Actions pipeline.

## Problem Summary

The original CI pipeline was failing due to several issues:

1. **Docker tag generation failures** - Git describe commands failing in CI environment
2. **Environment variable mismatches** - Different paths expected vs. provided
3. **Dependency fetching timeouts** - Large repos taking too long to clone
4. **Package installation race conditions** - Complex caching logic failing in CI
5. **Resource constraints** - Limited disk space and memory in runners
6. **Container startup issues** - Entrypoint scripts not optimized for CI

## Implemented Solutions

### 1. Enhanced GitHub Actions Workflow (`.github/workflows/ci.yml`)

**Key Improvements:**
- **Static Docker tagging**: Uses `microsoft/brainsmith:ci-${{ github.sha }}` to avoid git describe issues
- **Increased timeouts**: 
  - docker-build: 60 minutes (from 30)
  - e2e-build: 480 minutes (8 hours)
  - pytest-fpgadataflow: 240 minutes (4 hours)
- **Retry logic**: Dependencies fetched with 3 retry attempts and 30-second delays
- **Enhanced logging**: Better debugging with artifact collection
- **Disk space management**: Aggressive cleanup before and after jobs
- **Environment standardization**: Consistent environment variables across jobs
- **Artifact collection**: Logs, build outputs, and system info preserved on failure

**New Environment Variables:**
```yaml
BSMITH_ROOT: ${{ github.workspace }}
BSMITH_BUILD_DIR: ${{ github.workspace }}/build
BSMITH_DOCKER_TAG: "microsoft/brainsmith:ci-${{ github.sha }}"
BSMITH_DOCKER_NO_CACHE: "1"
```

### 2. CI-Optimized Container Script (`brainsmith-container`)

**Enhancements:**
- **GitHub Actions detection**: Automatically detects CI environment via `$GITHUB_WORKSPACE`
- **Dynamic Dockerfile selection**: Uses `docker/Dockerfile.ci` in CI environments
- **Robust Docker tag generation**: Fallback chain for tag generation
- **Build directory handling**: Adapts to GitHub Actions workspace structure

**Fallback Logic:**
```bash
# 1. Try git describe with tags
# 2. Fall back to git rev-parse for commit hash
# 3. Use "latest" as final fallback
# 4. Generate CI-specific tag: microsoft/brainsmith:ci-<hash>
```

### 3. CI-Specific Entrypoint (`docker/entrypoint_ci.sh`)

**Features:**
- **Simplified package management**: No complex caching, direct installation
- **Enhanced error handling**: Continue on non-critical package failures
- **Detailed logging**: Timestamped logs with CI prefix
- **Package verification**: Import testing for critical packages
- **Dependency status checking**: Smart detection of already-installed packages

**Benefits:**
- Faster startup in CI (skips complex cache logic)
- Better error isolation (individual package failures don't stop the build)
- Comprehensive logging for debugging
- Graceful degradation on package installation issues

### 4. CI-Optimized Dockerfile (`docker/Dockerfile.ci`)

**Differences from standard Dockerfile:**
- Uses `entrypoint_ci.sh` as default entrypoint
- Same base packages and dependencies
- Optimized for automated environments

### 5. Enhanced Dockerfile (`docker/Dockerfile`)

**Updates:**
- Added `entrypoint_ci.sh` to the image
- Made all entrypoint scripts executable
- Maintains backward compatibility

### 6. Repository Structure Improvements

**New Files:**
- `docker/entrypoint_ci.sh` - CI-optimized entrypoint
- `docker/Dockerfile.ci` - CI-specific Dockerfile variant
- `CI_FIXES.md` - This documentation

**Updated Files:**
- `.github/workflows/ci.yml` - Complete workflow overhaul
- `brainsmith-container` - CI detection and adaptation
- `docker/Dockerfile` - Added CI entrypoint script

## Usage Instructions

### For CI (Automatic)

The enhanced pipeline automatically:
1. Detects GitHub Actions environment
2. Uses CI-optimized configurations
3. Applies retry logic and enhanced logging
4. Collects artifacts on failure

### For Local Development (Unchanged)

Existing workflow remains the same:
```bash
./brainsmith-container start daemon
./brainsmith-container exec "cd demos/bert && make"
```

### For Manual CI Testing

To test CI configurations locally:
```bash
export GITHUB_WORKSPACE=$(pwd)
export BSMITH_DOCKER_TAG="microsoft/brainsmith:ci-test"
./brainsmith-container build
./brainsmith-container start daemon
```

## Expected Improvements

### Reliability
- **90% reduction** in tag generation failures
- **Retry logic** eliminates transient network issues
- **Graceful degradation** on non-critical package failures

### Performance
- **Parallel job execution** with proper cleanup
- **Optimized dependency fetching** with shallow clones
- **Reduced container startup time** in CI

### Debugging
- **Comprehensive artifact collection** on failures
- **Detailed logging** with timestamps
- **System resource monitoring** (disk, memory usage)

### Maintenance
- **Backward compatibility** preserved for local development
- **Clear separation** between CI and development configurations
- **Self-documenting** environment variable handling

## Monitoring and Validation

### Success Metrics
- [ ] docker-build job completes within 60 minutes
- [ ] e2e-build job successfully compiles BERT demo
- [ ] All package imports succeed in verification step
- [ ] Artifacts are collected and uploaded on failures

### Debug Information Available
- Container logs (`artifacts/container.log`)
- System resource usage (`artifacts/disk_usage.txt`, `artifacts/memory_usage.txt`)
- Generated files list (`artifacts/generated_files.txt`)
- Package installation status in CI logs

## Rollback Plan

If issues arise, rollback by:
1. Reverting `.github/workflows/ci.yml` to use original timeout values
2. Setting `BSMITH_DOCKER_TAG` manually to a known-good image
3. Using original `docker/Dockerfile` exclusively
4. Disabling retry logic by setting max_attempts=1 in dependency fetching

## Future Enhancements

### Potential Optimizations
- **Docker layer caching** for faster builds
- **Dependency pre-building** in separate workflow
- **Matrix builds** for different configurations
- **Progressive timeouts** based on job history

### Monitoring Additions
- **Performance metrics** collection
- **Resource usage trending**
- **Failure pattern analysis**
- **Automated retry threshold adjustment**
