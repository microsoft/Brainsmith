# Brainsmith CI Setup

This document describes the Continuous Integration (CI) setup for the Brainsmith project.

## Overview

The CI system uses GitHub Actions to run automated tests on:
- Pull requests to the `develop` branch
- Direct commits to the `develop` branch
- Weekly scheduled runs (Sunday at 00:00 UTC)

## Workflow Types

### 1. PR/Commit Builds
- Runs on every PR and commit to develop branch
- 30-minute timeout
- Basic Docker build and launch verification
- Uses dummy Xilinx paths (no license required)

### 2. Weekly Full Tests
- Runs every Sunday at 00:00 UTC
- 2-hour timeout
- Increased resources (2 CPUs, 8GB RAM)
- Runs complete pytest suite
- Requires Xilinx license and tools

## Required Secrets

The following secrets need to be configured in GitHub repository settings:

1. `BSMITH_XILINX_PATH`
   - Path to Xilinx tools installation
   - Example: `/opt/Xilinx`

2. `BSMITH_XILINX_VERSION`
   - Version of Xilinx tools
   - Example: `2024.2`

3. `XILINXD_LICENSE_FILE`
   - Path to Xilinx license file
   - Required for weekly tests only

## Resource Configuration

### PR Builds
- Default GitHub runner resources
- 30-minute maximum runtime

### Weekly Tests
- 2 CPU cores
- 8GB RAM total
- 2GB shared memory
- Increased file handle limits
- 2-hour maximum runtime

## Artifacts

The following artifacts are preserved from test runs:

1. Test Results
   - Location: `brainsmith/tests/`
   - Retention: 7 days
   - Uploaded for all runs

2. Error Logs (on failure)
   - Location: 
     - `brainsmith/tests/**/*.log`
     - `$BSMITH_HOST_BUILD_DIR/**/*.log`
   - Retention: 7 days
   - Uploaded only on test failures

## Environment Variables

### Common Variables
```bash
DOCKER_BUILDKIT=1
BSMITH_ROOT=${github.workspace}
BSMITH_HOST_BUILD_DIR=${github.workspace}/build
BSMITH_DOCKER_PREBUILT="0"
BSMITH_DOCKER_NO_CACHE="1"
BSMITH_SKIP_DEP_REPOS="0"
```

### Weekly Test Additional Variables
```bash
BSMITH_DOCKER_EXTRA="-v /opt/Xilinx:/opt/Xilinx -e XILINXD_LICENSE_FILE=<license_path> --memory=6g --memory-swap=8g --shm-size=2g"
```

## Future Enhancements

1. Docker layer caching for faster builds
2. More sophisticated error reporting and notifications
3. Automated issue creation for test failures
4. Team notifications via GitHub mentions