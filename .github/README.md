# Brainsmith CI System

## Overview

Automated testing for Brainsmith using persistent Docker containers and modular actions.

## Current Workflows

### PR Validation (`pr-validation.yml`)
- **Triggers**: Pull requests, pushes to develop
- **Runtime**: ~4 hours
- **Tests**: BERT single layer end-to-end
- **Disk**: 20GB required

### Biweekly Tests (`biweekly-tests.yml`)  
- **Triggers**: Monday/Thursday 00:00 UTC
- **Runtime**: ~24 hours
- **Tests**: BERT large model
- **Disk**: 40GB required

## Actions Available

### Core Actions
- `workflow-setup` - Standard checkout, disk check, cleanup
- `smithy-build` - Build and verify Docker image  
- `smithy-exec` - Execute commands in container
- `run-test` - Execute test with artifact collection

### Utility Actions
- `check-disk` - Validate disk space
- `docker-cleanup` - Clean container resources
- `collect-artifacts` - Collect debugging artifacts
- `pytest-fpgadataflow` - Unit tests (disabled - tests broken)

## Adding New Workflows

Standard pattern:

```yaml
jobs:
  my-test:
    runs-on: pre-release
    timeout-minutes: 120
    steps:
      - name: Setup
        uses: ./.github/actions/workflow-setup
        with:
          disk-threshold-gb: 30

      - name: Build
        uses: ./.github/actions/smithy-build

      - name: Test
        uses: ./.github/actions/run-test
        with:
          test-command: "cd demos/bert && make my_test"
          timeout-minutes: 60
          artifact-name: my-test-artifacts
          artifact-retention: 7
```

## Environment Variables

Required in workflow environment:
```yaml
env:
  DOCKER_BUILDKIT: 1
  BSMITH_DOCKER_PREBUILT: "0"
  BSMITH_DOCKER_NO_CACHE: "1"
  BSMITH_SKIP_DEP_REPOS: "0"
  BSMITH_XILINX_VERSION: ${{ vars.BSMITH_XILINX_VERSION }}
  BSMITH_XILINX_PATH: ${{ vars.BSMITH_XILINX_PATH }}
  NUM_DEFAULT_WORKERS: ${{ vars.NUM_DEFAULT_WORKERS }}
  BSMITH_DOCKER_TAG: "microsoft/brainsmith:unique-tag"
  BSMITH_DOCKER_FLAGS: "-e XILINXD_LICENSE_FILE=${{ secrets.XILINXD_LICENSE_FILE }}"
```

## Debugging Failed Tests

1. **Check logs**: Actions tab → Failed workflow → Expand failed step
2. **Download artifacts**: Available for 7-14 days after failure
3. **Local testing**: Use `./smithy daemon && ./smithy exec "command"`
4. **Container issues**: Check `./smithy status` and `./smithy logs`

## Common Issues

- **Disk space**: Ensure runner has required space (20-40GB)
- **Container reuse**: Build artifacts cleaned between runs automatically
- **Timeout**: BERT tests can take hours, timeouts are generous
- **Dependencies**: Will be fetched automatically if missing

## Test Commands

Available make targets in `demos/bert/`:
- `single_layer` - BERT single layer (used in PR validation)  
- `bert_large_single_layer` - BERT large model (used in biweekly)

## Container Management

The CI uses persistent containers via the `smithy` script:
- `./smithy daemon` - Start background container
- `./smithy exec "command"` - Run command in container
- `./smithy cleanup` - Remove container and artifacts
- `./smithy status` - Check container state
