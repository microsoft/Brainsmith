# CI Workflows

## Architecture

The CI system uses **composite actions** for step-level composition and **reusable workflows** for job-level orchestration.

| File | Role |
|------|------|
| [`ci.yml`](ci.yml) | Main workflow orchestrator |
| [`run-smithy-test.yml`](run-smithy-test.yml) | Test execution (uses composite action) |
| [`build-and-push.yml`](build-and-push.yml) | Docker build and push (uses composite action) |
| [`../actions/setup-and-test/action.yml`](../actions/setup-and-test/action.yml) | Environment setup composite action |
| [`../scripts/ci-common.sh`](../scripts/ci-common.sh) | Shell operations |

### Architecture Flow
```
ci.yml (Main Orchestrator)
├── run-smithy-test.yml (3 test jobs)
│   └── setup-and-test action (step-level composition)
└── build-and-push.yml (1 build job)
    └── setup-and-test action (step-level composition)
```

## Adding a New Test

Use the reusable workflow pattern:

```yaml
my-new-test:
  uses: ./.github/workflows/run-smithy-test.yml
  needs: [validate-environment, docker-build-and-test]
  with:
    test-name: "My Test"
    test-command: "make test"
    timeout-minutes: 60
  secrets:
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

## Available Reusable Workflows

### run-smithy-test.yml
Complete test execution with setup, image pull, test run, and artifact collection.

**Inputs:**
- `test-name` (required) - Name for the test
- `test-command` (required) - Command to execute
- `timeout-minutes` (default: 60) - Test timeout
- `collect-artifacts` (default: true) - Whether to collect artifacts
- `runner` (default: pre-release) - Runner to use

**Secrets:**
- `github-token` (required) - For GHCR access
- `docker-flags` (optional) - Additional Docker flags

### build-and-push.yml
Docker image build, test, and push to GHCR.

**Inputs:**
- `runner` (default: pre-release) - Runner to use
- `test-image` (default: true) - Whether to test built image

**Secrets:**
- `github-token` (required) - For GHCR access

## Composite Action

### setup-and-test
Environment setup composite action used by all workflows.

**Inputs:**
- `checkout` (default: true) - Whether to checkout repository
- `check-disk` (default: true) - Whether to check disk space
- `disk-threshold` (default: 20) - Disk space threshold in GB
- `pull-image` (default: true) - Whether to pull image from GHCR
- `docker-cleanup` (default: false) - Whether to clean Docker resources first

## Available Operations

| Operation | Description | Usage |
|-----------|-------------|-------|
| `smithy-test` | Run test with container lifecycle | `smithy-test "Test Name" "command" timeout_mins` |
| `check-disk` | Validate available disk space | `check-disk [threshold_gb]` |
| `ghcr-pull` | Pull and verify image from registry | `ghcr-pull` |
| `docker-cleanup` | Clean Docker resources | `docker-cleanup` |
| `collect-artifacts` | Gather debug info | `collect-artifacts [dir]` |
| `build-verify` | Build and verify Docker image | `build-verify` |
| `push-ghcr` | Push to registry with digest | `push-ghcr` |

## Example Test Commands

- `"make test"`
- `"cd demos/bert && make clean && make"`
- `"cd tests && pytest -v ./"`
- `"python --version && pytest tests/"`

## Benefits

- **5-line test jobs** instead of 40+ lines
- **Native GitHub Actions** composability
- **Single source of truth** for common patterns
- **Easy maintenance** - update one workflow, affects all users
- **Clear interfaces** via inputs and outputs

*See [`../docs/ci-reusable-workflows-transition-plan.md`](../docs/ci-reusable-workflows-transition-plan.md) for implementation details*
