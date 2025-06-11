# CI Workflows

## Architecture

| File | Role |
|------|------|
| [`ci.yml`](ci.yml) | Main workflow |
| [`../scripts/ci-common.sh`](../scripts/ci-common.sh) | Reusable operations |
| [`../actions/setup-and-test/action.yml`](../actions/setup-and-test/action.yml) | Common setup |

## Standard Test Job Pattern

```yaml
my-test:
  runs-on: pre-release
  needs: [validate-environment, docker-build-and-test]
  steps:
    - uses: ./.github/actions/setup-and-test
    - uses: actions/download-artifact@v4
      with: { name: image-digest, path: /tmp/ }
    - run: .github/scripts/ci-common.sh smithy-test "Test Name" "command" timeout
```

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

## Adding a New Test

1. Copy the standard pattern above
2. Change the job name and test command
3. Adjust timeout as needed

Example commands:
- `"make test"`
- `"cd demos/bert && make clean && make"`
- `"cd tests && pytest -v ./"`

## Setup Options

```yaml
- uses: ./.github/actions/setup-and-test
  with:
    docker-cleanup: 'true'    # Clean before start
    check-disk: 'false'       # Skip disk check
    pull-image: 'false'       # Don't pull (for build jobs)
```
