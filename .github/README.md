# Brainsmith CI/CD System

## Architecture

```
.github/
├── actions/          # 8 secure composite actions
│   ├── check-disk/
│   ├── collect-artifacts/
│   ├── docker-cleanup/
│   ├── docker-login/
│   ├── docker-pull/
│   ├── docker-push/
│   ├── smithy-build/
│   └── smithy-test/
└── workflows/        # 5 reusable workflows
    ├── pr-tests.yml           # Fast PR/push validation
    ├── scheduled-tests.yml    # Comprehensive biweekly testing
    ├── build-and-push.yml     # Docker build & push
    ├── run-smithy-test.yml    # Test execution
```

## Workflows

### Main CI Pipeline (`pr-tests.yml`)
Fast validation for pull requests and develop branch pushes.

**Triggers**: Push to `develop`, Pull Requests
**Runtime**: Approximately 2.5 hours
**Jobs**:
- `validate-environment` - System validation and setup
- `docker-build-and-test` - Build Docker image and push to GHCR
- `e2e-test` - Essential BERT end-to-end tests
- `cleanup` - Resource cleanup

### Scheduled Testing (`scheduled-tests.yml`)
Comprehensive testing for complete system validation.

**Triggers**: Biweekly schedule (Monday/Thursday 00:00 UTC)
**Runtime**: Approximately 28 hours
**Jobs**:
- `validate-environment` - System validation and setup
- `docker-build-and-test` - Build Docker image and push to GHCR
- `full-test-suite` - Complete unit tests (240 minutes)
- `bert-large-biweekly` - Large model tests (1440 minutes)
- `cleanup` - Resource cleanup

**Design**: Separates essential development validation from resource-intensive comprehensive testing.

### Reusable Workflows

These workflows provide standardized operations that are called by the main CI pipelines.

#### `build-and-push.yml`
Standardized Docker build and publish pipeline used by both main CI workflows.

**Purpose**: Eliminates code duplication and ensures consistent build processes across different execution contexts.

**Process**:
- Environment validation and resource cleanup
- Docker image build via composite actions
- Optional container functionality testing
- GHCR authentication and image push
- Artifact preservation for debugging

**Usage:**
```yaml
build-job:
  uses: ./.github/workflows/build-and-push.yml
  with:
    runner: 'pre-release'      # Optional: specify runner
    test-image: true           # Optional: enable container testing
  secrets:
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

#### `run-smithy-test.yml`
Secure test execution workflow with predefined test types to prevent command injection.

**Purpose**: Provides type-safe test execution with comprehensive artifact collection and error handling.

**Process**:
- Repository checkout and environment setup
- Docker image authentication and pull
- Predefined test command execution with timeout protection
- Artifact collection and upload
- Resource cleanup

**Usage:**
```yaml
test-job:
  uses: ./.github/workflows/run-smithy-test.yml
  with:
    test-name: "My Test"
    test-type: "e2e-bert"      # Predefined types only
    test-variant: "default"    # default | large | clean
    timeout-minutes: 60
    collect-artifacts: true    # Optional: artifact collection
    runner: 'pre-release'      # Optional: specify runner
  secrets:
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

**Available Test Types:**
- `e2e-bert` - BERT end-to-end tests (variants: default, large, clean)
- `unit-tests` - Unit test suite
- `integration-tests` - Integration test suite
- `python-tests` - Python environment validation

## Composite Actions

### Infrastructure Actions
- `check-disk` - Validates available disk space
- `docker-cleanup` - Cleans Docker resources
- `collect-artifacts` - Collects CI artifacts with path validation

### Docker Actions
- `docker-login` - GHCR authentication
- `docker-pull` - Pull images with digest verification
- `docker-push` - Push images with digest generation
- `smithy-build` - Build Docker images

### Test Actions
- `smithy-test` - Execute predefined test commands

## Adding New Tests

Use the workflow pattern with predefined test types:

```yaml
my-new-test:
  uses: ./.github/workflows/run-smithy-test.yml
  needs: [validate-environment, docker-build-and-test]
  with:
    test-name: "Custom Test"
    test-type: "unit-tests"
    test-variant: "default"
    timeout-minutes: 30
  secrets:
    github-token: ${{ secrets.GITHUB_TOKEN }}
```
