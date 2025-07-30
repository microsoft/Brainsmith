# Brainsmith CI/CD System

## Architecture

```
.github/
├── actions/          # 8 modular composite actions
│   ├── build-docker/        # Docker build with verification
│   ├── check-disk/          # Disk space validation
│   ├── collect-artifacts/   # Safe artifact collection
│   ├── docker-cleanup/      # Container & build cleanup
│   ├── run-test-with-artifacts/  # Complete test lifecycle
│   ├── smithy-exec/         # Command execution with daemon
│   └── workflow-setup/      # Standard initialization
└── workflows/        # 2 focused workflows
    ├── pr-validation.yml     # BERT Quicktest
    └── biweekly-tests.yml    # BERT Large Model Test
```

## Workflows

### PR Validation (`pr-validation.yml`)
Fast validation for pull requests and develop branch pushes.

**Triggers**: Push to `develop`, Pull Requests  
**Runtime**: ~5 hours (4 hours test + 1 hour setup/cleanup)  
**Job**: `bert-quicktest` (BERT Quicktest)

**Steps**:
1. Checkout repository
2. Setup workflow (disk check, cleanup, build)
3. Run E2E test with artifact collection using `./demos/bert/scripts/quicktest.sh`

### Biweekly Tests (`biweekly-tests.yml`)
Comprehensive testing for large model validation.

**Triggers**: Biweekly schedule (Monday/Thursday 00:00 UTC)  
**Runtime**: ~24 hours  
**Job**: `bert-large-comprehensive-test` (BERT Large Model Comprehensive Test)

**Steps**:
1. Checkout repository
2. Setup workflow (disk check, cleanup, build)
3. Run BERT Large test with artifact collection

## Action Architecture

### Layer 2: Composite Actions (Orchestration)

#### `workflow-setup`
Standard initialization for all workflows.
```yaml
- uses: ./.github/actions/workflow-setup
  with:
    disk-threshold-gb: 20  # or 40 for biweekly
```
**Process**: Check disk → Clean Docker → Build image

#### `run-test-with-artifacts`
Complete test lifecycle with conditional artifact collection.
```yaml
- uses: ./.github/actions/run-test-with-artifacts
  with:
    command: "cd demos/bert && make single_layer"
    timeout-minutes: 240
    artifact-name: "test-results"
    collect-on: "failure"  # or "always"
    retention-days: 7
```
**Process**: Execute test → Collect artifacts → Upload → Cleanup

### Layer 3: Core Actions (Specific Tasks)

#### Infrastructure Actions
- `check-disk` - Validates available disk space with configurable thresholds
- `docker-cleanup` - Cleans containers AND persistent build directories
- `collect-artifacts` - Collects system info, container logs, and test artifacts

#### Docker Actions  
- `build-docker` - Builds image with verification and timing fixes
- `smithy-exec` - Executes commands with daemon lifecycle management

## Adding New Workflows

The modular architecture makes adding new workflows trivial:

```yaml
name: New Test Type

on:
  workflow_dispatch:

jobs:
  my-test:
    runs-on: pre-release
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0

      - name: Setup workflow
        uses: ./.github/actions/workflow-setup
        with:
          disk-threshold-gb: 30

      - name: Run my test
        uses: ./.github/actions/run-test-with-artifacts
        with:
          command: "my test command"
          timeout-minutes: 60
          artifact-name: "my-test-results"
          collect-on: "failure"
          retention-days: 7
```
