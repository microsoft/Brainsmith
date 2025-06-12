# CI Common Script Alternatives - Better Code Reuse Without Security Risks

## Executive Summary

The current `ci-common.sh` approach has inherent security risks due to its shell script nature and dynamic command execution. This analysis presents **safer alternatives** for achieving code reuse while eliminating security vulnerabilities.

## 🚨 CURRENT PROBLEMS WITH ci-common.sh

### Security Issues
1. **Command injection risks** in `smithy-test` function
2. **Dynamic shell execution** of user-provided commands
3. **Complex input validation** required for shell safety
4. **Environment variable dependencies** creating coupling
5. **Path traversal vulnerabilities** in file operations

### Architectural Issues
1. **Monolithic design** - all operations in one script
2. **Shell-specific logic** - not portable across platforms
3. **Hard to test** - complex shell functions
4. **Error handling complexity** - bash error handling is fragile
5. **Maintenance burden** - shell scripts become unwieldy

## 💡 ALTERNATIVE APPROACHES

### Option 1: Individual Composite Actions (RECOMMENDED)
**Concept**: Replace monolithic script with individual composite actions for each operation.

#### Benefits
- **Eliminate command injection** - no dynamic shell execution
- **Native GitHub Actions** integration and validation
- **Type-safe inputs** with GitHub Actions schema validation
- **Better error handling** with GitHub Actions built-ins
- **Easier testing** - each action can be tested independently

#### Implementation Structure
```
.github/actions/
├── check-disk/action.yml
├── docker-login/action.yml
├── docker-pull/action.yml
├── docker-push/action.yml
├── smithy-build/action.yml
├── smithy-test/action.yml
├── collect-artifacts/action.yml
└── docker-cleanup/action.yml
```

#### Example: Safe Test Execution
```yaml
# .github/actions/smithy-test/action.yml
name: 'Smithy Test'
description: 'Run test with smithy container'
inputs:
  test-name:
    description: 'Name of the test'
    type: string
    required: true
  make-target:
    description: 'Make target to execute'
    type: string
    required: true
    # Restricted to predefined safe values
  timeout-minutes:
    description: 'Test timeout in minutes'
    type: number
    default: 60

runs:
  using: 'composite'
  steps:
    - name: Validate inputs
      shell: bash
      run: |
        # Input validation with allowlist approach
        case "${{ inputs.make-target }}" in
          "test"|"clean"|"build"|"lint"|"check")
            echo "✓ Valid make target: ${{ inputs.make-target }}"
            ;;
          *)
            echo "ERROR: Invalid make target. Allowed: test, clean, build, lint, check"
            exit 1
            ;;
        esac

    - name: Run smithy test
      shell: bash
      run: |
        chmod +x smithy
        ./smithy daemon
        sleep 5
        
        # Safe execution - no user command injection
        cd ${{ inputs.test-directory || '.' }}
        make ${{ inputs.make-target }}
        
        ./smithy stop || true
```

### Option 2: Docker Container Actions
**Concept**: Package operations as Docker container actions for maximum isolation.

#### Benefits
- **Complete isolation** from runner environment
- **Reproducible execution** across different runners
- **Language flexibility** - can use Python, Go, etc. instead of bash
- **Built-in security** - container boundaries prevent many attacks
- **Easy testing** - can test containers locally

#### Example Structure
```yaml
# .github/actions/smithy-test/action.yml
name: 'Smithy Test'
description: 'Run test with smithy container'
runs:
  using: 'docker'
  image: './Dockerfile'
  args:
    - ${{ inputs.test-name }}
    - ${{ inputs.make-target }}
    - ${{ inputs.timeout-minutes }}
```

```dockerfile
# .github/actions/smithy-test/Dockerfile
FROM python:3.9-slim

RUN pip install pyyaml click

COPY entrypoint.py /entrypoint.py
COPY smithy_test.py /smithy_test.py

ENTRYPOINT ["python", "/entrypoint.py"]
```

```python
# .github/actions/smithy-test/entrypoint.py
#!/usr/bin/env python3
import sys
import click
import subprocess
from enum import Enum

class MakeTarget(Enum):
    TEST = "test"
    CLEAN = "clean" 
    BUILD = "build"
    LINT = "lint"
    CHECK = "check"

@click.command()
@click.argument('test_name')
@click.argument('make_target', type=click.Choice([t.value for t in MakeTarget]))
@click.argument('timeout_minutes', type=int)
def smithy_test(test_name: str, make_target: str, timeout_minutes: int):
    """Run smithy test with validated inputs."""
    
    # Input validation
    if not test_name.replace(' ', '').replace('-', '').isalnum():
        click.echo("ERROR: Invalid test name format", err=True)
        sys.exit(1)
    
    if timeout_minutes <= 0 or timeout_minutes > 1440:
        click.echo("ERROR: Invalid timeout value", err=True)
        sys.exit(1)
    
    # Safe execution
    try:
        subprocess.run(['chmod', '+x', 'smithy'], check=True)
        subprocess.run(['./smithy', 'daemon'], check=True)
        time.sleep(5)
        
        # Safe make execution - no shell injection possible
        result = subprocess.run(
            ['make', make_target], 
            timeout=timeout_minutes * 60,
            check=True
        )
        
        click.echo(f"✓ {test_name} passed")
        
    except subprocess.TimeoutExpired:
        click.echo(f"✗ {test_name} timed out after {timeout_minutes} minutes", err=True)
        sys.exit(1)
    except subprocess.CalledProcessError:
        click.echo(f"✗ {test_name} failed", err=True)
        subprocess.run(['./smithy', 'logs', '--tail', '50'], check=False)
        sys.exit(1)
    finally:
        subprocess.run(['./smithy', 'stop'], check=False)

if __name__ == '__main__':
    smithy_test()
```

### Option 3: Predefined Command Templates
**Concept**: Use predefined, parameterized command templates instead of arbitrary command execution.

#### Implementation
```yaml
# .github/actions/smithy-test/action.yml
inputs:
  test-type:
    description: 'Type of test to run'
    type: string
    required: true
    # Options: e2e-bert, unit-tests, integration-tests, performance-tests
  test-variant:
    description: 'Test variant'
    type: string
    default: 'default'
    # Options: default, large, small, clean

runs:
  using: 'composite'
  steps:
    - name: Execute predefined test
      shell: bash
      run: |
        case "${{ inputs.test-type }}" in
          "e2e-bert")
            case "${{ inputs.test-variant }}" in
              "default") COMMAND="cd demos/bert && make clean && make" ;;
              "large") COMMAND="cd demos/bert && make bert_large_single_layer" ;;
              *) echo "ERROR: Invalid bert variant"; exit 1 ;;
            esac
            ;;
          "unit-tests")
            COMMAND="cd tests && pytest -v ./"
            ;;
          "integration-tests")
            COMMAND="cd tests && pytest -v integration/"
            ;;
          *)
            echo "ERROR: Invalid test type"
            exit 1
            ;;
        esac
        
        # Execute predefined command - no injection possible
        chmod +x smithy
        ./smithy daemon
        sleep 5
        eval "$COMMAND"  # Safe because COMMAND is predefined
        ./smithy stop || true
```

### Option 4: Configuration-Driven Approach
**Concept**: Use YAML configuration files to define test operations instead of shell commands.

#### Test Configuration
```yaml
# .github/test-configs/bert-e2e.yml
name: "E2E BERT"
type: "e2e"
working_directory: "demos/bert"
timeout_minutes: 120
steps:
  - type: "make"
    target: "clean"
  - type: "make"
    target: "build"
  - type: "make" 
    target: "test"
```

```yaml
# .github/test-configs/unit-tests.yml
name: "Unit Tests"
type: "unit"
working_directory: "tests"
timeout_minutes: 30
steps:
  - type: "pytest"
    args: ["-v", "./"]
```

#### Action Implementation
```python
# .github/actions/smithy-test/config_runner.py
import yaml
import subprocess
from pathlib import Path

def run_test_config(config_file: str):
    """Run test based on YAML configuration."""
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Validation
    allowed_types = ['make', 'pytest', 'python']
    allowed_targets = ['test', 'clean', 'build', 'lint', 'check']
    
    working_dir = Path(config.get('working_directory', '.'))
    
    for step in config['steps']:
        if step['type'] not in allowed_types:
            raise ValueError(f"Invalid step type: {step['type']}")
        
        if step['type'] == 'make' and step['target'] not in allowed_targets:
            raise ValueError(f"Invalid make target: {step['target']}")
        
        # Safe execution based on step type
        if step['type'] == 'make':
            subprocess.run(['make', step['target']], cwd=working_dir, check=True)
        elif step['type'] == 'pytest':
            subprocess.run(['pytest'] + step.get('args', []), cwd=working_dir, check=True)
```

## 📊 COMPARISON ANALYSIS

| Approach | Security | Maintainability | Performance | Complexity |
|----------|----------|-----------------|-------------|------------|
| **Current Shell Script** | ❌ Low | ❌ Poor | ✅ Fast | ❌ High |
| **Individual Composite Actions** | ✅ High | ✅ Good | ✅ Fast | ✅ Low |
| **Docker Container Actions** | ✅ Excellent | ✅ Good | ⚠️ Slower | ⚠️ Medium |
| **Command Templates** | ✅ Good | ✅ Good | ✅ Fast | ✅ Low |
| **Configuration-Driven** | ✅ Excellent | ✅ Excellent | ✅ Fast | ⚠️ Medium |

## 🎯 RECOMMENDED MIGRATION STRATEGY

### Phase 1: Split Critical Operations
Convert the most security-sensitive operations first:
1. **smithy-test** → Individual composite action with predefined commands
2. **collect-artifacts** → Safe composite action with path validation
3. **ghcr-pull/push** → Docker operations composite actions

### Phase 2: Utility Operations
Convert utility operations:
1. **check-disk** → Simple composite action
2. **docker-cleanup** → Simple composite action
3. **build-verify** → Build-specific composite action

### Phase 3: Eliminate Shell Script
Remove `ci-common.sh` entirely once all operations are converted.

## 🛡️ SECURITY BENEFITS

### Elimination of Attack Vectors
- ✅ **No command injection** - predefined operations only
- ✅ **No path traversal** - actions validate inputs natively
- ✅ **No shell evaluation** - direct API calls instead
- ✅ **Type safety** - GitHub Actions schema validation
- ✅ **Input validation** - built into action definitions

### Enhanced Security Posture
- **Principle of least privilege** - each action does one thing
- **Input sanitization** - automatic with action schema
- **Audit trail** - each action execution is logged
- **Reproducible builds** - consistent execution environment

## 📝 CONCLUSION

**Recommendation**: Migrate from the monolithic `ci-common.sh` to **individual composite actions** (Option 1) for these reasons:

1. **Eliminates security vulnerabilities** - no dynamic shell execution
2. **Maintains performance** - fast execution like current approach
3. **Improves maintainability** - smaller, focused actions
4. **Leverages GitHub Actions** - native features and validation
5. **Easier migration** - can be done incrementally

The current shell script approach is fundamentally insecure due to its need to execute arbitrary commands. Moving to predefined, validated operations through composite actions provides the same code reuse benefits while eliminating the security attack surface entirely.