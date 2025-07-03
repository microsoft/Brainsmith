# Phase 3 Container Usage Guide

## Environment Requirements

**Critical**: Phase 3 must be executed within the Brainsmith Docker container where all dependencies are properly configured.

## Container Setup

### Start Container
```bash
# Start persistent container in background
./smithy daemon

# Check container status
./smithy status
```

### Execute Phase 3 Commands
```bash
# Run Python scripts in container
./smithy exec "python your_script.py"

# Interactive shell for development
./smithy shell
```

## Usage Examples

### Basic Phase 3 Usage
```bash
# Test backend availability
./smithy exec "python -c 'from brainsmith.core.phase3 import LegacyFINNBackend; print(\"✅ Real FINN backend available\")'"

# Create build runner
./smithy exec "python -c '
from brainsmith.core.phase3 import create_build_runner_factory
factory = create_build_runner_factory(\"auto\")
runner = factory()
print(f\"Backend: {runner.get_backend_name()}\")
'"
```

### Integration with Phase 1-2
```bash
# Complete DSE workflow in container
./smithy exec "python -c '
from brainsmith.core.phase1 import forge
from brainsmith.core.phase2 import explore
from brainsmith.core.phase3 import create_build_runner_factory

# Phase 1: Create design space
design_space = forge(\"model.onnx\", \"blueprint.yaml\")

# Phase 2: Explore with real FINN backend
results = explore(
    design_space=design_space,
    build_runner_factory=create_build_runner_factory(\"legacy_finn\")
)

print(f\"Completed {results.success_count} successful builds\")
'"
```

## Why Container is Required

### Missing Dependencies in Host Environment
- `importlib_resources` package not installed on host
- FINN framework requires specific Python environment
- QONNX dependencies and custom operators
- Xilinx tools and environment variables

### Container Provides
- **Complete Python Environment**: All packages including `importlib_resources`
- **FINN Framework**: Properly installed and configured FINN builder
- **QONNX Integration**: Custom operators and transforms available  
- **Xilinx Tools**: Vivado, HLS, and synthesis tools mounted and configured
- **Build Environment**: Proper directory structure and permissions

## Troubleshooting

### Host Environment Errors
```bash
# ❌ This will fail - missing dependencies
python -c "from brainsmith.core.phase3 import LegacyFINNBackend"
# ModuleNotFoundError: No module named 'importlib_resources'

# ✅ This works - container has all dependencies  
./smithy exec "python -c 'from brainsmith.core.phase3 import LegacyFINNBackend'"
```

### Container Not Running
```bash
# Check container status
./smithy status

# Start if needed
./smithy daemon

# View logs if startup fails
./smithy logs
```

## Perfect Code Compliance

Using the container environment ensures:
- **Real implementations** instead of mocks
- **Complete functionality** with all dependencies
- **Production environment** matching deployment conditions
- **No compromise** on code quality due to environment issues