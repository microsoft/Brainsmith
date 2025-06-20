# Additional Brainsmith Commands Not Documented in CLAUDE.md

## Smithy Container Management (Extended Commands)

### Container Lifecycle Management
```bash
# Build Docker image without cache
./smithy build          # Build image (honors BSMITH_DOCKER_NO_CACHE=1)

# Advanced container operations  
./smithy restart        # Stop and restart container
./smithy logs           # Show container logs
./smithy logs -f        # Follow container logs in real-time
./smithy cleanup        # Remove container completely

# Container status with details
./smithy status         # Shows container name, status, and ports
```

### Environment Variables for Container Configuration
```bash
# Docker build control
export BSMITH_DOCKER_NO_CACHE=1      # Force rebuild without cache
export BSMITH_DOCKER_PREBUILT=0      # Build image locally (default)
export BSMITH_DOCKER_BUILD_FLAGS=""  # Additional docker build flags
export BSMITH_DOCKER_FLAGS=""        # Additional docker run flags
export BSMITH_DOCKER_EXTRA=""        # Extra docker run options

# Performance tuning
export NUM_DEFAULT_WORKERS=8         # Number of parallel workers
export BSMITH_INIT_TIMEOUT=600       # Container init timeout (seconds)
export BSMITH_SHOW_INIT_LOGS=true    # Show detailed init logs

# GPU support
export NVIDIA_VISIBLE_DEVICES="0,1"  # Specific GPU devices
export BSMITH_DOCKER_GPU=1           # Enable GPU support

# Permissions
export BSMITH_DOCKER_RUN_AS_ROOT=1   # Run container as root user
```

## BERT Demo Commands (Extended)

### Makefile Targets for Different Configurations
```bash
cd demos/bert

# Pre-configured BERT model variants
make folding_three_layers         # 3-layer BERT with SIMD=24, PE=16
make max_folding_three_layers     # Maximum folding: SIMD=48, PE=32  
make small_folding_three_layers   # Small config: SIMD=12, PE=8
make bert_large_single_layer      # BERT Large with SIMD=16, PE=8

# Quick test without DCP generation
./quicktest.sh                    # Runs single layer test, skips DCP
```

### Parameter Sweep for Design Space Exploration
```bash
cd demos/bert/tests
./param_sweep.sh    # Sweeps across multiple configurations:
                    # - FPS: 1000
                    # - Heads: 12, 24
                    # - Hidden size: 384, 192
                    # - Bitwidth: 8, 4
                    # - Sequence length: 128, 64, 32
```

### Direct BERT Model Generation
```bash
# Root directory bert.py - Full ONNX to FINN pipeline
python bert.py <model_path> <output_path> [--options]

# Generate custom folding configurations
python demos/bert/gen_initial_folding.py \
    --simd 16 --pe 8 \
    --num_layers 2 \
    -t 2 \  # Number of threads
    -o ./configs/custom_config.json
```

## Legacy Compatibility Commands

### run-docker.sh Wrapper (Deprecated but functional)
```bash
# One-off container commands (uses smithy under the hood)
./run-docker.sh                    # Interactive shell
./run-docker.sh pytest             # Run basic import tests
./run-docker.sh e2e               # Run end-to-end validation
./run-docker.sh bert-large-biweekly  # BERT Large test
./run-docker.sh debugtest         # Debug imports
./run-docker.sh "custom command"  # Run any command
```

## Testing Commands

### Comprehensive Test Suite
```bash
# Run all Phase 3 DSE tests
python run_all_tests.py           # Runs unit + integration tests
                                  # Shows detailed results per module
                                  # Validates Phase 3 implementation

# Run specific test modules in container
./smithy exec "python -m pytest tests/dataflow/unit/test_dataflow_interface.py -v"
./smithy exec "python -m pytest tests/tools/hw_kernel_gen/ -v"
./smithy exec "python -m pytest tests/integration/ -v"
```

## CLI Tools

### BrainSmith Core CLI (if installed)
```bash
# Forge accelerator from model and blueprint
brainsmith forge <model.onnx> <blueprint.yaml> -o output_dir

# Validate blueprint
brainsmith validate <blueprint.yaml>

# Alias for forge
brainsmith run <model.onnx> <blueprint.yaml> -o output_dir
```

### Hardware Kernel Generator CLI
```bash
# Generate RTL wrapper templates from SystemVerilog
python -m brainsmith.libraries.analysis.tools.hw_kernel_gen.hkg \
    <rtl_file> <compiler_data> -o <output_dir>
```

## Blueprint V2 Commands

### Modern BERT Demo
```bash
cd demos/bert_new

# Test modern blueprint integration
python test_modern_bert.py

# End-to-end with modern architecture
python end2end_bert.py -o output_name [options]
```

## Development Utilities

### Disk Space Management
```bash
# Smithy automatically checks disk space before operations
# Required: 10GB for normal operations, 15GB for builds

# Manual cleanup
docker system prune -af           # Clean all Docker resources
rm -rf /tmp/brainsmith_dev_*      # Clean temp build directories
```

### Container Monitoring
```bash
# Watch container initialization in real-time
BSMITH_SHOW_INIT_LOGS=true ./smithy daemon

# Debug container issues
docker inspect <container_name>   # Full container details
docker logs --tail 100 <name>     # Last 100 log lines
```

## Environment Setup

### Xilinx Tool Configuration
```bash
# Add to ~/.bashrc for persistence
export BSMITH_XILINX_PATH="/opt/Xilinx"
export BSMITH_XILINX_VERSION="2024.2"
export XILINXD_LICENSE_FILE="port@server"
```

### Build Directory Configuration
```bash
export BSMITH_BUILD_DIR="/custom/build/path"
export VIVADO_IP_CACHE="$BSMITH_BUILD_DIR/vivado_ip_cache"
```

## CI/CD Commands

### GitHub Actions Workflows
- PR validation runs: `cd demos/bert && make single_layer`
- Biweekly tests include additional configurations
- All tests run in containers with proper cleanup

## Summary

These additional commands provide:
1. **Advanced container management** with smithy (restart, logs, cleanup)
2. **Pre-configured BERT model variants** via Makefile targets
3. **Parameter sweep capabilities** for design space exploration
4. **Legacy compatibility** through run-docker.sh wrapper
5. **Comprehensive testing** with run_all_tests.py
6. **Environment configuration** for optimization
7. **Debugging and monitoring** tools

The key insight is that smithy provides much more functionality than documented, including persistent container management, advanced logging, and performance optimization features that significantly improve the development experience.