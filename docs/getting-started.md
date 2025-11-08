# Getting Started

- [Installation](#installation)
  - [Local Development with Poetry](#option-a-local-development-with-poetry)
  - [Docker-based Development](#option-b-docker-based-development)
- [Project Management](#project-management)
  - [Creating Projects](#creating-projects)
  - [Configuration](#configuration)
    - [Core Paths](#core-paths)
    - [Xilinx Configuration](#xilinx-configuration)
    - [Runtime Configuration](#runtime-configuration)
    - [Logging Configuration](#logging-configuration)
- [Validate Installation](#validate-installation)
- [Quick Start](#quick-start)
  - [Run Your First DSE](#run-your-first-dse)
  - [Explore Results](#explore-results)
  - [Customize the Design](#customize-the-design)
  - [Understanding Blueprints](#understanding-blueprints)
  - [Troubleshooting](#troubleshooting)
  - [Getting Help](#getting-help)
- [Next Steps](#next-steps)
  - [Explore Design Space](#explore-design-space)
  - [Try Custom Models](#try-custom-models)
  - [Learn the Tools](#learn-the-tools)

---

## Installation

!!! note "Prerequisites"
    - **Ubuntu 22.04+** (primary development/testing platform)
    - **[Vivado Design Suite](https://www.xilinx.com/support/download.html) 2024.2** (migration to 2025.1 in process)
    - **[Optional]** Cmake for BERT example V80 shell integration

```bash
git clone https://github.com/microsoft/brainsmith.git ./brainsmith
cd brainsmith

```

### (Option A): Local Development with Poetry

!!! note "Prerequisites"
    - **Python 3.11+** and **[Poetry](https://python-poetry.org/docs/#installation)**
    - [Optional] **[direnv](https://direnv.net/)** for automatic environment activation

Run automated setup script

```bash
./setup-venv.sh
```

Edit the project configuration file to customize project settings

```bash
vim brainsmith.yaml
```

Activate environment manually or with direnv

```bash
source .venv/bin/activate && source .brainsmith/env.sh
# If using direnv, just reload directory
cd .
```

Query project settings to confirm your configuration is loaded correctly

```bash
brainsmith project info
```

### (Option B): Docker-based Development


!!! note "Prerequisites"
    - Docker configured to run [without root](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

Edit `ctl-docker.sh` or set environment variables to directly set brainsmith project settings

```bash
export BSMITH_XILINX_PATH=/tools/Xilinx/
export BSMITH_XILINX_VERSION=2024.2
```

Start container

```bash
./ctl-docker.sh start
```

Open an interactive shell to check your configuration

```bash
./ctl-docker.sh shell
brainsmith project info
```

Or send one-off commands to the container

```bash
./ctl-docker.sh "brainsmith project info"
```

---

## Project Management

Brainsmith operates from a single poetry `venv` from the repostiory root, but
you can create isolated workspaces via the *project* system with independent configurations,
build artifacts, and component registries.


### Creating Projects

```bash
# Activate brainsmith venv if not in an active project
source /path/to/brainsmith/.venv/bin/activate

# Create and initialize new project directory
brainsmith project init ~/my-fpga-project
cd ~/my-fpga-project
```

Edit the default generated config file

```bash
vim ~/my-fpga-project/brainsmith.yaml
```

[Optional] Enable auto-activation if using direnv

```bash
brainsmith project allow-direnv
cd .  # Triggers direnv
```

Otherwise, refresh env after any config changes:

```bash
source .brainsmith/env.sh
```

---


### Configuration

Configuration management for Brainsmith projects with hierarchical loading and type-safe validation using Pydantic.

Settings are loaded from multiple sources with the following priority (highest to lowest) for deep user control, but it
is strongly recommended to primarily configure your projects via the `brainsmith.yaml`.

1. **CLI arguments** - Passed to `load_config()` or command-line tools
2. **Environment variables** - `BSMITH_*` prefix (e.g., `BSMITH_BUILD_DIR`)
3. **Project config file** - `brainsmith.yaml` in project root
4. **Built-in defaults** - Field defaults in `SystemConfig`

It is highly recommended to primarily configure


### Configuration Fields



#### Core Paths

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `build_dir` | `Path` | `"build"` | Build directory for compilation artifacts. Relative paths resolve to `project_dir`. |
| `project_dir` | `Path` | *auto-detected* | Project root directory (parent of `.brainsmith/`). Auto-detected via upward directory walk or `BSMITH_PROJECT_DIR`. |
| `bsmith_dir` | `Path` | *auto-detected* | Brainsmith installation root (cached property). |

All relative paths are resolved from the *point of configuration*...


!!! note "Path Resolution"
    - **Absolute paths**: Used as-is
    - **Relative paths from CLI**: Resolve to current working directory
    - **Relative paths from YAML/env**: Resolve to `project_dir`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `component_sources` | `Dict[str, Path \| None]` | `{'project': None}` | Filesystem-based component source paths. `'project'` defaults to `project_dir` (supports `kernels/` and `steps/` subdirectories). Core namespace (`'brainsmith'`) and entry points (`'finn'`) are loaded automatically. |
| `source_priority` | `List[str]` | `['project', 'brainsmith', 'finn', 'custom']` | Component source resolution priority (first match wins). Custom sources are auto-appended if not listed. |
| `source_module_prefixes` | `Dict[str, str]` | `{'brainsmith.': 'brainsmith', 'finn.': 'finn'}` | Module prefix → source name mapping for component classification. |
| `components_strict` | `bool` | `True` | Enable strict component loading (fail on errors vs. warn). Set to `false` for development. |
| `cache_components` | `bool` | `True` | Enable manifest caching for component discovery (auto-invalidates on file changes). |



#### Xilinx Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `xilinx_path` | `Path` | `"/tools/Xilinx"` | Xilinx root installation path. |
| `xilinx_version` | `str` | `"2024.2"` | Xilinx tool version (e.g., `"2024.2"`, `"2025.1"`). |
| `vivado_path` | `Path \| None` | *auto-detected* | Path to Vivado. Auto-detected from `{xilinx_path}/Vivado/{xilinx_version}`. |
| `vitis_path` | `Path \| None` | *auto-detected* | Path to Vitis. Auto-detected from `{xilinx_path}/Vitis/{xilinx_version}`. |
| `vitis_hls_path` | `Path \| None` | *auto-detected* | Path to Vitis HLS. Auto-detected from `{xilinx_path}/Vitis_HLS/{xilinx_version}`. |
| `vivado_ip_cache` | `Path \| None` | `{build_dir}/vivado_ip_cache` | Vivado IP cache directory for faster builds. |
| `vendor_platform_paths` | `str` | `"/opt/xilinx/platforms"` | Colon-separated vendor platform repository paths. |


#### Runtime Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_workers` | `int` | `4` | Default number of workers for parallel operations. Exported as `NUM_DEFAULT_WORKERS`. |
| `netron_port` | `int` | `8080` | Port for Netron neural network visualization server. |

#### Logging Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logging.level` | `str` | `"normal"` | Console verbosity: `quiet` \| `normal` \| `verbose` \| `debug`. |
| `logging.finn_tools` | `Dict[str, str] \| None` | `None` | Per-tool log levels for FINN tools (e.g., `{'vivado': 'WARNING', 'hls': 'INFO'}`). |
| `logging.suppress_patterns` | `List[str] \| None` | `None` | Regex patterns to suppress from console output (file logs unaffected). |
| `logging.max_log_size_mb` | `int` | `0` | Maximum log file size in MB (0 = no rotation). |
| `logging.keep_backups` | `int` | `3` | Number of rotated log backups to keep. |


Additional configuration fields (FINN settings, direct Xilinx tool paths, etc.) can be set directly, are recommended to
let auto-configure from the core brainsmith fields.






---


## Validate Installation

Run the quicktest to verify everything is working:

```bash
./examples/bert/quicktest.sh
```

This runs a minimal BERT example (single layer) to verify:
- Environment is configured correctly
- Vivado is accessible
- Dependencies are installed
- Basic DSE pipeline works

!!! info "Build Time"
    The quicktest can take upwards of an hour, depending on your system due to
    RTL simulation based fifo sizing.

---

## Quick Start

Get started with Brainsmith in 5 minutes by running the BERT example.

### Prerequisites

Make sure you've completed the installation above and activated your environment:

```bash
# Option 1: direnv users
cd /path/to/brainsmith

# Option 2: Manual activation
source .venv/bin/activate && source .brainsmith/env.sh
```

### Run Your First DSE

#### 1. Navigate to the BERT Example

```bash
cd examples/bert
```

#### 2. Run the Quick Test

```bash
./quicktest.sh
```

This automated script will:

1. **Generate a quantized ONNX model** - Single-layer BERT for rapid testing
2. **Generate a folding configuration** - Minimal resource usage
3. **Build the dataflow accelerator** - RTL generation with FINN
4. **Run RTL simulation** - Verify correctness

!!! info "Build Time"
    The quicktest takes approximately **30-60 minutes**, depending on your system. The build process involves:

    - ONNX model transformations
    - Kernel inference and specialization
    - RTL code generation
    - IP packaging with Vivado
    - RTL simulation

#### 3. Monitor Progress

The script outputs progress to the console. Key stages include:

```
[INFO] Generating BERT ONNX model...
[INFO] Creating folding configuration...
[INFO] Running dataflow build...
  ├─ Streamlining
  ├─ QONNX to FINN conversion
  ├─ Dataflow partition creation
  ├─ Kernel inference
  ├─ RTL code generation
  ├─ IP packaging
  └─ RTL simulation
[SUCCESS] Build complete!
```

---

### Explore Results

Results are saved in `examples/bert/quicktest/`:

```
quicktest/
├── model.onnx              # Quantized ONNX model
├── final_output/           # Generated RTL and reports
│   ├── stitched_ip/       # Synthesizable RTL
│   │   ├── finn_design_wrapper.v
│   │   └── *.v
│   └── report/            # Performance estimates
│       └── estimate_reports.json
└── build_dataflow.log     # Detailed build log
```

#### Understanding the Output

##### Performance Report

Check `final_output/report/estimate_reports.json`:

```json
{
  "critical_path_cycles": 123,
  "max_cycles": 456,
  "estimated_throughput_fps": 1234.5,
  "resources": {
    "LUT": 12345,
    "FF": 23456,
    "BRAM_18K": 34,
    "DSP48": 56
  }
}
```

##### RTL Output

The generated RTL is in `final_output/stitched_ip/`:

- `finn_design_wrapper.v` - Top-level wrapper with AXI stream interfaces
- Individual kernel implementations (e.g., `MVAU_hls_*.v`, `Thresholding_rtl_*.v`)
- Stream infrastructure (FIFOs, width converters, etc.)

---

### Customize the Design

Now that you've run the basic example, try customizing it:

#### Adjust Target Performance

Edit `bert_quicktest.yaml`:

```yaml
design_space:
  steps:
    - step_target_fps_parallelization:
        target_fps: 100  # Increase target throughput
```

Higher target FPS will:
- Increase parallelization factors
- Use more FPGA resources
- Reduce latency

#### Change Backend

Try different hardware implementations:

```yaml
design_space:
  kernels:
    - MVAU: [mvau_hls, mvau_rtl]  # Try both HLS and RTL backends
```

#### Run Full Example

The full BERT demo (not just quicktest) processes actual text:

```bash
python bert_demo.py --config bert_demo.yaml
```

This uses a larger model and generates a complete accelerator.

---

### Understanding Blueprints

The `bert_quicktest.yaml` file is a **Blueprint** - a YAML configuration that defines your design space:

```yaml
name: "BERT Quicktest"
clock_ns: 5.0           # 200MHz target clock
output: "rtlsim"        # Generate RTL and run simulation

design_space:
  kernels:              # Hardware kernels to use
    - MVAU
    - Thresholding
    - StreamingFCLayer

  steps:                # Build pipeline steps
    - streamline
    - qonnx_to_finn
    - step_create_dataflow_partition
    - step_specialize_layers
    - step_target_fps_parallelization:
        target_fps: 50
    - step_minimize_bit_width
    - step_generate_estimate_reports
    - step_hw_codegen
    - step_hw_ipgen
    - step_set_fifo_depths
    - step_create_stitched_ip
    - step_synthesize_bitfile
    - step_make_pynq_driver
    - step_deployment_package
```

Learn more in the [Blueprint Reference](developer-guide/3-reference/blueprints.md).

---

### Troubleshooting

#### Build Fails During Kernel Inference

**Error:** `Could not find suitable kernel for operation X`

**Solution:** Add the missing kernel to your blueprint's `kernels` list.

#### RTL Simulation Fails

**Error:** `Simulation mismatch detected`

**Solution:** This usually indicates a configuration issue. Check:
- Folding factors are valid for your model dimensions
- FIFO depths are sufficient
- Input/output data types match model requirements

#### Out of Memory

**Error:** `MemoryError` during build

**Solution:**
- Close other applications
- Reduce model size for testing
- Use `--stop-step` to build incrementally

#### Vivado Not Found

**Error:** `vivado: command not found`

**Solution:**
1. Check config: `brainsmith project info`
2. Re-activate environment: `source .brainsmith/env.sh`
3. Verify Vivado path: `ls $XILINX_PATH`

---

### Getting Help

- Check the [build log](../examples/bert/quicktest/build_dataflow.log) for detailed error messages
- Search existing [GitHub Issues](https://github.com/microsoft/brainsmith/issues)
- Ask on [GitHub Discussions](https://github.com/microsoft/brainsmith/discussions)

---

## Next Steps

### Explore Design Space

Run multiple configurations to compare trade-offs:

```bash
# Low resource usage
smith dfc model.onnx blueprint_small.yaml --output-dir ./results_small

# High performance
smith dfc model.onnx blueprint_fast.yaml --output-dir ./results_fast
```

### Try Custom Models

1. Quantize your PyTorch model with Brevitas
2. Export to ONNX
3. Create a blueprint
4. Run DSE:
   ```bash
   smith dfc my_model.onnx my_blueprint.yaml
   ```

### Learn the Tools

- [Blueprint Reference](developer-guide/3-reference/blueprints.md) - Master the YAML format
- [CLI Reference](developer-guide/3-reference/cli.md) - Explore all commands
- [Design Space Exploration](developer-guide/2-core-systems/design-space-exploration.md) - Understand DSE concepts
