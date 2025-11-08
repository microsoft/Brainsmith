# Getting Started

## Installation

!!! note "Prerequisites"
    - **Ubuntu 22.04+** (primary development/testing platform)
    - **[Vivado Design Suite](https://www.xilinx.com/support/download.html) 2024.2** (migration to 2025.1 in process)
    - **[Optional]** Cmake for BERT example V80 shell integration

```bash
git clone https://github.com/microsoft/brainsmith.git ./brainsmith
cd brainsmith

```

---

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

---

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

Brainsmith operates from a single poetry `venv` from the repository root, but
you can create isolated workspaces via the *project* system with independent
configurations, build directories, and component registries.

---

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

Project settings are loaded from multiple sources with the following priority
(highest to lowest) for deep user control, but the recommended interface is the
**yaml config file** using CLI arguments to override as necessary.

1. **CLI arguments** - Passed to `load_config()` or command-line tools
2. **Environment variables** - `BSMITH_*` prefix (e.g., `BSMITH_BUILD_DIR`)
3. **Project config file** - `brainsmith.yaml` in project root
4. **Built-in defaults** - Field defaults in `SystemConfig`

#### Core Paths

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `build_dir` | `Path` | `"build"` | Build directory for compilation artifacts. Relative paths resolve to `project_dir`. |
| `project_dir` | `Path` | *auto-detected* | Project root directory (parent of `.brainsmith/`). Auto-detected via upward directory walk or `BSMITH_PROJECT_DIR`. |
| `bsmith_dir` | `Path` | *auto-detected* | Brainsmith installation root (cached property). |
| `component_sources` | `Dict[str, Path | None]` | `{'project': None}` | Filesystem-based component source paths. `'project'` defaults to `project_dir` (supports `kernels/` and `steps/` subdirectories). Core namespace (`'brainsmith'`) and entry points (`'finn'`) are loaded automatically. |
| `source_priority` | `List[str]` | `['project', 'brainsmith', 'finn', 'custom']` | Component source resolution priority (first match wins). Custom sources are auto-appended if not listed. |
| `source_module_prefixes` | `Dict[str, str]` | `{'brainsmith.': 'brainsmith', 'finn.': 'finn'}` | Module prefix → source name mapping for component classification. |
| `components_strict` | `bool` | `True` | Enable strict component loading (fail on errors vs. warn). Set to `false` for development. |
| `cache_components` | `bool` | `True` | Enable manifest caching for component discovery (auto-invalidates on file changes). |

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

## Running Design Space Exploration

---

### Prerequisites

Make sure you've completed the installation above and activated your environment:

```bash
# Option 1: direnv users
cd /path/to/brainsmith

# Option 2: Manual activation
source .venv/bin/activate && source .brainsmith/env.sh
```


---

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

The script outputs detailed progress to the console and log files. The build process transforms your model through several stages:

**Transformation Pipeline:**
```
PyTorch → ONNX → Hardware Kernels → HLS/RTL → IP Cores → Bitfile
```

**Key stages you'll see:**

- **Model transformation**: Converting ONNX operations to hardware kernels
- **Design space exploration**: Determining parallelization factors (PE/SIMD)
- **Code generation**: Generating HLS C++ and RTL (Verilog/VHDL)
- **IP packaging**: Creating Vivado IP cores
- **Simulation**: Verifying correctness with RTL simulation

Check `build/quicktest/build_dataflow.log` for detailed progress and diagnostics.

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

##### RTL Output

The generated RTL is in `final_output/stitched_ip/`:

- `finn_design_wrapper.v` — Top-level wrapper with AXI stream interfaces
- Individual kernel implementations (e.g., `MVAU_hls_*.v`, `Thresholding_rtl_*.v`)
- Stream infrastructure (FIFOs, width converters, etc.)

---

### Customize the Design

Now that you've run the basic example, try customizing it:

#### Adjust Target Performance

Edit `bert_quicktest.yaml` to increase throughput:

```yaml
finn_config:
  target_fps: 10  # Increase from 1 to 10 FPS
```

Higher target FPS will:
- Increase parallelization factors (PE/SIMD parameters)
- Use more FPGA resources (LUTs, DSPs, BRAM)
- Reduce latency per inference

#### Create Custom Configurations

Create your own blueprint that inherits from the BERT base:

```yaml
# my_custom_bert.yaml
name: "My Custom BERT"
extends: "${BSMITH_DIR}/examples/blueprints/bert.yaml"

clock_ns: 4.0           # 250MHz (faster clock)
output: "estimates"     # Just get resource estimates

finn_config:
  target_fps: 5000      # Very high throughput
```

Then run it:

```bash
python bert_demo.py -o my_output -l 2 --blueprint my_custom_bert.yaml
```

#### Run Full BERT Demo

The full demo processes larger models. From `examples/bert/`:

```bash
# Generate standard folding configuration
python gen_folding_config.py --simd 16 --pe 16 -o configs/demo_folding.json

# Run with 4 layers instead of 1
python bert_demo.py -o bert_demo_output -l 4 --blueprint bert_demo.yaml
```

This creates a more realistic accelerator but takes significantly longer to build.

---

### Understanding Blueprints

A **Blueprint** is a YAML configuration that defines your hardware design space - the kernels, transformation steps, and build parameters.

#### Blueprint Inheritance

Blueprints support inheritance via the `extends` key, allowing you to build on existing configurations:

```yaml
# bert_quicktest.yaml - Quick test configuration
name: "BERT Quicktest"
extends: "${BSMITH_DIR}/examples/bert/bert_demo.yaml"

output: "bitfile"

finn_config:
  target_fps: 1                     # Low FPS for quick testing
  folding_config_file: "configs/quicktest_folding.json"
  fifosim_n_inferences: 2           # Faster FIFO sizing
```

The parent blueprint (`bert_demo.yaml`) defines the core design space:

```yaml
# bert_demo.yaml - Inherits from blueprints/bert.yaml
name: "BERT Demo"
extends: "${BSMITH_DIR}/examples/blueprints/bert.yaml"

clock_ns: 5.0           # 200MHz target clock
output: "bitfile"
board: "V80"

finn_config:
  target_fps: 3000                  # High performance target
  standalone_thresholds: true
```

---


### Getting Help

**Build logs:** Check `<output_dir>/build_dataflow.log` for detailed error messages and transformation steps.

**Resources:**
- [GitHub Issues](https://github.com/microsoft/brainsmith/issues) - Report bugs or search existing issues
- [GitHub Discussions](https://github.com/microsoft/brainsmith/discussions) - Ask questions and share experiences

---

## Next Steps

### Explore Design Space Trade-offs

Compare different configurations by varying performance targets. From `examples/bert/`:

```bash
# Create low-resource configuration
cat > bert_lowres.yaml << 'EOF'
name: "BERT Low Resource"
extends: "${BSMITH_DIR}/examples/bert/bert_quicktest.yaml"
finn_config:
  target_fps: 1
EOF

# Create high-performance configuration
cat > bert_highperf.yaml << 'EOF'
name: "BERT High Performance"
extends: "${BSMITH_DIR}/examples/bert/bert_quicktest.yaml"
finn_config:
  target_fps: 50
EOF

# Compare resource usage
python bert_demo.py -o output_lowres -l 1 --blueprint bert_lowres.yaml
python bert_demo.py -o output_highperf -l 1 --blueprint bert_highperf.yaml

# Check results
cat output_lowres/final_output/report/estimate_reports.json
cat output_highperf/final_output/report/estimate_reports.json
```

### Try Custom Models

**From PyTorch to FPGA:**

1. **Quantize** - Use Brevitas to quantize your PyTorch model to low precision (INT8, INT4, etc.)
2. **Export** - Export to ONNX format with `brevitas.export`
3. **Create blueprint** - Start with an existing blueprint and modify the kernel list
4. **Build** - Run design space exploration:
   ```bash
   smith my_model.onnx my_blueprint.yaml -o my_output
   ```

### Learn More

- **[API Reference](api/index.md)** - Programmatic access to Brainsmith functions
- **[Example Blueprints](https://github.com/microsoft/brainsmith/tree/main/examples/blueprints)** - Pre-built configurations for common architectures
- **[GitHub Discussions](https://github.com/microsoft/brainsmith/discussions)** - Community support and examples
