
# Brainwave

Brainsmith v80 shell. 

It supports hardware generation (Vivado flow), a C++ runtime, and Python bindings through pybind11.

---

## Configuration Overview

The build is controlled by a layered CMake setup:

- Top-level CMake: selects build targets, sets paths, and invokes the main Brainwave CMake.
- Base CMake: defines all build logic for hardware, software, and Python modules (can be used as a submodule).

### Build Options

| Option | Description | Default |
|:-------|:-------------|:--------|
| `BUILD_HW` | Build FPGA hardware (Vivado flow) | `ON` |
| `BUILD_SW` | Build C++ runtime and main executable | `OFF` |
| `BUILD_PY` | Build Python bindings (pybind11 + LibTorch) | `ON` |

### Required Variables

| Variable | Description |
|:----------|:-------------|
| `BWAVE_DIR` | Path to the Brainwave base directory (if used as a submodule) |
| `CORE_PATH` | Path to the stitched IP core (must contain `shell_handover.json`) |

---

## Building

### 1. Configure

From your build directory:

```
bash
cmake -S ../ -B build
```

Or explicitly specify components:

```
cmake -S ../ -B build -DBUILD_SW=ON -DBUILD_PY=ON
```

### 2. Build Targets (make)

| Target       | Description                                    |
| :----------- | :--------------------------------------------- |
| `hw_project` | Creates the Vivado project                     |
| `hw_synth`   | Runs synthesis and generates checkpoints       |
| `hw_compile` | Runs implementation and generates bitstream    |
| `sw_build`   | Builds the shared library and main executable  |
| `sw_install` | Packages binaries and libraries into `sw/dist` |
| `sw_python`  | Builds the Python extension module `brainwave` |

## Requirements

- Requires CMake ≥ 3.18

- Hardware builds require Vivado (auto-detected)

- Python ≥ 3.8

- pybind11

- LibTorch (CPU or GPU build)

---
