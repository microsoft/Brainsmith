# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brainsmith is an open-source platform for FPGA AI accelerators developed collaboratively by Microsoft and AMD. It converts PyTorch models to RTL implementations for FPGA deployment using a sophisticated Blueprint-based design space exploration (DSE) approach.

### Core Pipeline
```
PyTorch Model → Brevitas Quantization → ONNX → FINN/Brainsmith → RTL Synthesis
Blueprint YAML → DSE v3 → Hardware Implementation → FPGA Deployment
```

### Key Innovation
Brainsmith introduces a **Blueprint-driven DSE system** that systematically explores hardware design spaces to find optimal FPGA configurations. Unlike traditional single-point solutions, it explores multiple kernel backends, transform combinations, and build configurations to maximize performance while meeting resource constraints.

## Dependency Management

### Repository Locations
- Our branch of QONNX is at deps/qonnx
- FINN is at deps/FINN

**External Dependencies** (via `docker/fetch-repos.sh`)
- **FINN** from `custom/brainsmith-patch` branch → `deps/finn/`
- **QONNX** from `custom/brainsmith` branch → `deps/qonnx/`
- **Brevitas** quantization framework → `deps/brevitas/`
- **FINN-Experimental** specific commit → `deps/finn-experimental/`
- **Board files** and **pyxsi** for Xilinx integration

### Repository Branch Details
- Our branch of qonnx is called `custom\brainsmith` and is at path deps/qonnx