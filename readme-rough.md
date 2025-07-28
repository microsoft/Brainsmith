Brainsmith is a PyTorch to RTL toolchain for AI accelerators on FPGA. 


It systematically explores hardware implementation options for neural networks, evaluating different algorithms, parallelization strategies, and resource allocations to find optimal configurations.

**This repository is in a pre-release state and under active co-development by Microsoft and AMD.**


Pre-release features:
- Component library - define 
- Blueprint interface -  
- BERT Demo - Example end-to-end demo (PyTorch to stitched-IP RTL accelerator)


Planned features:
- FINN Kernel backend rework - from RTLBackend/HLSBackend to generic Kernel backend
- Automated Kernel Integrator - Generate full compiler integration python code from RTL code alone
- Branching tree execution - Execute multiple builds in parallel, intelligently re-using build artefacts



## Overview

Brainsmith is a comprehensive platform for dataflow AI accelerator development on FPGA.

uses Blueprints that define a Design Space to explore different configurations for a given neural network definition to be implemented on FPGAs. A Blueprint is a yaml file which can configure the following:
- Model optimization and network surgery by specifying combinations of graph transformations, such as expanding / fusing multiple ONNX-level operations
- Hardware configuration parameters, e.g., different FPGA targets
- FPGA compiler step variations, e.g., assembly of FINN compiler build steps
- Future support for multiple kernel implementations for the same layer, e.g. optimized RTL or HLS for specific network settings

The system currently uses the Legacy FINN backend for compilation, as FINN does not yet support the new entrypoint-based plugin system. The architecture is designed to support future backends with kernel-level customization.

## Core Pipeline
