# HW Kernel Generator – Integrate RTL Source Code into FINN Compiler 

## Context
The FINN toolflow generates custom FPGA dataflow accelerators for AI models. FINN matches each layer in the neural network to a generalized implementation in RTL (System Verilog) called a "HW Kernel". Each HW Kernel is then parameterized with model-specific information such as the dimensions, datatypes, and parallelism factors. The final implementation must have functional parity with an associated ONNX node (or subgraph of ONNX nodes for multi-layer HW Kernels). During runtime, FINN uses these to generate the final RTL code for synthesis. 

Brainsmith is an extension of FINN that has two key goals: hosting and maintaining a library of high-quality HW Kernels and providing automated Design Space Exploration (DSE) utilities. It has a focus on accessibility for a wide range of users, helping both hardware and software focused engineers utilize the full toolchain.

## Objective
You are creating a Hardware Kernel Generator (HKG) tool for the Brainsmith library that integrates custom RTL implementations into FINN's hardware library. The HKG has two primary responsibilities:

1. Create a parameterized wrapper template that instantiates the RTL module, enabling FINN to configure the design at runtime
2. Generate the compiler integration files (HWCustomOp and RTLBackend instances) that FINN uses for design space exploration and RTL implementation

The HKG examines only the top-level module interface of the input RTL, extracting parameters and identifying interface signals. This tool will be used by contributors to the Brainsmith project to integrate their digital circuit designs into the open-source HW Kernel library for use with FINN.

## Requirements
### 1. *Inputs*
- *Manual implementation*: RTL implementation of the target layer or subgraph with custom Pragmas to register parameters and features for the compiler (.sv). Only the top-level module interface needs to be parsed.
- *Compiler data*: A python file with supplementary information for integration with the compiler.
    - *ONNX Pattern*: FINN matches the HW Kernel to an ONNX node or subgraph that represents the pure software implementation. This should be supplied as an ONNX "model" or "graph" object.
    - *HW cost functions*: Multiple HW Kernels will be considered during design space exploration to build the optimal design. The HW cost functions model various FPGA resource usage (LUTs, BRAM, DSPs, etc.) based on design parameters, and must be specified by the user. In the future, this will be replaced with automatic code profiling.
- (Optional input) *Custom Documentation* - Markdown documentation describing the HW Kernel.


### 3. *Outputs*
- *RTL Template*: A wrapper module that instantiates the input RTL with parameterizable values for FINN to configure at runtime.
- *HWCustomOp instance*: ONNX node representing the HW Kernel used for Design Space Exploration in FINN.
- *RTLBackend instance*: ONNX node representing the attributes of the HW Kernel specific to an RTL implementation.
- *Documentation*: Auto-generated descriptions of the interfaces, parameters, assumptions, and limitations of the HW Kernel. This is combined with any provided input documentation.

## *Implementation Details*
### 1. *Technologies*
- *Languages*: Use Python for implementation

### 2. *Coding Style*
- The HKG is designed to help Hardware Engineers with little to no understanding of the FINN toolchain to contribute to the library of open source HW Kernels. 
- This will be part of an open source release from a prestigious company, so code quality and documentation must be exemplary.
- Extensibility is a key design goal. This specification is for the initial implementation of the Hardware Kernel Generator: many features are yet to come. 

### 3. *Assumptions*
- Assume all RTL source code is functional. The HKG's only responsibility is to create the wrapper and integration files needed to register the input with FINN if it meets the Spec for a Hardware Kernel.
- Only the top-level module interface needs to be parsed - internal implementation details can be ignored.

### 4. *Execution Phases*
Split implementation into multiple phases, pausing for the user to debug and analyze between each phase.

## *Sources*
- FINN Docs: @https://finn.readthedocs.io/en/latest/developers.html 

