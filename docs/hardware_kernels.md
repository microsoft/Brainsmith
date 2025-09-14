# Brainsmith Hardware Kernels

## Definition

Hardware Kernels are synthesizable hardware modules that implement neural network operators. They serve as the foundational components from which Brainsmith constructs accelerators.

### Interfaces

To ensure complete modularity, kernel interfaces are limited to the following:

- *Control* - Clock and reset (optional second clock for double-pumped designs)
- *Input/Output* - AXI-Stream for activations and weights. A minium of one input and one output.
- *Config* - (Optional) Maximum of a single AXI-Lite interface for runtime configuration/debugging

See [protocol_validator.py](brainsmith/tools/kernel_integrator/rtl_parser/protocol_validator.py) for complete interface signal definitions, and the name suffix rules for RTL interfaces.


## Why Kernels?

Brainsmith chooses **kernels** as its fundamental abstraction for several reasons:
1. **Preserve Design Space** - AI models are designed and expressed in terms of Layers/nodes, and preserving this design granularity allows for natural extension from AI frameworks like PyTorch/ONNX.
2. **Prevent Exponential Explosion** - Although there is theoretical value in fully decomposing models to individual operations, this exponetially explodes the design space. 
3. **Hand Optimization** - Allows hardware engineers to design kernels hand-optimized on the kernel scale without requireing deep knowledge of the AI model. This allows for hand-optimized performance that fully generated designs lack, while maintaining flexibility by composing the final hardware graph through Brainsmith.

## Implementation

Each kernel requires four components. For RTL Kernels, use of the Kernel Integrator is highly recommended.

### RTL/HLS Source Code

RTL/HLS implementation with top-level interfaces matching the required spec.

RTL example: 

HLS example: [`brainsmith/kernels/layernorm/layernorm.hpp`](brainsmith/kernels/layernorm/layernorm.hpp)

### Kernel Operator (HWCustomOp)

- HWCustomOp - ONNX node modeling the high

[HWCustomOp](https://github.com/Xilinx/finn/blob/main/src/finn/custom_op/fpgadataflow/hwcustomop.py)

RTL example: 

HLS example: [`brainsmith/kernels/layernorm/layernorm.py`](brainsmith/kernels/layernorm/layernorm.py)

### Backend Operator (RTLBackend/HLSBackend)

- Backend

[HLSBackend](https://github.com/Xilinx/finn/blob/main/src/finn/custom_op/fpgadataflow/hlsbackend.py)

[RTLBackend](https://github.com/Xilinx/finn/blob/main/src/finn/custom_op/fpgadataflow/rtlbackend.py)

RTL example: 
HLS example: 

### Codegen Template 

- Template

RTL example: 
HLS example: 

