# Kernel Integrator User Guide

## ***PRE-RELEASE NOTE***
**The Kernel Integrator is an experimental feature that offers signficiant automation potential and will work for most simple kernels, it has some rough edges and limitations for complex corner cases (particularly including AXI-Lite config signals).**

## Overview

The Kernel Integrator is an automated tool that bridges the gap between SystemVerilog RTL hardware designs and the FINN compiler framework. It generates Python integration code that allows custom RTL kernels to be seamlessly used within neural network accelerator designs.

### What It Does

The Kernel Integrator takes a SystemVerilog RTL file annotated with special pragmas and automatically generates:

1. **HWCustomOp Class** - A FINN-compatible hardware operator that encapsulates your RTL kernel
2. **RTL Backend** - Python code that handles the RTL implementation details
3. **Verilog Wrapper** - SystemVerilog wrapper that adapts your kernel to FINN's interface requirements
4. **Python Package** - Complete Python package structure with proper imports

### Key Benefits

- **Automated Integration**: No manual Python code writing required
- **Protocol Validation**: Automatic detection and validation of AXI-Stream and AXI-Lite interfaces
- **Type Safety**: Enforced datatype constraints between RTL and Python
- **FINN Compatibility**: Generated code follows FINN best practices
- **Incremental Development**: Validate RTL before generating code

## How It Works

The Kernel Integrator follows a sophisticated pipeline:

```
RTL File + Pragmas → Parser → Metadata → Generator → Python/Verilog Files
```

### 1. Input Processing

The tool reads your SystemVerilog RTL file and extracts:
- Module definitions, ports, and parameters
- Special `@brainsmith` pragma annotations
- Interface protocols (AXI-Stream, AXI-Lite)

### 2. Metadata Construction

Parsed information is organized into a structured metadata model:
- **KernelMetadata**: Top-level kernel information
- **InterfaceMetadata**: Input/output interfaces with protocols
- **ParameterMetadata**: RTL parameters and their relationships

### 3. Code Generation

Templates transform metadata into production code:
- Python classes that inherit from FINN base classes
- Verilog wrappers that handle interface adaptation
- Complete package structure with proper imports

## Basic Usage

### Command Line Interface

```bash
# Generate all files in the same directory as RTL
python -m brainsmith.tools.kernel_integrator design.sv

# Generate in a specific output directory
python -m brainsmith.tools.kernel_integrator design.sv -o output/

# Validate RTL without generating files
python -m brainsmith.tools.kernel_integrator design.sv --validate

# Display parsed metadata
python -m brainsmith.tools.kernel_integrator design.sv --info

# Generate specific artifacts only
python -m brainsmith.tools.kernel_integrator design.sv --artifacts autohwcustomop,wrapper
```

### With Brainsmith Container

```bash
# Using the smithy wrapper script
./smithy kernel design.sv -o output/
```

## Pragma System

Pragmas are special comments that provide additional metadata to the Kernel Integrator. They must start with `@brainsmith` and appear as single-line comments in your RTL.

See the [Pragma Reference](./kernel-integrator-pragma-reference.md) guide.

## RTL Requirements

### Module Structure

Your RTL module should follow these conventions:

1. **Clear Port Definitions**: Use standard SystemVerilog ANSI-style port declarations
2. **Parameter Declaration**: Use `parameter` or `localparam` appropriately
3. **Protocol Compliance**: Follow AXI-Stream or AXI-Lite protocols for interfaces

### AXI-Stream Interfaces

Input/output interfaces should follow AXI-Stream naming:
```systemverilog
// Input stream
input  [WIDTH-1:0] in_tdata,
input              in_tvalid,
output             in_tready,

// Output stream  
output [WIDTH-1:0] out_tdata,
output             out_tvalid,
input              out_tready
```

### AXI-Lite Interfaces

Configuration interfaces should follow AXI-Lite naming:
```systemverilog
// AXI-Lite slave interface
input  [ADDR_WIDTH-1:0] s_axi_awaddr,
input                   s_axi_awvalid,
output                  s_axi_awready,
// ... (other AXI-Lite signals)
```

## Troubleshooting

### Debug Options

```bash
# Enable verbose output
python -m brainsmith.tools.kernel_integrator design.sv --verbose

# Disable strict validation for experimentation
python -m brainsmith.tools.kernel_integrator design.sv --no-strict

# Check parsed metadata without generating files
python -m brainsmith.tools.kernel_integrator design.sv --info
```

## Integration with FINN

The generated HWCustomOp can be used in FINN workflows:

```python
from generated_module import MyKernelHWCustomOp

# Use in ONNX graph construction
node = helper.make_node(
    op_type="MyKernelHWCustomOp",
    inputs=["input_tensor"],
    outputs=["output_tensor"],
    domain="brainsmith.custom_ops",
    # Set attributes
    DATA_WIDTH=8,
    NUM_CHANNELS=64
)
```

## Best Practices

1. **Start Simple**: Begin with basic pragmas and add complexity incrementally
2. **Validate Early**: Use `--validate` flag during development
3. **Use Meaningful Names**: Clear interface and parameter names improve generated code
4. **Document Pragmas**: Add comments explaining pragma choices
5. **Test Generated Code**: Verify the generated HWCustomOp in your FINN workflow

## Next Steps

- See the [Pragma Reference](kernel-integrator-pragma-reference.md) for detailed pragma documentation
- Check the [Quick Start Guide](kernel-integrator-quickstart.md) for a step-by-step tutorial
- Explore examples in `examples/kernel_integrator/` directory