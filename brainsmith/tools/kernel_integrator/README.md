# Kernel Integrator

> **DEPRECATION NOTICE**
> The Kernel Integrator is currently deprecated in favor of the KernelOp system and will be updated for compatibility in a future release.
> For new kernel development, please use the KernelOp framework directly. See the [Kernel Development Guide](../../../docs/developer-guide/experimental/kernel_ops/) for details.

## Overview

The Kernel Integrator is a code generation tool that bridges SystemVerilog RTL implementations with Brainsmith's KernelOp abstraction layer. It automates the creation of FINN-compatible hardware custom operations from annotated RTL files.

## What It Does

Given a SystemVerilog file with `@brainsmith` pragmas, the Kernel Integrator generates:

1. **KernelOp subclass** - Python class defining the kernel interface and graph transformation logic
2. **RTL Backend** - Code generator for producing synthesizable hardware
3. **Verilog Wrapper** - FINN-compatible wrapper with AXI-Stream interfaces
4. **Package Init** - Python module exports for registry integration

## Basic Usage

```bash
# Via CLI command
brainsmith smith kernel design.sv

# Or directly
python -m brainsmith.tools.kernel_integrator design.sv
```

### Common Options

```bash
# Specify output directory
brainsmith smith kernel design.sv -o /path/to/output

# Validate RTL only (no generation)
brainsmith smith kernel design.sv --validate

# Display parsed metadata
brainsmith smith kernel design.sv --info

# Generate specific artifacts only
brainsmith smith kernel design.sv --artifacts kernelop,wrapper

# Include additional RTL dependencies
brainsmith smith kernel design.sv --include-rtl helper.sv --include-rtl constants.sv
```

## Generated Artifacts

| Artifact | Template | Output | Description |
|----------|----------|--------|-------------|
| `kernelop` | `auto_hw_custom_op.py.j2` | `{name}.py` | KernelOp subclass with transformation logic |
| `rtlbackend` | `auto_rtl_backend.py.j2` | `{name}_rtl.py` | RTL backend for code generation |
| `wrapper` | `rtl_wrapper.v.j2` | `{name}_wrapper.v` | FINN-compatible Verilog wrapper |
| `init` | `__init__.py.j2` | `__init__.py` | Package initialization and exports |

## RTL Pragma Format

The Kernel Integrator expects SystemVerilog files with `@brainsmith` pragmas:

```systemverilog
// @brainsmith parameter <name> <type> [default_value]
// @brainsmith interface <type> <name> <direction> [options]

module MatMul #(
    // @brainsmith parameter M int 8
    parameter M = 8,
    // @brainsmith parameter N int 8
    parameter N = 8
) (
    // @brainsmith interface axis input_a input tdata[M*N-1:0]
    input wire [M*N-1:0] input_a_tdata,
    input wire input_a_tvalid,
    output wire input_a_tready,

    // @brainsmith interface axis output_c output tdata[M*N-1:0]
    output wire [M*N-1:0] output_c_tdata,
    output wire output_c_tvalid,
    input wire output_c_tready
);
    // Implementation...
endmodule
```

## Integration with KernelOp System

Generated classes automatically register with Brainsmith's kernel registry using the `@kernel` decorator, making them available for:

- Dataflow graph transformations
- Design space exploration
- Hardware generation workflows
- FINN integration

## Example

See `examples/kernel_integrator/` for a complete working example.

## Architecture

```
kernel_integrator/
├── cli.py              # Command-line interface
├── generator.py        # KernelGenerator (Jinja2 templating)
├── metadata.py         # Data structures (KernelMetadata, InterfaceMetadata)
├── rtl_parser/         # SystemVerilog parser with pragma support
│   ├── parser.py
│   └── pragmas/
└── templates/          # Jinja2 templates for code generation
    ├── auto_hw_custom_op.py.j2
    ├── auto_rtl_backend.py.j2
    ├── rtl_wrapper.v.j2
    └── __init__.py.j2
```

## Migration Path

For new development, use the KernelOp framework directly:

1. **Manual KernelOp Implementation** - Create KernelOp subclasses following the [Kernel Development Guide](../../../docs/developer-guide/experimental/kernel_ops/)
2. **Direct Backend Integration** - Implement RTL backends without code generation
3. **Registry Integration** - Use `@kernel` decorator for automatic registration

The Kernel Integrator will be updated in a future release to maintain compatibility with the evolving KernelOp system.

## Further Reading

- [Hardware Kernels Guide](../../../docs/developer-guide/hardware-kernels.md)
- [KernelOp Development Guide](../../../docs/developer-guide/experimental/kernel_ops/)
- [Dataflow Modeling](../../../docs/developer-guide/dataflow.md)
