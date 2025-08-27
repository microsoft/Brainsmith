# Kernel Integrator User Guide

This guide provides comprehensive instructions for using the Kernel Integrator tool, including CLI usage, pragma reference, and best practices.

## Table of Contents
- [Installation](#installation)
- [Command Line Interface](#command-line-interface)
- [Pragma Reference](#pragma-reference)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Installation

The Kernel Integrator is part of the Brainsmith toolkit:

```bash
# Run as a module
python -m brainsmith.tools.kernel_integrator

# Or if installed in PATH
kernel_integrator
```

## Command Line Interface

### Basic Usage

```bash
kernel_integrator <rtl_file> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output DIR` | Output directory for generated files | Same as RTL file |
| `-v, --verbose` | Enable detailed logging | False |
| `--validate` | Validate RTL only (no generation) | False |
| `--info` | Display parsed metadata only | False |
| `--artifacts LIST` | Comma-separated artifacts to generate | All |
| `--no-strict` | Disable strict validation | False |
| `--include-rtl FILE` | Additional RTL file to include (repeatable) | None |
| `--rtl-path PATHS` | Colon-separated search paths for RTL files | None |

### Artifacts

You can selectively generate specific files:

- `autohwcustomop` - FINN HWCustomOp Python class
- `rtlbackend` - RTL backend implementation
- `wrapper` - SystemVerilog wrapper

Example:
```bash
# Generate only the HWCustomOp class
kernel_integrator design.sv --artifacts autohwcustomop

# Generate HWCustomOp and wrapper
kernel_integrator design.sv --artifacts autohwcustomop,wrapper
```

### Validation Modes

#### Strict Mode (Default)
Enforces:
- Presence of `ap_clk` and `ap_rst_n`
- At least one input and output interface
- Valid BDIM/SDIM for streaming interfaces
- Complete AXI protocol signals

#### Relaxed Mode
```bash
kernel_integrator experimental.sv --no-strict
```
Allows partial interfaces and missing dimensions for experimentation.

## Pragma Reference

Pragmas are special comments that provide metadata for code generation:

```systemverilog
// @brainsmith <PRAGMA_TYPE> <arguments>
```

### Interface Pragmas

#### DATATYPE_CONSTRAINT
Constrains the allowed data types for an interface:
```systemverilog
// @brainsmith DATATYPE_CONSTRAINT <interface_name> <sign> <width> [<shape>]
// @brainsmith DATATYPE_CONSTRAINT s_axis_input UINT 8
// @brainsmith DATATYPE_CONSTRAINT m_axis_weights INT 16 SHAPE [3, 3]
```

#### DATATYPE
Links RTL parameters to datatype properties:
```systemverilog
// @brainsmith DATATYPE <interface_name> WIDTH <param_name>
// @brainsmith DATATYPE s_axis_input WIDTH DATA_WIDTH
```

#### WEIGHT
Marks an interface as containing weight/parameter data:
```systemverilog
// @brainsmith WEIGHT <interface_name>
// @brainsmith WEIGHT s_axis_weights
```

### Dimension Pragmas

#### BDIM (Block Dimension)
Defines tiling dimensions for block processing:
```systemverilog
// @brainsmith BDIM <interface_name> <dimension_list>
// @brainsmith BDIM s_axis_input [16]
// @brainsmith BDIM s_axis_input [8, 8]  // 2D tiling
```

#### SDIM (Stream Dimension)
Defines streaming dimensions:
```systemverilog
// @brainsmith SDIM <interface_name> <dimension_list>
// @brainsmith SDIM s_axis_input [256]
// @brainsmith SDIM s_axis_input [32, 32]  // 2D streaming
```

#### Shape Expressions
Both BDIM and SDIM support SHAPE expressions:
```systemverilog
// @brainsmith BDIM s_axis_input SHAPE [H/16, W/16, 16, 16]
// @brainsmith SDIM s_axis_weights SHAPE [K, C, 3, 3]
```

### Parameter Pragmas

#### ALIAS
Creates user-friendly parameter names:
```systemverilog
// @brainsmith ALIAS <rtl_param> <user_name>
// @brainsmith ALIAS STRM_WIDTH stream_width
```

#### AXILITE_PARAM
Maps parameters to AXI-Lite registers:
```systemverilog
// @brainsmith AXILITE_PARAM <param_name> OFFSET <hex_offset>
// @brainsmith AXILITE_PARAM threshold OFFSET 0x10
```

#### DERIVED_PARAMETER
Defines computed parameters:
```systemverilog
// @brainsmith DERIVED_PARAMETER <name> EXPRESSION <expr>
// @brainsmith DERIVED_PARAMETER output_size EXPRESSION "input_size * 2"
```

### Module Pragmas

#### TOP_MODULE
Selects which module to process in multi-module files:
```systemverilog
// @brainsmith TOP_MODULE <module_name>
// @brainsmith TOP_MODULE accelerator_top
```

### Relationship Pragmas

Define dependencies between interfaces:
```systemverilog
// @brainsmith RELATIONSHIP <interface1> DEPENDS_ON <interface2>
// @brainsmith RELATIONSHIP m_axis_output DEPENDS_ON s_axis_input
```

## Common Patterns

### Simple Processing Module

```systemverilog
// @brainsmith DATATYPE_CONSTRAINT s_axis_input UINT 8
// @brainsmith DATATYPE_CONSTRAINT m_axis_output UINT 8
// @brainsmith BDIM s_axis_input [16]
// @brainsmith BDIM m_axis_output [16]

module processor #(
    parameter DATA_WIDTH = 8
) (
    input ap_clk,
    input ap_rst_n,
    
    // Input stream
    input [DATA_WIDTH-1:0] s_axis_input_tdata,
    input s_axis_input_tvalid,
    output s_axis_input_tready,
    input s_axis_input_tlast,
    
    // Output stream
    output [DATA_WIDTH-1:0] m_axis_output_tdata,
    output m_axis_output_tvalid,
    input m_axis_output_tready,
    output m_axis_output_tlast
);
```

### Configurable Module with AXI-Lite

```systemverilog
// @brainsmith DATATYPE_CONSTRAINT s_axis_input UINT 8
// @brainsmith BDIM s_axis_input [32]
// @brainsmith AXILITE_PARAM threshold OFFSET 0x10
// @brainsmith AXILITE_PARAM mode OFFSET 0x14

module configurable #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 32
) (
    input ap_clk,
    input ap_rst_n,
    
    // AXI-Lite interface
    input [ADDR_WIDTH-1:0] s_axilite_araddr,
    input s_axilite_arvalid,
    output s_axilite_arready,
    output [31:0] s_axilite_rdata,
    output [1:0] s_axilite_rresp,
    output s_axilite_rvalid,
    input s_axilite_rready,
    
    input [ADDR_WIDTH-1:0] s_axilite_awaddr,
    input s_axilite_awvalid,
    output s_axilite_awready,
    input [31:0] s_axilite_wdata,
    input [3:0] s_axilite_wstrb,
    input s_axilite_wvalid,
    output s_axilite_wready,
    output [1:0] s_axilite_bresp,
    output s_axilite_bvalid,
    input s_axilite_bready,
    
    // Data interface
    input [DATA_WIDTH-1:0] s_axis_input_tdata,
    input s_axis_input_tvalid,
    output s_axis_input_tready
);
```

### Module with Weights

```systemverilog
// @brainsmith DATATYPE_CONSTRAINT s_axis_input UINT 8
// @brainsmith DATATYPE_CONSTRAINT s_axis_weights INT 8 SHAPE [3, 3]
// @brainsmith WEIGHT s_axis_weights
// @brainsmith BDIM s_axis_input [16, 16]

module convolution #(
    parameter DATA_WIDTH = 8,
    parameter WEIGHT_WIDTH = 8
) (
    input ap_clk,
    input ap_rst_n,
    
    // Input features
    input [DATA_WIDTH-1:0] s_axis_input_tdata,
    input s_axis_input_tvalid,
    output s_axis_input_tready,
    
    // Weights
    input [WEIGHT_WIDTH-1:0] s_axis_weights_tdata,
    input s_axis_weights_tvalid,
    output s_axis_weights_tready,
    
    // Output
    output [DATA_WIDTH-1:0] m_axis_output_tdata,
    output m_axis_output_tvalid,
    input m_axis_output_tready
);
```

## Multi-File RTL Support

The Kernel Integrator supports RTL designs that span multiple files, allowing you to organize complex designs with helper modules and shared utilities.

### Including Additional RTL Files

There are two ways to include additional RTL files:

#### 1. Using INCLUDE_RTL Pragma

Add pragma directives in your main RTL file to specify dependencies:

```systemverilog
// @brainsmith INCLUDE_RTL helper_modules.sv
// @brainsmith INCLUDE_RTL ../common/utilities.sv
// @brainsmith INCLUDE_RTL /absolute/path/to/shared.sv

module my_kernel (
    // ... module ports ...
);
```

#### 2. Using CLI Options

Include additional files via command line:

```bash
# Single additional file
kernel_integrator design.sv --include-rtl helper.sv

# Multiple files
kernel_integrator design.sv \
    --include-rtl helper_modules.sv \
    --include-rtl utilities.sv \
    --include-rtl shared_types.sv
```

### Path Resolution

RTL files are resolved in the following order:

1. **Absolute paths**: `/absolute/path/to/file.sv`
2. **Relative to source file**: `helper_modules.sv` (relative to main RTL file location)
3. **Relative to current directory**: `path/from/cwd/file.sv`

Example directory structure:
```
project/
├── rtl/
│   ├── kernels/
│   │   ├── adder.sv          # Main kernel
│   │   └── adder_utils.sv    # Helper for adder
│   └── common/
│       └── shared_types.sv   # Shared definitions
└── generated/                # Output directory
```

From `project/`, you can run:
```bash
kernel_integrator rtl/kernels/adder.sv -o generated/
```

With pragmas in `adder.sv`:
```systemverilog
// @brainsmith INCLUDE_RTL adder_utils.sv      # Found relative to adder.sv
// @brainsmith INCLUDE_RTL ../common/shared_types.sv  # Relative path
```

### Viewing Included Files

Use the `--info` flag to see all included files:

```bash
kernel_integrator design.sv --info

# Output includes:
# 📁 Included RTL Files (3):
#   - design.sv
#   - helper_modules.sv
#   - ../common/utilities.sv
```

### Generated Code Behavior

When generating the RTL backend, all included files are:

1. **Copied to output directory**: All dependencies are copied alongside generated files
2. **Added to IP integration**: TCL scripts include all files for synthesis
3. **Preserved in hierarchy**: Relative paths are maintained when possible

Example generated structure:
```
generated/
├── my_kernel.py           # Generated HWCustomOp
├── my_kernel_rtl.py       # Generated RTL backend
├── my_kernel_wrapper.v    # Generated wrapper
├── my_kernel.sv           # Copied main RTL
├── helper_modules.sv      # Copied dependencies
└── shared_types.sv        # Copied dependencies
```

### Best Practices for Multi-File RTL

1. **Use relative paths** in pragmas for portability
2. **Keep related modules together** in the same directory
3. **Create a common directory** for shared utilities
4. **Document dependencies** with comments
5. **Validate all files exist** before generation (default behavior)

### Example: Complex Multi-File Design

```systemverilog
// Main kernel file: conv2d_kernel.sv
// @brainsmith INCLUDE_RTL line_buffer.sv
// @brainsmith INCLUDE_RTL multiply_accumulate.sv
// @brainsmith INCLUDE_RTL ../common/axi_utilities.sv
// @brainsmith DATATYPE_CONSTRAINT s_axis_input UINT 8
// @brainsmith DATATYPE_CONSTRAINT s_axis_weights INT 8
// @brainsmith BDIM s_axis_input SHAPE [H/16, W/16, 16, 16]

module conv2d_kernel #(
    parameter DATA_WIDTH = 8,
    parameter WEIGHT_WIDTH = 8,
    parameter KERNEL_SIZE = 3
) (
    input ap_clk,
    input ap_rst_n,
    // ... interface ports ...
);
    
    // Instantiate helper modules
    line_buffer #(.WIDTH(DATA_WIDTH)) lb_inst (
        .clk(ap_clk),
        // ...
    );
    
    multiply_accumulate #(.DW(DATA_WIDTH), .WW(WEIGHT_WIDTH)) mac_inst (
        .clk(ap_clk),
        // ...
    );
    
endmodule
```

Generate with:
```bash
kernel_integrator rtl/kernels/conv2d_kernel.sv \
    --include-rtl rtl/common/debug_utils.sv \
    -o generated/conv2d/
```

## Troubleshooting

### Common Issues

#### "No module found in RTL file"
- Ensure your file contains valid SystemVerilog module syntax
- Check for syntax errors in the RTL
- Use `--validate` to check parsing

#### "Missing required control signals"
- Add `ap_clk` and `ap_rst_n` to your module
- Or use `--no-strict` for experimental modules

#### "Invalid pragma syntax"
- Check pragma format: `// @brainsmith TYPE args`
- Ensure no typos in pragma types
- Verify argument count and format

#### "Interface protocol not recognized"
- Follow naming conventions:
  - `s_axis_*` for AXI-Stream inputs
  - `m_axis_*` for AXI-Stream outputs
  - Include all required signals (tdata, tvalid, tready)

### Debug Mode

Use verbose mode for detailed information:
```bash
kernel_integrator design.sv --verbose
```

This shows:
- Parsing steps
- Detected interfaces
- Applied pragmas
- Validation results

### Validation Workflow

1. Start with `--validate`:
   ```bash
   kernel_integrator design.sv --validate
   ```

2. Check parsed metadata:
   ```bash
   kernel_integrator design.sv --info
   ```

3. Generate files:
   ```bash
   kernel_integrator design.sv
   ```

## Best Practices

1. **Always add pragmas** for DATATYPE and dimensions
2. **Use consistent naming** for interface signals
3. **Validate before generating** to catch issues early
4. **Start simple** - get basic module working before adding complexity
5. **Document pragmas** near the module declaration
6. **Test generated code** with FINN before deployment

## Advanced Usage

### Custom Output Organization

```bash
# Organize by module type
kernel_integrator nn_layers/*.sv -o ./generated/layers/
kernel_integrator preprocessing/*.sv -o ./generated/preproc/
```

### Batch Processing

```bash
# Process multiple files
for f in rtl/*.sv; do
    kernel_integrator "$f" -o ./generated/
done
```

### Integration with Build Systems

```makefile
# Makefile example
GENERATED_DIR = ./generated
RTL_SOURCES = $(wildcard rtl/*.sv)
GENERATED_PY = $(patsubst rtl/%.sv,$(GENERATED_DIR)/%.py,$(RTL_SOURCES))

$(GENERATED_DIR)/%.py: rtl/%.sv
	kernel_integrator $< -o $(GENERATED_DIR)

all: $(GENERATED_PY)
```