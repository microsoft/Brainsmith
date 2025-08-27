# Kernel Integrator RTL Guide

This guide covers everything you need to know about writing SystemVerilog RTL that works seamlessly with the Kernel Integrator, including pragma syntax, formatting requirements, and best practices.

## Table of Contents
- [RTL Requirements](#rtl-requirements)
- [Interface Naming Conventions](#interface-naming-conventions)
- [Pragma Reference](#pragma-reference)
- [Common Patterns](#common-patterns)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## RTL Requirements

### Mandatory Elements

Every RTL module must include:

1. **Control Signals**
   ```systemverilog
   input wire ap_clk,      // System clock
   input wire ap_rst_n,    // Active-low reset
   ```

2. **At least one input and one output interface**

3. **Valid SystemVerilog syntax** (SV-2017 compatible)

### Module Structure

```systemverilog
// Pragmas go here (before module declaration)
// @brainsmith DATATYPE s_axis_input UINT 8

module my_kernel #(
    // Parameters with defaults
    parameter DATA_WIDTH = 8,
    parameter FEATURE_SIZE = 16
) (
    // Control signals (required)
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Interface ports
    // ... your interfaces here
);
    // Implementation
endmodule
```

## Interface Naming Conventions

The Kernel Integrator automatically detects interface protocols based on naming patterns:

### AXI-Stream Interfaces

| Direction | Prefix | Example | Description |
|-----------|--------|---------|-------------|
| Input | `s_axis_` | `s_axis_data` | Slave/input stream |
| Output | `m_axis_` | `m_axis_result` | Master/output stream |

Required signals:
- `<prefix>_tdata` - Data channel
- `<prefix>_tvalid` - Data valid signal
- `<prefix>_tready` - Ready signal

Optional signals:
- `<prefix>_tlast` - Last transfer in burst
- `<prefix>_tkeep` - Byte enable
- `<prefix>_tstrb` - Byte strobe
- `<prefix>_tid` - Stream ID
- `<prefix>_tdest` - Destination ID
- `<prefix>_tuser` - User-defined sideband

### AXI-Lite Interface

| Prefix | Example | Description |
|--------|---------|-------------|
| `s_axilite_` | `s_axilite` | Memory-mapped configuration |

Can be read-only, write-only, or full:

```systemverilog
// Full AXI-Lite (read + write)
input  [31:0] s_axilite_araddr,
input         s_axilite_arvalid,
output        s_axilite_arready,
output [31:0] s_axilite_rdata,
output [1:0]  s_axilite_rresp,
output        s_axilite_rvalid,
input         s_axilite_rready,

input  [31:0] s_axilite_awaddr,
input         s_axilite_awvalid,
output        s_axilite_awready,
input  [31:0] s_axilite_wdata,
input  [3:0]  s_axilite_wstrb,
input         s_axilite_wvalid,
output        s_axilite_wready,
output [1:0]  s_axilite_bresp,
output        s_axilite_bvalid,
input         s_axilite_bready
```

## Pragma Reference

### Pragma Syntax

```systemverilog
// @brainsmith <PRAGMA_TYPE> <arguments>
```

- Must start with `// @brainsmith`
- Case-sensitive pragma types
- Arguments vary by pragma type
- Place before module declaration for best organization

### Complete Pragma Reference

#### DATATYPE_CONSTRAINT
Constrains data types for an interface:

```systemverilog
// @brainsmith DATATYPE_CONSTRAINT <interface> <sign> <width> [SHAPE <dimensions>]

// Examples:
// @brainsmith DATATYPE_CONSTRAINT s_axis_input UINT 8
// @brainsmith DATATYPE_CONSTRAINT m_axis_output INT 16
// @brainsmith DATATYPE_CONSTRAINT s_axis_weights UINT 8 SHAPE [3, 3, 64, 128]
```

- **Sign**: `UINT` or `INT`
- **Width**: Bit width (1-64)
- **SHAPE**: Optional tensor shape

#### DATATYPE
Links RTL parameters to interface properties:

```systemverilog
// @brainsmith DATATYPE <interface> <property> <parameter>

// Examples:
// @brainsmith DATATYPE s_axis_input WIDTH DATA_WIDTH
// @brainsmith DATATYPE m_axis_output WIDTH OUTPUT_WIDTH
```

Properties:
- `WIDTH`: Links to data type bit width

#### BDIM (Block Dimension)
Defines tiling dimensions for block-based processing:

```systemverilog
// @brainsmith BDIM <interface> <dimensions>
// @brainsmith BDIM <interface> SHAPE <expression>

// Examples:
// @brainsmith BDIM s_axis_input [16]
// @brainsmith BDIM s_axis_input [8, 8]
// @brainsmith BDIM s_axis_input SHAPE [H/16, W/16, 16, 16]
```

#### SDIM (Stream Dimension)
Defines streaming dimensions:

```systemverilog
// @brainsmith SDIM <interface> <dimensions>
// @brainsmith SDIM <interface> SHAPE <expression>

// Examples:
// @brainsmith SDIM s_axis_input [256]
// @brainsmith SDIM s_axis_input [32, 32]
// @brainsmith SDIM s_axis_input SHAPE [BATCH, CHANNELS, HEIGHT, WIDTH]
```

#### WEIGHT
Marks an interface as containing trained parameters:

```systemverilog
// @brainsmith WEIGHT <interface>

// Example:
// @brainsmith WEIGHT s_axis_weights
```

#### ALIAS
Creates user-friendly parameter names:

```systemverilog
// @brainsmith ALIAS <rtl_parameter> <user_name>

// Examples:
// @brainsmith ALIAS STRM_W stream_width
// @brainsmith ALIAS THR threshold_value
```

#### AXILITE_PARAM
Maps parameters to AXI-Lite register offsets:

```systemverilog
// @brainsmith AXILITE_PARAM <parameter> OFFSET <hex_offset>

// Examples:
// @brainsmith AXILITE_PARAM threshold OFFSET 0x10
// @brainsmith AXILITE_PARAM mode OFFSET 0x14
// @brainsmith AXILITE_PARAM scale_factor OFFSET 0x18
```

#### DERIVED_PARAMETER
Defines computed parameters:

```systemverilog
// @brainsmith DERIVED_PARAMETER <name> EXPRESSION <expr>

// Examples:
// @brainsmith DERIVED_PARAMETER output_size EXPRESSION "input_size * 2"
// @brainsmith DERIVED_PARAMETER total_pixels EXPRESSION "height * width"
```

#### TOP_MODULE
Selects target module in multi-module files:

```systemverilog
// @brainsmith TOP_MODULE <module_name>

// Example:
// @brainsmith TOP_MODULE accelerator_core
```

#### RELATIONSHIP
Defines interface dependencies:

```systemverilog
// @brainsmith RELATIONSHIP <interface1> DEPENDS_ON <interface2>

// Example:
// @brainsmith RELATIONSHIP m_axis_output DEPENDS_ON s_axis_input
```

## Common Patterns

### Basic Stream Processing

```systemverilog
// @brainsmith DATATYPE_CONSTRAINT s_axis_input UINT 8
// @brainsmith DATATYPE_CONSTRAINT m_axis_output UINT 8
// @brainsmith BDIM s_axis_input [16]
// @brainsmith BDIM m_axis_output [16]
// @brainsmith RELATIONSHIP m_axis_output DEPENDS_ON s_axis_input

module stream_processor #(
    parameter DATA_WIDTH = 8
) (
    // Control
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Input stream
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    input wire s_axis_input_tlast,
    
    // Output stream  
    output reg [DATA_WIDTH-1:0] m_axis_output_tdata,
    output reg m_axis_output_tvalid,
    input wire m_axis_output_tready,
    output reg m_axis_output_tlast
);
```

### Parameterized Processing

```systemverilog
// @brainsmith DATATYPE s_axis_input UINT 8
// @brainsmith DATATYPE m_axis_output UINT 1  
// @brainsmith DATATYPE_PARAM s_axis_input WIDTH IN_WIDTH
// @brainsmith DATATYPE_PARAM m_axis_output WIDTH OUT_WIDTH
// @brainsmith ALIAS IN_WIDTH input_width
// @brainsmith ALIAS OUT_WIDTH output_width

module parameterized #(
    parameter IN_WIDTH = 8,
    parameter OUT_WIDTH = 1
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    input wire [IN_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [OUT_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);
```

### Multi-Stream with Weights

```systemverilog
// @brainsmith DATATYPE s_axis_input UINT 8 SHAPE [1, 32, 32]
// @brainsmith DATATYPE s_axis_weights INT 8 SHAPE [3, 3, 32, 64]
// @brainsmith DATATYPE m_axis_output UINT 8 SHAPE [1, 30, 30]
// @brainsmith WEIGHT s_axis_weights
// @brainsmith BDIM s_axis_input [1, 16, 16]
// @brainsmith BDIM m_axis_output [1, 15, 15]

module conv2d #(
    parameter DATA_WIDTH = 8,
    parameter WEIGHT_WIDTH = 8,
    parameter IN_CHANNELS = 32,
    parameter OUT_CHANNELS = 64
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    // Feature input
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    // Weight input
    input wire [WEIGHT_WIDTH-1:0] s_axis_weights_tdata,
    input wire s_axis_weights_tvalid,
    output wire s_axis_weights_tready,
    
    // Feature output
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);
```

### Configurable via AXI-Lite

```systemverilog
// @brainsmith DATATYPE s_axis_input UINT 8
// @brainsmith AXILITE_PARAM threshold OFFSET 0x10
// @brainsmith AXILITE_PARAM enable OFFSET 0x14
// @brainsmith ALIAS threshold threshold_value

module configurable_filter #(
    parameter DATA_WIDTH = 8
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    // AXI-Lite configuration
    input wire [31:0] s_axilite_awaddr,
    input wire s_axilite_awvalid,
    output wire s_axilite_awready,
    input wire [31:0] s_axilite_wdata,
    input wire [3:0] s_axilite_wstrb,
    input wire s_axilite_wvalid,
    output wire s_axilite_wready,
    output wire [1:0] s_axilite_bresp,
    output wire s_axilite_bvalid,
    input wire s_axilite_bready,
    
    input wire [31:0] s_axilite_araddr,
    input wire s_axilite_arvalid,
    output wire s_axilite_arready,
    output wire [31:0] s_axilite_rdata,
    output wire [1:0] s_axilite_rresp,
    output wire s_axilite_rvalid,
    input wire s_axilite_rready,
    
    // Data path
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    output wire [DATA_WIDTH-1:0] m_axis_output_tdata,
    output wire m_axis_output_tvalid,
    input wire m_axis_output_tready
);
```

## Examples

### Example 1: Simple Threshold

```systemverilog
// Threshold module that binarizes input data
// @brainsmith DATATYPE s_axis_input UINT 8
// @brainsmith DATATYPE m_axis_output UINT 1
// @brainsmith BDIM s_axis_input [16]
// @brainsmith BDIM m_axis_output [16]
// @brainsmith ALIAS THRESHOLD threshold_value

module threshold #(
    parameter DATA_WIDTH = 8,
    parameter THRESHOLD = 128
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    // 8-bit input stream
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    
    // 1-bit output stream
    output reg m_axis_output_tdata,
    output reg m_axis_output_tvalid,
    input wire m_axis_output_tready
);
    // Simple flow control
    assign s_axis_input_tready = m_axis_output_tready;
    
    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            m_axis_output_tdata <= 1'b0;
            m_axis_output_tvalid <= 1'b0;
        end else if (m_axis_output_tready) begin
            m_axis_output_tvalid <= s_axis_input_tvalid;
            m_axis_output_tdata <= (s_axis_input_tdata >= THRESHOLD) ? 1'b1 : 1'b0;
        end
    end
endmodule
```

### Example 2: Pooling Operation

```systemverilog
// Max pooling with 2x2 window
// @brainsmith DATATYPE s_axis_input UINT 8
// @brainsmith DATATYPE m_axis_output UINT 8  
// @brainsmith BDIM s_axis_input SHAPE [H/2, W/2, 2, 2]
// @brainsmith BDIM m_axis_output SHAPE [H/2, W/2]
// @brainsmith SDIM s_axis_input SHAPE [H, W]
// @brainsmith SDIM m_axis_output SHAPE [H/2, W/2]

module maxpool_2x2 #(
    parameter DATA_WIDTH = 8,
    parameter IMG_WIDTH = 32,
    parameter IMG_HEIGHT = 32
) (
    input wire ap_clk,
    input wire ap_rst_n,
    
    input wire [DATA_WIDTH-1:0] s_axis_input_tdata,
    input wire s_axis_input_tvalid,
    output wire s_axis_input_tready,
    input wire s_axis_input_tlast,
    
    output reg [DATA_WIDTH-1:0] m_axis_output_tdata,
    output reg m_axis_output_tvalid,
    input wire m_axis_output_tready,
    output reg m_axis_output_tlast
);
    // Pooling implementation
endmodule
```

## Troubleshooting

### Common Issues and Solutions

#### "Interface protocol not recognized"

**Problem**: Interface isn't detected properly
**Solution**: Check naming convention:
```systemverilog
// ❌ Wrong
input [7:0] input_data,
input input_valid,

// ✅ Correct  
input [7:0] s_axis_input_tdata,
input s_axis_input_tvalid,
```

#### "Missing required signals"

**Problem**: AXI protocol incomplete
**Solution**: Add all required signals:
```systemverilog
// ❌ Missing tready
input [7:0] s_axis_data_tdata,
input s_axis_data_tvalid,

// ✅ Complete
input [7:0] s_axis_data_tdata,
input s_axis_data_tvalid,
output s_axis_data_tready,
```

#### "Invalid pragma syntax"

**Problem**: Pragma not parsed correctly
**Solution**: Check exact syntax:
```systemverilog
// ❌ Wrong
// @Brainsmith DATATYPE input UINT 8     // Wrong case
// @ brainsmith DATATYPE input UINT 8    // Extra space
// @brainsmith DATATYPE input UINT8       // Missing space

// ✅ Correct
// @brainsmith DATATYPE s_axis_input UINT 8
```

#### "Parameter not linked to interface"

**Problem**: Width parameter not associated
**Solution**: Use explicit linking:
```systemverilog
// @brainsmith DATATYPE_PARAM s_axis_input WIDTH MY_WIDTH

module example #(
    parameter MY_WIDTH = 8  // Now linked
) (
    input [MY_WIDTH-1:0] s_axis_input_tdata,
    // ...
);
```

### Best Practices

1. **Organize Pragmas**: Place all pragmas together before module
2. **Consistent Naming**: Use clear, descriptive interface names
3. **Document Parameters**: Add comments explaining parameter purposes
4. **Validate Early**: Use `--validate` flag during development
5. **Start Simple**: Get basic version working before adding complexity

### Validation Checklist

Before running Kernel Integrator:

- [ ] Module has `ap_clk` and `ap_rst_n`
- [ ] All interfaces follow naming conventions
- [ ] Required signals present for each protocol
- [ ] Pragmas have correct syntax
- [ ] Parameters have meaningful names
- [ ] At least one input and output interface

### Parameter Width Matching

Ensure parameter widths match port declarations:

```systemverilog
// @brainsmith DATATYPE_PARAM s_axis_data WIDTH DATA_W

module example #(
    parameter DATA_W = 8
) (
    input [DATA_W-1:0] s_axis_data_tdata,  // Uses DATA_W
    // ...
);
```