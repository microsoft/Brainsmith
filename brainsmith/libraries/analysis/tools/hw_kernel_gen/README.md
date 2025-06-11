# Hardware Kernel Generator (HKG)

The Hardware Kernel Generator (HKG) is a tool for integrating custom RTL (SystemVerilog) implementations into the FINN compiler toolchain. It automates the creation of wrapper templates and integration files needed to make custom hardware kernels available for FINN's design space exploration and RTL synthesis pipeline.

## Overview

The HKG takes a SystemVerilog RTL implementation with custom compiler pragmas and generates the necessary files for FINN integration. Currently, the HKG focuses on generating parameterized RTL wrapper templates, with additional generators planned for future releases.

### Key Features

- **RTL Interface Analysis**: Automatically parses SystemVerilog files to extract module parameters, ports, and interface information
- **Template Generation**: Creates parameterized Verilog wrapper templates with placeholder substitution for FINN runtime configuration
- **Multi-Phase Pipeline**: Modular execution pipeline allowing debugging and analysis at each stage
- **Extensive Validation**: Built-in error checking and debugging output for troubleshooting integration issues

### Integration Pipeline

```
SystemVerilog RTL → RTL Parser → Interface Analysis → Wrapper Template Generation
     ↓                                                         ↓
Compiler Data ────────────────── (Future: HWCustomOp & RTLBackend Generation)
```

## Quick Start

### Basic Usage

```bash
# Generate RTL wrapper template from SystemVerilog source
python -m brainsmith.tools.hw_kernel_gen.hkg \
    path/to/module.sv \
    path/to/compiler_data.py \
    -o output_directory/
```

### Example

```bash
# Using the thresholding example
python -m brainsmith.tools.hw_kernel_gen.hkg \
    examples/thresholding/thresholding_axi.sv \
    examples/thresholding/dummy_compiler_data.py \
    -o generated_output/
```

## Command Line Interface

### Required Arguments

- `rtl_file`: Path to the SystemVerilog RTL source file (.sv)
- `compiler_data`: Path to Python file containing compiler data (currently placeholder format)
- `-o, --output-dir`: Directory where generated files will be saved

### Optional Arguments

- `-d, --custom-doc`: Path to Markdown file with custom documentation sections
- `--stop-after`: Stop execution after specified phase for debugging

#### Available Stop Points

- `parse_rtl`: Stop after RTL parsing
- `parse_compiler_data`: Stop after compiler data loading  
- `load_custom_documentation`: Stop after documentation loading
- `generate_rtl_template`: Stop after RTL template generation
- `generate_hw_custom_op`: Stop after HWCustomOp generation (placeholder)
- `generate_rtl_backend`: Stop after RTLBackend generation (placeholder)
- `generate_documentation`: Stop after documentation generation (placeholder)

## Input Requirements

### SystemVerilog RTL File

The RTL file must contain a SystemVerilog module with:

- **ANSI-style port declarations** (ports declared in module header)
- **Standard interface naming conventions** for automatic interface detection:
  - Global control: `ap_clk`, `ap_rst_n`, `ap_clk2x` (optional)
  - AXI-Stream: `*_TDATA`, `*_TVALID`, `*_TREADY`, `*_TLAST` (optional)
  - AXI-Lite: `s_axilite_*` signals for configuration interfaces

**Example Interface Structure:**
```systemverilog
module thresholding_axi #(
    int unsigned N,     // output precision
    int unsigned WI,    // input precision  
    int unsigned WT     // threshold precision
)(
    // Global Control
    input  logic ap_clk,
    input  logic ap_rst_n,
    
    // AXI-Lite Configuration
    input  logic                 s_axilite_AWVALID,
    output logic                 s_axilite_AWREADY,
    input  logic [ADDR_BITS-1:0] s_axilite_AWADDR,
    // ... additional AXI-Lite signals
    
    // AXI-Stream Input
    output logic s_axis_tready,
    input  logic s_axis_tvalid,
    input  logic [((PE*WI+7)/8)*8-1:0] s_axis_tdata,
    
    // AXI-Stream Output  
    input  logic m_axis_tready,
    output logic m_axis_tvalid,
    output logic [((PE*O_BITS+7)/8)*8-1:0] m_axis_tdata
);
```

### Compiler Data File

Currently a placeholder Python file that must contain basic structure for import validation:

```python
# Placeholder compiler data format
onnx_patterns = []

def cost_function(*args, **kwargs):
    return 1.0
```

*Note: The compiler data format will be fully defined in future releases during the parallelism refactor.*

## Generated Output

### RTL Wrapper Template

The primary output is a parameterized Verilog wrapper template (`{module_name}_wrapper.v`) that:

- **Preserves Original Parameters**: All module parameters are exposed with placeholder substitution
- **Maintains Interface Organization**: Groups and orders interfaces by type (Global Control, AXI-Stream, AXI-Lite)
- **Enables Runtime Configuration**: Uses `$PARAMETER_NAME$` placeholders for FINN runtime substitution

**Example Generated Template:**
```verilog
module $THRESHOLDING_AXI_WRAPPER_NAME$ #(
    parameter N = $N$,
    parameter WI = $WI$,
    parameter WT = $WT$
    // ... additional parameters
)(
    // --- Global Control ---
    input ap_clk,
    input ap_rst_n,
    
    // --- AXI-Lite (s_axilite) ---  
    input s_axilite_AWVALID,
    output s_axilite_AWREADY,
    // ... additional ports
);

    // Instantiate the wrapped kernel
    thresholding_axi #(
        .N(N),
        .WI(WI),
        .WT(WT)
    ) thresholding_axi_inst (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        // ... port connections
    );
endmodule
```

## Dependencies

The HKG requires the following Python packages (included in Brainsmith requirements):

- **tree-sitter**: SystemVerilog parsing via py-tree-sitter
- **Jinja2**: Template generation engine
- **pathlib**: Path handling utilities

## Architecture

### Core Components

| Component | Purpose |
|-----------|---------|
| **`hkg.py`** | Main orchestrator and CLI interface |
| **`rtl_parser/`** | SystemVerilog parsing and interface analysis |
| **`generators/rtl_template_generator.py`** | RTL wrapper template generation |
| **`templates/rtl_wrapper.v.j2`** | Jinja2 template for Verilog wrapper |

### Execution Phases

1. **RTL Parsing**: Extract module parameters, ports, and interface information
2. **Compiler Data Loading**: Import and validate compiler data file
3. **Documentation Loading**: Load optional custom documentation
4. **RTL Template Generation**: Generate parameterized wrapper template
5. **Integration File Generation**: Generate HWCustomOp and RTLBackend files *(planned)*
6. **Documentation Generation**: Auto-generate kernel documentation *(planned)*

## Programming Interface

### Python API Usage

```python
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator

# Initialize generator
hkg = HardwareKernelGenerator(
    rtl_file_path="path/to/module.sv",
    compiler_data_path="path/to/compiler_data.py", 
    output_dir="output/",
    custom_doc_path="optional_docs.md"  # Optional
)

# Generate RTL template only
generated_files = hkg.run(stop_after="generate_rtl_template")
print(f"RTL template: {generated_files['rtl_template']}")

# Or access parsed data directly
hw_kernel_data = hkg.get_parsed_rtl_data()
```

### Testing

The HKG includes comprehensive test coverage using the thresholding example:

```bash
# Run RTL template generation tests
python -m pytest tests/tools/hw_kernel_gen/test_rtl_template_generator.py -v
```

## Current Status

**Implemented:**
- ✅ RTL parsing and interface analysis
- ✅ RTL wrapper template generation
- ✅ Command-line interface
- ✅ Multi-phase execution pipeline

**Planned (Future Releases):**
- 🔄 HWCustomOp instance generation
- 🔄 RTLBackend instance generation  
- 🔄 Automated documentation generation
- 🔄 Enhanced pragma support
- 🔄 Compiler data format specification

The HKG currently focuses on RTL template generation as the foundation for FINN integration, with additional generators to be implemented based on FINN compiler requirements and parallelism architecture decisions.