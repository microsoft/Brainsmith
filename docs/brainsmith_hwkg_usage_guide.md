# Brainsmith Hardware Kernel Generator - Usage Guide

## Quick Start

The Brainsmith Hardware Kernel Generator (HWKG) automatically creates FINN-compatible HWCustomOp classes from SystemVerilog RTL descriptions. This guide shows you how to use the system effectively.

## Prerequisites

### Required Dependencies
```bash
# Core dependencies
pip install jinja2 numpy tree-sitter

# FINN integration (optional for development)
pip install finn-base qonnx

# Development dependencies
pip install pytest logging
```

### SystemVerilog Grammar
The system requires a compiled tree-sitter SystemVerilog grammar:
```bash
# Grammar is included at: brainsmith/tools/hw_kernel_gen/rtl_parser/sv.so
# If you need to rebuild it, follow tree-sitter documentation
```

## Basic Usage

### 1. Command Line Interface

The simplest way to generate HWCustomOp classes:

```bash
python -m brainsmith.tools.hw_kernel_gen.hkg \
    examples/thresholding/thresholding_axi.sv \
    examples/thresholding/dummy_compiler_data.py \
    -o output_directory/
```

**Arguments:**
- `rtl_file`: Path to SystemVerilog RTL file
- `compiler_data`: Path to Python file with ONNX metadata
- `-o/--output-dir`: Output directory for generated files
- `--custom-doc`: Optional custom documentation file
- `--stop-after`: Stop after specific phase (for debugging)

### 2. Programmatic API

```python
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator

# Initialize generator
hkg = HardwareKernelGenerator(
    rtl_file_path="path/to/kernel.sv",
    compiler_data_path="path/to/compiler_data.py",
    output_dir="generated_output/"
)

# Generate complete package
generated_files = hkg.run()

# Or generate single HWCustomOp
hwcustomop_path = hkg.generate_auto_hwcustomop(
    template_path="path/to/template.j2",
    output_path="output/kernel_hwcustomop.py"
)
```

### 3. Individual Component Usage

```python
# Use RTL Parser standalone
from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser

parser = RTLParser(debug=True)
hw_kernel = parser.parse_file("kernel.sv")
print(f"Found {len(hw_kernel.interfaces)} interfaces")

# Use HWCustomOp Generator standalone
from brainsmith.tools.hw_kernel_gen.generators.hw_custom_op_generator import HWCustomOpGenerator

generator = HWCustomOpGenerator()
generated_code = generator.generate_hwcustomop(
    hw_kernel=hw_kernel,
    output_path="output.py",
    class_name="MyKernelHWCustomOp"
)
```

## RTL File Requirements

### Supported SystemVerilog Features

#### 1. Module Declaration (ANSI Style)
```systemverilog
module thresholding_axi #(
    parameter DATA_WIDTH = 32,
    parameter THRESHOLD = 128
)(
    // Global control
    input  logic ap_clk,
    input  logic ap_rst_n,
    
    // AXI-Stream input
    input  logic [DATA_WIDTH-1:0] s_axis_input_tdata,
    input  logic                  s_axis_input_tvalid,
    output logic                  s_axis_input_tready,
    
    // AXI-Stream output  
    output logic [DATA_WIDTH-1:0] m_axis_output_tdata,
    output logic                  m_axis_output_tvalid,
    input  logic                  m_axis_output_tready
);
```

#### 2. Required Interfaces

**Global Control** (mandatory):
```systemverilog
input  logic ap_clk,    // Clock
input  logic ap_rst_n   // Active-low reset
```

**AXI-Stream Interfaces** (at least one input and one output):
```systemverilog
// Input stream (slave)
input  logic [WIDTH-1:0] s_axis_<name>_tdata,
input  logic            s_axis_<name>_tvalid,
output logic            s_axis_<name>_tready,

// Output stream (master)
output logic [WIDTH-1:0] m_axis_<name>_tdata,
output logic            m_axis_<name>_tvalid,
input  logic            m_axis_<name>_tready
```

**AXI-Lite Configuration** (optional):
```systemverilog
// AXI-Lite slave interface
input  logic [ADDR_WIDTH-1:0] s_axilite_awaddr,
input  logic                  s_axilite_awvalid,
output logic                  s_axilite_awready,
// ... other AXI-Lite signals
```

### Enhanced Pragma System

#### 1. TDIM Pragmas (Tensor Dimensions)
```systemverilog
// @brainsmith TDIM s_axis_input [BATCH_SIZE, HEIGHT, WIDTH, CHANNELS]
// @brainsmith TDIM m_axis_output [BATCH_SIZE, HEIGHT, WIDTH, 1]
```

**Features:**
- Parameter evaluation: `[N, H*W, C]` where N, H, W, C are module parameters
- Validation: Ensures parameters exist and are valid
- Default fallback: Sensible defaults if pragma missing

#### 2. DATATYPE Pragmas (Datatype Constraints)
```systemverilog
// @brainsmith DATATYPE s_axis_input INT UINT 1 16
// Allows INT and UINT types, 1-16 bit widths
```

**Format:** `DATATYPE <interface> <base_types...> <min_bits> <max_bits>`

#### 3. WEIGHT Pragmas (Weight Interface Marking)
```systemverilog
// @brainsmith WEIGHT s_axis_weights
// Marks interface as containing model weights
```

#### 4. TOP_MODULE Pragma (Multi-module Files)
```systemverilog
// @brainsmith TOP_MODULE thresholding_axi
// Specifies which module to process in multi-module files
```

## Compiler Data File

Create a Python file with ONNX metadata and cost functions:

```python
# dummy_compiler_data.py

# ONNX model pattern (optional)
onnx_pattern = """
<pattern>
    <input name="input" />
    <operation type="Threshold" />
    <output name="output" />
</pattern>
"""

# ONNX metadata for tensor shapes
onnx_metadata = {
    "s_axis_input_shape": [1, 64, 64, 3],
    "s_axis_input_layout": "NHWC",
    "m_axis_output_shape": [1, 64, 64, 1], 
    "m_axis_output_layout": "NHWC"
}

# Cost functions for FINN optimization
def estimate_cycles(node):
    """Estimate execution cycles for the operation."""
    input_shape = node.input_shapes[0]
    return np.prod(input_shape)

def estimate_resources(node, fpga_part):
    """Estimate FPGA resource usage."""
    return {
        "BRAM": 2,
        "LUT": 150,
        "DSP": 0
    }
```

## Generated Output Files

### 1. HWCustomOp Class (`<kernel>_hwcustomop.py`)
```python
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata

class ThresholdingHWCustomOp(AutoHWCustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node, self.get_interface_metadata())
    
    def get_interface_metadata(self):
        return [
            InterfaceMetadata(
                name="s_axis_input",
                interface_type=DataflowInterfaceType.INPUT,
                # ... generated metadata
            ),
            # ... other interfaces
        ]
    
    def get_nodeattr_types(self):
        attrs = super().get_enhanced_nodeattr_types()
        attrs.update({
            "threshold": ("i", False, 128),
            # ... kernel-specific attributes
        })
        return attrs
```

### 2. RTL Backend (`<kernel>_rtlbackend.py`)
```python
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

class ThresholdingRTLBackend(RTLBackend):
    def __init__(self, onnx_node):
        super().__init__(onnx_node)
    
    def generate_hdl(self, model, fpgapart, clk):
        # Generated HDL wrapper and instantiation
        pass
```

### 3. Test Suite (`test_<kernel>.py`)
```python
import pytest
from <kernel>_hwcustomop import ThresholdingHWCustomOp

class TestThresholdingHWCustomOp:
    def test_node_creation(self):
        # Test ONNX node creation
        pass
    
    def test_datatype_constraints(self):
        # Test datatype validation
        pass
    
    def test_resource_estimation(self):
        # Test resource estimation
        pass
```

### 4. Documentation (`<kernel>_README.md`)
Comprehensive documentation including:
- Interface specifications
- Usage examples
- Resource estimation details
- Testing instructions

## Advanced Usage Patterns

### 1. Custom Template Development

Create custom Jinja2 templates for specialized output:

```python
# custom_generator.py
from brainsmith.tools.hw_kernel_gen.generators.hw_custom_op_generator import HWCustomOpGenerator

class CustomGenerator(HWCustomOpGenerator):
    def __init__(self):
        super().__init__(template_dir="path/to/custom/templates")
    
    def generate_custom_output(self, hw_kernel, output_path):
        template = self.jinja_env.get_template("custom_template.j2")
        # ... custom generation logic
```

### 2. Batch Processing

Process multiple kernels in a batch:

```python
import os
from pathlib import Path

def batch_generate(kernel_directory, output_directory):
    kernel_files = list(Path(kernel_directory).glob("*.sv"))
    
    for rtl_file in kernel_files:
        compiler_data = rtl_file.with_suffix(".py")
        if compiler_data.exists():
            hkg = HardwareKernelGenerator(
                rtl_file_path=str(rtl_file),
                compiler_data_path=str(compiler_data),
                output_dir=str(output_directory / rtl_file.stem)
            )
            
            try:
                generated_files = hkg.run()
                print(f"✓ Generated {len(generated_files)} files for {rtl_file.stem}")
            except Exception as e:
                print(f"✗ Failed to generate {rtl_file.stem}: {e}")
```

### 3. Integration with FINN Workflows

```python
# integration_example.py
from finn.core.modelwrapper import ModelWrapper
from <generated_kernel> import ThresholdingHWCustomOp

# Load ONNX model
model = ModelWrapper("model.onnx")

# Get thresholding nodes
threshold_nodes = model.get_nodes_by_op_type("Threshold")

for node in threshold_nodes:
    # Create HWCustomOp instance
    hw_op = ThresholdingHWCustomOp(node)
    
    # Configure parallelism
    hw_op.set_nodeattr("s_axis_input_parallel", 4)
    hw_op.set_nodeattr("s_axis_input_dtype", "UINT8")
    
    # Verify configuration
    hw_op.verify_node()
    
    # Get resource estimates
    resources = hw_op.derive_characteristic_fxns()
    print(f"Resources: {resources}")
```

## Debugging and Troubleshooting

### 1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific components
logger = logging.getLogger("brainsmith.tools.hw_kernel_gen")
logger.setLevel(logging.DEBUG)
```

### 2. Stop After Specific Phases
```bash
# Stop after RTL parsing to debug syntax issues
python -m brainsmith.tools.hw_kernel_gen.hkg kernel.sv data.py -o output/ --stop-after parse_rtl

# Stop after interface analysis to debug interface detection
python -m brainsmith.tools.hw_kernel_gen.hkg kernel.sv data.py -o output/ --stop-after analyze_interfaces
```

### 3. Common Issues and Solutions

#### RTL Parsing Errors
```
Error: Invalid SystemVerilog syntax near line 45, column 12
```
**Solution:** Check SystemVerilog syntax. Only ANSI-style port declarations are supported.

#### Interface Detection Issues
```
Error: No input AXI-Stream interface found
```
**Solution:** Ensure proper AXI-Stream signal naming (`s_axis_*`, `m_axis_*`) and required signals (`tdata`, `tvalid`, `tready`).

#### Missing Global Control
```
Error: Module 'kernel' is missing a valid Global Control interface
```
**Solution:** Add required global control signals (`ap_clk`, `ap_rst_n`).

#### Pragma Syntax Errors
```
Error: Invalid TDIM pragma syntax
```
**Solution:** Check pragma format: `// @brainsmith TDIM <interface> [dim1, dim2, ...]`

## Performance Optimization

### 1. Template Caching
```python
# Enable template caching for repeated generation
from jinja2 import Environment, FileSystemLoader

env = Environment(
    loader=FileSystemLoader("templates/"),
    cache_size=400,  # Cache compiled templates
    auto_reload=False  # Disable in production
)
```

### 2. Parallel Processing
```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def generate_kernel(args):
    rtl_file, compiler_data, output_dir = args
    hkg = HardwareKernelGenerator(rtl_file, compiler_data, output_dir)
    return hkg.run()

# Process multiple kernels in parallel
with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    futures = [executor.submit(generate_kernel, args) for args in kernel_args]
    results = [future.result() for future in futures]
```

### 3. Incremental Generation
```python
# Only regenerate if source files changed
import os
from pathlib import Path

def needs_regeneration(rtl_file, output_file):
    if not output_file.exists():
        return True
    
    rtl_mtime = os.path.getmtime(rtl_file)
    output_mtime = os.path.getmtime(output_file)
    
    return rtl_mtime > output_mtime
```

## Best Practices

### 1. RTL Design Guidelines
- Use consistent naming conventions for AXI interfaces
- Include comprehensive parameter documentation
- Add meaningful pragmas for optimization hints
- Test RTL thoroughly before generation

### 2. Pragma Usage
- Use TDIM pragmas for non-standard tensor layouts
- Apply DATATYPE constraints for mixed-precision designs
- Mark weight interfaces explicitly with WEIGHT pragma
- Document pragma meanings in RTL comments

### 3. Testing Strategy
- Test generated HWCustomOps with representative data
- Validate resource estimations against synthesis results
- Verify FINN integration with complete models
- Use continuous integration for regression testing

### 4. Documentation
- Maintain up-to-date compiler data files
- Document custom pragmas and their meanings
- Include usage examples in RTL header comments
- Keep generated documentation in version control

## Integration Examples

### 1. Continuous Integration Pipeline
```yaml
# .github/workflows/hwkg.yml
name: Hardware Kernel Generation
on: [push, pull_request]

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Generate HWCustomOps
      run: |
        python scripts/batch_generate.py \
          --input-dir kernels/ \
          --output-dir generated/ \
          --validate
    - name: Run tests
      run: pytest generated/tests/
```

### 2. Development Workflow Integration
```python
# scripts/dev_workflow.py
import os
import subprocess
from pathlib import Path

def development_workflow(kernel_path):
    """Complete development workflow for a kernel."""
    
    # 1. Generate HWCustomOp
    output_dir = Path("generated") / kernel_path.stem
    hkg = HardwareKernelGenerator(
        rtl_file_path=str(kernel_path),
        compiler_data_path=str(kernel_path.with_suffix(".py")),
        output_dir=str(output_dir)
    )
    
    generated_files = hkg.run()
    
    # 2. Run tests
    test_file = output_dir / f"test_{kernel_path.stem}.py"
    if test_file.exists():
        result = subprocess.run(["pytest", str(test_file), "-v"])
        if result.returncode != 0:
            print(f"Tests failed for {kernel_path.stem}")
            return False
    
    # 3. Generate documentation
    doc_file = output_dir / f"{kernel_path.stem}_README.md"
    print(f"Documentation generated: {doc_file}")
    
    # 4. Validate with FINN (if available)
    try:
        validate_finn_integration(generated_files["hwcustomop"])
    except ImportError:
        print("FINN not available, skipping integration validation")
    
    return True
```

This usage guide provides comprehensive information for effectively using the Brainsmith Hardware Kernel Generator in both simple and advanced scenarios.