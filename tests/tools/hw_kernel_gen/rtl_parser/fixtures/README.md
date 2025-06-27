# RTL Parser Test Fixtures

This directory contains SystemVerilog fixture files for testing the RTL parser. The fixtures are organized by category to support different testing scenarios.

## Directory Structure

```
fixtures/
├── minimal/           # Minimal valid/invalid modules
├── protocols/         # Protocol-specific examples (AXI-Stream, AXI-Lite)
├── pragmas/          # Pragma demonstration files
├── complex/          # Complex multi-feature examples
└── errors/           # Error cases for validation testing
```

## Fixture Categories

### Minimal (`minimal/`)

Basic modules for fundamental parsing tests:

- **`strict_minimal.sv`**: Minimal module that passes strict validation
  - Has global control interface (ap_clk, ap_rst_n)
  - One input with BDIM/SDIM parameters
  - One output with BDIM parameter
  - Use for testing strict mode compliance

- **`non_strict_minimal.sv`**: Simple module for non-strict tests
  - Basic clk/rst (not standard names)
  - Simple ports (not AXI-Stream)
  - Use with `RTLParser(strict=False)`

### Protocols (`protocols/`)

Standard protocol implementations:

- **`axi_stream_complete.sv`**: Full AXI-Stream with all signals
  - TDATA, TVALID, TREADY, TLAST, TKEEP, TUSER
  - Multiple interfaces showing different patterns
  - Proper BDIM/SDIM parameters

- **`axi_stream_minimal.sv`**: Minimal AXI-Stream (only required signals)
  - Just TDATA, TVALID, TREADY
  - Multiple naming patterns
  - Shows auto-linking behavior

- **`axi_lite_complete.sv`**: Complete AXI-Lite slave
  - All five channels (AW, W, B, AR, R)
  - Configuration registers
  - Combined with data interfaces

- **`global_control.sv`**: Control signal patterns
  - Standard ap_clk/ap_rst_n
  - Comments show alternative patterns

### Pragmas (`pragmas/`)

Comprehensive pragma demonstrations:

- **`all_pragmas.sv`**: Every pragma type in one module
  - TOP_MODULE, ALIAS, DERIVED_PARAMETER
  - BDIM/SDIM (single and multi-dimensional)
  - DATATYPE, DATATYPE_PARAM, WEIGHT
  - RELATIONSHIP, AXILITE_PARAM

- **`bdim_sdim_variations.sv`**: Dimension parameter patterns
  - Single parameters (name_BDIM)
  - Indexed parameters (name_BDIM0, name_BDIM1)
  - List pragmas with multiple dimensions
  - Mixed singleton dimensions

- **`datatype_variations.sv`**: Datatype constraints
  - Basic DATATYPE (base type, min/max bits)
  - DATATYPE_PARAM mappings
  - Internal datatypes (accumulator, threshold)
  - Multiple constraints per interface

- **`relationship_examples.sv`**: Interface relationships
  - EQUAL relationships
  - DEPENDENT with scaling
  - DIVISIBLE constraints
  - Complex multi-interface dependencies

### Complex Examples (`complex/`)

Real-world scenarios:

- **`multi_interface.sv`**: Multiple interfaces of each type
  - Multiple inputs with different properties
  - Multiple outputs and weights
  - AXI-Lite control
  - Complex parameter relationships

- **`parametric_dimensions.sv`**: Computed parameters
  - DERIVED_PARAMETER usage
  - Parameters computed from others
  - Conditional parameter usage
  - Complex dimension calculations

- **`hierarchical_module.sv`**: Module hierarchy
  - Multiple modules in one file
  - TOP_MODULE pragma selection
  - Module instantiation
  - Parameter passing through hierarchy

### Error Cases (`errors/`)

For testing error handling:

- **`missing_control.sv`**: No global control interface
  - Has clk/rst but wrong names
  - Should fail: "missing valid Global Control interface"

- **`missing_bdim.sv`**: AXI-Stream without BDIM
  - Has interfaces but missing required parameters
  - Should fail: "missing required BDIM parameter"

- **`missing_sdim_input.sv`**: Input/weight without SDIM
  - Shows that inputs/weights need SDIM
  - Outputs only need BDIM

- **`invalid_pragmas.sv`**: Various pragma errors
  - Unknown pragma types
  - Missing arguments
  - Invalid syntax
  - Should generate warnings but not crash

- **`no_interfaces.sv`**: No AXI-Stream interfaces
  - Has ports but not standard protocols
  - Should fail: "at least one interface required"

## Usage Examples

### Basic Parsing Test
```python
from pathlib import Path
from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser

# Load fixture
fixture_path = Path("fixtures/minimal/strict_minimal.sv")
with open(fixture_path) as f:
    rtl = f.read()

# Parse with strict validation
parser = RTLParser()  # strict=True by default
result = parser.parse(rtl, str(fixture_path))
```

### Non-Strict Parsing
```python
# For minimal RTL without full requirements
parser = RTLParser(strict=False)
fixture_path = Path("fixtures/minimal/non_strict_minimal.sv")
```

### Using Test Utilities
```python
from tests.tools.hw_kernel_gen.rtl_parser.utils.rtl_builder import StrictRTLBuilder

# Generate compliant RTL programmatically
rtl = (StrictRTLBuilder()
       .module("test")
       .add_stream_input("s_axis_in", bdim_value="32", sdim_value="512")
       .add_stream_output("m_axis_out", bdim_value="32")
       .build())
```

### Error Testing
```python
# Test specific validation errors
fixture_path = Path("fixtures/errors/missing_control.sv")
with open(fixture_path) as f:
    rtl = f.read()

try:
    parser = RTLParser()  # strict=True
    result = parser.parse(rtl, str(fixture_path))
except Exception as e:
    assert "Global Control interface" in str(e)
```

## Adding New Fixtures

When adding new fixtures:

1. Choose appropriate category directory
2. Include header comment explaining the fixture's purpose
3. For strict fixtures, ensure:
   - Global control with ap_clk/ap_rst_n
   - Proper BDIM/SDIM parameters
   - At least one input and output
4. For error fixtures, document expected error
5. Update this README with the new fixture

## Notes

- Fixture files use `.sv` extension for SystemVerilog
- All fixtures should be valid SystemVerilog syntax (even error cases)
- Comments in fixtures explain key features and usage
- Use fixtures instead of inline RTL strings in tests when possible