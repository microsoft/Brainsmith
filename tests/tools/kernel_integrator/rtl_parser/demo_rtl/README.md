# RTL Parser Demo Files

This directory contains example SystemVerilog files demonstrating various features of the Brainsmith RTL Parser. Each file showcases specific pragma types, auto-linking capabilities, and best practices.

## Demo Files Overview

### Basic Examples
- `01_basic_module.sv` - Minimal RTL module with basic interfaces
- `02_all_interface_types.sv` - Demonstrates all interface types (INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL)
- `03_datatype_pragmas.sv` - DATATYPE and DATATYPE_PARAM pragma usage

### Auto-linking Examples
- `04_auto_linking_datatypes.sv` - Automatic datatype parameter detection
- `05_auto_linking_dimensions.sv` - BDIM/SDIM auto-linking with naming conventions
- `06_indexed_dimensions.sv` - Multi-dimensional parameters with indexed naming

### Advanced Features
- `07_alias_pragmas.sv` - ALIAS pragma for user-friendly parameter names
- `08_derived_parameters.sv` - DERIVED_PARAMETER pragma for computed values
- `09_weight_interfaces.sv` - WEIGHT pragma and weight-specific features
- `10_multi_interface.sv` - Modules with multiple interfaces of the same type

### Complex Examples
- `11_interface_relationships.sv` - Demonstrating interface dimension relationships
- `12_complete_accelerator.sv` - Full-featured accelerator combining all pragma types
- `13_edge_cases.sv` - Common pitfalls and edge cases

## Running the Demos

To parse any demo file:

```bash
cd /path/to/brainsmith-2
./smithy exec "python -m brainsmith.tools.kernel_integrator tests/tools/hw_kernel_gen/rtl_parser/demo_rtl/<filename> -o output/"
```

Or use the RTL parser demo script to see the parsed metadata:

```bash
./smithy exec "python tests/tools/hw_kernel_gen/rtl_parser/rtl_parser_demo.py tests/tools/hw_kernel_gen/rtl_parser/demo_rtl/<filename>"
```

## Key Concepts Demonstrated

### Interface Types
- **INPUT**: Data input streams (supports BDIM, SDIM, datatypes)
- **OUTPUT**: Data output streams (supports BDIM, datatypes, no SDIM)
- **WEIGHT**: Weight/parameter inputs (supports BDIM, SDIM, datatypes)
- **CONFIG**: AXI-Lite configuration (supports datatypes only)
- **CONTROL**: Clock/reset signals (no parameterization)

### Pragma Precedence
1. Explicit pragmas (highest priority)
2. Single parameter auto-linking
3. Indexed parameter auto-linking
4. Default parameter names (lowest priority)

### Naming Conventions
- Datatype: `<interface>_WIDTH`, `<interface>_SIGNED`
- Single dimension: `<interface>_BDIM`, `<interface>_SDIM`
- Indexed dimension: `<interface>_BDIM0`, `<interface>_BDIM1`, etc.

## Best Practices

1. **Use consistent naming**: Follow the naming conventions for auto-linking
2. **Be explicit when needed**: Use pragmas to override auto-linking behavior
3. **Document intent**: Add comments explaining pragma usage
4. **Test your patterns**: Verify parser output matches expectations
5. **Handle all interface types correctly**: Respect interface type restrictions

## Common Issues

1. **SDIM on OUTPUT**: SDIM only applies to INPUT and WEIGHT interfaces
2. **Dimensions on CONFIG**: CONFIG interfaces don't support BDIM/SDIM
3. **Parameterizing CONTROL**: Clock/reset signals can't be parameterized
4. **Name conflicts**: Ensure parameter names don't conflict across interfaces
5. **Missing indices**: Gaps in indexed parameters are filled with "1"