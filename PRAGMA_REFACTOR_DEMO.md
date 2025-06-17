# Pragma System Refactoring Demonstration

This directory contains demonstration scripts that showcase the refactored pragma system for the Brainsmith Hardware Kernel Generator.

## What Was Refactored

### 🏗️ **Architecture Changes**

1. **InterfacePragma Base Class**: Replaced `InterfaceNameMatcher` mixin with proper `InterfacePragma` inheritance
2. **Centralized Pragma Application**: All interface pragmas applied in one sweep via `PragmaHandler.apply_interface_pragmas()`
3. **Comprehensive Validation**: Extracted validation logic into `_validate_interface_metadata()`
4. **Consolidated Parser Methods**: Removed duplicate `_initial_parse_string()` method
5. **Clean Separation**: InterfaceBuilder (AST) → PragmaHandler (pragmas) → Parser (validation)

### 📁 **Files Modified**

- `brainsmith/tools/hw_kernel_gen/rtl_parser/data.py`
- `brainsmith/tools/hw_kernel_gen/rtl_parser/pragma.py`
- `brainsmith/tools/hw_kernel_gen/rtl_parser/parser.py`
- `brainsmith/tools/hw_kernel_gen/rtl_parser/interface_builder.py`

## Demo Scripts

### 🚀 **Quick Demo**

```bash
./smithy exec "python demo_pragma_refactor.py"
```

Shows a quick overview of the refactoring and validates it works with the test file.

### 🧪 **Comprehensive Test**

```bash
./smithy exec "python test_pragma_system.py"
```

Demonstrates the full pragma system with detailed KernelMetadata visualization.

### 🔍 **Test Specific File**

```bash
./smithy exec "python test_pragma_system.py --file your_rtl_file.sv"
./smithy exec "python test_pragma_system.py --file your_rtl_file.sv --debug"
```

## Key Features Demonstrated

### ✅ **Interface Pragma Application**

- **BDIM Pragmas**: `@brainsmith BDIM <interface> <param> [SHAPE=...] [RINDEX=...]`
- **SDIM Pragmas**: `@brainsmith SDIM <interface> <param>`
- **DATATYPE Pragmas**: `@brainsmith DATATYPE <interface> <type> <min> <max>`
- **WEIGHT Pragmas**: `@brainsmith WEIGHT <interface>`
- **DATATYPE_PARAM Pragmas**: `@brainsmith DATATYPE_PARAM <interface> <prop> <param>`

### 🏷️ **Flexible Interface Naming**

Users can name interfaces with any prefix:
```systemverilog
// These all work now:
input wire potato_input_TDATA,     // pragma: potato_input
input wire carrot_weights_TDATA,   // pragma: carrot_weights  
input wire my_custom_data_TDATA,   // pragma: my_custom_data
```

### 🔍 **Comprehensive Validation**

- Parameter existence checking
- Interface type consistency
- BDIM/SDIM parameter linkage
- Shape parameter validation
- Datatype parameter availability

### 📊 **Rich Metadata Output**

The scripts output detailed KernelMetadata including:
- Module parameters with defaults
- Pragma parsing results
- Interface metadata with chunking strategies
- Parameter linkage mappings
- Validation warnings

## Example Output

```
🔍 KERNEL METADATA: test_new_format
============================================================

📁 Source File: test_new_pragma_format.sv
⚠️  Warnings: 0

🔧 PARAMETERS (9):
   • INPUT0_WIDTH = 8
   • SIGNED_INPUT0 = 0
   • C = 64
   • PE = 4

📝 PRAGMAS (7):
   • Line 3: @brainsmith bdim s_axis_input0 INPUT0_BDIM SHAPE=[C,PE] RINDEX=0

🔌 INTERFACES (3):
   [2] s_axis_input0 (input)
       Chunking: [C, PE] (rindex=0)  ← Applied from BDIM pragma
       Parameter Linkage:
         • width → INPUT0_WIDTH
         • bdim → s_axis_input0_BDIM
```

## Architecture Benefits

- **🔄 Scalable**: Easy to add new interface pragmas without touching RTLParser
- **🧹 Clean**: No code duplication, unified parsing methods
- **🎯 Focused**: Clear separation of concerns
- **🔍 Validated**: Comprehensive parameter and interface validation
- **🏷️ Flexible**: Support for any interface naming convention
- **⚡ Efficient**: Single-pass pragma application

## Integration

The refactored system is fully backward compatible and integrates seamlessly with:
- Template generation pipeline
- FINN HWCustomOp creation
- RTL backend generation
- Existing pragma syntax

All existing functionality is preserved while providing a much cleaner and more maintainable architecture.