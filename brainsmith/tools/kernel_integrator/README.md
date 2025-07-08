# Kernel Integrator (KI)

The Kernel Integrator automatically creates FINN-compatible Python wrappers from SystemVerilog RTL kernels using pragma-driven analysis.

## Key Features

- **Explicit Code Generation**: Produces human-readable, self-documenting code
- **Pragma-Driven Analysis**: Uses SystemVerilog comments to extract kernel metadata
- **Clean Architecture**: Modular design with clear separation of concerns
- **FINN Integration**: Automatic generation of HWCustomOp and RTL backend classes

## Quick Start

```bash
# Basic usage
./smithy exec "python -m brainsmith.tools.kernel_integrator <rtl_file> -o <output_dir>"

# Example
./smithy exec "python -m brainsmith.tools.kernel_integrator tests/test_kernel_e2e.sv -o output/"
```

## Generated Files

For each kernel, KI generates three complementary files:

1. **`{kernel}_hw_custom_op.py`** - FINN HWCustomOp subclass with explicit node attributes
2. **`{kernel}_rtl.py`** - FINN RTL backend with explicit parameter resolution  
3. **`{kernel}_wrapper.v`** - SystemVerilog wrapper with parameter substitution

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for complete documentation of the unified kernel generation architecture, including:

- System architecture and data flow
- Code generation pipeline details
- Template system and explicit parameter binding
- Performance characteristics and benefits
- Migration notes and breaking changes

## Directory Structure

```
brainsmith/tools/hw_kernel_gen/
├── ARCHITECTURE.md              # Complete architecture documentation
├── codegen_binding.py          # Compile-time parameter binding system
├── cli.py                      # Command-line interface
├── kernel_integrator.py        # Main workflow orchestration
├── generators/                 # Modular code generators
├── templates/                  # Jinja2 templates for code generation
├── rtl_parser/                 # SystemVerilog parsing and pragma analysis
└── tests/                      # End-to-end tests and examples
```

## Testing

```bash
# Run end-to-end tests
./smithy exec "cd brainsmith/tools/hw_kernel_gen/tests && python test_e2e_generation.py"

# Run specific test
./smithy exec "pytest brainsmith/tools/hw_kernel_gen/tests/"
```

## Pragma Reference

See [rtl_parser/PRAGMA_GUIDE.md](./rtl_parser/PRAGMA_GUIDE.md) for complete pragma documentation.

---

For detailed architectural information and development guidelines, see [ARCHITECTURE.md](./ARCHITECTURE.md).