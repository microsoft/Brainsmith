# Hardware Kernel Generator (HKG) – Current Design (Updated)

## Overview
The Hardware Kernel Generator orchestrates the integration of custom RTL designs into the FINN/Brainsmith ecosystem. It combines RTL analysis with compiler metadata to generate parameterizable templates and integration files for FPGA acceleration workflows.

## Architecture

### Core Components
1. **HardwareKernelGenerator** (`hkg.py`) - Main orchestrator class
2. **RTL Template Generator** (`rtl_template_generator.py`) - Jinja2-based wrapper generation
3. **RTL Parser Integration** - Leverages full RTL Parser pipeline
4. **Compiler Data Handler** - Python module loading and AST analysis

### Pipeline Phases

#### 1. Input Validation
- **RTL File**: SystemVerilog source with Brainsmith pragmas
- **Compiler Data**: Python module with ONNX patterns and cost functions
- **Output Directory**: Target location for generated files
- **Custom Documentation**: Optional Markdown content

#### 2. RTL Analysis
- Full RTL Parser pipeline execution
- Interface identification and validation
- Parameter extraction with template mapping
- Pragma processing and application

#### 3. Compiler Data Processing
- Dynamic Python module importing
- AST parsing for code analysis
- Validation of required objects (ONNX patterns, cost functions)

#### 4. Code Generation
Currently Implemented:
- **RTL Template**: Parameterizable Verilog wrapper using Jinja2

Planned Components:
- **HWCustomOp**: FINN DSE integration class
- **RTLBackend**: FINN synthesis backend
- **Documentation**: Auto-generated kernel documentation

### RTL Template Generation

#### Template Features
- **Interface Sorting**: Consistent ordering (Global, AXI-Stream, AXI-Lite)
- **Parameter Substitution**: Template variables ($PARAM$) for runtime configuration
- **Port Generation**: Automatic wire declarations and connections
- **Metadata Injection**: Timestamps, source file references

#### Jinja2 Environment
- **Template Location**: `templates/rtl_wrapper.v.j2`
- **Extensions**: `jinja2.ext.do` for complex logic
- **Error Handling**: Strict undefined variables
- **Formatting**: Trimmed blocks and left-stripped blocks

#### Template Context
```python
{
    'kernel_name': str,           # Module name
    'timestamp': str,             # Generation timestamp
    'source_file': str,           # Original RTL file
    'parameters': [Parameter],    # Template parameters
    'interfaces': [Interface],    # Sorted interface list
    'sorted_interfaces': [Interface]  # Explicitly sorted
}
```

### Error Handling

#### Exception Hierarchy
- **HardwareKernelGeneratorError**: Base HWKG exception
- **ParserError**: RTL parsing failures
- **FileNotFoundError**: Missing input files
- **Template Errors**: Jinja2 rendering issues

#### Validation Strategy
- Input file existence checks
- Output directory creation
- Module import validation
- Template rendering verification

### CLI Interface

#### Command Structure
```bash
python -m brainsmith.tools.hw_kernel_gen.hkg \
    <rtl_file> <compiler_data> \
    -o <output_dir> \
    [-d <custom_doc>] \
    [--stop-after <phase>]
```

#### Execution Control
- **Phase-based execution**: Stop after specific phases for debugging
- **Error reporting**: Detailed error messages with phase context
- **Output tracking**: File generation monitoring

### Development Status

#### Completed Features
- ✅ RTL Parser integration
- ✅ RTL template generation with Jinja2
- ✅ CLI interface with phase control
- ✅ Comprehensive error handling
- ✅ Input validation and directory management

#### In Development
- 🔄 HWCustomOp generation (placeholder)
- 🔄 RTLBackend generation (placeholder)
- 🔄 Documentation generation (placeholder)

#### Future Enhancements
- Advanced template customization
- Multi-kernel support
- Integration testing framework
- Performance optimization

### Integration Points

#### FINN Ecosystem
- **HWCustomOp**: Design space exploration interface
- **RTLBackend**: Synthesis and implementation backend
- **Template System**: Runtime parameterization

#### Brainsmith Library
- **Kernel Registry**: Automated kernel discovery
- **Quality Assurance**: Validation and testing workflows
- **Documentation**: Auto-generated kernel documentation

## Technology Stack
- **Core**: Python 3.8+ with pathlib and importlib
- **Templating**: Jinja2 with custom extensions
- **Parsing**: RTL Parser with tree-sitter backend
- **CLI**: argparse with comprehensive error handling
- **Testing**: pytest framework integration

## Output Artifacts
- **RTL Wrapper**: Parameterizable Verilog template
- **Integration Files**: FINN-compatible Python classes (planned)
- **Documentation**: Kernel usage and interface documentation (planned)
- **Metadata**: Generation tracking and validation reports
