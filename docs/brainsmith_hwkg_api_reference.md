# Brainsmith Hardware Kernel Generator - API Reference

## Overview

This document provides comprehensive API documentation for all classes and functions in the Brainsmith Hardware Kernel Generator system. The API is organized by functional modules.

## Table of Contents

1. [Hardware Kernel Generator](#hardware-kernel-generator)
2. [RTL Parser](#rtl-parser)
3. [HWCustomOp Generator](#hwcustomop-generator)
4. [Dataflow Framework](#dataflow-framework)
5. [RTL Conversion](#rtl-conversion)
6. [Data Structures](#data-structures)
7. [Utilities](#utilities)

---

## Hardware Kernel Generator

### `HardwareKernelGenerator`

Main orchestrator class that coordinates the complete generation pipeline.

```python
class HardwareKernelGenerator:
    def __init__(self, rtl_file_path: str, compiler_data_path: str, 
                 output_dir: str, custom_doc_path: Optional[str] = None)
```

**Purpose**: Orchestrates the generation of FINN integration files for a custom RTL HW Kernel.

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `rtl_file_path` | `str` | Path to the SystemVerilog RTL source file |
| `compiler_data_path` | `str` | Path to the Python file containing compiler data |
| `output_dir` | `str` | Directory where generated files will be saved |
| `custom_doc_path` | `Optional[str]` | Optional path to a Markdown file with custom documentation |

#### Public Methods

##### `run(stop_after: Optional[str] = None) -> Dict[str, Path]`

Executes the complete HKG pipeline.

**Parameters:**
- `stop_after`: Optional phase name to stop execution after (for debugging)

**Returns:** Dictionary containing paths to generated files

**Phases:**
1. `parse_rtl` - Parse RTL file using RTLParser
2. `parse_compiler_data` - Import and parse compiler data
3. `load_custom_documentation` - Load optional custom documentation
4. `build_dataflow_model` - Create dataflow model from interfaces
5. `generate_rtl_template` - Generate RTL wrapper template
6. `generate_hw_custom_op` - Generate HWCustomOp class
7. `generate_rtl_backend` - Generate RTLBackend class
8. `generate_test_suite` - Generate test suite
9. `generate_documentation` - Generate documentation

**Example:**
```python
hkg = HardwareKernelGenerator("kernel.sv", "data.py", "output/")
generated_files = hkg.run()
print(f"Generated {len(generated_files)} files")
```

##### `generate_auto_hwcustomop(template_path: str, output_path: str) -> str`

Public method for generating AutoHWCustomOp with Phase 3 enhancements.

**Parameters:**
- `template_path`: Path to Jinja2 template file (for compatibility)
- `output_path`: Output file path for generated class

**Returns:** Path to generated file

##### `generate_complete_package(output_dir: Optional[str] = None) -> Dict[str, Path]`

Generate complete package of all files for the kernel.

**Parameters:**
- `output_dir`: Optional override for output directory

**Returns:** Dictionary mapping file types to generated file paths

##### `get_parsed_rtl_data() -> HWKernel`

Returns the parsed RTL data for testing purposes.

**Returns:** The parsed HWKernel data, parsing RTL first if needed

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dataflow_enabled` | `bool` | Whether dataflow framework is available |
| `generated_files` | `Dict[str, Path]` | Dictionary of generated file paths |
| `hw_kernel_data` | `Optional[HWKernel]` | Parsed RTL kernel data |

#### Example Usage

```python
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator

# Basic usage
hkg = HardwareKernelGenerator(
    rtl_file_path="examples/thresholding/thresholding_axi.sv",
    compiler_data_path="examples/thresholding/dummy_compiler_data.py",
    output_dir="generated_output/"
)

# Generate all files
generated_files = hkg.run()

# Or generate just HWCustomOp
hwcustomop_path = hkg.generate_auto_hwcustomop(
    template_path="templates/hw_custom_op_slim.py.j2",
    output_path="output/thresholding_hwcustomop.py"
)
```

---

## RTL Parser

### `RTLParser`

Parser for SystemVerilog RTL files using tree-sitter.

```python
class RTLParser:
    def __init__(self, grammar_path: Optional[str] = None, debug: bool = False)
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `grammar_path` | `Optional[str]` | Optional path to compiled tree-sitter grammar library |
| `debug` | `bool` | Enable detailed debug logging |

#### Public Methods

##### `parse_file(file_path: str) -> HWKernel`

Main public method to parse a SystemVerilog file.

**Parameters:**
- `file_path`: Absolute path to the SystemVerilog file to parse

**Returns:** `HWKernel` object containing parsed information

**Raises:**
- `ParserError`: For logical errors, ambiguity, or validation failures
- `SyntaxError`: For SystemVerilog syntax errors
- `FileNotFoundError`: If input file cannot be found

**Processing Stages:**
1. Initial parse: AST generation and module selection
2. Extract components: Name, parameters, ports
3. Analyze interfaces: Build and validate AXI interfaces
4. Apply pragmas: Process @brainsmith pragmas

**Example:**
```python
parser = RTLParser(debug=True)
hw_kernel = parser.parse_file("kernel.sv")
print(f"Parsed kernel '{hw_kernel.name}' with {len(hw_kernel.interfaces)} interfaces")
```

### `InterfaceBuilder`

Builds logical interfaces from raw port lists.

```python
class InterfaceBuilder:
    def __init__(self, debug: bool = False)
```

#### Public Methods

##### `build_interfaces(ports: List[Port]) -> Tuple[Dict[str, Interface], List[Port]]`

Build logical interfaces from a list of ports.

**Parameters:**
- `ports`: List of Port objects to process

**Returns:** Tuple of (interfaces_dict, unassigned_ports)

**Interface Types:**
- `GLOBAL_CONTROL`: Clock and reset signals (`ap_clk`, `ap_rst_n`)
- `AXI_STREAM`: Data streaming interfaces (`s_axis_*`, `m_axis_*`)
- `AXI_LITE`: Configuration/control interfaces (`s_axilite_*`)

### `PragmaHandler`

Handles extraction and application of @brainsmith pragmas.

```python
class PragmaHandler:
    def __init__(self, debug: bool = False)
```

#### Public Methods

##### `extract_pragmas(root_node: Node) -> List[Pragma]`

Extract all @brainsmith pragmas from the AST.

**Parameters:**
- `root_node`: Root AST node to search

**Returns:** List of extracted Pragma objects

**Supported Pragma Types:**
- `TDIM`: Tensor dimension specifications
- `DATATYPE`: Datatype constraints
- `WEIGHT`: Weight interface marking
- `TOP_MODULE`: Module selection in multi-module files

**Example:**
```python
pragma_handler = PragmaHandler(debug=True)
pragmas = pragma_handler.extract_pragmas(ast_root)
print(f"Found {len(pragmas)} pragmas")
```

---

## HWCustomOp Generator

### `HWCustomOpGenerator`

Phase 3 HWCustomOp generator with enhanced TDIM pragma integration.

```python
class HWCustomOpGenerator:
    def __init__(self, template_dir: Optional[Path] = None)
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `template_dir` | `Optional[Path]` | Template directory (defaults to built-in templates) |

#### Public Methods

##### `generate_hwcustomop(hw_kernel: HWKernel, output_path: Path, class_name: Optional[str] = None, source_file: str = "unknown.sv") -> str`

Generate a slim HWCustomOp class from parsed RTL data.

**Parameters:**
- `hw_kernel`: Parsed RTL kernel data with pragmas
- `output_path`: Where to write the generated Python file
- `class_name`: Optional override for class name
- `source_file`: Original RTL source file name

**Returns:** Generated Python code as string

**Features:**
- Slim template generation (68% code reduction)
- Enhanced TDIM pragma support
- Automatic chunking strategy generation
- AXI interface type classification

**Example:**
```python
generator = HWCustomOpGenerator()
generated_code = generator.generate_hwcustomop(
    hw_kernel=parsed_kernel,
    output_path=Path("output/kernel_hwcustomop.py"),
    class_name="ThresholdingHWCustomOp",
    source_file="thresholding.sv"
)
```

#### Template Context

The generator builds rich template contexts with:

| Field | Type | Description |
|-------|------|-------------|
| `class_name` | `str` | Generated class name |
| `kernel_name` | `str` | Original kernel name |
| `interfaces` | `List[InterfaceTemplateData]` | Interface specifications |
| `rtl_parameters` | `List[Dict]` | RTL parameters |
| `kernel_type` | `str` | Inferred kernel type (matmul, conv, etc.) |
| `kernel_complexity` | `str` | Complexity level (low, medium, high) |

### `create_hwcustomop(rtl_file: Path, output_dir: Path, class_name: Optional[str] = None) -> Path`

Convenience function to generate HWCustomOp from RTL file.

**Parameters:**
- `rtl_file`: Path to SystemVerilog RTL file
- `output_dir`: Directory to write generated Python file
- `class_name`: Optional override for class name

**Returns:** Path to generated Python file

---

## Dataflow Framework

### `DataflowModel`

Core computational model implementing mathematical relationships between interfaces and parallelism parameters.

```python
class DataflowModel:
    def __init__(self, interfaces: List[DataflowInterface], parameters: Dict[str, Any])
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `interfaces` | `List[DataflowInterface]` | List of dataflow interfaces |
| `parameters` | `Dict[str, Any]` | Model parameters |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `input_interfaces` | `List[DataflowInterface]` | All INPUT type interfaces |
| `output_interfaces` | `List[DataflowInterface]` | All OUTPUT type interfaces |
| `weight_interfaces` | `List[DataflowInterface]` | All WEIGHT type interfaces |
| `config_interfaces` | `List[DataflowInterface]` | All CONFIG type interfaces |

#### Public Methods

##### `calculate_initiation_intervals(iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals`

Unified calculation of cII, eII, L for given parallelism parameters.

**Parameters:**
- `iPar`: Input parallelism per input interface `{interface_name: parallelism}`
- `wPar`: Weight parallelism per weight interface `{interface_name: parallelism}`

**Returns:** `InitiationIntervals` containing:
- `cII`: Calculation Initiation Interval per input interface
- `eII`: Execution Initiation Interval per input interface
- `L`: Inference Cycle Latency
- `bottleneck_analysis`: Performance bottleneck information

**Mathematical Model:**
```
cII_i = ∏(tDim_i / sDim_i)
eII_i = cII_i × max_weight_cycles
L = eII_bottleneck × ∏(qDim_bottleneck)
```

##### `get_parallelism_bounds() -> Dict[str, ParallelismBounds]`

Calculate valid bounds for iPar/wPar parameters for FINN optimization.

**Returns:** Dictionary mapping parameter names to their parallelism bounds

##### `get_resource_requirements(parallelism_config: ParallelismConfiguration) -> Dict[str, Any]`

Estimate resource requirements for given parallelism configuration.

**Parameters:**
- `parallelism_config`: Complete parallelism configuration

**Returns:** Dictionary with resource estimates:
- `memory_bits`: Memory requirements in bits
- `transfer_bandwidth`: Bandwidth requirements
- `computation_cycles`: Computation cycles

##### `validate_mathematical_constraints() -> ValidationResult`

Validate mathematical relationships between dimensions.

**Returns:** ValidationResult with any constraint violations

### `DataflowInterface`

Represents a single hardware interface in the dataflow model.

```python
class DataflowInterface:
    def __init__(self, name: str, interface_type: DataflowInterfaceType, 
                 qDim: List[int], tDim: List[int], sDim: List[int],
                 dtype: DataflowDataType, **kwargs)
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Interface name |
| `interface_type` | `DataflowInterfaceType` | Type (INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL) |
| `qDim` | `List[int]` | Quantization dimensions |
| `tDim` | `List[int]` | Tensor dimensions |
| `sDim` | `List[int]` | Stream dimensions |
| `dtype` | `DataflowDataType` | Data type specification |

#### Public Methods

##### `calculate_stream_width() -> int`

Calculate the stream width in bits for this interface.

**Returns:** Stream width in bits

##### `get_memory_footprint() -> int`

Calculate memory footprint for this interface.

**Returns:** Memory requirement in bits

##### `reconstruct_tensor_shape() -> List[int]`

Reconstruct the original tensor shape from qDim and tDim.

**Returns:** Reconstructed tensor shape

##### `validate_constraints() -> ValidationResult`

Validate interface constraints and mathematical relationships.

**Returns:** ValidationResult with any violations

##### `validate_datatype_string(datatype: str) -> bool`

Validate if a datatype string is allowed for this interface.

**Parameters:**
- `datatype`: Datatype string to validate

**Returns:** True if datatype is valid

### `AutoHWCustomOp`

Base class for auto-generated HWCustomOp implementations.

```python
class AutoHWCustomOp(HWCustomOp):
    def __init__(self, onnx_node, interface_metadata: List[InterfaceMetadata], **kwargs)
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `onnx_node` | `Node` | ONNX node for this operation |
| `interface_metadata` | `List[InterfaceMetadata]` | Interface metadata defining the operation |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dataflow_model` | `DataflowModel` | Lazily-built dataflow model |
| `interface_metadata` | `InterfaceMetadataCollection` | Interface metadata collection |
| `input_interfaces` | `List[str]` | Input interface names |
| `output_interfaces` | `List[str]` | Output interface names |
| `weight_interfaces` | `List[str]` | Weight interface names |
| `config_interfaces` | `List[str]` | Config interface names |

#### Core FINN Methods

##### `get_input_datatype(ind: int = 0) -> DataType`

Get input datatype from DataflowModel interface.

**Parameters:**
- `ind`: Input index

**Returns:** FINN DataType for the input

##### `get_output_datatype(ind: int = 0) -> DataType`

Get output datatype from DataflowModel interface.

**Parameters:**
- `ind`: Output index

**Returns:** FINN DataType for the output

##### `get_normal_input_shape(ind: int = 0) -> List[int]`

Get normal input shape from DataflowModel interface.

**Parameters:**
- `ind`: Input index

**Returns:** Normal tensor shape

##### `get_normal_output_shape(ind: int = 0) -> List[int]`

Get normal output shape from DataflowModel interface.

**Parameters:**
- `ind`: Output index

**Returns:** Normal tensor shape

##### `get_folded_input_shape(ind: int = 0) -> List[int]`

Get folded input shape considering parallelism configuration.

**Parameters:**
- `ind`: Input index

**Returns:** Folded tensor shape

##### `get_folded_output_shape(ind: int = 0) -> List[int]`

Get folded output shape considering parallelism configuration.

**Parameters:**
- `ind`: Output index

**Returns:** Folded tensor shape

##### `get_instream_width(ind: int = 0) -> int`

Get input stream width in bits.

**Parameters:**
- `ind`: Input index

**Returns:** Stream width in bits

##### `get_outstream_width(ind: int = 0) -> int`

Get output stream width in bits.

**Parameters:**
- `ind`: Output index

**Returns:** Stream width in bits

##### `get_exp_cycles() -> int`

Get expected cycles using DataflowModel's unified computational model.

**Returns:** Expected execution cycles

#### Resource Estimation Methods

##### `estimate_bram_usage() -> int`

Estimate BRAM usage using DataflowModel resource requirements.

**Returns:** Estimated BRAM usage

##### `estimate_lut_usage() -> int`

Estimate LUT usage using DataflowModel resource requirements.

**Returns:** Estimated LUT usage

##### `estimate_dsp_usage(fpgapart: str = "xczu7ev") -> int`

Estimate DSP usage using DataflowModel resource requirements.

**Parameters:**
- `fpgapart`: Target FPGA part

**Returns:** Estimated DSP usage

#### Utility Methods

##### `get_enhanced_nodeattr_types() -> Dict[str, Tuple[str, bool, Any]]`

Get enhanced node attribute types with dataflow modeling support.

**Returns:** Dictionary of node attributes with types and defaults

##### `get_interface_config(interface_name: str) -> Dict[str, Any]`

Get configuration for a specific interface.

**Parameters:**
- `interface_name`: Name of the interface

**Returns:** Interface configuration dictionary

##### `set_model_wrapper(model_wrapper)`

Set ModelWrapper for accurate tensor shape extraction.

**Parameters:**
- `model_wrapper`: FINN ModelWrapper instance

---

## RTL Conversion

### `RTLInterfaceConverter`

Converts RTL Parser Interface objects to DataflowInterface objects.

```python
class RTLInterfaceConverter:
    def __init__(self, onnx_metadata: Optional[Dict] = None)
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `onnx_metadata` | `Optional[Dict]` | Optional ONNX model metadata for tensor shape inference |

#### Public Methods

##### `convert_interfaces(rtl_interfaces: Dict[str, RTLInterface], parameters: Optional[Dict[str, Any]] = None) -> List[DataflowInterface]`

Convert RTL Parser interfaces to DataflowInterface objects.

**Parameters:**
- `rtl_interfaces`: Dictionary of RTL Parser Interface objects
- `parameters`: Module parameters for TDIM pragma evaluation

**Returns:** List of DataflowInterface objects

**Conversion Process:**
1. Interface type mapping (AXI-Stream → INPUT/OUTPUT/WEIGHT)
2. Dimension extraction from ONNX metadata and TDIM pragmas
3. Datatype constraint conversion from DATATYPE pragmas
4. Default constraint assignment for unconstrained interfaces

**Example:**
```python
converter = RTLInterfaceConverter(onnx_metadata)
dataflow_interfaces = converter.convert_interfaces(
    rtl_interfaces=hw_kernel.interfaces,
    parameters={param.name: param.default_value for param in hw_kernel.parameters}
)
```

### `validate_conversion_result(dataflow_interfaces: List[DataflowInterface]) -> List[ValidationError]`

Validate the result of RTL to Dataflow interface conversion.

**Parameters:**
- `dataflow_interfaces`: List of converted DataflowInterface objects

**Returns:** List of validation errors (empty if all valid)

**Validation Checks:**
- Required interface types (INPUT, OUTPUT)
- Individual interface constraint validation
- Mathematical relationship validation

---

## Data Structures

### `HWKernel`

Container for parsed hardware kernel information.

```python
@dataclass
class HWKernel:
    name: str
    parameters: List[Parameter]
    interfaces: Dict[str, Interface]
    pragmas: List[Pragma]
```

### `Interface`

Represents a logical hardware interface.

```python
@dataclass
class Interface:
    name: str
    type: InterfaceType
    ports: Dict[str, Port]
    metadata: Dict[str, Any]
```

### `Port`

Represents a single hardware port.

```python
@dataclass
class Port:
    name: str
    direction: Direction
    width: str
    description: Optional[str] = None
```

### `Parameter`

Represents a module parameter.

```python
@dataclass
class Parameter:
    name: str
    param_type: Optional[str]
    default_value: Optional[str]
    description: Optional[str] = None
```

### `Pragma`

Base class for all pragma types.

```python
@dataclass
class Pragma:
    type: PragmaType
    line_number: int
    raw_text: str
    parsed_data: Dict[str, Any]
```

### Enumerations

#### `InterfaceType`
```python
class InterfaceType(Enum):
    AXI_STREAM = "axi_stream"
    AXI_LITE = "axi_lite"
    GLOBAL_CONTROL = "global_control"
```

#### `DataflowInterfaceType`
```python
class DataflowInterfaceType(Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    WEIGHT = "WEIGHT"
    CONFIG = "CONFIG"
    CONTROL = "CONTROL"
```

#### `Direction`
```python
class Direction(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
```

#### `PragmaType`
```python
class PragmaType(Enum):
    TDIM = "TDIM"
    DATATYPE = "DATATYPE"
    WEIGHT = "WEIGHT"
    TOP_MODULE = "TOP_MODULE"
```

---

## Utilities

### Error Classes

#### `HardwareKernelGeneratorError`
Base exception for HKG errors.

#### `ParserError`
Base class for parser errors.

#### `SyntaxError`
Raised when SystemVerilog syntax is invalid.

#### `ConversionValidationError`
Exception raised when interface conversion validation fails.

### Validation Framework

#### `ValidationResult`
```python
@dataclass
class ValidationResult:
    errors: List[ValidationError]
    warnings: List[ValidationError]
```

#### `ValidationError`
```python
@dataclass
class ValidationError:
    component: str
    error_type: str
    message: str
    severity: ValidationSeverity
    context: Dict[str, Any]
```

#### `ValidationSeverity`
```python
class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
```

---

## Usage Examples

### Complete Workflow Example

```python
from brainsmith.tools.hw_kernel_gen.hkg import HardwareKernelGenerator
from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser
from brainsmith.tools.hw_kernel_gen.generators.hw_custom_op_generator import HWCustomOpGenerator

# 1. Parse RTL file
parser = RTLParser(debug=True)
hw_kernel = parser.parse_file("kernel.sv")

# 2. Generate HWCustomOp
generator = HWCustomOpGenerator()
generated_code = generator.generate_hwcustomop(
    hw_kernel=hw_kernel,
    output_path=Path("output/kernel_hwcustomop.py"),
    class_name="KernelHWCustomOp"
)

# 3. Use generated class
exec(generated_code)  # In practice, import the generated module

# 4. Or use complete pipeline
hkg = HardwareKernelGenerator(
    rtl_file_path="kernel.sv",
    compiler_data_path="data.py",
    output_dir="output/"
)
generated_files = hkg.run()
```

### Advanced Configuration Example

```python
# Custom interface metadata
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp

interface_metadata = [
    InterfaceMetadata(
        name="s_axis_input",
        interface_type=DataflowInterfaceType.INPUT,
        default_datatype=DataTypeConstraint(
            base_types=["UINT"],
            min_bitwidth=8,
            max_bitwidth=16
        )
    )
]

# Use with custom HWCustomOp
class CustomHWCustomOp(AutoHWCustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node, interface_metadata)
    
    def get_nodeattr_types(self):
        attrs = super().get_enhanced_nodeattr_types()
        attrs.update({
            "custom_param": ("i", False, 42)
        })
        return attrs
```

This API reference provides comprehensive documentation for using all components of the Brainsmith Hardware Kernel Generator system.