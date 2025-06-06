# Brainsmith Dataflow Framework API Reference

Complete API documentation for the Interface-Wise Dataflow Modeling Framework.

## Table of Contents

- [Core Module](#core-module)
  - [DataflowInterface](#dataflowinterface)
  - [DataflowModel](#dataflowmodel)
  - [AutoHWCustomOp](#autohwcustomop)
  - [AutoRTLBackend](#autortlbackend)
  - [Validation Framework](#validation-framework)
  - [Tensor Chunking](#tensor-chunking)
  - [Class Naming](#class-naming)
- [Integration Module](#integration-module)
  - [RTL Conversion](#rtl-conversion)
- [Examples Module](#examples-module)

---

## Core Module

### DataflowInterface

The core interface abstraction for hardware kernel interfaces.

#### Class: `DataflowInterface`

**Location**: [`brainsmith.dataflow.core.dataflow_interface`](core/dataflow_interface.py:142)

Unified abstraction for hardware kernel interfaces providing standardized representation of interface characteristics.

```python
@dataclass
class DataflowInterface:
    name: str                           # Interface identifier
    interface_type: DataflowInterfaceType  # INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL
    qDim: List[int]                     # Query dimensions (complete data set)
    tDim: List[int]                     # Tensor dimensions (per calculation)
    sDim: List[int]                     # Stream dimensions (per clock cycle)
    dtype: DataflowDataType             # Element data type specification
    allowed_datatypes: List[DataTypeConstraint] = field(default_factory=list)
    axi_metadata: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    pragma_metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Methods

##### `calculate_stream_width() -> int`

Calculate AXI stream width based on sDim and dtype.

**Returns**: Stream width in bits, aligned to 8-bit boundaries

**Example**:
```python
interface = DataflowInterface(...)
width = interface.calculate_stream_width()  # Returns: 64 (for 8 elements × 8 bits)
```

##### `validate_constraints() -> ValidationResult`

Validate interface constraints and configuration.

**Returns**: [`ValidationResult`](#validationresult) with any errors or warnings

**Example**:
```python
result = interface.validate_constraints()
if result.success:
    print("Interface is valid")
else:
    for error in result.errors:
        print(f"Error: {error.message}")
```

##### `apply_parallelism(iPar: Optional[int] = None, wPar: Optional[int] = None) -> None`

Update sDim based on parallelism parameters.

**Parameters**:
- `iPar`: Input parallelism factor
- `wPar`: Weight parallelism factor

**Example**:
```python
interface.apply_parallelism(iPar=4)  # Updates sDim[0] = 4 for INPUT interface
```

##### `get_axi_signals() -> Dict[str, Dict[str, Any]]`

Generate AXI signal specifications for this interface.

**Returns**: Dictionary mapping signal names to signal specifications

**Example**:
```python
signals = interface.get_axi_signals()
# Returns: {
#   "input0_TDATA": {"direction": "input", "width": 32, "description": "..."},
#   "input0_TVALID": {"direction": "input", "width": 1, "description": "..."},
#   "input0_TREADY": {"direction": "output", "width": 1, "description": "..."}
# }
```

##### `validate_datatype(target_dtype: DataflowDataType) -> bool`

Validate that target datatype is allowed for this interface.

**Parameters**:
- `target_dtype`: [`DataflowDataType`](#dataflowdatatype) to validate

**Returns**: True if datatype is valid

##### `validate_datatype_string(dtype_string: str) -> bool`

Validate if a datatype string is allowed for this interface.

**Parameters**:
- `dtype_string`: FINN datatype string (e.g., "UINT8", "INT16")

**Returns**: True if datatype is valid

**Example**:
```python
if interface.validate_datatype_string("UINT16"):
    print("UINT16 is allowed")
```

##### `get_memory_footprint() -> int`

Calculate total memory footprint in bits.

**Returns**: Total memory requirement in bits

##### `get_transfer_cycles() -> int`

Calculate number of transfer cycles needed.

**Returns**: Number of cycles required for complete transfer

##### `reconstruct_tensor_shape() -> List[int]`

Reconstruct original tensor shape from qDim and tDim using broadcasting.

**Returns**: Original tensor shape before chunking

**Example**:
```python
# For qDim=[30], tDim=[50]
original_shape = interface.reconstruct_tensor_shape()  # Returns: [1500]
```

##### `validate_tensor_chunking(original_shape: List[int]) -> ValidationResult`

Validate that qDim/tDim correctly chunk the original tensor shape.

**Parameters**:
- `original_shape`: Original tensor shape to validate against

**Returns**: [`ValidationResult`](#validationresult) with any chunking errors

#### Class Methods

##### `from_tensor_chunking(name: str, interface_type: DataflowInterfaceType, original_shape: List[int], tDim: List[int], dtype: DataflowDataType, chunking_mode: str = "broadcast", **kwargs) -> DataflowInterface`

Factory method to create DataflowInterface from tensor chunking specification.

**Parameters**:
- `name`: Interface name
- `interface_type`: Type of interface
- `original_shape`: Original tensor shape before chunking
- `tDim`: Desired tensor dimensions per calculation
- `dtype`: Data type specification
- `chunking_mode`: How to compute qDim ("broadcast", "divide", "explicit")

**Returns**: DataflowInterface with computed qDim

**Example**:
```python
interface = DataflowInterface.from_tensor_chunking(
    name="conv_input",
    interface_type=DataflowInterfaceType.INPUT,
    original_shape=[1, 64, 32, 32],
    tDim=[32, 32],
    dtype=DataflowDataType("UINT", 8, False, "UINT8")
)
```

#### Supporting Classes

##### `DataflowInterfaceType`

**Location**: [`brainsmith.dataflow.core.dataflow_interface`](core/dataflow_interface.py:24)

Enumeration of interface types for dataflow modeling.

```python
class DataflowInterfaceType(Enum):
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data  
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    CONFIG = "config"    # AXI-Lite for runtime configuration
    CONTROL = "control"  # Global control signals (clk, rst, etc.)
```

##### `DataflowDataType`

**Location**: [`brainsmith.dataflow.core.dataflow_interface`](core/dataflow_interface.py:33)

Enhanced datatype specification supporting FINN compatibility.

```python
@dataclass
class DataflowDataType:
    base_type: str       # INT, UINT, FLOAT, FIXED
    bitwidth: int        # Bit precision
    signed: bool         # Sign specification
    finn_type: str       # FINN DataType string representation
```

**Example**:
```python
dtype = DataflowDataType(
    base_type="UINT",
    bitwidth=8,
    signed=False,
    finn_type="UINT8"
)
```

##### `DataTypeConstraint`

**Location**: [`brainsmith.dataflow.core.dataflow_interface`](core/dataflow_interface.py:71)

Constraint on allowed datatypes for an interface.

```python
@dataclass
class DataTypeConstraint:
    base_types: List[str]        # Allowed base types
    min_bitwidth: int            # Minimum allowed bitwidth
    max_bitwidth: int            # Maximum allowed bitwidth
    signed_allowed: bool = True  # Allow signed types
    unsigned_allowed: bool = True # Allow unsigned types
```

**Methods**:

###### `is_valid_datatype(dtype: DataflowDataType) -> bool`

Check if a datatype satisfies this constraint.

---

### DataflowModel

The core computational model implementing mathematical relationships between interfaces.

#### Class: `DataflowModel`

**Location**: [`brainsmith.dataflow.core.dataflow_model`](core/dataflow_model.py:39)

Core computational model implementing mathematical relationships between interfaces and parallelism parameters.

```python
class DataflowModel:
    def __init__(self, interfaces: List[DataflowInterface], parameters: Dict[str, Any]):
        self.interfaces = self._organize_interfaces(interfaces)
        self.parameters = parameters
        self.constraints = self._extract_constraints()
        self.computation_graph = self._build_computation_graph()
```

#### Properties

##### `input_interfaces -> List[DataflowInterface]`

All INPUT type interfaces.

##### `output_interfaces -> List[DataflowInterface]`

All OUTPUT type interfaces.

##### `weight_interfaces -> List[DataflowInterface]`

All WEIGHT type interfaces.

##### `config_interfaces -> List[DataflowInterface]`

All CONFIG type interfaces.

##### `control_interfaces -> List[DataflowInterface]`

All CONTROL type interfaces.

#### Methods

##### `calculate_initiation_intervals(iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals`

Unified calculation of cII, eII, L for given parallelism parameters.

**Parameters**:
- `iPar`: Input parallelism per input interface {interface_name: parallelism}
- `wPar`: Weight parallelism per weight interface {interface_name: parallelism}

**Returns**: [`InitiationIntervals`](#initiationintervals) containing performance metrics

**Example**:
```python
model = DataflowModel(interfaces, parameters)
intervals = model.calculate_initiation_intervals(
    iPar={"input0": 4},
    wPar={"weights": 8}
)
print(f"Latency: {intervals.L} cycles")
print(f"cII: {intervals.cII}")
print(f"eII: {intervals.eII}")
```

##### `validate_mathematical_constraints() -> ValidationResult`

Validate mathematical relationships between dimensions.

**Returns**: [`ValidationResult`](#validationresult) with constraint validation results

##### `get_parallelism_bounds() -> Dict[str, ParallelismBounds]`

Calculate valid bounds for iPar/wPar parameters for FINN optimization.

**Returns**: Dictionary mapping parameter names to [`ParallelismBounds`](#parallelismbounds)

**Example**:
```python
bounds = model.get_parallelism_bounds()
for param_name, bound in bounds.items():
    print(f"{param_name}: [{bound.min_value}, {bound.max_value}]")
    print(f"Valid divisors: {bound.divisibility_constraints}")
```

##### `get_resource_requirements(parallelism_config: ParallelismConfiguration) -> Dict[str, Any]`

Estimate resource requirements for given parallelism configuration.

**Parameters**:
- `parallelism_config`: [`ParallelismConfiguration`](#parallelismconfiguration) specifying parallelism settings

**Returns**: Dictionary with resource estimates

**Example**:
```python
resources = model.get_resource_requirements(config)
print(f"Memory: {resources['memory_bits']} bits")
print(f"Bandwidth: {resources['transfer_bandwidth']} bits/cycle")
```

##### `optimize_parallelism(constraints: Dict[str, Any]) -> ParallelismConfiguration`

Find optimal parallelism configuration within given constraints.

**Parameters**:
- `constraints`: Resource and performance constraints

**Returns**: [`ParallelismConfiguration`](#parallelismconfiguration) with optimal settings

#### Supporting Classes

##### `InitiationIntervals`

**Location**: [`brainsmith.dataflow.core.dataflow_model`](core/dataflow_model.py:17)

Container for initiation interval calculations.

```python
@dataclass
class InitiationIntervals:
    cII: Dict[str, int]  # Per input interface calculation intervals
    eII: Dict[str, int]  # Per input interface execution intervals
    L: int               # Overall inference latency
    bottleneck_analysis: Dict[str, Any]  # Performance bottleneck information
```

##### `ParallelismBounds`

**Location**: [`brainsmith.dataflow.core.dataflow_model`](core/dataflow_model.py:25)

Valid bounds for parallelism parameters.

```python
@dataclass 
class ParallelismBounds:
    interface_name: str
    min_value: int
    max_value: int
    divisibility_constraints: List[int]  # Values that must divide evenly
```

##### `ParallelismConfiguration`

**Location**: [`brainsmith.dataflow.core.dataflow_model`](core/dataflow_model.py:33)

Complete parallelism configuration for a kernel.

```python
@dataclass
class ParallelismConfiguration:
    iPar: Dict[str, int]  # Input parallelism per input interface
    wPar: Dict[str, int]  # Weight parallelism per weight interface
    derived_sDim: Dict[str, List[int]]  # Computed stream dimensions
```

---

### AutoHWCustomOp

Base class for auto-generated HWCustomOp implementations.

#### Class: `AutoHWCustomOp`

**Location**: [`brainsmith.dataflow.core.auto_hw_custom_op`](core/auto_hw_custom_op.py:44)

Base class for auto-generated HWCustomOp implementations providing standardized method implementations.

```python
class AutoHWCustomOp(HWCustomOp):
    def __init__(self, onnx_node, dataflow_model: DataflowModel, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.dataflow_model = dataflow_model
```

#### Properties

##### `input_interfaces -> List[str]`

Get input interface names from DataflowModel.

##### `output_interfaces -> List[str]`

Get output interface names from DataflowModel.

##### `weight_interfaces -> List[str]`

Get weight interface names from DataflowModel.

##### `config_interfaces -> List[str]`

Get config interface names from DataflowModel.

#### Methods

##### `get_enhanced_nodeattr_types() -> Dict[str, Tuple[str, bool, Any]]`

Get enhanced node attribute types with dataflow modeling support.

**Returns**: Dictionary of attribute specifications for FINN integration

**Example**:
```python
attrs = self.get_enhanced_nodeattr_types()
# Returns attributes like:
# {
#   "input0_parallel": ("i", False, 1),
#   "input0_dtype": ("s", False, "UINT8"),
#   "resource_estimation_mode": ("s", False, "automatic"),
#   ...
# }
```

##### `get_input_datatype(ind: int = 0) -> Any`

Get input datatype from DataflowModel interface.

**Parameters**:
- `ind`: Input interface index

**Returns**: FINN DataType object or string representation

##### `get_output_datatype(ind: int = 0) -> Any`

Get output datatype from DataflowModel interface.

**Parameters**:
- `ind`: Output interface index

**Returns**: FINN DataType object or string representation

##### `get_normal_input_shape(ind: int = 0) -> List[int]`

Get normal input shape from DataflowModel interface.

**Parameters**:
- `ind`: Input interface index

**Returns**: Normal (unfolded) input shape

##### `get_normal_output_shape(ind: int = 0) -> List[int]`

Get normal output shape from DataflowModel interface.

**Parameters**:
- `ind`: Output interface index

**Returns**: Normal (unfolded) output shape

##### `get_folded_input_shape(ind: int = 0) -> List[int]`

Get folded input shape considering parallelism configuration.

**Parameters**:
- `ind`: Input interface index

**Returns**: Folded input shape based on parallelism

##### `get_folded_output_shape(ind: int = 0) -> List[int]`

Get folded output shape considering parallelism configuration.

**Parameters**:
- `ind`: Output interface index

**Returns**: Folded output shape based on parallelism

##### `get_instream_width(ind: int = 0) -> int`

Get input stream width in bits.

**Parameters**:
- `ind`: Input interface index

**Returns**: Stream width in bits

##### `get_outstream_width(ind: int = 0) -> int`

Get output stream width in bits.

**Parameters**:
- `ind`: Output interface index

**Returns**: Stream width in bits

##### `get_number_output_values() -> int`

Get total number of output values.

**Returns**: Total number of output values across all interfaces

##### `get_exp_cycles() -> int`

Get expected cycles using DataflowModel's unified computational model.

**Returns**: Expected execution cycles

**Example**:
```python
cycles = self.get_exp_cycles()  # Uses current parallelism configuration
```

##### `get_op_and_param_counts() -> Dict[str, int]`

Get operation and parameter counts.

**Returns**: Dictionary with operation and parameter statistics

**Example**:
```python
counts = self.get_op_and_param_counts()
# Returns: {
#   "ops": 1000,
#   "params": 500,
#   "weight_params": 400,
#   "config_params": 100
# }
```

##### `derive_characteristic_fxns() -> Dict[str, Any]`

Derive characteristic functions for the operation.

**Returns**: Dictionary with performance characteristics

##### `generate_params(model, path)`

Generate parameter files for weight interfaces.

**Parameters**:
- `model`: ONNX model containing weights
- `path`: Output path for parameter files

##### `estimate_bram_usage() -> int`

Estimate BRAM usage using DataflowModel resource requirements.

**Returns**: Estimated BRAM count

##### `estimate_lut_usage() -> int`

Estimate LUT usage using DataflowModel resource requirements.

**Returns**: Estimated LUT count

##### `estimate_dsp_usage(fpgapart: str = "xczu7ev") -> int`

Estimate DSP usage using DataflowModel resource requirements.

**Parameters**:
- `fpgapart`: Target FPGA part number

**Returns**: Estimated DSP count

##### `get_interface_config(interface_name: str) -> Dict[str, Any]`

Get configuration for a specific interface using DataflowModel.

**Parameters**:
- `interface_name`: Name of the interface

**Returns**: Interface configuration dictionary

**Example**:
```python
config = self.get_interface_config("input0")
# Returns: {
#   "interface_type": "INPUT",
#   "dtype": {"finn_type": "UINT8", "signed": False},
#   "qDim": [64],
#   "tDim": [16],
#   "parallel": 4,
#   "runtime_dtype": "UINT8"
# }
```

#### Static Methods

##### `generate_class_name(kernel_name: str) -> str`

Generate proper CamelCase class name from kernel name.

**Parameters**:
- `kernel_name`: Underscore-separated kernel name

**Returns**: CamelCase class name

**Example**:
```python
class_name = AutoHWCustomOp.generate_class_name("thresholding_axi")
# Returns: "AutoThresholdingAxi"
```

---

### AutoRTLBackend

Base class for auto-generated RTLBackend implementations.

#### Class: `AutoRTLBackend`

**Location**: [`brainsmith.dataflow.core.auto_rtl_backend`](core/auto_rtl_backend.py:41)

Base class for auto-generated RTLBackend implementations providing standardized methods for RTL code generation.

```python
class AutoRTLBackend(RTLBackend):
    def __init__(self):
        super().__init__()
        self.dataflow_interfaces = {}  # Set by subclass
```

#### Properties

##### `input_interfaces -> List[str]`

Get list of input interface names.

##### `output_interfaces -> List[str]`

Get list of output interface names.

##### `weight_interfaces -> List[str]`

Get list of weight interface names.

##### `config_interfaces -> List[str]`

Get list of config interface names.

#### Methods

##### `get_enhanced_nodeattr_types() -> Dict[str, Any]`

Get enhanced node attribute types for RTL backend configuration.

**Returns**: Dictionary of RTL-specific attributes

**Example**:
```python
attrs = self.get_enhanced_nodeattr_types()
# Returns: {
#   "clk_name": ("s", False, "ap_clk"),
#   "rst_name": ("s", False, "ap_rst_n"),
#   "generate_wrapper": ("b", False, True),
#   ...
# }
```

##### `generate_interface_definitions() -> List[Dict[str, Any]]`

Generate RTL interface definitions from dataflow interfaces.

**Returns**: List of interface definition dictionaries

##### `generate_signal_assignments() -> List[Dict[str, str]]`

Generate RTL signal assignments for interfaces.

**Returns**: List of signal assignment specifications

##### `generate_parameter_overrides() -> Dict[str, Any]`

Generate RTL parameter overrides based on current configuration.

**Returns**: Dictionary of parameter overrides

##### `generate_clock_assignments() -> List[str]`

Generate clock signal assignments.

**Returns**: List of clock assignment strings

##### `generate_reset_assignments() -> List[str]`

Generate reset signal assignments.

**Returns**: List of reset assignment strings

##### `calculate_interface_width(interface_name: str) -> int`

Calculate bit width for an interface.

**Parameters**:
- `interface_name`: Name of the interface

**Returns**: Interface width in bits

##### `generate_enhanced_code_dict() -> Dict[str, Any]`

Generate enhanced code generation dictionary with dataflow metadata.

**Returns**: Enhanced code generation parameters for RTL templates

**Example**:
```python
codegen_dict = self.generate_enhanced_code_dict()
# Returns: {
#   "interfaces": [...],
#   "signals": [...],
#   "parameters": {...},
#   "clocks": [...],
#   "resets": [...]
# }
```

##### `generate_params(model, path)`

Generate RTL parameter files based on dataflow interface configuration.

**Parameters**:
- `model`: ONNX model node
- `path`: Output directory for parameter files

#### Static Methods

##### `generate_class_name(kernel_name: str) -> str`

Generate proper CamelCase backend class name from kernel name.

**Parameters**:
- `kernel_name`: Underscore-separated kernel name

**Returns**: CamelCase backend class name

**Example**:
```python
class_name = AutoRTLBackend.generate_class_name("thresholding_axi")
# Returns: "AutoThresholdingAxiRTLBackend"
```

---

### Validation Framework

Comprehensive validation framework for dataflow models and interfaces.

#### Class: `ValidationResult`

**Location**: [`brainsmith.dataflow.core.validation`](core/validation.py:31)

Container for validation results.

```python
@dataclass
class ValidationResult:
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
```

#### Methods

##### `add_error(error: ValidationError)`

Add an error to the validation result.

##### `is_valid() -> bool`

Check if validation passed (no errors).

##### `has_warnings() -> bool`

Check if there are any warnings.

##### `success -> bool`

Alias for `is_valid()` to match test expectations.

##### `merge(other: ValidationResult)`

Merge another validation result into this one.

#### Class: `ValidationError`

**Location**: [`brainsmith.dataflow.core.validation`](core/validation.py:21)

Represents a validation error.

```python
@dataclass
class ValidationError:
    component: str
    error_type: str
    message: str
    severity: ValidationSeverity
    context: Dict[str, Any] = field(default_factory=dict)
```

#### Enum: `ValidationSeverity`

**Location**: [`brainsmith.dataflow.core.validation`](core/validation.py:12)

Severity levels for validation results.

```python
class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

#### Utility Functions

##### `create_validation_result() -> ValidationResult`

Create a new validation result.

##### `create_divisibility_error(interface_name: str, param_name: str, value: int, divisor: int) -> ValidationError`

Create a divisibility constraint error.

##### `create_range_error(interface_name: str, param_name: str, value: int, min_val: int, max_val: int) -> ValidationError`

Create a range constraint error.

##### `create_datatype_error(interface_name: str, datatype: str, allowed_types: List[str]) -> ValidationError`

Create a datatype constraint error.

##### `validate_dataflow_model(model) -> ValidationResult`

Validate a complete dataflow model.

**Parameters**:
- `model`: DataflowModel to validate

**Returns**: ValidationResult with any errors or warnings

---

### Tensor Chunking

Tensor chunking utilities for dataflow modeling.

#### Enum: `ChunkingStrategy`

**Location**: [`brainsmith.dataflow.core.tensor_chunking`](core/tensor_chunking.py:12)

Strategies for chunking tensors.

```python
class ChunkingStrategy(Enum):
    BROADCAST = "broadcast"
    DIVIDE = "divide"
    EXPLICIT = "explicit"
```

#### Class: `TensorChunk`

**Location**: [`brainsmith.dataflow.core.tensor_chunking`](core/tensor_chunking.py:20)

Represents a chunk of a tensor.

```python
@dataclass
class TensorChunk:
    original_shape: List[int]
    chunk_shape: List[int]
    chunk_index: List[int]
    strategy: ChunkingStrategy
```

#### Functions

##### `calculate_tensor_chunks(original_shape: List[int], chunk_size: List[int], strategy: ChunkingStrategy = ChunkingStrategy.DIVIDE) -> List[TensorChunk]`

Calculate tensor chunks based on original shape and chunk size.

**Parameters**:
- `original_shape`: Original tensor shape
- `chunk_size`: Desired chunk size
- `strategy`: Chunking strategy to use

**Returns**: List of TensorChunk objects

#### Class: `TensorChunking`

**Location**: [`brainsmith.dataflow.core.tensor_chunking`](core/tensor_chunking.py:58)

Utility class for tensor chunking operations in dataflow modeling.

##### Methods

###### `infer_dimensions(onnx_layout: str, onnx_shape: List[int]) -> tuple[List[int], List[int]]`

Infer qDim and tDim from ONNX layout and shape.

**Parameters**:
- `onnx_layout`: ONNX tensor layout (e.g., "NCHW", "NHWC")
- `onnx_shape`: ONNX tensor shape

**Returns**: Tuple of (qDim, tDim) lists

##### Static Methods

###### `_compute_qDim_from_chunking(original_shape: List[int], tDim: List[int]) -> List[int]`

Compute qDim from original shape and tDim.

**Parameters**:
- `original_shape`: Original tensor shape
- `tDim`: Target tensor dimensions

**Returns**: Computed qDim

---

### Class Naming

Class naming utilities for auto-generated hardware kernel classes.

#### Functions

##### `generate_class_name(kernel_name: str, prefix: str = "Auto") -> str`

**Location**: [`brainsmith.dataflow.core.class_naming`](core/class_naming.py:10)

Convert kernel_name to proper CamelCase class name.

**Parameters**:
- `kernel_name`: Underscore-separated kernel name
- `prefix`: Class name prefix (default: "Auto")

**Returns**: Properly formatted CamelCase class name

**Examples**:
```python
generate_class_name("thresholding_axi")     # Returns: "AutoThresholdingAxi"
generate_class_name("conv_layer")          # Returns: "AutoConvLayer"
generate_class_name("batch_norm")          # Returns: "AutoBatchNorm"
```

##### `generate_test_class_name(kernel_name: str) -> str`

**Location**: [`brainsmith.dataflow.core.class_naming`](core/class_naming.py:33)

Generate test class name for a kernel.

**Parameters**:
- `kernel_name`: Underscore-separated kernel name

**Returns**: Test class name in proper CamelCase

**Examples**:
```python
generate_test_class_name("thresholding_axi")  # Returns: "TestAutoThresholdingAxi"
generate_test_class_name("conv_layer")        # Returns: "TestAutoConvLayer"
```

##### `generate_backend_class_name(kernel_name: str) -> str`

**Location**: [`brainsmith.dataflow.core.class_naming`](core/class_naming.py:51)

Generate RTL backend class name for a kernel.

**Parameters**:
- `kernel_name`: Underscore-separated kernel name

**Returns**: RTL backend class name in proper CamelCase

**Examples**:
```python
generate_backend_class_name("thresholding_axi")  # Returns: "AutoThresholdingAxiRTLBackend"
generate_backend_class_name("conv_layer")        # Returns: "AutoConvLayerRTLBackend"
```

---

## Integration Module

### RTL Conversion

RTL to Dataflow Interface Conversion Pipeline.

#### Class: `RTLInterfaceConverter`

**Location**: [`brainsmith.dataflow.integration.rtl_conversion`](integration/rtl_conversion.py:23)

Converts RTL Parser Interface objects to DataflowInterface objects.

```python
class RTLInterfaceConverter:
    def __init__(self, onnx_metadata: Optional[Dict] = None):
        self.onnx_metadata = onnx_metadata or {}
        self.tensor_chunking = TensorChunking()
```

#### Methods

##### `convert_interfaces(rtl_interfaces: Dict[str, RTLInterface], parameters: Optional[Dict[str, Any]] = None) -> List[DataflowInterface]`

Convert RTL Parser interfaces to DataflowInterface objects.

**Parameters**:
- `rtl_interfaces`: Dictionary of RTL Parser Interface objects
- `parameters`: Module parameters for TDIM pragma evaluation

**Returns**: List of DataflowInterface objects

**Raises**: ValueError if conversion fails for critical interfaces

**Example**:
```python
converter = RTLInterfaceConverter(onnx_metadata)
dataflow_interfaces = converter.convert_interfaces(
    rtl_interfaces=rtl_parser_output.interfaces,
    parameters={"WIDTH": 32, "DEPTH": 64}
)
```

#### Exceptions

##### `ConversionValidationError`

**Location**: [`brainsmith.dataflow.integration.rtl_conversion`](integration/rtl_conversion.py:449)

Exception raised when interface conversion validation fails.

#### Functions

##### `validate_conversion_result(dataflow_interfaces: List[DataflowInterface]) -> List[ValidationError]`

**Location**: [`brainsmith.dataflow.integration.rtl_conversion`](integration/rtl_conversion.py:454)

Validate the result of RTL to Dataflow interface conversion.

**Parameters**:
- `dataflow_interfaces`: List of converted DataflowInterface objects

**Returns**: List of validation errors (empty if all valid)

**Example**:
```python
errors = validate_conversion_result(dataflow_interfaces)
if not errors:
    print("Conversion successful")
else:
    for error in errors:
        print(f"Validation error: {error.message}")
```

---

## Examples Module

### Basic Usage Example

**Location**: [`brainsmith.dataflow.examples.basic_usage`](examples/basic_usage.py:17)

Comprehensive example demonstrating framework usage.

#### Function: `main()`

Demonstrate basic dataflow framework usage including:

1. Creating datatype with constraints
2. Creating dataflow interfaces
3. Validating interface constraints
4. Testing datatype constraint validation
5. Creating dataflow model
6. Performing unified initiation interval calculations
7. Generating parallelism bounds for FINN optimization
8. Demonstrating tensor chunking
9. Generating AXI signal specifications
10. Resource estimation

**Usage**:
```bash
python -m brainsmith.dataflow.examples.basic_usage
```

---

## Usage Patterns

### Integration Pattern

Complete integration with HW Kernel Generator:

```python
from brainsmith.dataflow.integration.rtl_conversion import RTLInterfaceConverter
from brainsmith.dataflow.core.dataflow_model import DataflowModel

# 1. Convert RTL interfaces
converter = RTLInterfaceConverter(onnx_metadata)
dataflow_interfaces = converter.convert_interfaces(rtl_interfaces)

# 2. Create unified model
model = DataflowModel(dataflow_interfaces, parameters)

# 3. Generate optimized configuration
bounds = model.get_parallelism_bounds()
optimal_config = model.optimize_parallelism(constraints)

# 4. Calculate performance
intervals = model.calculate_initiation_intervals(
    optimal_config.iPar, 
    optimal_config.wPar
)
```

### Template Generation Pattern

Minimal template code using base classes:

```python
# Generated HWCustomOp
class {{class_name}}(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        dataflow_model = self._create_model()
        super().__init__(onnx_node, dataflow_model, **kwargs)
    
    def get_nodeattr_types(self):
        return self.get_enhanced_nodeattr_types()
    
    def _create_model(self):
        # Template-specific model creation
        return DataflowModel(interfaces, parameters)

# Generated RTLBackend  
class {{backend_class_name}}(AutoRTLBackend):
    def __init__(self):
        super().__init__()
        self.dataflow_interfaces = {...}  # Template-generated
    
    def get_nodeattr_types(self):
        return self.get_enhanced_nodeattr_types()
```

### Validation Pattern

Comprehensive validation workflow:

```python
# Interface validation
interface_result = interface.validate_constraints()
if not interface_result.success:
    handle_interface_errors(interface_result.errors)

# Model validation
model_result = model.validate_mathematical_constraints()
if not model_result.success:
    handle_model_errors(model_result.errors)

# Conversion validation
conversion_errors = validate_conversion_result(dataflow_interfaces)
if conversion_errors:
    handle_conversion_errors(conversion_errors)

# Tensor chunking validation
chunking_result = interface.validate_tensor_chunking(original_shape)
if not chunking_result.success:
    handle_chunking_errors(chunking_result.errors)
```

---

## Cross-References

- [`DataflowInterface`](#dataflowinterface) ↔ [`DataflowModel`](#dataflowmodel): Interfaces are organized and managed by the model
- [`AutoHWCustomOp`](#autohwcustomop) ↔ [`DataflowModel`](#dataflowmodel): Base class uses model as single source of truth
- [`RTLInterfaceConverter`](#rtlinterfaceconverter) → [`DataflowInterface`](#dataflowinterface): Converts RTL to dataflow representations
- [`ValidationResult`](#validationresult) ← All validation methods: Common return type for validation operations
- [`TensorChunking`](#tensor-chunking) ↔ [`DataflowInterface`](#dataflowinterface): Used for dimension inference and validation

---

This API reference provides complete documentation for all public interfaces in the Brainsmith Dataflow Framework. For implementation examples, see the [README](README.md) and [examples](examples/) directory.