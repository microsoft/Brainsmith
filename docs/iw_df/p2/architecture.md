# Interface-Wise Dataflow Modeling Framework Architecture

## Executive Summary

This document defines the complete technical architecture for the Interface-Wise Dataflow Modeling framework. The framework provides a unified abstraction layer for hardware kernel design that simplifies the complexity of integrating custom RTL implementations into the FINN/Brainsmith ecosystem through standardized interface-based modeling and automated code generation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Framework Components](#core-framework-components)
3. [Data Structures](#data-structures)
4. [Computational Model](#computational-model)
5. [Integration Architecture](#integration-architecture)
6. [Code Generation System](#code-generation-system)
7. [Extension Points](#extension-points)
8. [Interface Specifications](#interface-specifications)
9. [Validation and Testing Architecture](#validation-and-testing-architecture)

## Architecture Overview

### Design Philosophy

The Interface-Wise Dataflow Modeling framework follows a layered architecture approach with clear separation of concerns:

1. **Abstraction Layer**: Unified interface representation independent of implementation details
2. **Computational Layer**: Mathematical modeling of dataflow relationships and performance characteristics
3. **Integration Layer**: Seamless connection with existing RTL Parser and HW Kernel Generator infrastructure
4. **Generation Layer**: Automated production of complete, functional HWCustomOp classes

### System Context

```
┌───────────────────────────────────────────────────────────────┐
│                    RTL Source + Pragmas                       │
└─────────────────────┬─────────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│                    RTL Parser                                 │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│  │ Interface   │ │ Parameter    │ │ Pragma                  │ │
│  │ Scanner     │ │ Extractor    │ │ Handler                 │ │
│  └─────────────┘ └──────────────┘ └─────────────────────────┘ │
└─────────────────────┬─────────────────────────────────────────┘
                      │ HWKernel
┌─────────────────────▼─────────────────────────────────────────┐
│            Interface-Wise Dataflow Framework                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │DataflowInterface│ │ DataflowModel   │ │ TensorChunking  │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │ HWKernelDataflow│ │ AutoHWCustomOp  │ │ ValidationSuite │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────┬─────────────────────────────────────────┘
                      │ Generated Code
┌─────────────────────▼─────────────────────────────────────────┐
│                FINN/Brainsmith Integration                    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │ AutoHWCustomOp  │ │ AutoRTLBackend  │ │ Documentation   │  │
│  │ Classes         │ │ Classes         │ │ Generation      │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Separation of Concerns**: Clear boundaries between interface modeling, computational analysis, and code generation
2. **Composability**: Framework components can be used independently or in combination
3. **Extensibility**: Well-defined extension points for new interface types, computational models, and generation targets
4. **Validation**: Comprehensive validation at every layer to ensure correctness and reliability
5. **Performance**: Efficient algorithms and data structures optimized for common use cases

## Core Framework Components

### 1. DataflowInterface

The foundational abstraction for all hardware interfaces in the system.

#### Class Definition
```python
@dataclass
class DataflowInterface:
    """
    Unified abstraction for hardware kernel interfaces providing 
    standardized representation of interface characteristics.
    """
    name: str                           # Interface identifier (e.g., "in0", "out0", "weights")
    interface_type: DataflowInterfaceType  # INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL
    qDim: List[int]                     # Query dimensions (complete data set)
    tDim: List[int]                     # Tensor dimensions (per calculation)
    sDim: List[int]                     # Stream dimensions (per clock cycle)
    dtype: DataflowDataType             # Element data type specification
    allowed_datatypes: List[DataTypeConstraint]  # Allowed datatypes from DATATYPE pragma
    axi_metadata: Dict[str, Any]        # Protocol-specific metadata
    constraints: List[Constraint]       # Interface-specific constraints
    pragma_metadata: Dict[str, Any]    # Pragma-derived information
```

#### Interface Type Hierarchy
```python
class DataflowInterfaceType(Enum):
    INPUT = "input"      # AXI-Stream input for activation data
    OUTPUT = "output"    # AXI-Stream output for result data  
    WEIGHT = "weight"    # AXI-Stream input for weight/parameter data
    CONFIG = "config"    # AXI-Lite for runtime configuration
    CONTROL = "control"  # Global control signals (clk, rst, etc.)

class DataflowDataType:
    """Enhanced datatype specification supporting FINN compatibility"""
    base_type: str       # INT, UINT, FLOAT, FIXED
    bitwidth: int        # Bit precision
    signed: bool         # Sign specification
    finn_type: str       # FINN DataType string representation

@dataclass
class DataTypeConstraint:
    """Constraint on allowed datatypes for an interface"""
    base_types: List[str]        # Allowed base types (INT, UINT, FLOAT, FIXED)
    min_bitwidth: int            # Minimum allowed bitwidth
    max_bitwidth: int            # Maximum allowed bitwidth
    signed_allowed: bool         # Whether signed types are allowed
    unsigned_allowed: bool       # Whether unsigned types are allowed
```

#### Key Methods
```python
def calculate_stream_width(self) -> int:
    """Calculate AXI stream width based on sDim and dtype"""
    
def validate_constraints(self) -> List[ValidationError]:
    """Validate interface constraints and configuration"""
    
def apply_parallelism(self, iPar: int, wPar: int) -> None:
    """Update sDim based on parallelism parameters"""
    
def get_axi_signals(self) -> Dict[str, Signal]:
    """Generate AXI signal specifications for this interface"""
    
def validate_datatype(self, target_dtype: DataflowDataType) -> bool:
    """Validate that target datatype is allowed for this interface"""
    for constraint in self.allowed_datatypes:
        if (target_dtype.base_type in constraint.base_types and
            constraint.min_bitwidth <= target_dtype.bitwidth <= constraint.max_bitwidth and
            ((target_dtype.signed and constraint.signed_allowed) or 
             (not target_dtype.signed and constraint.unsigned_allowed))):
            return True
    return False
```

### 2. DataflowModel

The computational core implementing mathematical relationships between interfaces.

#### Class Definition
```python
class DataflowModel:
    """
    Core computational model implementing mathematical relationships
    between interfaces and parallelism parameters.
    """
    
    def __init__(self, interfaces: List[DataflowInterface], parameters: Dict[str, Any]):
        self.interfaces = self._organize_interfaces(interfaces)
        self.parameters = parameters
        self.constraints = self._extract_constraints()
        self.computation_graph = self._build_computation_graph()
```

#### Interface Organization
```python
@property
def input_interfaces(self) -> List[DataflowInterface]:
    """All INPUT type interfaces"""
    
@property  
def output_interfaces(self) -> List[DataflowInterface]:
    """All OUTPUT type interfaces"""
    
@property
def weight_interfaces(self) -> List[DataflowInterface]:
    """All WEIGHT type interfaces"""
    
@property
def config_interfaces(self) -> List[DataflowInterface]:
    """All CONFIG type interfaces"""
    
@property
def control_interfaces(self) -> List[DataflowInterface]:
    """All CONTROL type interfaces"""
```

#### Computational Methods
```python
def calculate_initiation_intervals(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals:
    """
    Unified calculation of cII, eII, L for given parallelism parameters.
    Handles both simple and multi-interface cases automatically.
    
    Returns:
        InitiationIntervals containing:
        - cII: Calculation Initiation Interval per input interface
        - eII: Execution Initiation Interval per input interface
        - L: Inference Cycle Latency
        - bottleneck_analysis: Performance bottleneck information
    """
    
def validate_mathematical_constraints(self) -> List[ConstraintViolation]:
    """
    Validate divisibility and other mathematical constraints
    """
    
def get_parallelism_bounds(self) -> Dict[str, ParallelismBounds]:
    """
    Calculate valid bounds for iPar/wPar parameters for FINN optimization
    
    Returns dictionary mapping interface names to their parallelism bounds
    """
```

#### Mathematical Implementation
```python
class InitiationIntervals:
    """Container for initiation interval calculations"""
    cII: Dict[str, int]  # Per input interface calculation intervals
    eII: Dict[str, int]  # Per input interface execution intervals
    L: int               # Overall inference latency
    bottleneck_analysis: Dict[str, Any]  # Performance bottleneck information

@dataclass
class ParallelismBounds:
    """Valid bounds for parallelism parameters"""
    interface_name: str
    min_value: int
    max_value: int
    divisibility_constraints: List[int]  # Values that must divide evenly

def calculate_initiation_intervals(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals:
    """
    Unified initiation interval calculation for all interface configurations
    
    Algorithm:
    1. Update stream dimensions for all interfaces based on parallelism
    2. Calculate cII for each input interface: cII_i = ∏(tDim_i / sDim_i)
    3. Calculate eII for each input considering weight constraints
    4. Determine overall latency L and bottleneck analysis
    """
    
    input_interfaces = self.input_interfaces
    weight_interfaces = self.weight_interfaces
    output_interfaces = self.output_interfaces
    
    if not input_interfaces:
        return InitiationIntervals(cII={}, eII={}, L=1, bottleneck_analysis={})
    
    cII_per_input = {}
    eII_per_input = {}
    
    # Calculate for each input interface
    for input_if in input_interfaces:
        # Update input stream dimensions
        input_if.sDim = [iPar[input_if.name]]
        
        # Calculate cII for this input
        cII_per_input[input_if.name] = np.prod([tdim // sdim for tdim, sdim in zip(input_if.tDim, input_if.sDim)])
        
        # Find maximum weight constraint for this input
        max_weight_cycles = 1
        for weight_if in weight_interfaces:
            # Update weight stream dimensions relative to this input
            weight_if.sDim = [wPar[weight_if.name] * iPar[input_if.name] * 
                             (weight_if.tDim[0] // input_if.tDim[0])]
            weight_cycles = np.prod([qdim // wPar[weight_if.name] for qdim in weight_if.qDim])
            max_weight_cycles = max(max_weight_cycles, weight_cycles)
        
        # Calculate eII for this input
        eII_per_input[input_if.name] = cII_per_input[input_if.name] * max_weight_cycles
    
    # Determine bottleneck and overall latency
    bottleneck_input = max(input_interfaces, key=lambda iface: eII_per_input[iface.name])
    L = eII_per_input[bottleneck_input.name] * np.prod(bottleneck_input.qDim)
    
    # Update output stream dimensions based on bottleneck
    for output_if in output_interfaces:
        output_if.sDim = [iPar[bottleneck_input.name] * (output_if.tDim[0] // bottleneck_input.tDim[0])]
    
    return InitiationIntervals(
        cII=cII_per_input,
        eII=eII_per_input,
        L=L,
        bottleneck_analysis={"bottleneck_input": bottleneck_input.name}
    )
```

### 3. TensorChunking

Handles mapping between ONNX tensor layouts and dataflow interface dimensions.

#### Class Definition
```python
class TensorChunking:
    """
    Handles ONNX layout to qDim/tDim mapping with support for
    pragma-based overrides and complex chunking patterns.
    """
    
    # Standard layout mapping table
    LAYOUT_MAPPINGS = {
        "[N, C]": {"qDim_func": lambda shape: [1], "tDim_func": lambda shape: [shape[1]]},
        "[N, C, H, W]": {"qDim_func": lambda shape: [shape[1]], "tDim_func": lambda shape: [shape[2] * shape[3]]},
        "[N, H, W, C]": {"qDim_func": lambda shape: [shape[1] * shape[2]], "tDim_func": lambda shape: [shape[3]]},
        "[N, L, C]": {"qDim_func": lambda shape: [shape[1]], "tDim_func": lambda shape: [shape[2]]},
        "[N, C, L]": {"qDim_func": lambda shape: [shape[1]], "tDim_func": lambda shape: [shape[2]]},
        "[N, L, h, d]": {"qDim_func": lambda shape: [shape[1]], "tDim_func": lambda shape: [shape[2] * shape[3]]}
    }
```

#### Core Methods
```python
@staticmethod
def infer_dimensions(onnx_layout: str, shape: List[int]) -> Tuple[List[int], List[int]]:
    """
    Map ONNX tensor layout to qDim/tDim using standard patterns
    
    Args:
        onnx_layout: Layout string (e.g., "[N, C, H, W]")
        shape: Tensor shape dimensions
        
    Returns:
        Tuple of (qDim, tDim) lists
    """
    
@staticmethod
def apply_tdim_pragma(pragma: TDimPragma, parameters: Dict[str, Any]) -> List[int]:
    """
    Apply TDIM pragma to override default chunking
    
    Args:
        pragma: Parsed TDIM pragma with dimension expressions
        parameters: Module parameters for expression evaluation
        
    Returns:
        Computed tDim list
    """
    
def validate_chunking(self, interface: DataflowInterface) -> List[ValidationError]:
    """
    Validate that qDim, tDim, sDim relationships are mathematically sound
    """
```

### 4. HWKernelDataflow

Enhanced kernel representation with dataflow modeling capabilities.

#### Class Definition
```python
class HWKernelDataflow:
    """
    Enhanced HWKernel representation with integrated dataflow modeling
    providing unified interface for kernel analysis and code generation.
    """
    
    def __init__(self, hw_kernel: HWKernel, onnx_metadata: Optional[Dict] = None):
        self.hw_kernel = hw_kernel
        self.onnx_metadata = onnx_metadata or {}
        self.dataflow_interfaces = self._build_dataflow_interfaces()
        self.dataflow_model = self._build_dataflow_model()
        self.parallelism_constraints = self._extract_parallelism_constraints()
        self.validation_results = self._validate_configuration()
```

#### Interface Conversion
```python
def _build_dataflow_interfaces(self) -> List[DataflowInterface]:
    """
    Convert RTL Parser interfaces to DataflowInterface objects
    
    Process:
    1. Identify interface types from RTL Parser Interface objects
    2. Extract qDim/tDim from ONNX metadata or pragma overrides
    3. Apply TDIM pragma modifications if present
    4. Initialize sDim with default values (updated during parallelism optimization)
    5. Extract datatype information and constraints from DATATYPE pragmas
    """
    
def _extract_parallelism_constraints(self) -> Dict[str, ParallelismBounds]:
    """
    Extract parallelism constraints from pragmas and interface characteristics
    
    Returns constraints for:
    - iPar bounds per input interface
    - wPar bounds per weight interface  
    - Cross-interface coupling constraints
    - Mathematical divisibility requirements
    """
```

#### Code Generation Interface
```python
def generate_auto_hwcustomop(self, template_path: str) -> str:
    """
    Generate complete AutoHWCustomOp Python class code
    
    Args:
        template_path: Path to Jinja2 template file
        
    Returns:
        Complete Python class implementation
    """
    
def get_template_context(self) -> Dict[str, Any]:
    """
    Build complete template context for code generation
    
    Returns dictionary containing:
    - kernel_metadata: Name, parameters, source info
    - dataflow_interfaces: Complete interface specifications
    - computational_model: Mathematical relationships and constraints
    - method_implementations: Standardized method bodies
    - resource_estimation_stubs: Placeholder method signatures
    """
```

### 5. AutoHWCustomOp

The standardized base class implementing most HWCustomOp functionality.

#### Class Definition
```python
class AutoHWCustomOp(HWCustomOp):
    """
    Standardized HWCustomOp base class implementing common functionality
    through dataflow modeling, reducing manual implementation effort.
    """
    
    def __init__(self, onnx_node, dataflow_interfaces: List[DataflowInterface], 
                 dataflow_model: DataflowModel, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.dataflow_interfaces = {iface.name: iface for iface in dataflow_interfaces}
        self.dataflow_model = dataflow_model
        self._validate_initialization()
```

#### Standardized Method Implementations
```python
def get_input_datatype(self, ind: int = 0) -> DataType:
    """
    Standardized implementation based on dataflow_interfaces
    
    Returns FINN DataType for specified input interface index
    """
    input_interfaces = [iface for iface in self.dataflow_interfaces.values() 
                       if iface.interface_type == DataflowInterfaceType.INPUT]
    if ind >= len(input_interfaces):
        raise IndexError(f"Input index {ind} exceeds available inputs {len(input_interfaces)}")
    return DataType[input_interfaces[ind].dtype.finn_type]

def get_output_datatype(self, ind: int = 0) -> DataType:
    """Standardized implementation for output datatypes"""
    
def get_normal_input_shape(self, ind: int = 0) -> Tuple[int, ...]:
    """
    Return ONNX tensor shape (qDim * tDim)
    """
    input_iface = self._get_input_interface(ind)
    return tuple(np.array(input_iface.qDim) * np.array(input_iface.tDim))

def get_folded_input_shape(self, ind: int = 0) -> Tuple[int, ...]:
    """
    Return hardware-folded shape based on parallelism parameters
    
    Folding applied as: original_shape[:-1] + [tDim//sDim, sDim]
    """
    
def get_instream_width(self, ind: int = 0) -> int:
    """
    Stream width calculation: sDim * dtype.bitwidth with 8-bit alignment
    """
    input_iface = self._get_input_interface(ind)
    raw_width = np.prod(input_iface.sDim) * input_iface.dtype.bitwidth
    return ((raw_width + 7) // 8) * 8  # Align to 8-bit boundaries

def get_outstream_width(self, ind: int = 0) -> int:
    """Output stream width calculation"""
    
def get_exp_cycles(self) -> int:
    """
    Expected cycles calculation using dataflow model
    
    Returns L (Inference Cycle Latency) from current parallelism configuration
    """
    current_parallelism = self._get_current_parallelism()
    intervals = self.dataflow_model.calculate_initiation_intervals(**current_parallelism)
    return intervals.L
```

#### Resource Estimation Placeholders
```python
def bram_estimation(self) -> int:
    """
    BRAM resource estimation placeholder
    
    Raises:
        NotImplementedError with detailed instructions for user implementation
    """
    raise NotImplementedError(
        f"BRAM estimation for {self.__class__.__name__} must be implemented by user.\n"
        f"Please implement this method considering:\n"
        f"- Weight memory requirements: {self._get_weight_memory_summary()}\n"
        f"- Activation buffer requirements: {self._get_activation_buffer_summary()}\n"  
        f"- Current parallelism: {self._get_current_parallelism()}\n"
        f"See FINN documentation for estimation guidelines."
    )

def lut_estimation(self) -> int:
    """LUT resource estimation placeholder with guidance"""
    
def dsp_estimation(self) -> int:
    """DSP resource estimation placeholder with guidance"""
    
def uram_estimation(self) -> int:
    """UltraRAM resource estimation placeholder with guidance"""
```

#### Helper Methods
```python
def _get_current_parallelism(self) -> Dict[str, Dict[str, int]]:
    """Extract current iPar/wPar from node attributes"""
    
def _validate_parallelism_constraints(self) -> List[ConstraintViolation]:
    """Validate current parallelism against interface constraints"""
    
def _get_weight_memory_summary(self) -> Dict[str, Any]:
    """Summary of weight memory requirements for resource estimation guidance"""
    
def _get_activation_buffer_summary(self) -> Dict[str, Any]:
    """Summary of activation buffer requirements for resource estimation guidance"""
```

## Data Structures

### Core Data Types

#### Constraint System
```python
@dataclass
class Constraint:
    """Base constraint representation"""
    name: str
    constraint_type: ConstraintType
    parameters: Dict[str, Any]
    error_message: str

class ConstraintType(Enum):
    DIVISIBILITY = "divisibility"      # Mathematical divisibility requirements
    RANGE = "range"                    # Parameter value ranges
    DEPENDENCY = "dependency"          # Inter-parameter dependencies
    RESOURCE = "resource"              # Resource limit constraints

@dataclass
class DivisibilityConstraint(Constraint):
    """Divisibility constraint (e.g., tDim % sDim == 0)"""
    dividend: str      # Parameter name or expression
    divisor: str       # Parameter name or expression
    
@dataclass
class RangeConstraint(Constraint):
    """Range constraint (e.g., 1 <= iPar <= tDim)"""
    parameter: str
    min_value: Union[int, str]  # Literal or parameter reference
    max_value: Union[int, str]  # Literal or parameter reference
```

#### Validation Framework
```python
@dataclass
class ValidationError:
    """Standardized validation error representation"""
    component: str              # Component where error occurred
    error_type: str            # Error classification
    message: str               # Human-readable error description
    severity: ValidationSeverity  # ERROR, WARNING, INFO
    context: Dict[str, Any]    # Additional context for debugging

class ValidationSeverity(Enum):
    ERROR = "error"      # Blocks code generation
    WARNING = "warning"  # Should be addressed but not blocking
    INFO = "info"        # Informational only

@dataclass
class ValidationResult:
    """Complete validation result set"""
    success: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    info: List[ValidationError]
    
    def has_blocking_errors(self) -> bool:
        return any(err.severity == ValidationSeverity.ERROR for err in self.errors)
```

#### Parallelism Management
```python
@dataclass
class ParallelismConfiguration:
    """Complete parallelism configuration for a kernel"""
    iPar: Dict[str, int]  # Input parallelism per input interface
    wPar: Dict[str, int]  # Weight parallelism per weight interface
    derived_sDim: Dict[str, List[int]]  # Computed stream dimensions
```

## Computational Model

### Mathematical Relationships

The computational model provides a unified approach to calculating initiation intervals for any interface configuration, automatically handling both simple and complex multi-interface scenarios.

#### Unified Initiation Interval Calculation

```python
def calculate_initiation_intervals(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals:
    """
    Unified calculation handling all interface configurations
    
    Algorithm:
    1. Update stream dimensions: sDim_I = iPar, sDim_W = wPar * iPar * (tDim_W / tDim_I)
    2. Calculate cII per input: cII_i = ∏(tDim_I / sDim_I)
    3. Calculate eII per input: eII_i = cII_i * max(∏(qDim_W / wPar)) over all weights
    4. Determine bottleneck: L = max(eII_i * ∏(qDim_I)) over all inputs
    """
    
    input_interfaces = self.input_interfaces
    weight_interfaces = self.weight_interfaces
    output_interfaces = self.output_interfaces
    
    if not input_interfaces:
        return InitiationIntervals(cII={}, eII={}, L=1, bottleneck_analysis={})
    
    cII_per_input = {}
    eII_per_input = {}
    
    # Calculate for each input interface
    for input_if in input_interfaces:
        # Update input stream dimensions
        input_if.sDim = [iPar[input_if.name]]
        
        # Calculate cII for this input
        cII_per_input[input_if.name] = np.prod([tdim // sdim for tdim, sdim in zip(input_if.tDim, input_if.sDim)])
        
        # Find maximum weight constraint for this input
        max_weight_cycles = 1
        for weight_if in weight_interfaces:
            # Update weight stream dimensions relative to this input
            weight_if.sDim = [wPar[weight_if.name] * iPar[input_if.name] * 
                             (weight_if.tDim[0] // input_if.tDim[0])]
            weight_cycles = np.prod([qdim // wPar[weight_if.name] for qdim in weight_if.qDim])
            max_weight_cycles = max(max_weight_cycles, weight_cycles)
        
        # Calculate eII for this input
        eII_per_input[input_if.name] = cII_per_input[input_if.name] * max_weight_cycles
    
    # Determine bottleneck and overall latency
    bottleneck_input = max(input_interfaces, key=lambda iface: eII_per_input[iface.name])
    L = eII_per_input[bottleneck_input.name] * np.prod(bottleneck_input.qDim)
    
    # Update output stream dimensions based on bottleneck
    for output_if in output_interfaces:
        output_if.sDim = [iPar[bottleneck_input.name] * (output_if.tDim[0] // bottleneck_input.tDim[0])]
    
    return InitiationIntervals(
        cII=cII_per_input,
        eII=eII_per_input,
        L=L,
        bottleneck_analysis={"bottleneck_input": bottleneck_input.name}
    )
```

### Constraint Validation

#### Mathematical Constraint Checking
```python
def validate_mathematical_constraints(interfaces: List[DataflowInterface]) -> List[ValidationError]:
    """
    Validate mathematical relationships between dimensions
    
    Checks:
    - Stream dimensions tile into tensor dimensions: tDim % sDim == 0
    - Tensor dimensions tile into query dimensions: qDim % tDim == 0  
    - Parallelism parameters within valid ranges
    - Cross-interface consistency requirements
    """
    errors = []
    
    for interface in interfaces:
        # Validate tiling relationships
        for i, (tdim, sdim) in enumerate(zip(interface.tDim, interface.sDim)):
            if tdim % sdim != 0:
                errors.append(ValidationError(
                    component=f"interface.{interface.name}",
                    error_type="divisibility_violation", 
                    message=f"tDim[{i}]={tdim} must be divisible by sDim[{i}]={sdim}",
                    severity=ValidationSeverity.ERROR,
                    context={"interface": interface.name, "dimension": i, "tDim": tdim, "sDim": sdim}
                ))
                
        for i, (qdim, tdim) in enumerate(zip(interface.qDim, interface.tDim)):
            if qdim % tdim != 0:
                errors.append(ValidationError(
                    component=f"interface.{interface.name}",
                    error_type="divisibility_violation",
                    message=f"qDim[{i}]={qdim} must be divisible by tDim[{i}]={tdim}", 
                    severity=ValidationSeverity.ERROR,
                    context={"interface": interface.name, "dimension": i, "qDim": qdim, "tDim": tdim}
                ))
    
    return errors
```

### FINN Optimization Integration

The framework provides all necessary information for FINN's optimization algorithms without implementing optimization itself:

```python
def get_parallelism_bounds(self) -> Dict[str, ParallelismBounds]:
    """
    Calculate valid bounds for iPar/wPar parameters for FINN optimization
    
    Returns comprehensive bounds information that FINN can use for:
    - SetFolding transformation optimization
    - Resource-constrained optimization
    - Performance target optimization
    """
    bounds = {}
    
    # Calculate bounds for input interfaces (iPar)
    for input_if in self.input_interfaces:
        min_val = 1
        max_val = np.prod(input_if.tDim)  # Maximum possible parallelism
        divisibility = input_if.tDim.copy()  # Must divide tensor dimensions
        
        bounds[f"{input_if.name}_iPar"] = ParallelismBounds(
            interface_name=input_if.name,
            min_value=min_val,
            max_value=max_val,
            divisibility_constraints=divisibility
        )
    
    # Calculate bounds for weight interfaces (wPar)
    for weight_if in self.weight_interfaces:
        min_val = 1
        max_val = np.prod(weight_if.qDim)  # Maximum possible parallelism
        divisibility = weight_if.qDim.copy()  # Must divide query dimensions
        
        bounds[f"{weight_if.name}_wPar"] = ParallelismBounds(
            interface_name=weight_if.name,
            min_value=min_val,
            max_value=max_val,
            divisibility_constraints=divisibility
        )
    
    return bounds
```

## Integration Architecture

### RTL Parser Integration

#### Enhanced Pragma Support
```python
class TDimPragma(Pragma):
    """
    TDIM pragma for custom tensor dimension specification
    
    Format: // @brainsmith TDIM <interface_name> <dim1_expr> <dim2_expr> ... <dimN_expr>
    Example: // @brainsmith TDIM in0 PE*CHANNELS 1
    """
    
    def _parse_inputs(self) -> Dict:
        if len(self.inputs) < 2:
            raise PragmaError("TDIM pragma requires interface name and at least one dimension expression")
        
        interface_name = self.inputs[0]
        dimension_expressions = self.inputs[1:]
        
        return {
            "interface_name": interface_name,
            "dimension_expressions": dimension_expressions
        }
    
    def apply(self, **kwargs) -> Any:
        """Apply TDIM pragma to override default tensor chunking"""
        interfaces = kwargs.get('interfaces', {})
        parameters = kwargs.get('parameters', {})
        
        interface_name = self.parsed_data["interface_name"]
        dimension_exprs = self.parsed_data["dimension_expressions"]
        
        # Find matching interface
        target_interface = None
        for iface in interfaces.values():
            if iface.name == interface_name:
                target_interface = iface
                break
                
        if not target_interface:
            logger.warning(f"TDIM pragma: interface '{interface_name}' not found")
            return
            
        # Evaluate dimension expressions
        try:
            evaluated_dims = []
            for expr in dimension_exprs:
                # Simple expression evaluator (could be enhanced)
                evaluated_dims.append(self._evaluate_expression(expr, parameters))
            
            # Store in metadata for later processing by TensorChunking
            target_interface.metadata["tdim_override"] = evaluated_dims
            logger.info(f"Applied TDIM pragma: {interface_name} tDim set to {evaluated_dims}")
            
        except Exception as e:
            logger.error(f"TDIM pragma evaluation failed: {e}")

class DataTypePragma(Pragma):
    """
    Enhanced DATATYPE pragma supporting datatype constraints
    
    Format: // @brainsmith DATATYPE <interface_name> <base_types> <min_bits> <max_bits> <signed> <unsigned>
    Example: // @brainsmith DATATYPE in0 INT,UINT 1 16 true true
    Example: // @brainsmith DATATYPE weights FIXED 8 8 true false
    """
    
    def _parse_inputs(self) -> Dict:
        if len(self.inputs) < 6:
            raise PragmaError("DATATYPE pragma requires interface_name, base_types, min_bits, max_bits, signed, unsigned")
        
        interface_name = self.inputs[0]
        base_types = [t.strip() for t in self.inputs[1].split(',')]
        min_bits = int(self.inputs[2])
        max_bits = int(self.inputs[3])
        signed_allowed = self.inputs[4].lower() == 'true'
        unsigned_allowed = self.inputs[5].lower() == 'true'
        
        return {
            "interface_name": interface_name,
            "base_types": base_types,
            "min_bitwidth": min_bits,
            "max_bitwidth": max_bits,
            "signed_allowed": signed_allowed,
            "unsigned_allowed": unsigned_allowed
        }
    
    def apply(self, **kwargs) -> Any:
        """Apply DATATYPE pragma to set interface datatype constraints"""
        interfaces = kwargs.get('interfaces', {})
        
        interface_name = self.parsed_data["interface_name"]
        
        # Find matching interface
        target_interface = None
        for iface in interfaces.values():
            if iface.name == interface_name:
                target_interface = iface
                break
                
        if not target_interface:
            logger.warning(f"DATATYPE pragma: interface '{interface_name}' not found")
            return
            
        # Create datatype constraint
        constraint = DataTypeConstraint(
            base_types=self.parsed_data["base_types"],
            min_bitwidth=self.parsed_data["min_bitwidth"],
            max_bitwidth=self.parsed_data["max_bitwidth"],
            signed_allowed=self.parsed_data["signed_allowed"],
            unsigned_allowed=self.parsed_data["unsigned_allowed"]
        )
        
        # Store constraint in interface metadata
        if "datatype_constraints" not in target_interface.metadata:
            target_interface.metadata["datatype_constraints"] = []
        target_interface.metadata["datatype_constraints"].append(constraint)
        
        logger.info(f"Applied DATATYPE pragma: {interface_name} constraints set")
```

### HW Kernel Generator Integration

The HardwareKernelGenerator class is directly enhanced rather than creating a separate wrapper class:

#### Enhanced HardwareKernelGenerator
```python
class HardwareKernelGenerator:
    """Enhanced HKG with integrated dataflow modeling support"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataflow_enabled = True
    
    def generate_auto_hwcustomop(self, hw_kernel_dataflow: HWKernelDataflow, 
                                template_path: str, output_path: str) -> str:
        """
        Generate complete AutoHWCustomOp class using dataflow modeling
        
        Process:
        1. Validate HWKernelDataflow configuration
        2. Build template context with dataflow metadata
        3. Render Jinja2 template with standardized method implementations
        4. Validate generated code syntax and logic
        5. Write to output file with proper formatting
        """
        
        # Validation
        validation_result = hw_kernel_dataflow.validate_configuration()
        if validation_result.has_blocking_errors():
            raise GenerationError(f"Cannot generate code with validation errors: {validation_result.errors}")
            
        # Template context
        context = self._build_template_context(hw_kernel_dataflow)
        
        # Render template
        try:
            template = self.jinja_env.get_template(template_path)
            generated_code = template.render(**context)
        except Exception as e:
            raise GenerationError(f"Template rendering failed: {e}")
            
        # Validate generated code
        self._validate_generated_code(generated_code)
        
        # Write output
        with open(output_path, 'w') as f:
            f.write(generated_code)
            
        return output_path
        
    def _build_template_context(self, hw_kernel_dataflow: HWKernelDataflow) -> Dict[str, Any]:
        """Build comprehensive template context for code generation"""
        
        return {
            # Kernel metadata
            "kernel_name": hw_kernel_dataflow.hw_kernel.name,
            "class_name": f"Auto{hw_kernel_dataflow.hw_kernel.name.title()}",
            "source_file": hw_kernel_dataflow.hw_kernel.metadata.get("source_file"),
            "generation_timestamp": datetime.now().isoformat(),
            
            # Interface specifications
            "dataflow_interfaces": hw_kernel_dataflow.dataflow_interfaces,
            "input_interfaces": hw_kernel_dataflow.dataflow_model.input_interfaces,
            "output_interfaces": hw_kernel_dataflow.dataflow_model.output_interfaces,
            "weight_interfaces": hw_kernel_dataflow.dataflow_model.weight_interfaces,
            "config_interfaces": hw_kernel_dataflow.dataflow_model.config_interfaces,
            
            # Parameters and attributes
            "kernel_parameters": hw_kernel_dataflow.hw_kernel.parameters,
            "nodeattr_types": self._generate_nodeattr_types(hw_kernel_dataflow),
            
            # Computational model
            "computational_relationships": self._generate_computational_methods(hw_kernel_dataflow),
            "constraint_validation": self._generate_constraint_validation(hw_kernel_dataflow),
            
            # Resource estimation stubs
            "resource_estimation_stubs": self._generate_resource_stubs(hw_kernel_dataflow),
            
            # Helper methods
            "helper_methods": self._generate_helper_methods(hw_kernel_dataflow)
        }
```

## Code Generation System

### Template Architecture

#### AutoHWCustomOp Template Structure
```jinja2
{# templates/auto_hwcustomop.py.j2 #}

############################################################################
# Copyright (C) {{ generation_timestamp.year }}, Advanced Micro Devices, Inc.
# All rights reserved.
# 
# SPDX-License-Identifier: MIT
#
# Auto-generated HWCustomOp for {{ kernel_name }}
# Generated from: {{ source_file }}
# Generation timestamp: {{ generation_timestamp }}
############################################################################

import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType

# Dataflow framework imports
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
from brainsmith.dataflow.core.dataflow_model import DataflowModel
from brainsmith.dataflow.auto_hwcustomop import AutoHWCustomOp

class {{ class_name }}(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for {{ kernel_name }} kernel.
    
    This class was automatically generated using the Interface-Wise Dataflow
    Modeling framework. Most methods are standardized implementations from
    the AutoHWCustomOp base class.
    
    Generated from RTL: {{ source_file }}
    
    Interfaces:
    {% for interface in dataflow_interfaces %}
    - {{ interface.name }}: {{ interface.interface_type.value }} ({{ interface.dtype.finn_type }})
    {% endfor %}
    """
    
    def __init__(self, onnx_node, **kwargs):
        # Interface definitions
        self._dataflow_interfaces = [
            {% for interface in dataflow_interfaces %}
            DataflowInterface(
                name="{{ interface.name }}",
                interface_type=DataflowInterfaceType.{{ interface.interface_type.name }},
                qDim={{ interface.qDim }},
                tDim={{ interface.tDim }},
                sDim={{ interface.sDim }},
                dtype={{ interface.dtype|to_datatype_spec }},
                allowed_datatypes={{ interface.allowed_datatypes|to_constraint_spec }},
                axi_metadata={{ interface.axi_metadata }},
                constraints={{ interface.constraints|to_constraint_spec }},
                pragma_metadata={{ interface.pragma_metadata }}
            ),
            {% endfor %}
        ]
        
        # Initialize dataflow model
        dataflow_model = DataflowModel(self._dataflow_interfaces, {})
        
        super().__init__(onnx_node, self._dataflow_interfaces, dataflow_model, **kwargs)
    
    def get_nodeattr_types(self):
        """Node attribute type specifications"""
        my_attrs = {
            {% for param in kernel_parameters %}
            "{{ param.name }}": {{ param|to_nodeattr_spec }},
            {% endfor %}
            # Parallelism parameters
            {% for interface in input_interfaces %}
            "{{ interface.name }}_iPar": ("i", False, 1),
            {% endfor %}
            {% for interface in weight_interfaces %}  
            "{{ interface.name }}_wPar": ("i", False, 1),
            {% endfor %}
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs
    
    # Resource estimation methods (must be implemented by user)
    def bram_estimation(self) -> int:
        """
        BRAM resource estimation for {{ kernel_name }}
        
        TODO: Implement based on your kernel's memory requirements
        
        Consider:
        {% for interface in weight_interfaces %}
        - {{ interface.name }}: Weight storage requirements
        {% endfor %}
        - Activation buffering requirements
        - Current parallelism configuration
        
        Example implementation:
        ```python
        # Calculate weight memory requirements
        weight_bits = 0
        {% for interface in weight_interfaces %}
        {{ interface.name }}_bits = (
            np.prod({{ interface.qDim }}) *  # Total weights
            np.prod({{ interface.tDim }}) *  # Per-calculation weights  
            {{ interface.dtype.bitwidth }}   # Bits per weight
        )
        weight_bits += {{ interface.name }}_bits
        {% endfor %}
        
        # Estimate BRAMs needed
        bram_capacity = 36 * 512  # BRAM18 capacity in bits
        return math.ceil(weight_bits / bram_capacity)
        ```
        """
        raise NotImplementedError(
            f"BRAM estimation for {{ class_name }} must be implemented by user. "
            f"See method docstring for implementation guidance."
        )
    
    def lut_estimation(self) -> int:
        """
        LUT resource estimation for {{ kernel_name }}
        
        TODO: Implement based on your kernel's logic requirements
        """
        raise NotImplementedError(
            f"LUT estimation for {{ class_name }} must be implemented by user."
        )
    
    def dsp_estimation(self) -> int:
        """
        DSP resource estimation for {{ kernel_name }}
        
        TODO: Implement based on your kernel's arithmetic requirements
        """
        raise NotImplementedError(
            f"DSP estimation for {{ class_name }} must be implemented by user."
        )
    
    # Kernel-specific overrides (customize as needed)
    def verify_node(self):
        """
        Verify node configuration and constraints
        
        This implementation checks standard dataflow constraints.
        Override if additional kernel-specific validation is needed.
        """
        super().verify_node()
        
        # Add any kernel-specific validation here
        {% if constraint_validation %}
        {{ constraint_validation|indent(8) }}
        {% endif %}
```

#### Template Filters and Functions
```python
def to_datatype_spec(dtype: DataflowDataType) -> str:
    """Convert DataflowDataType to template-ready specification"""
    return f"DataflowDataType(base_type='{dtype.base_type}', bitwidth={dtype.bitwidth}, signed={dtype.signed}, finn_type='{dtype.finn_type}')"

def to_constraint_spec(constraints: List[Constraint]) -> str:
    """Convert constraint list to template-ready specification"""
    constraint_specs = []
    for constraint in constraints:
        spec = f"{constraint.__class__.__name__}({constraint.__dict__})"
        constraint_specs.append(spec)
    return f"[{', '.join(constraint_specs)}]"

def to_nodeattr_spec(parameter: Parameter) -> str:
    """Convert Parameter to FINN nodeattr specification"""
    if parameter.param_type == "int":
        return f'("i", False, {parameter.default_value or 0})'
    elif parameter.param_type == "string":
        return f'("s", True, "{parameter.default_value or ""}")'
    elif parameter.param_type == "ints":
        return f'("ints", True, {parameter.default_value or []})'
    else:
        return f'("s", False, "{parameter.default_value or ""}")'
```

### Validation System

#### Generated Code Validation
```python
class GeneratedCodeValidator:
    """Validates generated AutoHWCustomOp code for correctness"""
    
    def validate_syntax(self, code: str) -> ValidationResult:
        """Validate Python syntax of generated code"""
        try:
            ast.parse(code)
            return ValidationResult(success=True, errors=[], warnings=[], info=[])
        except SyntaxError as e:
            error = ValidationError(
                component="code_generation",
                error_type="syntax_error",
                message=f"Generated code has syntax error: {e}",
                severity=ValidationSeverity.ERROR,
                context={"line": e.lineno, "offset": e.offset}
            )
            return ValidationResult(success=False, errors=[error], warnings=[], info=[])
    
    def validate_method_completeness(self, code: str, required_methods: List[str]) -> ValidationResult:
        """Validate that all required methods are implemented"""
        tree = ast.parse(code)
        implemented_methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        
        missing_methods = set(required_methods) - implemented_methods
        if missing_methods:
            error = ValidationError(
                component="code_generation",
                error_type="missing_methods",
                message=f"Generated code missing required methods: {missing_methods}",
                severity=ValidationSeverity.ERROR,
                context={"missing_methods": list(missing_methods)}
            )
            return ValidationResult(success=False, errors=[error], warnings=[], info=[])
            
        return ValidationResult(success=True, errors=[], warnings=[], info=[])
    
    def validate_dataflow_integration(self, hw_kernel_dataflow: HWKernelDataflow, code: str) -> ValidationResult:
        """Validate that generated code properly integrates dataflow model"""
        errors = []
        warnings = []
        
        # Check interface definitions match
        # Check method implementations use dataflow model
        # Check constraint validation is present
        
        return ValidationResult(success=len(errors)==0, errors=errors, warnings=warnings, info=[])
```

## Extension Points

### Interface Type Extension
```python
class CustomDataflowInterfaceType(DataflowInterfaceType):
    """Extended interface types for custom protocols"""
    CUSTOM_STREAMING = "custom_streaming"
    MEMORY_MAPPED = "memory_mapped"
    
class CustomInterfaceHandler:
    """Handler for custom interface types"""
    
    def supports_interface_type(self, interface_type: DataflowInterfaceType) -> bool:
        return interface_type in [CustomDataflowInterfaceType.CUSTOM_STREAMING, 
                                CustomDataflowInterfaceType.MEMORY_MAPPED]
    
    def calculate_stream_dimensions(self, interface: DataflowInterface, parallelism: Dict) -> List[int]:
        """Custom stream dimension calculation for non-standard interfaces"""
        if interface.interface_type == CustomDataflowInterfaceType.CUSTOM_STREAMING:
            return self._calculate_custom_streaming_dimensions(interface, parallelism)
        elif interface.interface_type == CustomDataflowInterfaceType.MEMORY_MAPPED:
            return self._calculate_memory_mapped_dimensions(interface, parallelism)
```

### Computational Model Extension
```python
class CustomComputationalModel(DataflowModel):
    """Extended computational model for specialized kernels"""
    
    def calculate_initiation_intervals(self, iPar: Dict[str, int], wPar: Dict[str, int]) -> InitiationIntervals:
        """Override for custom computational patterns"""
        
        # Check if this kernel requires custom calculation
        if self._requires_custom_calculation():
            return self._calculate_custom_intervals(iPar, wPar)
        else:
            return super().calculate_initiation_intervals(iPar, wPar)
    
    def _requires_custom_calculation(self) -> bool:
        """Determine if kernel requires custom computational model"""
        # Check for specific interface patterns, pragmas, or kernel characteristics
        return False
```

### Template Extension
```python
class CustomTemplateGenerator:
    """Custom template generator for specialized kernels"""
    
    def __init__(self, base_template_path: str):
        self.base_template_path = base_template_path
        self.custom_filters = {
            "custom_filter": self._custom_filter,
            "specialized_conversion": self._specialized_conversion
        }
    
    def generate_custom_methods(self, hw_kernel_dataflow: HWKernelDataflow) -> Dict[str, str]:
        """Generate custom method implementations for specialized kernels"""
        custom_methods = {}
        
        # Generate kernel-specific methods based on interface patterns
        if self._has_complex_interface_pattern(hw_kernel_dataflow):
            custom_methods["custom_validation"] = self._generate_custom_validation(hw_kernel_dataflow)
            custom_methods["specialized_execution"] = self._generate_specialized_execution(hw_kernel_dataflow)
        
        return custom_methods
```

## Interface Specifications

### AXI Interface Generation
```python
def generate_axi_interface_spec(interface: DataflowInterface) -> Dict[str, Any]:
    """Generate complete AXI interface specification from DataflowInterface"""
    
    if interface.interface_type == DataflowInterfaceType.INPUT:
        return {
            "interface_type": "s_axis",
            "signals": {
                f"{interface.name}_TDATA": {
                    "direction": "input",
                    "width": interface.calculate_stream_width(),
                    "description": f"Input data stream for {interface.name}"
                },
                f"{interface.name}_TVALID": {
                    "direction": "input", 
                    "width": 1,
                    "description": f"Valid signal for {interface.name}"
                },
                f"{interface.name}_TREADY": {
                    "direction": "output",
                    "width": 1, 
                    "description": f"Ready signal for {interface.name}"
                }
            }
        }
    elif interface.interface_type == DataflowInterfaceType.OUTPUT:
        return {
            "interface_type": "m_axis",
            "signals": {
                f"{interface.name}_TDATA": {
                    "direction": "output",
                    "width": interface.calculate_stream_width(),
                    "description": f"Output data stream for {interface.name}"
                },
                f"{interface.name}_TVALID": {
                    "direction": "output",
                    "width": 1,
                    "description": f"Valid signal for {interface.name}"
                },
                f"{interface.name}_TREADY": {
                    "direction": "input",
                    "width": 1,
                    "description": f"Ready signal for {interface.name}"
                }
            }
        }
```

## Validation and Testing Architecture

### Testing Framework Structure
```
tests/
├── unit/
│   ├── test_dataflow_interface.py      # DataflowInterface unit tests
│   ├── test_dataflow_model.py          # DataflowModel computational tests
│   ├── test_tensor_chunking.py         # TensorChunking mapping tests
│   └── test_hw_kernel_dataflow.py      # HWKernelDataflow integration tests
├── integration/
│   ├── test_rtl_parser_integration.py  # RTL Parser to dataflow conversion
│   ├── test_hkg_integration.py         # HKG code generation pipeline  
│   └── test_auto_hwcustomop.py         # Generated class functionality
├── end_to_end/
│   ├── test_thresholding_e2e.py        # Complete thresholding pipeline
│   ├── test_validation_pipeline.py     # Validation system testing
│   └── test_performance_modeling.py    # Computational model accuracy
└── fixtures/
    ├── sample_rtl_kernels/             # RTL test cases
    ├── expected_outputs/               # Golden reference outputs
    └── validation_data/                # Test data for validation
```

### Test Coverage Requirements

#### Unit Test Coverage
- **DataflowInterface**: Interface creation, constraint validation, dimension calculations
- **DataflowModel**: Mathematical relationships, parallelism optimization, constraint checking
- **TensorChunking**: ONNX layout mapping, TDIM pragma application, dimension validation  
- **HWKernelDataflow**: RTL conversion, template context generation, validation integration

#### Integration Test Coverage
- **RTL Parser Integration**: Complete pipeline from SystemVerilog to DataflowInterface objects
- **HKG Integration**: Code generation with dataflow modeling, template rendering validation
- **AutoHWCustomOp Integration**: Generated class instantiation, method functionality, FINN compatibility

#### End-to-End Test Coverage
- **Complete Pipeline**: RTL analysis → Dataflow modeling → Code generation → Functional validation
- **Performance Validation**: Computational model accuracy against actual hardware measurements
- **Constraint Validation**: Mathematical constraint checking across diverse kernel configurations

---

## Summary

This architecture provides a comprehensive foundation for the Interface-Wise Dataflow Modeling framework that addresses the core requirements while maintaining extensibility and integration with existing systems. The layered design enables independent development and testing of components while ensuring seamless integration with FINN/Brainsmith infrastructure.

Key architectural improvements based on feedback:

1. **Unified Initiation Interval Calculation**: Single `calculate_initiation_intervals` method handles all cases automatically
2. **Direct HKG Enhancement**: `HardwareKernelGenerator` class is enhanced directly rather than creating wrapper classes  
3. **Added Datatype Constraints**: `allowed_datatypes` attribute enables RTL creators to specify supported data types and bitwidths
4. **FINN Optimization Integration**: Framework exposes parallelism bounds and constraints for FINN's optimization algorithms without implementing optimization itself

The computational model provides mathematically sound relationships between interfaces and parallelism parameters, enabling automated optimization and validation. The code generation system produces complete, functional HWCustomOp classes that reduce manual implementation effort while maintaining quality and maintainability standards.

The extension points ensure the framework can evolve to support new interface types, computational patterns, and generation targets without requiring architectural changes. This design provides a solid foundation for both immediate implementation and long-term framework evolution.
