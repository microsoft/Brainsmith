# Interface-Wise Dataflow Modeling Implementation Strategy

## Executive Summary

This document outlines the comprehensive strategy for implementing the Interface-Wise Dataflow Modeling system in FINN/Brainsmith, based on expert consultation and analysis of the existing codebase. The system will provide seamless integration with existing FINN infrastructure while enabling automated HWCustomOp generation.

## Key Context and Constraints

### Expert Guidance Summary
1. **Seamless Integration**: PE/SIMD maps cleanly to wPar/iPar, ensuring compatibility with current FINN architecture
2. **Automation Focus**: Primary goal is automating and accelerating HW kernel development
3. **New Interface Types**: Create dedicated interface types for the dataflow model
4. **Hybrid Generation**: Use both template-based and programmatic approaches for AutoHWCustomOp generation
5. **API Compatibility**: Maintain full compatibility with existing FINN tools (iPar→SIMD, wPar→PE)
6. **Validation Targets**: Focus on Thresholding and MVAU kernels for validation

### Current FINN Architecture Analysis

#### Existing PE/SIMD System
- **PE (Processing Elements)**: Output parallelism - how many outputs computed simultaneously
- **SIMD**: Input parallelism - how many inputs processed simultaneously  
- **Folding Factor**: `(MH/PE) × (MW/SIMD)` determines cycle count and resource usage
- **Constraints**: `MW % SIMD == 0` and `MH % PE == 0`
- **Current Limitation**: Primarily supports MVAU/VVAU operations

#### Three-Tier HWCustomOp Architecture
1. **Tier 1**: Generic HWCustomOp base class with abstract methods
2. **Tier 2**: Backend interfaces (HLSBackend, RTLBackend) 
3. **Tier 3**: Concrete implementations inheriting from both operation and backend

#### RTL Parser Current Capabilities
- **Interface Detection**: GLOBAL_CONTROL, AXI_STREAM, AXI_LITE
- **Pragma System**: TOP_MODULE, DATATYPE, DERIVED_PARAMETER, WEIGHT
- **Parameter Extraction**: Module parameters with template mapping
- **Validation**: Protocol compliance and constraint checking

#### Hardware Kernel Generator Status
- **Completed**: RTL template generation with Jinja2, CLI interface, error handling
- **Planned**: HWCustomOp generation, RTLBackend generation, documentation

## Interface-Wise Dataflow Model Framework

### Core Concepts

#### Interface Types (New)
Extending beyond current AXI protocol detection:
- **INPUT**: AXI-Stream input for activation data (`iPar` parallelism)
- **OUTPUT**: AXI-Stream output for activation data
- **WEIGHT**: AXI-Stream input for weight data (`wPar` parallelism)
- **CONFIG**: AXI-Lite configuration interface (existing)
- **CONTROL**: Clock/reset signals (existing)

#### Data Hierarchy
- **Query**: Complete dataset for inference/execution
- **Tensor**: Data for single calculation
- **Stream**: Data per clock cycle
- **Element**: Single value with defined datatype

#### Parallelism Parameters
- **iPar**: Input parallelism (maps to SIMD)
- **wPar**: Weight parallelism (maps to PE)
- **Seamless Mapping**: Ensures existing FINN tools work unchanged

### Interface Parameters
- **qDim**: Query dimensions (number of tensors per inference/execution)
- **tDim**: Tensor dimensions (shape for single calculation)
- **sDim**: Stream dimensions (shape per clock cycle)
- **dtype**: Data type of elements

### Computational Model
- **cII**: Calculation Initiation Interval (cycles per calculation)
- **eII**: Execution Initiation Interval (cycles per execution)
- **L**: Inference Cycle Latency (cycles per inference)

## Implementation Architecture

### Phase 2: Dataflow Modeling Framework

#### New Data Structures
```python
# Extend existing InterfaceType enum
class DataflowInterfaceType(Enum):
    INPUT = "input"           # AXI-Stream activation input
    OUTPUT = "output"         # AXI-Stream activation output  
    WEIGHT = "weight"         # AXI-Stream weight input
    CONFIG = "config"         # AXI-Lite configuration
    CONTROL = "control"       # Clock/reset signals

# New dataflow-specific interface representation
@dataclass
class DataflowInterface:
    name: str
    interface_type: DataflowInterfaceType
    q_dim: List[int]          # Query dimensions
    t_dim: List[int]          # Tensor dimensions  
    s_dim: List[int]          # Stream dimensions (runtime determined)
    dtype: str                # Data type
    parallelism_param: str    # iPar or wPar
    axi_ports: Dict[str, Port] # Underlying AXI-Stream ports
```

#### Enhanced Pragma System
```python
# New pragmas for dataflow modeling
class DataflowPragmaType(Enum):
    INPUT_INTERFACE = "input_interface"      # Mark AXI-Stream as input
    OUTPUT_INTERFACE = "output_interface"    # Mark AXI-Stream as output
    WEIGHT_INTERFACE = "weight_interface"    # Mark AXI-Stream as weight
    TENSOR_SHAPE = "tensor_shape"           # Define tensor dimensions
    CALCULATION_PATTERN = "calculation_pattern" # Define calc relationships
    PARALLELISM_CONSTRAINT = "parallelism_constraint" # iPar/wPar constraints
```

#### Dataflow Model Class
```python
@dataclass
class DataflowModel:
    """Complete dataflow representation of a hardware kernel."""
    kernel_name: str
    input_interfaces: List[DataflowInterface]
    output_interfaces: List[DataflowInterface]
    weight_interfaces: List[DataflowInterface]
    config_interfaces: List[DataflowInterface]
    control_interface: DataflowInterface
    
    # Computational model
    calculation_pattern: str  # How inputs/weights/outputs relate
    initiation_intervals: Dict[str, int]  # cII, eII, L calculations
    
    # Parallelism constraints
    parallelism_constraints: Dict[str, Any]
    
    def validate_model(self) -> ValidationResult:
        """Validate the complete dataflow model."""
        
    def generate_pe_simd_mapping(self) -> Dict[str, int]:
        """Generate PE/SIMD values for FINN compatibility."""
        return {
            "PE": self.get_wpar_total(),
            "SIMD": self.get_ipar_total()
        }
```

### Phase 3: HWKG Integration

#### AutoHWCustomOp Architecture
```python
class AutoHWCustomOp(HWCustomOp):
    """Automatically generated HWCustomOp based on dataflow model."""
    
    def __init__(self, onnx_node, dataflow_model: DataflowModel, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.dataflow_model = dataflow_model
        
    # Standardized method implementations
    def get_input_datatype(self, ind=0): 
        return self.dataflow_model.input_interfaces[ind].dtype
        
    def get_output_datatype(self, ind=0):
        return self.dataflow_model.output_interfaces[ind].dtype
        
    def get_normal_input_shape(self, ind=0):
        return self.dataflow_model.input_interfaces[ind].q_dim
        
    def get_folded_input_shape(self, ind=0):
        iface = self.dataflow_model.input_interfaces[ind]
        # Apply folding based on parallelism
        return self._apply_folding(iface.q_dim, iface.s_dim)
        
    def get_exp_cycles(self):
        return self.dataflow_model.initiation_intervals.get("L", 0)
        
    def get_instream_width(self, ind=0):
        iface = self.dataflow_model.input_interfaces[ind]
        return len(iface.s_dim) * iface.dtype.bitwidth
```

#### Hybrid Generation Strategy

**Template-Based Components**:
- Class structure and method signatures
- Standard method implementations
- Import statements and decorators

**Programmatic Components**:
- Dataflow model integration
- Complex constraint validation
- Resource estimation functions

#### Generation Pipeline
```python
class AutoHWCustomOpGenerator:
    """Generates AutoHWCustomOp classes from dataflow models."""
    
    def generate(self, dataflow_model: DataflowModel, output_dir: Path) -> Path:
        # 1. Generate base class structure with templates
        base_code = self._generate_base_structure(dataflow_model)
        
        # 2. Generate method implementations programmatically
        methods = self._generate_methods(dataflow_model)
        
        # 3. Generate validation and constraints
        validation = self._generate_validation(dataflow_model)
        
        # 4. Combine and write to file
        return self._write_combined_output(base_code, methods, validation, output_dir)
```

## Validation Strategy

### Target Kernels
1. **Thresholding**: Simple elementwise operation for basic validation
2. **MVAU**: Complex matrix-vector operation for comprehensive testing

### Validation Approach
1. **Functional Equivalence**: AutoHWCustomOp produces same results as manual implementation
2. **Performance Validation**: Cycle counts and resource usage match expectations
3. **Integration Testing**: Works correctly with existing FINN transformations and DSE
4. **API Compatibility**: PE/SIMD mapping works with existing tools

### Validation Tools
```python
class DataflowModelValidator:
    """Validates dataflow models against manual implementations."""
    
    def validate_functional_equivalence(self, auto_op, manual_op, test_cases):
        """Compare outputs for identical inputs."""
        
    def validate_performance(self, auto_op, expected_cycles, expected_resources):
        """Verify cycle counts and resource estimates."""
        
    def validate_finn_integration(self, auto_op):
        """Test integration with FINN transformations."""
```

## Implementation Phases

### Phase 2: Dataflow Modeling Framework (Weeks 1-3)
- [ ] Extend RTL Parser with new interface types and pragmas
- [ ] Implement DataflowInterface and DataflowModel classes
- [ ] Create dataflow model validation logic
- [ ] Add PE/SIMD mapping functions
- [ ] Unit tests for all components

### Phase 3: HWKG Integration (Weeks 4-6)
- [ ] Design AutoHWCustomOp base class
- [ ] Implement hybrid generation strategy
- [ ] Create Jinja2 templates for class structure
- [ ] Implement programmatic method generation
- [ ] Create AutoRTLBackend generator
- [ ] Integration tests with HWKG pipeline

### Phase 4: Validation and Testing (Weeks 7-8)
- [ ] Implement validation tools
- [ ] Create test cases for Thresholding kernel
- [ ] Create test cases for MVAU kernel
- [ ] Validate API compatibility
- [ ] Performance benchmarking
- [ ] Documentation and examples

## Success Criteria

1. **Automated Generation**: HWKG can generate functional AutoHWCustomOp from RTL
2. **FINN Compatibility**: Generated operations work with existing FINN tools
3. **Performance Parity**: Cycle counts and resource usage match manual implementations
4. **Validation Passing**: Both Thresholding and MVAU kernels pass all validation tests
5. **API Preservation**: Existing PE/SIMD-based code works unchanged

## Risk Mitigation

1. **Complexity Management**: Incremental development with continuous testing
2. **Compatibility Issues**: Extensive validation against existing implementations  
3. **Performance Regression**: Benchmark against current manual implementations
4. **Integration Challenges**: Early integration testing with FINN pipeline

This strategy provides a clear roadmap for implementing the Interface-Wise Dataflow Modeling system while maintaining full compatibility with the existing FINN ecosystem.
