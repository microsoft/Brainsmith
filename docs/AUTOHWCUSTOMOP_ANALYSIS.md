# AutoHWCustomOp Architecture Analysis

## Overview

AutoHWCustomOp represents a revolutionary approach to HWCustomOp generation in the Brainsmith ecosystem. It replaces the manual, per-operation HWCustomOp implementations with a unified, template-driven system that automatically generates FINN-compatible hardware custom operations from RTL analysis.

## Core Architecture

### Three-Tier Information Model

AutoHWCustomOp implements a sophisticated 3-tier information architecture:

1. **Tier 1 - Kernel Data (Static)**
   - Interface metadata from RTL parsing
   - Chunking strategies and block shapes
   - Datatype constraints
   - Node attribute definitions

2. **Tier 2 - Model Data (Runtime)**
   - Tensor dimensions from ONNX context
   - Block dimensions resolved from parameters
   - User-specified datatypes via node attributes

3. **Tier 3 - Parallelism (Dynamic)**
   - Input parallelism (iPar) values
   - Weight parallelism (wPar) values  
   - Stream dimensions calculations
   - Performance characteristics

### Key Design Principles

1. **Separation of Concerns**: Each tier handles distinct responsibilities
2. **Runtime Flexibility**: Parallelism can be updated without rebuilding the model
3. **FINN Compatibility**: Maintains full compatibility with existing FINN infrastructure
4. **Legacy Support**: Maps modern concepts to legacy SIMD/PE terminology

## Template Generation System

### hw_custom_op_phase2.py.j2 Template

The template generates AutoHWCustomOp subclasses with:

- **Static Interface Metadata**: Pre-validated symbolic BDIM shapes
- **Dynamic Parameter Extraction**: RTL parameters become ONNX node attributes
- **Automatic Validation**: Constraint checking against RTL-derived requirements
- **FINN Integration**: Standard HWCustomOp pattern with simple constructor

### Key Template Features

```python
# Interface metadata with validated symbolic shapes
def get_interface_metadata() -> List[InterfaceMetadata]:
    return [
        InterfaceMetadata(
            name="in0",
            interface_type=InterfaceType.INPUT,
            chunking_strategy=BlockChunkingStrategy(
                block_shape=['batch_size', 'channels'],  # Validated symbols
                rindex=0
            )
        )
    ]

# RTL parameters as node attributes
def get_nodeattr_types(self) -> Dict[str, Tuple[str, bool, Any]]:
    my_attrs = {}
    # Required RTL parameters
    my_attrs["channels"] = ("i", True, None)
    # Optional with defaults
    my_attrs["pipeline_depth"] = ("i", False, 4)
    return my_attrs
```

## Comparison with Legacy HWCustomOp

### Legacy Approach (e.g., Thresholding)

```python
class Thresholding(HWCustomOp):
    def get_nodeattr_types(self):
        return {
            "PE": ("i", True, 0),
            "NumChannels": ("i", True, 0),
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # ... 20+ manual attributes
        }
    
    def get_instream_width(self, ind=0):
        # Manual calculation for each operation
        i_bits = self.get_input_datatype(0).bitwidth()
        return i_bits * self.get_nodeattr("PE")
    
    def get_folded_input_shape(self, ind=0):
        # Manual shape calculation
        pe = self.get_nodeattr("PE")
        fold = self.calc_tmem()
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [fold, pe])
```

### AutoHWCustomOp Approach

```python
class AutoThresholding(AutoHWCustomOp):
    @staticmethod
    def get_interface_metadata() -> List[InterfaceMetadata]:
        # Generated from RTL analysis
        return [/* RTL-derived metadata */]
    
    # All shape/width calculations handled by parent class
    # Stream widths computed from DataflowModel
    # Parallelism managed through unified interface
    # Resource estimation using standardized formulas
```

## Key Advantages

### 1. Automatic Generation
- **RTL-to-Template Pipeline**: Direct generation from SystemVerilog analysis
- **No Manual Implementation**: Eliminates 500+ lines of boilerplate per operation
- **Consistent Patterns**: All operations follow identical structure

### 2. Unified Interface System
- **Single Type System**: InterfaceType replaces dual RTL/Dataflow types
- **Constraint Validation**: Automatic datatype constraint checking
- **Symbolic Parameters**: BDIM shapes with validated symbolic references

### 3. Enhanced Maintainability
- **Template Updates**: Single template affects all generated operations
- **Bug Fixes**: Centralized fixes propagate to all operations
- **Feature Additions**: New capabilities added once, available everywhere

### 4. FINN Integration
- **Legacy Compatibility**: Maps modern concepts to SIMD/PE terminology
- **Standard Patterns**: Follows FINN's expected HWCustomOp structure
- **Resource Estimation**: Standardized BRAM/LUT/DSP calculation methods

## Implementation Details

### DataflowModel Integration

```python
def _build_dataflow_model_from_node(self) -> DataflowModel:
    """Build DataflowModel from ONNX node attributes."""
    interfaces = []
    for metadata in self.get_interface_metadata():
        if metadata.interface_type == InterfaceType.CONTROL:
            interface = self._create_control_interface(metadata)
        else:
            # Extract runtime datatype from node attributes
            dtype_attr = f"{metadata.name}_dtype"
            runtime_dtype = self.get_nodeattr(dtype_attr)
            
            interface = DataflowInterface.from_metadata_and_runtime_datatype(
                metadata=metadata,
                runtime_datatype=runtime_dtype,
                tensor_dims=self._get_tensor_dims_for_interface(metadata.name),
                block_dims=self._resolve_block_dims(metadata),
                stream_dims=[1] * len(self._get_block_shape(metadata))
            )
        interfaces.append(interface)
    
    return DataflowModel(interfaces, {})
```

### Legacy Attribute Mapping

```python
def get_legacy_attr(self) -> Dict[str, Any]:
    """Map modern DataflowModel to legacy FINN attributes."""
    legacy_attrs = {}
    
    # Input parallelism → SIMD
    if input_interfaces:
        legacy_attrs["SIMD"] = input_interfaces[0].stream_dims[0]
        legacy_attrs["inputDataType"] = str(input_interfaces[0].dtype)
    
    # Weight parallelism → PE  
    if weight_interfaces:
        legacy_attrs["PE"] = weight_interfaces[0].stream_dims[0]
        legacy_attrs["weightDataType"] = str(weight_interfaces[0].dtype)
    
    # Output datatype
    if output_interfaces:
        legacy_attrs["outputDataType"] = str(output_interfaces[0].dtype)
    
    return legacy_attrs
```

## Code Generation Requirements Analysis

Based on the FINN findings document, AutoHWCustomOp addresses all major requirements:

### ✅ Operation Specification
- **Automatic Classification**: RTL analysis determines operation type
- **Complexity Assessment**: Interface count and parameter complexity
- **Description Generation**: From RTL comments and pragma annotations

### ✅ Parallelization Strategy
- **PE Strategy**: Derived from weight interface parallelism
- **SIMD Strategy**: Derived from input interface parallelism  
- **Stream Width Formulas**: Automatic calculation from datatypes and parallelism

### ✅ Node Attributes Schema
- **Operation Parameters**: RTL parameters become node attributes
- **Parallelization Parameters**: iPar/wPar mapped to SIMD/PE
- **Datatype Parameters**: Interface-specific dtype attributes
- **Memory Parameters**: ram_style, runtime_writeable_weights

### ✅ Shape Management
- **Unified Calculations**: DataflowInterface handles all shape math
- **Folding Support**: Automatic folded/normal shape conversion
- **Constraint Validation**: Block dimension divisibility checks

### ✅ Execution Logic
- **Template-based**: Default pass-through with override capability
- **Simulation Support**: Integration with FINN's execute_node pattern

### ✅ Resource Estimation
- **Standardized Formulas**: BRAM/LUT/DSP/URAM estimation methods
- **Configurable Modes**: Conservative/optimistic/automatic scaling
- **Memory Architecture**: ram_style-aware resource calculations

## Migration Path

### Phase 1: Template Refinement
1. Enhance template with more sophisticated resource estimation
2. Add operation-specific optimizations
3. Improve constraint validation

### Phase 2: Legacy Replacement
1. Generate AutoHWCustomOp equivalents for existing operations
2. Maintain compatibility interfaces
3. Gradual migration of FINN infrastructure

### Phase 3: Advanced Features
1. Multi-interface parallelism support
2. Dynamic reconfiguration capabilities
3. Advanced optimization passes

## Conclusion

AutoHWCustomOp represents a paradigm shift from manual HWCustomOp implementation to automatic generation. By combining RTL analysis, template-based code generation, and a unified interface system, it eliminates the need for hundreds of lines of boilerplate code per operation while maintaining full FINN compatibility.

The three-tier architecture provides clean separation of concerns, the template system ensures consistency, and the DataflowModel integration enables sophisticated parallelism and resource management. This approach scales to support the entire spectrum of FPGA operations while reducing maintenance burden and improving code quality.