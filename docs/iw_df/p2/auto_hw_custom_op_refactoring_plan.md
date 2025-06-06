# AutoHWCustomOp and DataflowModel Integration Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to improve the integration between `AutoHWCustomOp` and `DataflowModel` in the Interface-Wise Dataflow Modeling Framework. The current implementation has `AutoHWCustomOp` duplicating functionality that should be delegated to `DataflowModel`, resulting in maintenance issues and architectural inconsistencies.

## Current Architecture Issues

### Problem 1: Duplicated Interface Management
- `AutoHWCustomOp` maintains its own `dataflow_interfaces` dictionary
- Has separate caching mechanisms for interface lists
- Duplicates logic that exists in `DataflowModel`

### Problem 2: No DataflowModel Integration
- `DataflowModel` is created in generated subclasses but never used
- `AutoHWCustomOp` reimplements computational logic instead of delegating
- No connection between the two core components

### Problem 3: Manual Calculations
- `get_exp_cycles()` performs manual calculations instead of using unified model
- Resource estimation doesn't leverage `DataflowModel` capabilities
- Shape calculations are reimplemented rather than delegated

## Proposed Architecture

### High-Level Design
```
Generated Subclass (e.g., AutoThresholdingAxi)
    ↓ Creates DataflowInterface objects
    ↓ Creates DataflowModel with interfaces
    ↓ Passes DataflowModel to AutoHWCustomOp
AutoHWCustomOp
    ↓ Stores DataflowModel instance
    ↓ Delegates all interface queries to model
    ↓ Uses model's computational methods
DataflowModel
    → Single source of truth for dataflow
    → Manages all interfaces
    → Provides computational model
```

## Detailed Refactoring Steps

### Step 1: Update AutoHWCustomOp Constructor

**Current:**
```python
def __init__(self, onnx_node, **kwargs):
    super().__init__(onnx_node, **kwargs)
    self.dataflow_interfaces = {}  # Dictionary storage
    self._input_interfaces = None   # Caching
```

**Proposed:**
```python
def __init__(self, onnx_node, dataflow_model: DataflowModel, **kwargs):
    super().__init__(onnx_node, **kwargs)
    self.dataflow_model = dataflow_model  # Single source of truth
    # No more interface storage or caching!
```

### Step 2: Refactor Interface Properties

Replace all cached property implementations with simple delegations:

```python
@property
def input_interfaces(self) -> List[str]:
    """Get input interface names from DataflowModel."""
    return [iface.name for iface in self.dataflow_model.input_interfaces]

@property
def output_interfaces(self) -> List[str]:
    """Get output interface names from DataflowModel."""
    return [iface.name for iface in self.dataflow_model.output_interfaces]

@property
def weight_interfaces(self) -> List[str]:
    """Get weight interface names from DataflowModel."""
    return [iface.name for iface in self.dataflow_model.weight_interfaces]

@property
def config_interfaces(self) -> List[str]:
    """Get config interface names from DataflowModel."""
    return [iface.name for iface in self.dataflow_model.config_interfaces]
```

### Step 3: Refactor get_exp_cycles()

Replace manual calculation with DataflowModel delegation:

```python
def get_exp_cycles(self) -> int:
    """Get expected cycles using DataflowModel's unified computational model."""
    # Extract current parallelism configuration
    iPar = {}
    wPar = {}
    
    for iface in self.dataflow_model.input_interfaces:
        iPar[iface.name] = self.get_nodeattr(f"{iface.name}_parallel") or 1
    
    for iface in self.dataflow_model.weight_interfaces:
        wPar[iface.name] = self.get_nodeattr(f"{iface.name}_parallel") or 1
    
    # Use unified calculation
    intervals = self.dataflow_model.calculate_initiation_intervals(iPar, wPar)
    return intervals.L
```

### Step 4: Refactor Shape Methods

Use DataflowInterface methods for shape calculations:

```python
def get_normal_input_shape(self, ind: int = 0) -> List[int]:
    """Get normal input shape from DataflowModel interface."""
    input_ifaces = self.dataflow_model.input_interfaces
    if ind >= len(input_ifaces):
        raise IndexError(f"Input index {ind} exceeds available inputs")
    
    interface = input_ifaces[ind]
    return interface.reconstruct_tensor_shape()

def get_normal_output_shape(self, ind: int = 0) -> List[int]:
    """Get normal output shape from DataflowModel interface."""
    output_ifaces = self.dataflow_model.output_interfaces
    if ind >= len(output_ifaces):
        raise IndexError(f"Output index {ind} exceeds available outputs")
    
    interface = output_ifaces[ind]
    return interface.reconstruct_tensor_shape()
```

### Step 5: Refactor Datatype Methods

Access datatypes through DataflowInterface objects:

```python
def get_input_datatype(self, ind: int = 0) -> Any:
    """Get input datatype from DataflowModel interface."""
    input_ifaces = self.dataflow_model.input_interfaces
    if ind >= len(input_ifaces):
        raise IndexError(f"Input index {ind} exceeds available inputs")
    
    interface = input_ifaces[ind]
    
    # Check for runtime configuration override
    configured_dtype = self.get_nodeattr(f"{interface.name}_dtype")
    if configured_dtype:
        if not interface.validate_datatype_string(configured_dtype):
            raise ValueError(f"Configured datatype {configured_dtype} violates constraints")
        return DataType[configured_dtype] if FINN_AVAILABLE else configured_dtype
    
    # Use interface's default datatype
    return DataType[interface.dtype.finn_type] if FINN_AVAILABLE else interface.dtype.finn_type
```

### Step 6: Refactor Resource Estimation

Leverage DataflowModel's resource calculation capabilities:

```python
def estimate_bram_usage(self) -> int:
    """Estimate BRAM usage using DataflowModel resource requirements."""
    parallelism_config = self._get_current_parallelism_config()
    resources = self.dataflow_model.get_resource_requirements(parallelism_config)
    
    memory_bits = resources["memory_bits"]
    bram_capacity = 18 * 1024  # BRAM18K
    
    # Apply estimation mode scaling
    estimation_mode = self.get_nodeattr("resource_estimation_mode")
    scale_factor = {"conservative": 1.5, "optimistic": 0.7, "automatic": 1.0}.get(estimation_mode, 1.0)
    
    return int(np.ceil((memory_bits * scale_factor) / bram_capacity))
```

### Step 7: Update Template Generation

Modify `hw_custom_op.py.j2` template:

```python
# Create DataflowInterface objects
dataflow_interfaces = [
    {% for interface in dataflow_interfaces %}
    DataflowInterface(
        name="{{ interface.name }}",
        interface_type=DataflowInterfaceType.{{ interface.interface_type.name }},
        qDim={{ interface.qDim }},
        tDim={{ interface.tDim }},
        sDim={{ interface.sDim }},
        dtype=DataflowDataType(
            base_type="{{ interface.dtype.base_type }}",
            bitwidth={{ interface.dtype.bitwidth }},
            signed={{ interface.dtype.signed }},
            finn_type="{{ interface.dtype.finn_type }}"
        ),
        allowed_datatypes={{ interface.allowed_datatypes|to_constraint_spec }},
        axi_metadata={{ interface.axi_metadata }},
        constraints={{ interface.constraints|to_constraint_spec }}
    ),
    {% endfor %}
]

# Create DataflowModel
kernel_parameters = {
    {% for param in kernel_parameters %}
    "{{ param.name }}": self.get_nodeattr("{{ param.name }}"),
    {% endfor %}
}

self._dataflow_model = DataflowModel(dataflow_interfaces, kernel_parameters)

# Pass model to parent class
super().__init__(onnx_node, self._dataflow_model, **kwargs)
```

### Step 8: Add Required Methods to DataflowInterface

Add missing methods to support the refactoring:

```python
class DataflowInterface:
    def reconstruct_tensor_shape(self) -> List[int]:
        """Reconstruct original tensor shape from qDim and tDim."""
        shape = []
        for dim in self.qDim + self.tDim:
            if dim > 1:
                shape.append(dim)
        return shape if shape else [1]
    
    def validate_datatype_string(self, dtype_string: str) -> bool:
        """Validate if a datatype string is allowed for this interface."""
        # Implementation would check against allowed_datatypes constraints
        return True  # Placeholder
```

### Step 9: Helper Method Updates

Add helper methods to AutoHWCustomOp:

```python
def _get_current_parallelism_config(self) -> ParallelismConfiguration:
    """Extract current parallelism configuration from node attributes."""
    iPar = {}
    wPar = {}
    
    for iface in self.dataflow_model.input_interfaces:
        iPar[iface.name] = self.get_nodeattr(f"{iface.name}_parallel") or 1
    
    for iface in self.dataflow_model.weight_interfaces:
        wPar[iface.name] = self.get_nodeattr(f"{iface.name}_parallel") or 1
    
    return ParallelismConfiguration(iPar=iPar, wPar=wPar, derived_sDim={})

def get_interface_by_name(self, name: str) -> DataflowInterface:
    """Get interface by name from DataflowModel."""
    return self.dataflow_model.interfaces.get(name)
```

## Implementation Timeline

### Phase 1: Core Refactoring (Days 1-2)
- Update AutoHWCustomOp constructor
- Refactor interface properties
- Update helper methods

### Phase 2: Method Delegation (Days 3-4)
- Refactor get_exp_cycles()
- Refactor shape methods
- Refactor datatype methods

### Phase 3: Advanced Features (Days 5-6)
- Refactor resource estimation
- Update template generation
- Add DataflowInterface methods

### Phase 4: Testing and Validation (Days 7-8)
- Update unit tests
- Verify template generation
- End-to-end validation

## Benefits

1. **Architectural Clarity**: Clear separation of concerns between components
2. **Reduced Duplication**: Single implementation of dataflow logic
3. **Improved Maintainability**: Changes only needed in one place
4. **Better Testing**: Can test DataflowModel independently
5. **Consistent Behavior**: All calculations use unified model
6. **Future Extensibility**: Easier to add new features

## Risks and Mitigation

### Risk 1: Breaking Existing Code
**Mitigation**: Implement changes incrementally with backward compatibility

### Risk 2: Template Generation Issues
**Mitigation**: Comprehensive testing of generated code

### Risk 3: Performance Impact
**Mitigation**: Profile critical paths to ensure no regression

## Success Criteria

1. All AutoHWCustomOp methods delegate to DataflowModel
2. No duplicated interface management logic
3. Generated code passes all existing tests
4. Performance is maintained or improved
5. Architecture is cleaner and more maintainable

## Conclusion

This refactoring will significantly improve the architecture of the Interface-Wise Dataflow Modeling Framework by establishing DataflowModel as the single source of truth for all dataflow-related functionality. The result will be a cleaner, more maintainable, and more correct implementation that better serves the needs of hardware kernel developers.