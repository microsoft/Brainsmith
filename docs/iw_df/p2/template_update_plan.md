# Template Update Plan: AutoHWCustomOp Integration

## Overview

Update the HWCustomOp and RTLBackend templates to properly inherit from the AutoHWCustomOp base classes, leveraging all standardized implementations from the dataflow framework.

## Template Architecture Changes

### 1. HWCustomOp Template (`hw_custom_op.py.j2`)

#### Current Structure:
```python
class {{ class_name }}(HWCustomOp):
    # Re-implements all methods
    def get_input_datatype(self, ind=0):
        # Manual implementation
    def get_normal_input_shape(self, ind=0):
        # Manual implementation
    # ... many more manual implementations
```

#### New Structure:
```python
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
from brainsmith.dataflow.core.dataflow_model import DataflowModel

class {{ class_name }}(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for {{ kernel_name }} using dataflow framework.
    
    Most methods are standardized in AutoHWCustomOp base class.
    Only kernel-specific resource estimation needs implementation.
    """
    
    def __init__(self, onnx_node, **kwargs):
        # Define dataflow interfaces from template context
        dataflow_interfaces = [
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
        
        # Create dataflow model
        dataflow_model = DataflowModel(dataflow_interfaces, {})
        
        # Initialize parent with dataflow components
        super().__init__(onnx_node, dataflow_interfaces, dataflow_model, **kwargs)
    
    # Only implement resource estimation (kernel-specific)
    def bram_estimation(self) -> int:
        """Kernel-specific BRAM estimation."""
        # TODO: Implement based on {{ kernel_name }} requirements
        raise NotImplementedError(
            f"BRAM estimation for {{ class_name }} must be implemented. "
            f"Consider weight storage: {self._get_weight_memory_summary()}"
        )
    
    def lut_estimation(self) -> int:
        """Kernel-specific LUT estimation."""
        # TODO: Implement based on {{ kernel_name }} requirements
        raise NotImplementedError(f"LUT estimation for {{ class_name }} must be implemented.")
    
    def dsp_estimation(self) -> int:
        """Kernel-specific DSP estimation."""
        # TODO: Implement based on {{ kernel_name }} requirements
        raise NotImplementedError(f"DSP estimation for {{ class_name }} must be implemented.")
```

### 2. RTLBackend Template (`rtl_backend.py.j2`)

#### Current Structure:
```python
class {{ backend_class_name }}(RTLBackend):
    # Re-implements all RTL generation methods
```

#### New Structure:
```python
from brainsmith.dataflow.core.auto_rtl_backend import AutoRTLBackend

class {{ backend_class_name }}(AutoRTLBackend):
    """
    Auto-generated RTLBackend for {{ kernel_name }} using dataflow framework.
    
    Leverages AutoRTLBackend for standardized RTL generation.
    """
    
    def __init__(self, model, dataflow_model=None):
        super().__init__(model, dataflow_model)
        self.rtl_template_path = "{{ rtl_template_path }}"
    
    # Most methods inherited from AutoRTLBackend
    # Only override if kernel-specific behavior needed
```

## Benefits of This Approach

### 1. Code Reduction
- **Before**: ~500 lines per generated HWCustomOp
- **After**: ~100 lines per generated HWCustomOp
- **Reduction**: 80% less generated code

### 2. Standardization
- All datatype handling standardized
- All shape inference standardized  
- All stream width calculations standardized
- All parallelism handling standardized

### 3. Maintainability
- Updates to base class automatically propagate
- Consistent behavior across all kernels
- Single source of truth for dataflow logic

### 4. Clear Separation
- Generated code only contains kernel-specific logic
- Framework handles all generic dataflow operations
- Easy to see what's unique about each kernel

## Implementation Steps

### Step 1: Update Template Imports
```python
# Add to template header
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.dataflow_interface import (
    DataflowInterface, DataflowInterfaceType, DataflowDataType
)
from brainsmith.dataflow.core.dataflow_model import DataflowModel
```

### Step 2: Update Class Definition
```python
class {{ class_name }}(AutoHWCustomOp):  # Changed from HWCustomOp
```

### Step 3: Update Constructor
```python
def __init__(self, onnx_node, **kwargs):
    # Build dataflow interfaces from template context
    dataflow_interfaces = self._build_dataflow_interfaces()
    dataflow_model = DataflowModel(dataflow_interfaces, {})
    
    # Call parent constructor with dataflow components
    super().__init__(onnx_node, dataflow_interfaces, dataflow_model, **kwargs)
```

### Step 4: Remove Standardized Methods
Remove these methods (now handled by AutoHWCustomOp):
- `get_nodeattr_types()` - only override to add kernel-specific attributes
- `get_input_datatype()` 
- `get_output_datatype()`
- `get_normal_input_shape()`
- `get_normal_output_shape()` 
- `get_folded_input_shape()`
- `get_folded_output_shape()`
- `get_instream_width()`
- `get_outstream_width()`
- `get_exp_cycles()`
- `execute_node()` - unless kernel-specific behavior needed
- `verify_node()` - base class handles standard validation

### Step 5: Implement Resource Estimation
Keep only kernel-specific methods:
```python
def bram_estimation(self) -> int:
    """
    Kernel-specific BRAM estimation for {{ kernel_name }}.
    
    Base class provides helper methods:
    - self._get_weight_memory_summary()
    - self._get_activation_buffer_summary()
    - self._get_current_parallelism()
    """
    # Kernel-specific implementation
    weight_memory = self._get_weight_memory_summary()
    # ... custom calculation based on kernel architecture
    return estimated_brams

def lut_estimation(self) -> int:
    # Kernel-specific LUT calculation
    
def dsp_estimation(self) -> int:
    # Kernel-specific DSP calculation
```

## Template Context Requirements

The HKG must provide in template context:
```python
context = {
    "kernel_name": str,
    "class_name": str,
    "dataflow_interfaces": List[DataflowInterface],
    "has_dataflow_model": bool,
    "rtl_parameters": List[Parameter],
    # ... other context
}
```

## Testing Strategy

### 1. Validate Generated Code
- Ensure imports are correct
- Verify inheritance chain
- Check method signatures

### 2. Functional Testing  
- Create instance with test ONNX node
- Verify all inherited methods work
- Test resource estimation placeholders

### 3. Integration Testing
- Run through complete HKG pipeline
- Verify generated code integrates with FINN
- Test with thresholding_axi example

## Expected Outcome

Generated HWCustomOp classes will be:
- **Concise**: Only kernel-specific code
- **Standardized**: Consistent behavior via base class
- **Maintainable**: Updates to framework propagate automatically
- **Clear**: Easy to understand what's unique about each kernel

This demonstrates the full value of the dataflow framework by showing how it eliminates boilerplate and standardizes implementations across all kernels.