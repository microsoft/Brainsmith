# AutoHWCustomOp Template Generation Plan

## Overview

This document outlines the plan for updating our template generation to create AutoHWCustomOp subclasses that use DataflowModel as the heart of the HWCustomOp, replacing manual hardcoded functionality with automated DataflowModel-driven operations.

## Key Architectural Insight

The `AutoHWCustomOp` base class already implements a sophisticated 3-tier architecture that eliminates manual calculations:

1. **Tier 1: Kernel Data** (Static, from RTL) - Interface metadata, chunking strategies
2. **Tier 2: Model Data** (Runtime, from ONNX) - Tensor shapes, datatypes
3. **Tier 3: Parallelism** (Dynamic) - iPar/wPar values, performance metrics

## Template Generation Architecture

```
RTL Parser → ParsedKernelData → TemplateContextGenerator → AutoHWCustomOp Template
                                            ↓
                                  Creates InterfaceMetadata list
                                            ↓
                           AutoHWCustomOp subclass (uses DataflowModel internally)
```

## What the Template Must Generate

### 1. InterfaceMetadata List Creation

From `ParsedKernelData.interfaces`, generate:

```python
interface_metadata = [
    InterfaceMetadata(
        name=interface.name,
        interface_type=InterfaceType.INPUT,  # Based on interface.type
        allowed_datatypes=[
            DataTypeConstraint(finn_type="UINT8", bit_width=8, signed=False),
            # More constraints from pragmas or defaults
        ],
        chunking_strategy=DefaultChunkingStrategy()  # Or from pragma
    ),
    # ... for each interface
]
```

### 2. AutoHWCustomOp Subclass Structure

```python
class {{ class_name }}(AutoHWCustomOp):
    """Auto-generated HWCustomOp for {{ kernel_name }} using DataflowModel."""
    
    def __init__(self, onnx_node, **kwargs):
        # Create interface metadata
        interface_metadata = [ ... ]
        
        # Parent creates DataflowModel automatically!
        super().__init__(onnx_node, interface_metadata=interface_metadata, **kwargs)
        
        # Store kernel info
        self.kernel_name = "{{ kernel_name }}"
        self.rtl_source = "{{ source_file }}"
    
    def get_nodeattr_types(self):
        # Get DataflowModel attributes from parent
        attrs = super().get_enhanced_nodeattr_types()
        
        # Add RTL parameters and algorithm-specific attributes
        kernel_attrs = {
            {% for param in rtl_parameters %}
            "{{ param.name }}": ("i", False, {{ param.default_value }}),
            {% endfor %}
            # Algorithm-specific attributes from analysis
            {% for name, spec in algorithm_attributes.items() %}
            "{{ name }}": {{ spec }},
            {% endfor %}
        }
        
        attrs.update(kernel_attrs)
        return attrs
    
    # Optional kernel-specific overrides only when needed
```

## Methods Automatically Provided by DataflowModel

The following methods are **automatically handled** by `AutoHWCustomOp` via DataflowModel:

- ✅ `get_input_datatype()` - From DataflowModel interfaces
- ✅ `get_output_datatype()` - From DataflowModel interfaces
- ✅ `get_normal_input_shape()` - From interface tensor reconstruction
- ✅ `get_folded_input_shape()` - With parallelism applied
- ✅ `get_instream_width()` - From stream calculations
- ✅ `get_outstream_width()` - From stream calculations
- ✅ `get_number_output_values()` - From output shapes
- ✅ `estimate_bram_usage()` - From resource requirements
- ✅ `estimate_lut_usage()` - From resource requirements
- ✅ `estimate_dsp_usage()` - From resource requirements

## Template Context Enhancement

The `TemplateContextGenerator` needs to provide:

1. **Interface metadata context** - Convert ParsedKernelData interfaces to template-ready format
2. **Datatype constraints** - Extract from pragmas or use sensible defaults
3. **Chunking strategies** - From pragmas or default strategies
4. **Algorithm attributes** - Infer from kernel type and parameters

## Benefits of This Approach

1. **Eliminates Manual Calculations** - No more hardcoded shape/stream/resource methods
2. **Automatic Parallelism Scaling** - Changes to iPar/wPar automatically update everything
3. **Interface Validation** - DataflowModel validates constraints automatically
4. **Resource Estimation** - Unified resource calculation from interface specifications
5. **FINN Compatibility** - All standard FINN HWCustomOp methods work out of the box

## Implementation Steps

1. **Update TemplateContextGenerator** to generate interface metadata context
2. **Modify hw_custom_op_slim.py.j2** template to:
   - Import from `brainsmith.dataflow.core` instead of FINN
   - Generate InterfaceMetadata list in `__init__`
   - Inherit from AutoHWCustomOp
   - Remove manual shape/stream/resource methods
3. **Test with thresholding example** to verify functionality

## Next Steps

With the rtl_integration module removed and this cleaner architecture in place, we can:

1. Generate much simpler HWCustomOp classes
2. Leverage DataflowModel for all complex calculations
3. Focus templates on kernel-specific functionality only
4. Maintain full FINN compatibility while reducing code complexity

This approach aligns perfectly with using DataflowModel as the heart of HWCustomOp operations.