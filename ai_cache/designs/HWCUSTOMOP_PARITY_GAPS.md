# AutoHWCustomOp vs Legacy HWCustomOp: Parity Gap Analysis

## Overview

This analysis compares AutoHWCustomOp with legacy HWCustomOp implementations (using Thresholding as the reference) to identify gaps that need to be addressed for full parity.

## Method-by-Method Comparison

### ✅ Fully Covered Methods

| Method | Legacy Implementation | AutoHWCustomOp Status |
|--------|---------------------|---------------------|
| `get_nodeattr_types()` | Manual attribute definition | ✅ Template-generated + parent class |
| `get_input_datatype()` | Manual index handling | ✅ Handled by parent class |
| `get_output_datatype()` | Manual index handling | ✅ Handled by parent class |
| `get_normal_input_shape()` | Manual calculation | ✅ DataflowModel-based |
| `get_normal_output_shape()` | Manual calculation | ✅ DataflowModel-based |
| `get_folded_input_shape()` | Manual calculation | ✅ DataflowModel-based |
| `get_folded_output_shape()` | Manual calculation | ✅ DataflowModel-based |
| `get_instream_width()` | Manual calculation | ✅ DataflowModel-based |
| `get_outstream_width()` | Manual calculation | ✅ DataflowModel-based |
| `get_number_output_values()` | Manual calculation | ✅ DataflowModel-based |
| `get_exp_cycles()` | Manual calculation | ✅ DataflowModel-based |
| `verify_node()` | Manual validation | ✅ Enhanced validation in parent |

### ❌ Major Gaps

#### 1. **Operation-Specific Data Processing Methods**

**Missing: `minimize_accumulator_width()`**
```python
# Legacy Thresholding (lines 124-151)
def minimize_accumulator_width(self, model):
    """Minimize threshold width based on actual data ranges"""
    idt = self.get_input_datatype(0)
    if str(idt).startswith("FLOAT") or self.get_nodeattr("weightDataType").startswith("FLOAT"):
        return DataType[self.get_nodeattr("weightDataType")]
    
    thresholds = model.get_initializer(self.onnx_node.input[1])
    min_threshold = thresholds.min()
    max_threshold = thresholds.max()
    # Complex datatype optimization logic...
```

**Gap**: AutoHWCustomOp has no mechanism for operation-specific datatype optimization.

#### 2. **Hardware-Specific Tensor Formatting**

**Missing: `get_hw_compatible_threshold_tensor()` (lines 207-247)**
```python
def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
    """Convert weight matrix for hlslib compatibility:
    * ensure MH % PE == 0
    * handle unsigned input constraints  
    * interleave rows between PEs
    * reshape into (PE, TMEM, n_thres_steps)
    """
    mh = self.get_nodeattr("NumChannels")
    pe = self.get_nodeattr("PE")
    tmem = mh // pe
    # Complex weight tensor reshaping logic...
```

**Gap**: AutoHWCustomOp has no framework for operation-specific weight tensor formatting.

#### 3. **Advanced Execution Logic**

**Missing: Sophisticated `execute_node()` Implementation**
```python
# Legacy Thresholding (lines 249-268)
def execute_node(self, context, graph):
    node = self.onnx_node
    inp_values = context[node.input[0]]
    th_val = context[node.input[1]]
    out_bias = self.get_nodeattr("ActVal")
    
    # Handle 4D vs 2D input shape differences
    is_4d = len(inp_values.shape) == 4
    if is_4d:
        inp_values = np.transpose(inp_values, (0, 3, 1, 2))
    
    # Call operation-specific compute function
    y = multithreshold(inp_values, th_val, out_bias=out_bias)
    
    # Handle output transformations
    if is_4d:
        y = y.transpose(0, 2, 3, 1)
    
    # Apply datatype-specific transformations
    act = DataType[self.get_nodeattr("outputDataType")]
    if act == DataType["BIPOLAR"]:
        y = 2 * y - 1  # binary to bipolar conversion
    
    context[node.output[0]] = y
```

**Gap**: AutoHWCustomOp provides only basic pass-through execution, missing operation-specific compute logic.

#### 4. **Operation-Specific Utility Methods**

**Missing: Domain-Specific Helper Methods**
```python
# Legacy Thresholding (lines 270-274)
def calc_tmem(self):
    """Calculates and returns TMEM."""
    num_channels = self.get_nodeattr("NumChannels")
    pe = self.get_nodeattr("PE")
    return num_channels // pe
```

**Gap**: No mechanism for operation-specific utility methods.

### ⚠️ Partial Gaps

#### 1. **Advanced `infer_node_datatype()` Logic**

**Current AutoHWCustomOp**: Basic validation only
```python
def infer_node_datatype(self, model):
    # Only validates that datatypes are specified
    for iface in self.dataflow_model.input_interfaces:
        if not self.get_nodeattr(f"{iface.name}_dtype"):
            raise ValueError(f"Datatype must be specified")
```

**Legacy Thresholding**: Dynamic datatype inference with warnings
```python
def infer_node_datatype(self, model):
    node = self.onnx_node
    idt = model.get_tensor_datatype(node.input[0])
    if idt != self.get_input_datatype(0):
        warn_str = "inputDataType changing for %s: %s -> %s" % (...)
        warnings.warn(warn_str)
    self.set_nodeattr("inputDataType", idt.name)
    odt = self.get_output_datatype()
    model.set_tensor_datatype(node.output[0], odt)
```

**Gap**: Missing dynamic datatype inference and model tensor synchronization.

#### 2. **Advanced Stream Width Calculations**

**Legacy Thresholding**: Mode-dependent width calculation
```python
def get_instream_width(self, ind=0):
    if ind == 0:
        i_bits = self.get_input_datatype(0).bitwidth()
        width = i_bits * self.get_nodeattr("PE")
    elif ind == 1:
        try:
            mem_mode = self.get_nodeattr("mem_mode")
        except AttributeError:
            mem_mode = 0
        if mem_mode == "internal_decoupled":
            pe = self.get_nodeattr("PE")
            wp = self.get_input_datatype(1).bitwidth()
            n_thres_steps = self.get_nodeattr("numSteps")
            width = pe * wp * n_thres_steps
        else:
            width = 0
```

**AutoHWCustomOp**: Uniform calculation for all interfaces
```python
def get_instream_width(self, ind=0):
    input_dtype = self.get_input_datatype(ind)
    folded_shape = self.get_folded_input_shape(ind)
    elements_per_cycle = folded_shape[-1] if folded_shape else 1
    return input_dtype.bitwidth() * elements_per_cycle
```

**Gap**: Missing mode-dependent and interface-specific width calculation logic.

## Critical Missing Capabilities

### 1. **Operation-Specific Code Generation Hooks**

Legacy HWCustomOps often need custom logic for:
- Weight tensor formatting
- Datatype optimization
- Memory layout transformation
- Backend-specific code generation

**Solution Needed**: Template extension points for operation-specific methods.

### 2. **Dynamic Model Synchronization**

Legacy HWCustomOps actively modify the ONNX model:
- Update tensor datatypes
- Optimize weight precision
- Modify node attributes based on analysis

**Solution Needed**: Model synchronization framework in AutoHWCustomOp.

### 3. **Complex Execution Simulation**

Legacy HWCustomOps provide sophisticated simulation:
- Operation-specific compute functions
- Datatype conversions
- Shape transformations
- Multi-mode behavior

**Solution Needed**: Template-based execution method generation.

### 4. **Advanced Validation and Constraints**

Legacy HWCustomOps implement complex validation:
- Cross-parameter constraints
- Hardware-specific limitations  
- Datatype compatibility checks
- Memory architecture validation

**Solution Needed**: Extensible constraint system.

## Proposed Solutions

### 1. **Template Extension Points**

Add optional template sections for operation-specific methods:
```jinja2
{% if custom_methods %}
# Operation-specific methods
{% for method in custom_methods %}
{{ method.implementation }}
{% endfor %}
{% endif %}
```

### 2. **Model Synchronization Framework**

Extend AutoHWCustomOp with model update capabilities:
```python
class AutoHWCustomOp(HWCustomOp):
    def sync_with_model(self, model):
        """Synchronize node attributes with model tensors"""
        # Update tensor datatypes
        # Optimize weight precision
        # Validate constraints
```

### 3. **Execution Method Templates**

Generate operation-specific execution methods:
```jinja2
def execute_node(self, context, graph):
    """Generated execution logic for {{ kernel_name }}"""
    {% if execution_template %}
    {{ execution_template }}
    {% else %}
    # Default pass-through implementation
    super().execute_node(context, graph)
    {% endif %}
```

### 4. **Advanced Resource Estimation**

Extend resource estimation with operation-specific formulas:
```python
def estimate_resources(self):
    base_resources = super().estimate_resources()
    
    # Apply operation-specific multipliers
    if self.has_weights():
        base_resources["BRAM"] *= self.get_weight_complexity_factor()
    
    return base_resources
```

## Priority Ranking

1. **High Priority**: Operation-specific execution logic and weight tensor formatting
2. **Medium Priority**: Advanced datatype optimization and model synchronization  
3. **Low Priority**: Utility methods and advanced validation (can be worked around)

## Conclusion

While AutoHWCustomOp provides excellent coverage of standard HWCustomOp functionality, it currently lacks support for operation-specific customizations that are critical for complex operations like Thresholding, MVAU, and ConvolutionInputGenerator. The gaps are primarily in areas requiring operation-specific logic rather than generic dataflow modeling.

The template system provides a foundation for addressing these gaps through extension points and specialized generation logic.