# AutoHWCustomOp Generation Analysis & Implementation Plan

## Understanding Hardcoded HWCustomOp Structure

Based on analysis of `examples/thresholding/thresholding.py` and the HWCustomOp architecture documentation, here's what a complete AutoHWCustomOp generation system must produce:

### Core HWCustomOp Requirements

#### 1. **Essential Base Class Structure**
```python
class Thresholding(HWCustomOp):
    """Abstraction layer for HW implementation of Thresholding."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
```

#### 2. **Node Attribute Configuration**
```python
def get_nodeattr_types(self):
    my_attrs = {
        # Hardware-specific attributes
        "PE": ("i", True, 0),                    # Parallelization elements
        "NumChannels": ("i", True, 0),           # Number of channels
        "runtime_writeable_weights": ("i", False, 0, {0, 1}),
        
        # Data type specifications
        "inputDataType": ("s", True, ""),
        "weightDataType": ("s", True, ""),
        "outputDataType": ("s", True, ""),
        
        # Shape specifications
        "numInputVectors": ("ints", False, [1]),
        
        # Algorithm-specific parameters
        "numSteps": ("i", True, 1),
        "ActVal": ("i", False, 0),
    }
    my_attrs.update(super().get_nodeattr_types())
    return my_attrs
```

#### 3. **Data Type Management**
```python
def get_input_datatype(self, ind=0):
    """Returns FINN DataType of input."""
    return DataType[self.get_nodeattr("inputDataType")]

def get_output_datatype(self, ind=0):
    """Returns FINN DataType of output."""
    return DataType[self.get_nodeattr("outputDataType")]

def get_weight_datatype(self):
    """Returns FINN DataType of weights/thresholds."""
    return DataType[self.get_nodeattr("weightDataType")]
```

#### 4. **Shape and Stream Width Calculations**
```python
def get_normal_input_shape(self, ind=0):
    ich = self.get_nodeattr("NumChannels")
    vecs = list(self.get_nodeattr("numInputVectors"))
    return tuple(vecs + [ich])

def get_folded_input_shape(self, ind=0):
    pe = self.get_nodeattr("PE")
    fold = self.calc_tmem()  # NumChannels // PE
    vecs = list(self.get_nodeattr("numInputVectors"))
    return tuple(vecs + [fold, pe])

def get_instream_width(self, ind=0):
    i_bits = self.get_input_datatype().bitwidth()
    return i_bits * self.get_nodeattr("PE")

def get_outstream_width(self, ind=0):
    o_bits = self.get_output_datatype().bitwidth()
    return o_bits * self.get_nodeattr("PE")
```

#### 5. **Resource and Performance Estimation**
```python
def get_exp_cycles(self):
    """Expected cycles for computation."""
    return np.prod(self.get_folded_output_shape()[:-1])

def calc_tmem(self):
    """Calculate memory depth (TMEM)."""
    return self.get_nodeattr("NumChannels") // self.get_nodeattr("PE")
```

#### 6. **Hardware-Specific Data Preparation**
```python
def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
    """Convert weights/thresholds to hardware-compatible format."""
    # PE-based data distribution
    # Interleaving for parallel processing
    # Shape validation and transformation
```

#### 7. **Node Verification**
```python
def verify_node(self):
    """Validate node configuration and attributes."""
    info_messages = []
    # Check backend is "fpgadataflow"
    # Verify all necessary attributes exist
    # Validate parallelism constraints (NumChannels % PE == 0)
    return info_messages
```

#### 8. **Functional Execution**
```python
def execute_node(self, context, graph):
    """Software simulation of the operation."""
    # Reference implementation for validation
    # Handle different tensor layouts (4D, 2D)
    # Apply algorithm with proper data transformations
```

## Key Insights from Analysis

### 1. **Three-Tier Architecture Implementation**
- **Tier 1**: Generic operation logic (shape inference, datatype handling)
- **Tier 2**: Backend interface (will be RTL backend in our case)
- **Tier 3**: Concrete implementation (AutoHWCustomOp generated code)

### 2. **Critical Dependencies on RTL Analysis**
The AutoHWCustomOp must extract from RTL:
- **Interface types** (INPUT, OUTPUT, WEIGHT, CONFIG, CONTROL)
- **Parallelism parameters** (PE/SIMD equivalents)
- **Data types and bit widths**
- **Memory requirements**
- **Performance characteristics**

### 3. **Template Context Requirements**
Based on the hardcoded example, our template must generate:
- Node attribute definitions with proper types and constraints
- Shape calculation methods (normal vs folded)
- Stream width calculations based on parallelism
- Hardware-compatible data transformation methods
- Verification logic for configuration validation

## AutoHWCustomOp Generation Strategy

### Phase 1: Core Template Enhancement
Update the existing `hw_custom_op_slim.py.j2` template to generate all required methods:

```python
class {{ class_name }}(AutoHWCustomOp):
    """Auto-generated HWCustomOp for {{ kernel_name }} kernel."""
    
    def get_nodeattr_types(self):
        my_attrs = {
            # Generated from RTL parameter analysis
            {% for param in rtl_parameters %}
            "{{ param.name }}": ("i", True, {{ param.default_value or 0 }}),
            {% endfor %}
            
            # Generated from interface analysis
            {% for interface in input_interfaces %}
            "{{ interface.name }}_datatype": ("s", True, ""),
            {% endfor %}
            
            # Parallelism parameters inferred from RTL
            "PE": ("i", True, {{ inferred_pe or 1 }}),
            {% if has_weights %}
            "NumChannels": ("i", True, {{ inferred_channels or 1 }}),
            {% endif %}
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs
```

### Phase 2: RTL Analysis Enhancement
Enhance the RTL Parser and TemplateContextGenerator to extract:

```python
# In TemplateContextGenerator
def _analyze_parallelism_parameters(self, parsed_data):
    """Extract PE/SIMD equivalent parallelism from RTL."""
    # Analyze port widths and array sizes
    # Infer parallel processing elements
    # Extract memory organization (TMEM calculations)
    
def _infer_algorithm_parameters(self, parsed_data):
    """Extract algorithm-specific parameters."""
    # Analyze RTL parameters for algorithm configuration
    # Map to FINN attribute types and constraints
    
def _generate_shape_calculation_methods(self, parsed_data):
    """Generate shape and stream width calculation code."""
    # Based on interface analysis and parallelism
    # Generate normal vs folded shape methods
```

### Phase 3: Integration with Existing AutoHWCustomOp Framework
Build upon the existing `brainsmith.dataflow.core.auto_hw_custom_op.AutoHWCustomOp`:

```python
# Extend existing AutoHWCustomOp with RTL-derived capabilities
class RTLDerivedAutoHWCustomOp(AutoHWCustomOp):
    """Enhanced AutoHWCustomOp with RTL-derived implementations."""
    
    def __init__(self, rtl_analysis_data, **kwargs):
        self.rtl_data = rtl_analysis_data
        super().__init__(**kwargs)
    
    # Override methods with RTL-derived implementations
    def get_nodeattr_types(self):
        # Use RTL analysis to generate attributes
        
    def get_folded_input_shape(self, ind=0):
        # Use RTL parallelism analysis for folding
```

## Implementation Roadmap

### Step 1: Enhanced Template Context (1-2 hours)
1. **Enhance TemplateContextGenerator** to extract:
   - Parallelism parameters from RTL port analysis
   - Algorithm parameters from RTL parameter analysis
   - Interface characteristics for shape calculations

2. **Update hw_custom_op_slim.py.j2** to generate:
   - Complete `get_nodeattr_types()` method
   - All required shape calculation methods
   - Stream width calculation methods
   - Basic verification logic

### Step 2: RTL Analysis Integration (2-3 hours)
1. **Extend RTL Parser** to identify:
   - Parallel processing patterns in port arrays
   - Memory organization from parameter analysis
   - Data flow patterns from interface structure

2. **Create RTL-to-FINN mapping logic**:
   - Map RTL parameters to FINN node attributes
   - Infer PE/SIMD equivalents from port analysis
   - Extract memory depth calculations (TMEM)

### Step 3: Validation and Testing (1-2 hours)
1. **Generate AutoHWCustomOp** for thresholding_axi.sv
2. **Compare generated code** with hardcoded thresholding.py
3. **Validate functional compatibility**
4. **Test with FINN integration**

## Success Criteria

### Functional Completeness
- Generated AutoHWCustomOp implements all methods from hardcoded version
- Proper node attribute configuration with types and constraints
- Correct shape and stream width calculations
- Hardware-compatible data transformation methods

### RTL Integration
- Automatic extraction of parallelism parameters from RTL
- Proper mapping of RTL parameters to FINN attributes
- Interface analysis driving shape calculation logic

### Template Quality
- Generated code follows FINN conventions and patterns
- Proper error handling and validation
- Clean, readable, and maintainable generated code

## Key Challenges and Solutions

### Challenge 1: RTL Parallelism Inference
**Problem**: Automatically identifying PE/SIMD equivalents from RTL port structures
**Solution**: Pattern-based analysis of port arrays and memory organization

### Challenge 2: Algorithm Parameter Mapping
**Problem**: Mapping RTL-specific parameters to FINN node attributes
**Solution**: Template-driven parameter analysis with type inference

### Challenge 3: Shape Calculation Generation
**Problem**: Generating correct folded shape calculations for arbitrary operations
**Solution**: Interface-driven template generation with parallelism context

This analysis provides the foundation for implementing a complete AutoHWCustomOp generation system that can produce FINN-compatible HWCustomOp implementations directly from RTL analysis.