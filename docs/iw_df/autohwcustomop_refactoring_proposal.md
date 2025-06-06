# AutoHWCustomOp Generation Refactoring Proposal

## Problem Statement

The current AutoHWCustomOp generation system has two critical issues:

1. **Unwieldy Generated Code**: Generated AutoHWCustomOps are extremely verbose (300+ lines) with giant static dictionaries and placeholder implementations
2. **Attribute Timing Mismatch**: FINN sets interface attributes (`qDim`, `tDim`, `dtype`) via `onnx.helper.make_node` at node creation, then sets parallelism (`iPar`, `wPar`) during DSE transformations, but the current AutoHWCustomOp expects a pre-built DataflowModel at construction time

## Root Cause Analysis

### Current Architecture Issues

The fundamental mismatch is in the constructor signature:

**AutoHWCustomOp expects:**
```python
def __init__(self, onnx_node, dataflow_model: DataflowModel, **kwargs):
```

**Generated template calls:**
```python
super().__init__(onnx_node, **kwargs)  # Missing dataflow_model!
```

This forces the template to compensate with:
- Static interface dictionaries in `get_kernel_interface_specs()` (83-104 lines)
- Placeholder resource estimation methods (136-232 lines)
- Manual parameter handling instead of leveraging DataflowModel

### FINN Workflow vs Current Implementation

**FINN's Natural Workflow:**
1. `onnx.helper.make_node` sets `qDim`, `tDim`, `dtype` at node creation
2. DSE transformations later set `iPar`, `wPar` parallelism parameters
3. Methods are called with current attribute values

**Current AutoHWCustomOp Workflow:**
1. Constructor requires fully-built DataflowModel
2. Static interface specifications never change
3. No support for dynamic attribute updates

## Comprehensive Solution

### Phase 1: Two-Phase Initialization Architecture

#### 1.1 Refactor AutoHWCustomOp Base Class

```python
class AutoHWCustomOp(HWCustomOp):
    """Base class supporting FINN's two-phase initialization workflow."""
    
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        
        # Store kernel metadata for lazy DataflowModel creation
        self._interface_metadata = None
        self._dataflow_model = None
        self._model_built = False
        
    def _get_kernel_interface_metadata(self) -> List['InterfaceMetadata']:
        """Override in subclasses to provide kernel-specific interface metadata."""
        if self._interface_metadata is None:
            raise NotImplementedError("Subclass must implement _get_kernel_interface_metadata()")
        return self._interface_metadata
        
    def _build_dataflow_model_if_needed(self):
        """Lazily build DataflowModel from current ONNX node attributes."""
        if self._model_built:
            return
            
        # Extract current attribute values from ONNX node
        interface_metadata = self._get_kernel_interface_metadata()
        interfaces = []
        
        for metadata in interface_metadata:
            # Build DataflowInterface from metadata + current ONNX attributes
            interface = self._create_interface_from_metadata_and_attributes(metadata)
            interfaces.append(interface)
            
        self._dataflow_model = DataflowModel(interfaces, {})
        self._model_built = True
    
    @property 
    def dataflow_model(self) -> DataflowModel:
        """Get DataflowModel, building it lazily if needed."""
        self._build_dataflow_model_if_needed()
        return self._dataflow_model
        
    def set_nodeattr(self, name: str, value: Any):
        """Override to invalidate DataflowModel when attributes change."""
        super().set_nodeattr(name, value)
        
        # If this affects interface dimensions/parallelism, rebuild model
        if any(suffix in name for suffix in ["_qDim", "_tDim", "_dtype", "_parallel", "_shape", "_layout"]):
            self._model_built = False
```

#### 1.2 Dynamic Interface Creation from ONNX Attributes

```python
def _create_interface_from_metadata_and_attributes(self, metadata: 'InterfaceMetadata') -> DataflowInterface:
    """Create DataflowInterface from metadata + current ONNX node attributes with tensor chunking support."""
    interface_name = metadata.name
    
    # Check for explicit qDim/tDim attributes (legacy support or manual override)
    explicit_qDim = self.get_nodeattr(f"{interface_name}_qDim")
    explicit_tDim = self.get_nodeattr(f"{interface_name}_tDim")
    
    if explicit_qDim is not None and explicit_tDim is not None:
        # Use explicit dimensions if provided (legacy mode)
        qDim = explicit_qDim
        tDim = explicit_tDim
    else:
        # Extract tensor shape from input tensor and compute qDim/tDim
        qDim, tDim = self._compute_dimensions_from_input_tensor(interface_name, metadata)
    
    # Get parallelism (set during DSE)
    parallelism = self.get_nodeattr(f"{interface_name}_parallel") or 1
    sDim = [parallelism] + [1] * (len(tDim) - 1)
    
    # Get datatype (must be set by FINN at runtime)
    dtype_str = self.get_nodeattr(f"{interface_name}_dtype")
    if dtype_str is None:
        raise ValueError(f"Interface {interface_name} requires dtype attribute to be set at runtime")
    
    # Parse FINN datatype string to create DataflowDataType
    dtype = self._parse_finn_datatype(dtype_str)
    
    return DataflowInterface(
        name=interface_name,
        interface_type=metadata.interface_type,
        qDim=qDim,
        tDim=tDim,
        sDim=sDim,
        dtype=dtype,
        allowed_datatypes=metadata.allowed_datatypes,
        pragma_metadata=metadata.pragma_metadata
    )

def _compute_dimensions_from_input_tensor(self, interface_name: str, metadata: 'InterfaceMetadata') -> Tuple[List[int], List[int]]:
    """Extract tensor shape from input tensor and compute qDim/tDim using enhanced chunking logic."""
    from brainsmith.dataflow.core.tensor_chunking import TensorChunking
    
    # Extract tensor shape from the corresponding input tensor
    tensor_shape = self._extract_tensor_shape_from_input(interface_name)
    
    # Get tensor layout (optional attribute, with smart defaults)
    tensor_layout = self.get_nodeattr(f"{interface_name}_layout") or self._infer_layout_from_shape(tensor_shape)
    
    # Check for enhanced TDIM pragma override in interface metadata
    enhanced_tdim = metadata.pragma_metadata.get("enhanced_tdim")
    
    if enhanced_tdim:
        # Use enhanced TDIM pragma with extracted tensor shape
        chunker = TensorChunking()
        qDim, tDim = chunker.process_enhanced_tdim_pragma(
            enhanced_tdim, tensor_shape, tensor_layout
        )
    else:
        # Use default layout-based chunking
        chunker = TensorChunking()
        qDim, tDim = chunker.infer_dimensions_with_layout(tensor_layout, tensor_shape)
    
    return qDim, tDim

def _extract_tensor_shape_from_input(self, interface_name: str) -> List[int]:
    """Extract tensor shape from the corresponding input tensor in the ONNX graph."""
    # Map interface name to input tensor (this mapping could be stored in metadata)
    input_index = self._get_input_index_for_interface(interface_name)
    
    if input_index is None:
        raise ValueError(f"Cannot find input tensor for interface {interface_name}")
    
    # Get input tensor shape from ONNX graph
    try:
        # Access the input tensor shape from the ONNX node
        input_tensor_name = self.onnx_node.input[input_index]
        
        # Get shape from the model context (this would need to be passed in)
        # For now, we'll try to get it from node attributes as fallback
        shape_attr = self.get_nodeattr(f"input_{input_index}_shape")
        if shape_attr is not None:
            return shape_attr
            
        # If no shape available, try to infer from the model wrapper
        if hasattr(self, '_model_wrapper') and self._model_wrapper:
            tensor_shape = self._model_wrapper.get_tensor_shape(input_tensor_name)
            if tensor_shape:
                return tensor_shape
                
        # Final fallback - require explicit shape attribute
        raise ValueError(f"Cannot extract shape for interface {interface_name}. Consider setting {interface_name}_shape attribute.")
        
    except (IndexError, AttributeError) as e:
        raise ValueError(f"Failed to extract tensor shape for interface {interface_name}: {e}")

def _get_input_index_for_interface(self, interface_name: str) -> Optional[int]:
    """Map interface name to input tensor index."""
    # This mapping could be stored in interface metadata or computed from naming conventions
    # For now, simple mapping based on interface naming patterns
    if "in0" in interface_name:
        return 0
    elif "in1" in interface_name:
        return 1
    elif "weights" in interface_name:
        # Weight interfaces might map to different input indices
        return 1  # or appropriate weight input index
    else:
        # For output interfaces, we might need to look at input shapes
        return 0  # Default to first input

def _infer_layout_from_shape(self, tensor_shape: List[int]) -> str:
    """Infer tensor layout from shape dimensions with smart defaults."""
    if len(tensor_shape) == 4:
        return "NCHW"  # Default for 4D tensors
    elif len(tensor_shape) == 3:
        return "CHW"   # Default for 3D tensors
    elif len(tensor_shape) == 2:
        return "NC"    # Default for 2D tensors
    elif len(tensor_shape) == 1:
        return "C"     # Default for 1D tensors
    else:
        return "UNKNOWN"

def _parse_finn_datatype(self, dtype_str: str) -> DataflowDataType:
    """Parse FINN datatype string to create DataflowDataType object."""
    import re
    match = re.match(r'([A-Z]+)(\d+)', dtype_str)
    if not match:
        raise ValueError(f"Invalid FINN datatype string: {dtype_str}")
    
    base_type = match.group(1)
    bitwidth = int(match.group(2))
    signed = base_type.startswith('INT') and not base_type.startswith('UINT')
    
    return DataflowDataType(
        base_type=base_type,
        bitwidth=bitwidth,
        signed=signed,
        finn_type=dtype_str
    )

def _compute_dimensions_from_tensor_info(self, interface_name: str, tensor_shape: List[int],
                                       tensor_layout: str, metadata: 'InterfaceMetadata') -> Tuple[List[int], List[int]]:
    """Compute qDim and tDim from tensor shape and layout using chunking logic."""
    from brainsmith.dataflow.core.tensor_chunking import TensorChunking
    
    # Check for TDIM pragma override in interface metadata
    tdim_pragma = metadata.pragma_metadata.get("TDIM")
    
    if tdim_pragma:
        # TDIM pragma specifies custom chunking logic
        tDim = self._parse_tdim_pragma(tdim_pragma, tensor_shape, tensor_layout)
        # Compute qDim from original shape and tDim
        qDim = TensorChunking._compute_qDim_from_chunking(tensor_shape, tDim)
    else:
        # Use default tensor chunking logic based on layout
        chunker = TensorChunking()
        qDim, tDim = chunker.infer_dimensions_with_layout(tensor_layout, tensor_shape)
    
    return qDim, tDim

def _parse_tdim_pragma(self, tdim_pragma: str, tensor_shape: List[int], tensor_layout: str) -> List[int]:
    """Parse TDIM pragma to extract custom tensor dimension chunking."""
    # Parse pragma format: "TDIM=32,16,1" or "TDIM=C,H/2,W"
    if "=" in tdim_pragma:
        tdim_spec = tdim_pragma.split("=")[1].strip()
    else:
        tdim_spec = tdim_pragma.strip()
    
    tDim = []
    layout_mapping = self._get_layout_dimension_mapping(tensor_layout, tensor_shape)
    
    for dim_spec in tdim_spec.split(","):
        dim_spec = dim_spec.strip()
        
        if dim_spec.isdigit():
            # Explicit dimension value
            tDim.append(int(dim_spec))
        elif "/" in dim_spec:
            # Division expression like "H/2"
            parts = dim_spec.split("/")
            if len(parts) == 2 and parts[0] in layout_mapping:
                base_dim = layout_mapping[parts[0]]
                divisor = int(parts[1])
                tDim.append(base_dim // divisor)
            else:
                tDim.append(1)  # Fallback
        elif dim_spec in layout_mapping:
            # Direct layout reference like "C"
            tDim.append(layout_mapping[dim_spec])
        else:
            tDim.append(1)  # Fallback
    
    return tDim

def _get_layout_dimension_mapping(self, layout: str, shape: List[int]) -> Dict[str, int]:
    """Create mapping from layout characters to tensor dimensions."""
    mapping = {}
    
    if layout == "NCHW" and len(shape) >= 4:
        mapping = {"N": shape[0], "C": shape[1], "H": shape[2], "W": shape[3]}
    elif layout == "NHWC" and len(shape) >= 4:
        mapping = {"N": shape[0], "H": shape[1], "W": shape[2], "C": shape[3]}
    elif layout == "CHW" and len(shape) >= 3:
        mapping = {"C": shape[0], "H": shape[1], "W": shape[2]}
    elif layout == "HWC" and len(shape) >= 3:
        mapping = {"H": shape[0], "W": shape[1], "C": shape[2]}
    elif layout == "NC" and len(shape) >= 2:
        mapping = {"N": shape[0], "C": shape[1]}
    
    return mapping

@dataclass
class InterfaceMetadata:
    """Metadata for creating DataflowInterface objects at runtime (no default values)."""
    name: str
    interface_type: DataflowInterfaceType
    allowed_datatypes: List[DataTypeConstraint]
    pragma_metadata: Dict[str, Any] = field(default_factory=dict)

# This InterfaceMetadata class needs to be added to brainsmith/dataflow/core/dataflow_interface.py
```

### Phase 2: Enhanced Resource Estimation

#### 2.1 DataflowModel Resource Requirements

Add to `DataflowModel` class:

```python
def get_resource_requirements(self, parallelism_config: Dict[str, int]) -> Dict[str, Any]:
    """Enhanced resource estimation using interface analysis."""
    requirements = {
        "memory_bits": 0,
        "lut_ops": 0, 
        "dsp_ops": 0,
        "bram_blocks": 0,
        "computation_cycles": 0
    }
    
    # Calculate memory requirements from weight interfaces
    for iface in self.weight_interfaces:
        weight_bits = np.prod(iface.qDim) * np.prod(iface.tDim) * iface.dtype.bitwidth
        requirements["memory_bits"] += weight_bits
        
    # Calculate LUT requirements from interface complexity
    for iface in self.input_interfaces:
        parallelism = parallelism_config.get(iface.name, 1)
        # LUT estimation based on datatype bitwidth and parallelism
        lut_per_element = iface.dtype.bitwidth * 2  # Rough estimate
        requirements["lut_ops"] += parallelism * lut_per_element
        
    # Calculate DSP requirements from arithmetic interfaces
    for iface in self.weight_interfaces:
        if iface.dtype.bitwidth >= 8:  # Likely uses DSPs
            parallelism = parallelism_config.get(iface.name, 1)
            requirements["dsp_ops"] += parallelism
            
    # Convert memory to BRAM blocks
    requirements["bram_blocks"] = (requirements["memory_bits"] + 18*1024 - 1) // (18*1024)
    
    return requirements

def get_weight_memory_summary(self) -> Dict[str, int]:
    """Get detailed weight memory breakdown."""
    summary = {}
    for iface in self.weight_interfaces:
        bits = np.prod(iface.qDim) * np.prod(iface.tDim) * iface.dtype.bitwidth
        summary[iface.name] = {
            "total_bits": bits,
            "bram_blocks": (bits + 18*1024 - 1) // (18*1024),
            "elements": np.prod(iface.qDim) * np.prod(iface.tDim)
        }
    return summary
```

### Phase 3: Slim Template Generation

#### 3.1 New Template Structure with Tensor Chunking

**Before (300+ lines):**
```python
class ThresholdingAxi(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        # 50+ lines of boilerplate
        
    def get_kernel_interface_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "in0_V_data_V",
                "interface_type": "INPUT",
                "direction": "input",
                "allowed_datatypes": [
                    {
                        "base_types": ["UINT", "INT"],
                        "min_bitwidth": 1,
                        "max_bitwidth": 32,
                        "signed_allowed": True,
                        "unsigned_allowed": True
                    },
                ],
                "pragma_metadata": {...},
                "axi_protocol": "axi_stream"
            },
            # ... more 20+ line interface specs
        ]
    
    def bram_estimation(self) -> int:
        # 50+ lines of placeholder logic
        
    def lut_estimation(self) -> int:
        # 30+ lines of placeholder logic
        
    def dsp_estimation(self) -> int:
        # 20+ lines of placeholder logic
```

**After (50 lines) with Tensor Chunking (No Defaults):**
```python
class ThresholdingAxi(AutoHWCustomOp):
    """Auto-generated HWCustomOp for thresholding_axi with tensor chunking support."""
    
    def __init__(self, onnx_node, **kwargs):
        from brainsmith.dataflow.core.dataflow_interface import DataTypeConstraint, DataflowInterfaceType, InterfaceMetadata
        
        # Interface metadata (no default values - set at runtime via ONNX node attributes)
        self._interface_metadata = [
            InterfaceMetadata(
                name="in0_V_data_V",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        base_types=["UINT", "INT"],
                        min_bitwidth=1,
                        max_bitwidth=32,
                        signed_allowed=True,
                        unsigned_allowed=True
                    )
                ],
                pragma_metadata={"TDIM": "32,16,1"}  # From RTL analysis
            ),
            InterfaceMetadata(
                name="out_V_data_V",
                interface_type=DataflowInterfaceType.OUTPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        base_types=["UINT", "INT"],
                        min_bitwidth=1,
                        max_bitwidth=32,
                        signed_allowed=True,
                        unsigned_allowed=True
                    )
                ]
            )
        ]
        super().__init__(onnx_node, **kwargs)
        
    # Kernel-specific resource estimation (leverages DataflowModel)
    def bram_estimation(self) -> int:
        resources = self.dataflow_model.get_resource_requirements(
            self._get_current_parallelism_config()
        )
        return resources["bram_blocks"]
        
    def lut_estimation(self) -> int:
        resources = self.dataflow_model.get_resource_requirements(
            self._get_current_parallelism_config()
        )
        return resources["lut_ops"]
        
    def dsp_estimation(self) -> int:
        return 0  # thresholding_axi doesn't use arithmetic operations
```

#### 3.2 Template Context Updates

Update `hkg.py` template context generation:

```python
def _build_enhanced_template_context(self) -> Dict[str, Any]:
    context = {
        "kernel_name": self.hw_kernel_data.name,
        "class_name": generate_class_name(self.hw_kernel_data.name),
        
        # Interface metadata (no default values)
        "interface_metadata": [
            {
                "name": iface.name,
                "interface_type": iface.interface_type.name,
                "allowed_datatypes": self._format_datatype_constraints(iface.allowed_datatypes),
                "pragma_metadata": iface.pragma_metadata,
                "supports_chunking": True,
            }
            for iface in (self.dataflow_interfaces or [])
        ],
        
        # Resource estimation hints (for kernel-specific customization)
        "resource_estimation": {
            "has_weights": bool(self.dataflow_model and self.dataflow_model.weight_interfaces),
            "has_arithmetic": self._analyze_arithmetic_operations(),
            "memory_intensive": self._analyze_memory_usage(),
        },
        
        # Tensor chunking configuration
        "tensor_chunking": {
            "enabled": True,
            "pragma_support": True
        }
    }
    return context

def _format_datatype_constraints(self, constraints: List[DataTypeConstraint]) -> List[Dict[str, Any]]:
    """Format DataTypeConstraint objects for template generation."""
    return [
        {
            "base_types": constraint.base_types,
            "min_bitwidth": constraint.min_bitwidth,
            "max_bitwidth": constraint.max_bitwidth,
            "signed_allowed": constraint.signed_allowed,
            "unsigned_allowed": constraint.unsigned_allowed
        }
        for constraint in constraints
    ]
```

### Phase 4: Template Engine Updates

#### 4.1 New Slim Template (`hw_custom_op.py.j2`)

```jinja2
{#- Slim AutoHWCustomOp Template with DataflowModel Integration (No Defaults) -#}
class {{ class_name }}(AutoHWCustomOp):
    """Auto-generated HWCustomOp for {{ kernel_name }}."""
    
    def __init__(self, onnx_node, **kwargs):
        from brainsmith.dataflow.core.dataflow_interface import DataTypeConstraint, DataflowInterfaceType, InterfaceMetadata
        
        # Interface metadata (no default values - all values from runtime ONNX attributes)
        self._interface_metadata = [
            {% for metadata in interface_metadata %}
            InterfaceMetadata(
                name="{{ metadata.name }}",
                interface_type=DataflowInterfaceType.{{ metadata.interface_type }},
                allowed_datatypes=[
                    {% for constraint in metadata.allowed_datatypes %}
                    DataTypeConstraint(
                        base_types={{ constraint.base_types }},
                        min_bitwidth={{ constraint.min_bitwidth }},
                        max_bitwidth={{ constraint.max_bitwidth }},
                        signed_allowed={{ constraint.signed_allowed }},
                        unsigned_allowed={{ constraint.unsigned_allowed }}
                    ),
                    {% endfor %}
                ],
                pragma_metadata={{ metadata.pragma_metadata }}
            ),
            {% endfor %}
        ]
        super().__init__(onnx_node, **kwargs)
        
    {% if resource_estimation.has_weights %}
    def get_weight_memory_summary(self) -> Dict[str, int]:
        """Get weight memory requirements for {{ kernel_name }}."""
        return self.dataflow_model.get_weight_memory_summary()
    {% endif %}
    
    # Kernel-specific resource estimation (leverages DataflowModel)
    def bram_estimation(self) -> int:
        resources = self.dataflow_model.get_resource_requirements(
            self._get_current_parallelism_config()
        )
        return resources["bram_blocks"]
        
    def lut_estimation(self) -> int:
        resources = self.dataflow_model.get_resource_requirements(
            self._get_current_parallelism_config()
        )
        return resources["lut_ops"]
        
    {% if resource_estimation.has_arithmetic %}
    def dsp_estimation(self) -> int:
        resources = self.dataflow_model.get_resource_requirements(
            self._get_current_parallelism_config()
        )
        return resources["dsp_ops"]
    {% else %}
    def dsp_estimation(self) -> int:
        return 0  # {{ kernel_name }} doesn't use arithmetic operations
    {% endif %}
    
    {% if resource_estimation.memory_intensive %}
    def uram_estimation(self) -> int:
        """{{ kernel_name }} uses large memory structures."""
        # Custom URAM estimation for memory-intensive kernels
        return super().uram_estimation()
    {% endif %}
```

## Tensor Chunking Integration

### 5.1 Enhanced FINN Node Creation Workflow

The enhanced AutoHWCustomOp generation now supports tensor chunking for intuitive interface configuration:

**Before (Manual qDim/tDim Configuration):**
```python
# FINN users had to manually calculate and set dimensions
node = onnx.helper.make_node(
    "ThresholdingAxi",
    inputs=["input_tensor"],
    outputs=["output_tensor"],
    domain="brainsmith.custom",
    in0_V_data_V_qDim=[1, 1, 32, 32],  # Manual calculation required
    in0_V_data_V_tDim=[1, 8, 1, 1],   # Manual calculation required
    in0_V_data_V_dtype="UINT8",        # Required: set at runtime
    out_V_data_V_qDim=[1, 1, 32, 32], # Manual calculation required
    out_V_data_V_tDim=[1, 8, 1, 1],   # Manual calculation required
    out_V_data_V_dtype="UINT8"        # Required: set at runtime
)
```

**After (Tensor Shape + Layout Configuration):**
```python
# FINN users specify intuitive tensor shape and layout (no defaults in class)
node = onnx.helper.make_node(
    "ThresholdingAxi",
    inputs=["input_tensor"],
    outputs=["output_tensor"],
    domain="brainsmith.custom",
    in0_V_data_V_shape=[1, 8, 32, 32],    # Required: tensor shape set at runtime
    in0_V_data_V_layout="NCHW",           # Required: tensor layout set at runtime
    in0_V_data_V_dtype="UINT8",           # Required: datatype set at runtime
    out_V_data_V_shape=[1, 8, 32, 32],    # Required: tensor shape set at runtime
    out_V_data_V_layout="NCHW",           # Required: tensor layout set at runtime
    out_V_data_V_dtype="UINT8"            # Required: datatype set at runtime
)

# qDim and tDim are automatically computed at method call time based on:
# 1. Current tensor shape and layout from ONNX node attributes
# 2. TDIM pragma overrides from RTL analysis (stored in class)
# 3. Interface-specific chunking logic
```

### 5.2 Automatic Dimension Computation

The enhanced system automatically computes qDim and tDim using intelligent chunking strategies:

```python
# Example: NCHW tensor [1, 8, 32, 32] with TDIM pragma override
interface_spec = {
    "name": "in0_V_data_V",
    "interface_type": "INPUT",
    "default_shape": [1, 8, 32, 32],
    "default_layout": "NCHW",
    "pragma_metadata": {"TDIM": "1,2,16,16"}  # Custom chunking from RTL
}

# Automatic computation:
# 1. Parse TDIM pragma: tDim = [1, 2, 16, 16]
# 2. Compute qDim from shape/tDim: qDim = [1, 4, 2, 2]
# 3. Result: Process 2x2 blocks of 16x16 pixels with 2 channels at a time
```

### 5.3 Layout-Aware Chunking Strategies

The tensor chunking system includes built-in strategies for common layouts:

| Layout | Default Chunking Strategy | Use Case |
|--------|---------------------------|----------|
| NCHW | Channel-parallel, spatial streaming | CNNs, image processing |
| NHWC | Spatial-parallel, channel streaming | Mobile optimized CNNs |
| CHW | Channel-wise processing | Feature extraction |
| HWC | Spatial processing with channel parallelism | Dense operations |
| NC | Batch processing with channel parallelism | Fully connected layers |
| C | Channel-wise vector processing | 1D operations |

### 5.4 Enhanced TDIM Pragma Support for Flexible Chunking

RTL kernels can specify sophisticated chunking strategies via enhanced TDIM pragmas:

**Example 1: Runtime Shape Chunking**
```systemverilog
// RTL: Always chunk the last dimension regardless of tensor shape
// @brainsmith TDIM in0_V_data_V [:] -1

module thresholding_axi #(
    parameter PE = 1
)(
    input  logic [PE*8-1:0] in0_V_data_V,  // Chunked data width
    output logic [PE*8-1:0] out_V_data_V
);
```

**Example 2: Parameter-Defined Shape Chunking**
```systemverilog
// RTL: Use parameter-defined shape, chunk at second-to-last dimension
// @brainsmith TDIM weights [BATCH_SIZE, FEATURES, HIDDEN_DIM] -2

module my_matmul #(
    parameter BATCH_SIZE = 1,
    parameter FEATURES = 128,
    parameter HIDDEN_DIM = 64,
    parameter PE = 4
)(
    input  logic [PE*8-1:0] weights,  // PE parallel elements per cycle
    // ...
);
```

**Generated Chunking Logic:**
```python
# Example 1: Runtime shape chunking
# For tensor shape [1, 8, 32, 32] with "[:] -1":
# - target_shape = [1, 8, 32, 32] (runtime)
# - chunk_index = -1 (last dimension)
# - Result: qDim=[1, 8, 32, 1], tDim=[1, 1, 1, 32]

# Example 2: Parameter-defined shape chunking
# For parameters BATCH_SIZE=1, FEATURES=128, HIDDEN_DIM=64 with "[BATCH_SIZE, FEATURES, HIDDEN_DIM] -2":
# - target_shape = [1, 128, 64] (from parameters)
# - chunk_index = -2 (second-to-last dimension)
# - Result: qDim=[1, 1, 64], tDim=[1, 128, 1]
```

### 5.5 Backward Compatibility

The enhanced system maintains full backward compatibility:

```python
# Legacy explicit dimension setting still works
node = onnx.helper.make_node(
    "ThresholdingAxi",
    inputs=["input_tensor"],
    outputs=["output_tensor"],
    in0_V_data_V_qDim=[1, 4, 2, 2],      # Explicit qDim (legacy)
    in0_V_data_V_tDim=[1, 2, 16, 16],    # Explicit tDim (legacy)
    in0_V_data_V_dtype="UINT8"
)

# New tensor chunking attributes are ignored when explicit dimensions provided
# No migration required for existing FINN workflows
```

### 5.6 Chunking Strategy Selection

Advanced users can specify chunking strategies for different use cases:

```python
# Streaming strategy: minimal latency
hw_op.set_nodeattr("in0_V_data_V_chunking_strategy", "streaming")

# Block strategy: maximum throughput
hw_op.set_nodeattr("in0_V_data_V_chunking_strategy", "block")

# Channel parallel strategy: maximum parallelism
hw_op.set_nodeattr("in0_V_data_V_chunking_strategy", "channel_parallel")

# Default strategy: balanced approach (automatic from layout)
hw_op.set_nodeattr("in0_V_data_V_chunking_strategy", "default")
```

### 5.7 Template Generation Enhancements

Generated AutoHWCustomOp classes automatically include tensor chunking support:

```python
# Generated class includes chunking-aware dimension computation
class ThresholdingAxi(AutoHWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        self._kernel_interface_specs = [
            {
                "name": "in0_V_data_V",
                "interface_type": "INPUT",
                "default_shape": [1, 8, 32, 32],
                "default_layout": "NCHW",
                "pragma_metadata": {"TDIM": "C,H/2,W/4"},
                "supports_chunking": True
            }
        ]
        super().__init__(onnx_node, **kwargs)
    
    # All dimension queries automatically use chunking logic
    def get_normal_input_shape(self, ind=0):
        # Returns original tensor shape from chunking computation
        return super().get_normal_input_shape(ind)
        
    def get_folded_input_shape(self, ind=0):
        # Returns folded shape based on computed qDim/tDim
        return super().get_folded_input_shape(ind)
```

## Benefits Analysis

### 1. Code Reduction
- **Before**: 300+ lines per generated AutoHWCustomOp
- **After**: 50-80 lines per generated AutoHWCustomOp  
- **Reduction**: 75-80% fewer lines of generated code

### 2. FINN Workflow Compatibility
- ✅ Supports `onnx.helper.make_node` attribute setting
- ✅ Compatible with DSE transformation workflow
- ✅ Dynamic response to attribute updates
- ✅ No constructor signature changes required

### 3. DataflowModel Leverage
- ✅ Automatic resource estimation from interface analysis
- ✅ Unified computational model for performance calculations
- ✅ Constraint validation and parallelism optimization
- ✅ No manual placeholder implementations needed

### 4. Maintainability
- ✅ Template logic dramatically simplified
- ✅ Base class handles complexity automatically
- ✅ Kernel-specific customization still possible
- ✅ Consistent behavior across all generated classes

## Implementation Plan

### Phase 1: Base Class Refactoring (Week 1)
- [ ] Modify `AutoHWCustomOp` for two-phase initialization
- [ ] Implement lazy DataflowModel building
- [ ] Add attribute change invalidation
- [ ] Update method signatures for compatibility

### Phase 2: Resource Estimation Enhancement (Week 1)
- [ ] Add `get_resource_requirements()` to `DataflowModel`
- [ ] Implement memory, LUT, and DSP estimation algorithms
- [ ] Add weight memory summary methods
- [ ] Create estimation mode support (conservative/optimistic/automatic)

### Phase 3: Template Simplification (Week 2)
- [ ] Create new slim template structure
- [ ] Update template context generation in `hkg.py`
- [ ] Add resource estimation hints and analysis
- [ ] Implement conditional template blocks
- [ ] Integrate tensor chunking support in templates
- [ ] Add TDIM pragma parsing in HKG pipeline

### Phase 4: Integration Testing (Week 2)
- [ ] Test with existing thresholding kernel
- [ ] Validate FINN workflow compatibility
- [ ] Test tensor chunking with various layouts (NCHW, NHWC, CHW)
- [ ] Validate TDIM pragma parsing and dimension computation
- [ ] Performance benchmarking of generated classes
- [ ] End-to-end integration testing with shape/layout attributes

### Phase 5: Migration and Documentation (Week 3)
- [ ] Migrate existing generated classes
- [ ] Update documentation and examples
- [ ] Create tensor chunking user guide
- [ ] Document TDIM pragma syntax and examples
- [ ] Create migration guide
- [ ] Performance comparison analysis

## Risk Assessment

### Low Risk
- **Template changes**: Well-isolated, easily reversible
- **Base class enhancement**: Backward-compatible additions
- **Resource estimation**: Additive functionality

### Medium Risk  
- **FINN workflow integration**: Requires careful testing with real FINN transformations
- **Attribute timing**: Need to validate lazy model building works correctly

### Mitigation Strategies
- Maintain backward compatibility in base class
- Extensive testing with existing kernels before migration
- Gradual rollout with fallback to current implementation
- Performance monitoring during transition

## Success Metrics

### Code Quality
- **Generated code reduction**: Target 75%+ reduction in lines
- **Template complexity**: Target 50%+ reduction in template logic
- **Maintenance burden**: Eliminate placeholder implementations
- **Tensor chunking accuracy**: 100% accurate qDim/tDim computation from shape/layout

### Functionality
- **FINN compatibility**: 100% compatibility with existing FINN workflow
- **Tensor chunking support**: Seamless shape/layout to qDim/tDim conversion
- **TDIM pragma support**: Full RTL pragma parsing and override capability
- **Layout awareness**: Intelligent chunking strategies for all common layouts
- **Resource estimation accuracy**: Improve accuracy by leveraging DataflowModel
- **Performance**: No regression in method call performance

### Developer Experience
- **Template maintainability**: Easier to understand and modify templates
- **Intuitive interface**: Shape/layout specification instead of manual qDim/tDim calculation
- **Automatic dimension computation**: No more manual tensor dimension calculations
- **Customization**: Preserved ability for kernel-specific customization via TDIM pragmas
- **Error messages**: Better error reporting from lazy model building and chunking validation

This refactoring proposal addresses both the verbosity and timing mismatch issues while maintaining full compatibility with FINN's workflow and enhancing the capabilities of the generated AutoHWCustomOp classes.