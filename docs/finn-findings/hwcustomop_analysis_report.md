# FINN HWCustomOp Comprehensive Analysis Report

**Date**: June 14, 2025  
**Analysis of**: All FINN HWCustomOp implementations in TMP_fpgadataflow/  
**Purpose**: Compare generated AutoHWCustomOp against existing FINN infrastructure

## Executive Summary

This report provides a systematic analysis of all 20+ FINN HWCustomOp implementations to understand nodeattr patterns, function implementation variations, and architectural patterns. The analysis reveals clear design patterns that our generated AutoHWCustomOp must follow for seamless FINN integration.

## 1. NODEATTR TYPE PATTERNS

### 1.1 Data Type Categories Analysis

**Integer Types ("i")** - Most Common
- **Required Usage**: Dimensions, counts, parallelization factors
- **Optional Usage**: Boolean flags, resource IDs, depth controls
- **Examples from codebase**:
  ```python
  "PE": ("i", True, 0),                    # Processing Elements
  "SIMD": ("i", True, 0),                  # SIMD width  
  "NumChannels": ("i", True, 0),           # Channel count
  "runtime_writeable_weights": ("i", False, 0, {0, 1})  # Boolean flag
  "depth": ("i", False, 32),               # FIFO depth
  ```

**String Types ("s")** - Configuration Control
- **Required Usage**: DataTypes, backend specification
- **Optional Usage**: Memory modes, resource styles, execution modes
- **Examples from codebase**:
  ```python
  "inputDataType": ("s", True, ""),        # FINN DataType
  "backend": ("s", True, "fpgadataflow"),  # Backend selector
  "mem_mode": ("s", False, "internal_decoupled", 
              {"internal_embedded", "internal_decoupled", "external"})
  "ram_style": ("s", False, "auto", {"auto", "block", "distributed", "ultra"})
  ```

**Integer List Types ("ints")** - Shape/Vector Data
- **Primary Usage**: Multi-dimensional shapes, vector counts
- **Common Defaults**: `[1]`, `[2]`, `[2, 2]`
- **Examples from codebase**:
  ```python
  "numInputVectors": ("ints", False, [1]),     # Batch shape
  "IFMDim": ("ints", True, []),               # Input feature map dims
  "ConvKernelDim": ("ints", True, []),        # Convolution kernel dims
  "inFIFODepths": ("ints", False, [2]),       # Multi-input FIFO depths
  ```

**Tensor Types ("t")** - Rarely Used
- **Usage**: Performance characterization, initialization data
- **Examples**: `io_chrc_in`, `io_chrc_out` for RTL characterization

### 1.2 Universal Base Attributes

Every FINN HWCustomOp inherits these from the base class:

```python
{
    # Core infrastructure
    "backend": ("s", True, "fpgadataflow"),
    "preferred_impl_style": ("s", False, "", {"", "hls", "rtl"}),
    "exec_mode": ("s", False, "", {"", "rtlsim", "cppsim"}),
    
    # FPGA placement
    "slr": ("i", False, -1),                    # SLR assignment
    "mem_port": ("s", False, ""),               # Memory port binding
    "partition_id": ("i", False, 0),            # Partition ID
    "device_id": ("i", False, 0),               # Multi-FPGA device
    
    # Interface configuration
    "inFIFODepths": ("ints", False, [2]),       # Input FIFO depths
    "outFIFODepths": ("ints", False, [2]),      # Output FIFO depths
    
    # Code generation paths
    "code_gen_dir_ipgen": ("s", False, ""),
    "ipgen_path": ("s", False, ""),
    "ip_path": ("s", False, ""),
    "ip_vlnv": ("s", False, ""),
    
    # Performance tracking
    "cycles_rtlsim": ("i", False, 0),
    "cycles_estimate": ("i", False, 0),
    "res_estimate": ("s", False, ""),
    "res_synth": ("s", False, ""),
}
```

### 1.3 Common Operation-Specific Patterns

**Compute Operations** (MVAU, VVAU, Thresholding)
```python
{
    "PE": ("i", True, 0),                       # Output parallelism
    "SIMD": ("i", True, 0),                     # Input parallelism (MVAU/VVAU)
    "inputDataType": ("s", True, ""),
    "outputDataType": ("s", True, ""),
    "weightDataType": ("s", True, ""),          # For weight-bearing ops
    "numInputVectors": ("ints", False, [1]),
    "ActVal": ("i", False, 0),                  # Activation offset
}
```

**Memory-Heavy Operations** (MVAU, VVAU)
```python
{
    "mem_mode": ("s", False, "internal_decoupled", 
                {"internal_embedded", "internal_decoupled", "external"}),
    "ram_style": ("s", False, "auto", {"auto", "block", "distributed", "ultra"}),
    "runtime_writeable_weights": ("i", False, 0, {0, 1}),
    "accDataType": ("s", False, "INT32"),
}
```

**Streaming Operations** (StreamingFIFO, DWC, Pool)
```python
{
    "depth": ("i", False, 32),                  # FIFO depth
    "impl_style": ("s", False, "vivado", {"vivado", "rtl"}),
    "ram_style": ("s", False, "auto", {"auto", "block", "distributed"}),
}
```

**Spatial Operations** (Conv, Pool, Padding)
```python
{
    "ConvKernelDim": ("ints", True, []),        # Kernel dimensions
    "IFMDim": ("ints", True, []),               # Input feature map
    "OFMDim": ("ints", True, []),               # Output feature map
    "IFMChannels": ("i", True, 0),
    "OFMChannels": ("i", True, 0),
    "Stride": ("ints", False, [1, 1]),
    "Dilation": ("ints", False, [1, 1]),
}
```

## 2. FUNCTION IMPLEMENTATION VARIATIONS

### 2.1 Abstract Methods (MANDATORY)

All HWCustomOp subclasses MUST implement these abstract methods:

```python
@abstractmethod
def get_number_output_values(self):
    """Total number of output values produced."""

@abstractmethod  
def get_input_datatype(self, ind=0):
    """FINN DataType of input stream ind."""

@abstractmethod
def get_output_datatype(self, ind=0):
    """FINN DataType of output stream ind."""

@abstractmethod
def get_normal_input_shape(self, ind=0):
    """Normal (unfolded) input shape."""

@abstractmethod
def get_normal_output_shape(self, ind=0):
    """Normal (unfolded) output shape."""

@abstractmethod
def get_folded_input_shape(self, ind=0):
    """Folded input shape according to parallelization."""

@abstractmethod
def get_folded_output_shape(self, ind=0):
    """Folded output shape according to parallelization."""

@abstractmethod
def get_instream_width(self, ind=0):
    """Input stream width in bits."""

@abstractmethod
def get_outstream_width(self, ind=0):
    """Output stream width in bits."""
```

### 2.2 Common Implementation Variations

**Resource Estimation Methods** (Default: return 0)
```python
def bram_estimation(self):
    # Simple ops: return 0
    # MVAU/VVAU: Complex calculation based on weight memory
    # StreamingFIFO: Based on depth and width

def lut_estimation(self):
    # Most ops: return 0  
    # StreamingDataWidthConverter: ~width conversion logic
    # StreamingFIFO: ~control logic

def dsp_estimation(self, fpgapart):
    # Most ops: return 0
    # MVAU/VVAU: Based on PE count and datatype
    # ChannelwiseOp: Based on operation type

def uram_estimation(self):
    # Most ops: return 0
    # MVAU/VVAU: When ram_style="ultra"
```

**Shape Inference Methods** (Usually implemented)
```python
def infer_node_datatype(self, model):
    # Standard pattern:
    # 1. Read input datatypes from model
    # 2. Set node attributes accordingly  
    # 3. Compute and set output datatypes
    # 4. Update model tensor datatypes

def verify_node(self):
    # Validation logic varies greatly:
    # - Simple ops: Check basic attributes
    # - Complex ops: Validate shapes, parallelization constraints
    # - Memory ops: Check memory configuration consistency
```

**Performance Methods**
```python
def get_exp_cycles(self):
    # Simple ops: return 1 or small constant
    # Compute ops: Based on folding and input size
    # Memory ops: Based on memory access patterns

def execute_node(self, context, graph):
    # Python reference implementation
    # Varies completely by operation
    # Used for verification and functional simulation
```

### 2.3 Weight/Parameter-Specific Methods

Only in weight-bearing operations (MVAU, VVAU, Thresholding):

```python
def generate_params(self, model, path):
    """Generate weight/threshold files for synthesis."""

def make_weight_file(self, weights, weight_file_mode, weight_file_name):
    """Create weight files in various formats."""

def calc_wmem(self):  # Weight memory
    """Calculate weight memory requirements."""

def calc_tmem(self):  # Threshold memory  
    """Calculate threshold memory requirements."""

def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
    """Transform weights for hardware layout."""

def minimize_accumulator_width(self, model):
    """Optimize accumulator bitwidth."""

def minimize_weight_bit_width(self, model):
    """Optimize weight bitwidth."""
```

### 2.4 Interface-Specific Methods

**Standard Interface Override**
```python
def get_verilog_top_module_intf_names(self):
    # Most ops use default AXI-Stream interfaces
    # Some ops add AXI-Lite for configuration
    # Complex ops may add AXI-MM for external memory
    return {
        "clk": ["ap_clk"],
        "rst": ["ap_rst_n"], 
        "s_axis": [("in0_V", width), ...],
        "m_axis": [("out0_V", width), ...],
        "axilite": ["s_axi_control"],  # Optional
        "aximm": ["m_axi_gmem0"],      # Optional
    }
```

## 3. ARCHITECTURAL PATTERNS

### 3.1 Memory Architecture Patterns

**Three-Tier Memory System**
1. **internal_embedded** - Weights compiled into bitfile
   - Pros: No external memory, predictable performance
   - Cons: Long synthesis, no runtime reconfiguration
   
2. **internal_decoupled** - Internal streaming memory (DEFAULT)
   - Pros: Faster synthesis, optional runtime writeable weights
   - Cons: Consumes block RAM resources
   
3. **external** - External memory via AXI-MM
   - Pros: Unlimited weight size, shared memory
   - Cons: Memory bandwidth constraints, complex integration

**RAM Style Selection Pattern**
```python
"ram_style": {
    "auto": "Let Vivado decide based on size/usage",
    "block": "Force BRAM usage (most common)",
    "distributed": "Force LUTRAM (small memories)",  
    "ultra": "Force URAM (UltraScale+ only, large memories)"
}
```

### 3.2 Parallelization Architecture

**Folding Strategy Pattern**
```python
# Standard folding calculation
def get_folded_input_shape(self):
    normal_shape = self.get_normal_input_shape()
    simd = self.get_nodeattr("SIMD")
    # Fold innermost dimension by SIMD factor
    folded_shape = (*normal_shape[:-1], normal_shape[-1] // simd, simd)
    return folded_shape

def get_folded_output_shape(self):
    normal_shape = self.get_normal_output_shape()  
    pe = self.get_nodeattr("PE")
    # Fold innermost dimension by PE factor
    folded_shape = (*normal_shape[:-1], normal_shape[-1] // pe, pe)
    return folded_shape
```

**Stream Width Calculation Pattern**
```python
def get_instream_width(self, ind=0):
    folded_shape = self.get_folded_input_shape(ind)
    simd = folded_shape[-1]  # Parallel elements
    dt = self.get_input_datatype(ind)
    return simd * dt.bitwidth()

def get_outstream_width(self, ind=0):
    folded_shape = self.get_folded_output_shape(ind)
    pe = folded_shape[-1]  # Parallel elements  
    dt = self.get_output_datatype(ind)
    return pe * dt.bitwidth()
```

### 3.3 Backend Selection Patterns

**Implementation Style Hierarchy**
```python
1. Check "preferred_impl_style" attribute
2. If "rtl" and RTL implementation exists → use RTL
3. If "hls" or RTL unavailable → use HLS  
4. If "" (auto) → choose based on operation characteristics
```

**Execution Mode Selection**
```python
"exec_mode": {
    "cppsim": "C++ simulation (fastest, less accurate)",
    "rtlsim": "RTL simulation (slower, cycle-accurate)", 
    "": "Auto-select based on available implementations"
}
```

### 3.4 Interface Protocol Patterns

**AXI-Stream Standard Pattern**
```verilog
// Input interface
input [WIDTH-1:0] s_axis_TDATA,
input s_axis_TVALID,
output s_axis_TREADY,

// Output interface  
output [WIDTH-1:0] m_axis_TDATA,
output m_axis_TVALID,
input m_axis_TREADY,
```

**AXI-Lite Configuration Pattern** (Optional)
```verilog
// Configuration interface for runtime parameters
input s_axi_control_AWVALID,
output s_axi_control_AWREADY,
input [31:0] s_axi_control_AWADDR,
// ... (full AXI-Lite interface)
```

**Control Signal Pattern**
```verilog
// Global control (always present)
input ap_clk,
input ap_rst_n,
input ap_start,
output ap_done,
output ap_idle,
output ap_ready,
```

## 4. KEY GAPS IN CURRENT AUTOHWCUSTOMOP

### 4.1 Missing Critical Nodeattrs

Our generated AutoHWCustomOp needs these essential nodeattrs:

```python
{
    # Parallelization (CRITICAL)
    "PE": ("i", True, 0),
    "SIMD": ("i", True, 0),  # If applicable
    
    # Datatypes (CRITICAL)  
    "inputDataType": ("s", True, ""),
    "outputDataType": ("s", True, ""),
    
    # Shape information (CRITICAL)
    "numInputVectors": ("ints", False, [1]),
    
    # Backend preferences (IMPORTANT)
    "preferred_impl_style": ("s", False, "rtl", {"", "hls", "rtl"}),
    
    # Resource configuration (IMPORTANT)
    "ram_style": ("s", False, "auto", {"auto", "block", "distributed", "ultra"}),
}
```

### 4.2 Missing Critical Methods

Must implement these for FINN compatibility:

```python
def get_folded_input_shape(self, ind=0):
    """Must handle parallelization folding."""

def get_folded_output_shape(self, ind=0):
    """Must handle parallelization folding."""

def infer_node_datatype(self, model):
    """Must update model tensor datatypes."""

def verify_node(self):
    """Must validate node configuration."""
```

### 4.3 Resource Estimation Gaps

Should implement basic resource estimation:

```python
def lut_estimation(self):
    """Basic LUT estimate based on RTL characteristics."""
    
def bram_estimation(self):
    """BRAM estimate if using memory interfaces."""
    
def get_exp_cycles(self):
    """Performance estimate for scheduling."""
```

## 5. RECOMMENDATIONS FOR AUTOHWCUSTOMOP

### 5.1 Immediate Priorities

1. **Add Standard Nodeattrs**: Implement the critical nodeattrs identified above
2. **Implement Abstract Methods**: All 9 abstract methods from base class
3. **Add Shape Folding**: Support PE-based parallelization folding
4. **Add Datatype Inference**: Standard infer_node_datatype implementation

### 5.2 Template Enhancement Priorities

1. **Nodeattr Template Section**: Auto-generate nodeattrs from RTL pragmas
2. **Shape Method Templates**: Generate folding logic from interface metadata  
3. **Resource Estimation Templates**: Basic estimates from RTL characteristics
4. **Verification Templates**: Standard validation based on interface constraints

### 5.3 Long-term Enhancements

1. **Memory Mode Support**: For ops with weight interfaces
2. **Multi-Interface Support**: For complex ops with multiple streams
3. **Advanced Resource Estimation**: From synthesis results or models
4. **Backend Selection Logic**: RTL vs HLS preference handling

## Conclusion

This analysis reveals that FINN HWCustomOps follow well-established patterns that our AutoHWCustomOp must emulate for seamless integration. The most critical gaps are in nodeattr completeness, shape folding methods, and datatype inference. Addressing these will ensure generated AutoHWCustomOps work correctly within the FINN compilation flow.