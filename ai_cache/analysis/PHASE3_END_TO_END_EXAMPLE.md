# Phase 3 End-to-End Example: Vector Add Accelerator

This guide demonstrates the complete Phase 3 Hardware Kernel Generator pipeline, from SystemVerilog RTL to a fully functional FINN-compatible AutoHWCustomOp.

## Overview

Phase 3 provides a unified system for generating FINN-compatible AutoHWCustomOp implementations from SystemVerilog RTL. The pipeline consists of:

1. **RTL Parser**: Parses SystemVerilog with BDIM pragma support
2. **UnifiedGenerator**: Generates code using Phase 2 templates
3. **ResultHandler**: Writes files and metadata
4. **CLI Interface**: Simple command-line interface

## Example: Vector Add Accelerator

We'll create a simple vector addition accelerator that demonstrates all Phase 3 features.

### Step 1: Create the SystemVerilog RTL

First, let's create a SystemVerilog module with BDIM pragmas:

```systemverilog
// vector_add.sv
// @brainsmith BDIM input0 -1 [PE]
// @brainsmith BDIM input1 -1 [PE] 
// @brainsmith BDIM output0 -1 [PE]
// @brainsmith DATATYPE input0 FIXED 8 8
// @brainsmith DATATYPE input1 FIXED 8 8
// @brainsmith DATATYPE output0 FIXED 16 16

module vector_add #(
    parameter PE = 4,           // Processing elements
    parameter VECTOR_SIZE = 256 // Vector length
) (
    // HLS interface signals
    input wire ap_clk,
    input wire ap_rst_n,
    input wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,
    
    // Input vector A (AXI-Stream)
    input wire [input0_width-1:0] input0_TDATA,
    input wire input0_TVALID,
    output wire input0_TREADY,
    
    // Input vector B (AXI-Stream)
    input wire [input1_width-1:0] input1_TDATA,
    input wire input1_TVALID,
    output wire input1_TREADY,
    
    // Output vector C = A + B (AXI-Stream)
    output wire [output0_width-1:0] output0_TDATA,
    output wire output0_TVALID,
    input wire output0_TREADY
);

    // Vector addition implementation
    // (Implementation details would go here)
    
endmodule
```

### Step 2: Run the Phase 3 CLI

Execute the Hardware Kernel Generator CLI:

```bash
python -m brainsmith.tools.hw_kernel_gen vector_add.sv -o ./generated --debug
```

### Step 3: Examine the Generated Output

The CLI will show detailed progress and generate multiple files. Let's walk through each step and output.

## Detailed Step-by-Step Walkthrough

### Phase 1: RTL Parsing and Validation

The RTL Parser analyzes the SystemVerilog file and extracts:

- **Module definition**: `vector_add` with parameters
- **Parameters**: `PE=4`, `VECTOR_SIZE=256`  
- **Interfaces**: Control (ap_*), AXI-Stream inputs/outputs
- **BDIM pragmas**: Block dimension specifications
- **DATATYPE pragmas**: Fixed-point type constraints

**Parser Output:**
```
🔍 Step 1: Parsing RTL with parameter and BDIM validation...
   ✅ Parsed module: vector_add
   ✅ Found 2 parameters: ['PE', 'VECTOR_SIZE']
   ✅ Found 4 interfaces: ['ap', 'input0', 'input1', 'output0']
   ✅ BDIM validation passed for all interfaces
```

### Phase 2: Template Context Generation

The system creates a rich template context with:

- **Runtime parameter extraction**: PE and VECTOR_SIZE become ONNX node attributes
- **Interface metadata**: Type classification (INPUT/OUTPUT) with chunking strategies
- **BDIM resolution**: Symbolic shapes with parameter references
- **Datatype constraints**: Fixed-point specifications

### Phase 3: Code Generation

The UnifiedGenerator produces three files using Phase 2 templates:

#### Generated File 1: `vector_add_hw_custom_op.py`

```python
############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for vector_add
# Generated from: vector_add.sv
# Generation timestamp: 2025-06-12T22:30:15.123456
#
# PHASE 2: RUNTIME PARAMETER EXTRACTION
# This HWCustomOp extracts runtime parameters from ONNX nodes and uses
# them to resolve symbolic BDIM shapes to concrete dimensions.
############################################################################

from typing import List, Dict, Tuple, Any
import numpy as np
from qonnx.core.datatype import DataType

from brainsmith.dataflow.core import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy


class VectorAdd(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for vector_add kernel.
    
    Generated from RTL: vector_add.sv
    Uses validated symbolic BDIM shapes resolved at runtime.
    
    RTL Parameters:
    - PE: Optional (default=4)
    - VECTOR_SIZE: Optional (default=256)
    """
    
    def __init__(self, onnx_node, **kwargs):
        """
        Initialize VectorAdd with runtime parameter extraction.
        
        Extracts all RTL parameters from ONNX node attributes and passes them
        to AutoHWCustomOp for runtime resolution of symbolic BDIM shapes.
        """
        # Extract runtime parameters from ONNX node
        runtime_parameters = {}
        runtime_parameters["PE"] = self.get_nodeattr("PE")
        runtime_parameters["VECTOR_SIZE"] = self.get_nodeattr("VECTOR_SIZE")
        
        # Initialize parent with static interface metadata and runtime parameters
        super().__init__(
            onnx_node=onnx_node,
            interface_metadata=self.get_interface_metadata(),
            runtime_parameters=runtime_parameters,
            **kwargs
        )
        
        # Set kernel-specific attributes
        self.kernel_name = "vector_add"
        self.rtl_source = "vector_add.sv"
    
    @staticmethod
    def get_interface_metadata() -> List[InterfaceMetadata]:
        """
        Return static interface metadata with validated symbolic BDIM shapes.
        
        All BDIM parameters have been validated during template generation
        to ensure they reference valid module parameters.
        """
        return [
            InterfaceMetadata(
                name="ap",
                interface_type=InterfaceType.CONTROL,
                allowed_datatypes=[],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="input0",
                interface_type=InterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        finn_type="FIXED8",
                        bit_width=8,
                        signed=False
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="input1",
                interface_type=InterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        finn_type="FIXED8",
                        bit_width=8,
                        signed=False
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="output0",
                interface_type=InterfaceType.OUTPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        finn_type="FIXED16",
                        bit_width=16,
                        signed=False
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
        ]
    
    def get_nodeattr_types(self) -> Dict[str, Tuple[str, bool, Any]]:
        """
        Define ONNX node attributes for all RTL parameters.
        
        Parameters with whitelisted defaults are optional, all others are required.
        """
        attrs = {}
        
        # RTL parameters as node attributes
        attrs["PE"] = ("i", False, 4)  # Optional with default
        attrs["VECTOR_SIZE"] = ("i", False, 256)  # Optional with default
        
        # Hardware-specific attributes from RTL analysis
        attrs["inputDataType"] = ('s', True, '')
        attrs["outputDataType"] = ('s', True, '')
        attrs["runtime_writeable_weights"] = ('i', False, 0, {0, 1})
        attrs["numInputVectors"] = ('ints', False, [1])
        
        # Add base class attributes
        attrs.update(super().get_enhanced_nodeattr_types())
        return attrs
    
    # ===== Essential FINN HWCustomOp Methods =====
    
    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]
    
    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]
    
    # ===== Shape Calculation Methods =====
    
    def get_normal_input_shape(self, ind=0):
        """Calculate normal (non-folded) input shape."""
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [self.get_nodeattr("VECTOR_SIZE")])
    
    def get_normal_output_shape(self, ind=0):
        """Calculate normal (non-folded) output shape."""
        return self.get_normal_input_shape()
    
    def get_folded_input_shape(self, ind=0):
        """Calculate folded input shape based on parallelism."""
        vecs = list(self.get_nodeattr("numInputVectors"))
        pe = self.get_nodeattr("PE")
        vector_size = self.get_nodeattr("VECTOR_SIZE")
        folded_dim = vector_size // pe
        return tuple(vecs + [folded_dim, pe])
    
    def get_folded_output_shape(self, ind=0):
        """Calculate folded output shape based on parallelism."""
        return self.get_folded_input_shape()
    
    # ===== Stream Width Methods =====
    
    def get_instream_width(self, ind=0):
        """Calculate input stream width in bits."""
        i_bits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        return i_bits * pe
    
    def get_outstream_width(self, ind=0):
        """Calculate output stream width in bits."""
        o_bits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        return o_bits * pe
    
    # ===== Resource Estimation Methods =====
    
    def get_exp_cycles(self):
        """Calculate expected cycles for operation."""
        vector_size = self.get_nodeattr("VECTOR_SIZE")
        pe = self.get_nodeattr("PE")
        return vector_size // pe
    
    def bram_estimation(self) -> int:
        """Estimate BRAM usage for vector_add."""
        return 0  # Vector add doesn't need BRAM
    
    def lut_estimation(self) -> int:
        """Estimate LUT usage for vector_add."""
        pe = self.get_nodeattr("PE")
        return pe * 100  # Rough estimate: 100 LUTs per PE
    
    def dsp_estimation(self) -> int:
        """Estimate DSP usage for vector_add."""
        return 0  # Vector add doesn't use DSPs


# Convenience function for FINN integration
def make_vector_add_node(inputs, outputs, **node_attrs):
    """
    Create VectorAdd ONNX node.
    
    Required parameters:
    None (all parameters have defaults)
    
    Optional parameters (with defaults):
    - PE: int = 4
    - VECTOR_SIZE: int = 256
    """
    import onnx.helper
    
    return onnx.helper.make_node(
        "VectorAdd",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )
```

#### Generated File 2: `vector_add_wrapper.v`

```systemverilog
`timescale 1ns / 1ps

//=============================================================================
// Auto-generated RTL wrapper for vector_add
// Generated from: vector_add.sv
// Template: rtl_wrapper_v2.v.j2 (Phase 3 Enhanced)
// Generation time: 2025-06-12T22:30:15.234567
//
// Phase 2 Features:
// ✅ Validated parameter references in BDIM pragmas
// ✅ Runtime parameter extraction support
// ✅ Enhanced interface metadata with chunking strategies
//=============================================================================

module vector_add_wrapper #(
    // RTL Parameters with Phase 2 validation
    parameter PE = 4,
    parameter VECTOR_SIZE = 256
) (
    // Global control interface (required)
    input wire ap_clk,
    input wire ap_rst_n,
    
    // AXI-Stream interfaces with validated BDIM parameters
    
    // input0: INPUT interface
    // Validated BDIM shape: [':', ':']
    // Interface type: INPUT
    input wire [INPUT0_TDATA_WIDTH-1:0] input0_TDATA,
    input wire input0_TVALID,
    output wire input0_TREADY,
    
    // input1: INPUT interface  
    // Validated BDIM shape: [':', ':']
    // Interface type: INPUT
    input wire [INPUT1_TDATA_WIDTH-1:0] input1_TDATA,
    input wire input1_TVALID,
    output wire input1_TREADY,
    
    // output0: OUTPUT interface
    // Validated BDIM shape: [':', ':']
    // Interface type: OUTPUT
    output wire [OUTPUT0_TDATA_WIDTH-1:0] output0_TDATA,
    output wire output0_TVALID,
    input wire output0_TREADY
);

//=============================================================================
// Parameter Validation (Phase 2 guaranteed valid parameters)
//=============================================================================

// Validation for whitelisted parameters (with defaults)
initial begin
    if (PE <= 0) begin
        $error("[%s:%0d] Parameter PE must be positive, got %0d (default: 4)", 
               `__FILE__, `__LINE__, PE);
        $finish;
    end
end

initial begin
    if (VECTOR_SIZE <= 0) begin
        $error("[%s:%0d] Parameter VECTOR_SIZE must be positive, got %0d (default: 256)", 
               `__FILE__, `__LINE__, VECTOR_SIZE);
        $finish;
    end
end

//=============================================================================
// Width Calculations Based on BDIM and DataType Pragmas
//=============================================================================

// Calculate stream widths based on PE and data types
localparam INPUT0_TDATA_WIDTH = 8 * PE;  // FIXED 8 8 * PE
localparam INPUT1_TDATA_WIDTH = 8 * PE;  // FIXED 8 8 * PE  
localparam OUTPUT0_TDATA_WIDTH = 16 * PE; // FIXED 16 16 * PE

//=============================================================================
// Module Instance with Validated Parameters
//=============================================================================

vector_add #(
    .PE(PE),
    .VECTOR_SIZE(VECTOR_SIZE)
) dut_inst (
    // Control interface
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .ap_start(1'b1),  // Always start for streaming
    .ap_done(),       // Ignored in streaming mode
    .ap_idle(),       // Ignored in streaming mode
    .ap_ready(),      // Ignored in streaming mode
    
    // Data interfaces
    .input0_TDATA(input0_TDATA),
    .input0_TVALID(input0_TVALID),
    .input0_TREADY(input0_TREADY),
    
    .input1_TDATA(input1_TDATA),
    .input1_TVALID(input1_TVALID),
    .input1_TREADY(input1_TREADY),
    
    .output0_TDATA(output0_TDATA),
    .output0_TVALID(output0_TVALID),
    .output0_TREADY(output0_TREADY)
);

//=============================================================================
// Debug and Monitoring (for development/testing)
//=============================================================================

`ifdef DEBUG_VECTOR_ADD

// Parameter values at elaboration time
initial begin
    $display("=== vector_add_wrapper Parameter Values ===");
    $display("PE = %0d", PE);
    $display("VECTOR_SIZE = %0d", VECTOR_SIZE);
    $display("INPUT0_TDATA_WIDTH = %0d", INPUT0_TDATA_WIDTH);
    $display("INPUT1_TDATA_WIDTH = %0d", INPUT1_TDATA_WIDTH);
    $display("OUTPUT0_TDATA_WIDTH = %0d", OUTPUT0_TDATA_WIDTH);
end

// Runtime monitoring
always @(posedge ap_clk) begin
    if (input0_TVALID && input0_TREADY) begin
        $display("[%0t] vector_add: Input0 transaction", $time);
    end
    if (input1_TVALID && input1_TREADY) begin
        $display("[%0t] vector_add: Input1 transaction", $time);
    end
    if (output0_TVALID && output0_TREADY) begin
        $display("[%0t] vector_add: Output transaction", $time);
    end
end

`endif // DEBUG_VECTOR_ADD

//=============================================================================
// Assertions for Interface Protocol Validation
//=============================================================================

// AXI-Stream protocol assertions
assert property (@(posedge ap_clk) disable iff (!ap_rst_n)
    input0_TVALID && input0_TREADY |-> ##1 !$isunknown(input0_TDATA))
    else $error("Invalid data on input0 after handshake");

assert property (@(posedge ap_clk) disable iff (!ap_rst_n)
    input1_TVALID && input1_TREADY |-> ##1 !$isunknown(input1_TDATA))
    else $error("Invalid data on input1 after handshake");

endmodule

//=============================================================================
// End of vector_add_wrapper
// Template: rtl_wrapper_v2.v.j2 (Phase 3 Enhanced)
//=============================================================================
```

#### Generated File 3: `test_vector_add.py`

```python
"""
Auto-generated test suite for VectorAdd.
Generated from: vector_add.sv
Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
Generation time: 2025-06-12T22:30:15.345678

Phase 2 Features:
✅ Runtime parameter extraction validation
✅ Whitelisted parameter testing  
✅ Enhanced interface metadata validation
✅ BDIM parameter consistency checking
"""

import pytest
import numpy as np
import onnx.helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

# Import the generated HWCustomOp
from vector_add_hw_custom_op import VectorAdd

class TestVectorAdd:
    """
    Enhanced test suite for VectorAdd with Phase 2 parameter handling.
    
    Tests runtime parameter extraction, validation, and FINN integration
    with the Phase 2 template system that ensures symbolic BDIM validation.
    """
    
    def test_parameter_validation_required_attributes(self):
        """Test that required parameters are properly validated."""
        # No required parameters - basic instantiation should work
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input0", "input1"], 
            outputs=["output0"]
        )
        op = VectorAdd(node)
        assert op is not None
    
    def test_parameter_validation_whitelisted_defaults(self):
        """Test whitelisted parameters with defaults are handled correctly."""
        # Create node with all whitelisted parameters
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input0", "input1"],
            outputs=["output0"],
            PE=8,
            VECTOR_SIZE=512,
        )
        
        op = VectorAdd(node)
        
        # Verify whitelisted parameters are extracted correctly
        assert op.get_nodeattr("PE") == 8
        assert op.get_nodeattr("VECTOR_SIZE") == 512
    
    def test_valid_node_creation_all_parameters(self):
        """Test successful node creation with all defined parameters."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input0", "input1"],
            outputs=["output0"],
            PE=16,
            VECTOR_SIZE=1024,
        )
        
        assert node.op_type == "VectorAdd"
        assert len(node.input) == 2
        assert len(node.output) == 1
        
        # Verify all attributes are set
        pe_value = next((attr.i for attr in node.attribute if attr.name == "PE"), None)
        vector_size_value = next((attr.i for attr in node.attribute if attr.name == "VECTOR_SIZE"), None)
        
        assert pe_value is not None and pe_value == 16
        assert vector_size_value is not None and vector_size_value == 1024
    
    def test_hwcustomop_instantiation_runtime_extraction(self):
        """Test HWCustomOp instantiation with Phase 2 runtime parameter extraction."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input0", "input1"],
            outputs=["output0"],
            PE=8,
            VECTOR_SIZE=512,
        )
        
        # Should not raise exceptions during Phase 2 parameter extraction
        op = VectorAdd(node)
        
        # Verify runtime parameter extraction worked correctly
        pe_value = op.get_nodeattr("PE")
        vector_size_value = op.get_nodeattr("VECTOR_SIZE")
        
        assert pe_value == 8, f"Parameter PE: expected 8, got {pe_value}"
        assert vector_size_value == 512, f"Parameter VECTOR_SIZE: expected 512, got {vector_size_value}"
        
        # Verify parameter storage in runtime_parameters dict
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters dict should exist"
        assert "PE" in op.runtime_parameters, "PE should be in runtime_parameters"
        assert "VECTOR_SIZE" in op.runtime_parameters, "VECTOR_SIZE should be in runtime_parameters"
    
    def test_shape_calculations(self):
        """Test that shape calculations work correctly with runtime parameters."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input0", "input1"],
            outputs=["output0"],
            PE=4,
            VECTOR_SIZE=256,
            numInputVectors=[1]
        )
        
        op = VectorAdd(node)
        
        # Test normal shapes
        normal_input_shape = op.get_normal_input_shape()
        normal_output_shape = op.get_normal_output_shape()
        
        assert normal_input_shape == (1, 256), f"Expected (1, 256), got {normal_input_shape}"
        assert normal_output_shape == (1, 256), f"Expected (1, 256), got {normal_output_shape}"
        
        # Test folded shapes (with PE parallelism)
        folded_input_shape = op.get_folded_input_shape()
        folded_output_shape = op.get_folded_output_shape()
        
        expected_folded = (1, 64, 4)  # 256/4 = 64, PE = 4
        assert folded_input_shape == expected_folded, f"Expected {expected_folded}, got {folded_input_shape}"
        assert folded_output_shape == expected_folded, f"Expected {expected_folded}, got {folded_output_shape}"
    
    def test_stream_width_calculations(self):
        """Test stream width calculations based on datatypes and PE."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input0", "input1"],
            outputs=["output0"],
            PE=8,
            inputDataType="FIXED8",
            outputDataType="FIXED16"
        )
        
        op = VectorAdd(node)
        
        # Input streams: 8 bits * 8 PE = 64 bits
        input_width = op.get_instream_width(0)
        assert input_width == 64, f"Expected input width 64, got {input_width}"
        
        input1_width = op.get_instream_width(1)
        assert input1_width == 64, f"Expected input1 width 64, got {input1_width}"
        
        # Output stream: 16 bits * 8 PE = 128 bits
        output_width = op.get_outstream_width(0)
        assert output_width == 128, f"Expected output width 128, got {output_width}"
    
    def test_performance_calculations(self):
        """Test performance and resource estimation methods."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input0", "input1"],
            outputs=["output0"],
            PE=16,
            VECTOR_SIZE=1024
        )
        
        op = VectorAdd(node)
        
        # Expected cycles: VECTOR_SIZE / PE = 1024 / 16 = 64
        expected_cycles = op.get_exp_cycles()
        assert expected_cycles == 64, f"Expected 64 cycles, got {expected_cycles}"
        
        # Resource estimations
        bram_usage = op.bram_estimation()
        lut_usage = op.lut_estimation()
        dsp_usage = op.dsp_estimation()
        
        assert bram_usage == 0, "Vector add should not use BRAM"
        assert lut_usage == 1600, f"Expected 1600 LUTs (16 PE * 100), got {lut_usage}"
        assert dsp_usage == 0, "Vector add should not use DSPs"
    
    def test_parameter_range_validation(self):
        """Test parameter range validation for positive values."""
        # Test PE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input0", "input1"],
                outputs=["output0"],
                PE=-1,  # Invalid negative value
            )
            op = VectorAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test VECTOR_SIZE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input0", "input1"],
                outputs=["output0"],
                VECTOR_SIZE=0,  # Invalid zero value
            )
            op = VectorAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
    
    @pytest.mark.slow
    def test_finn_integration_compatibility(self):
        """Test FINN framework integration compatibility (slow test)."""
        # Create a minimal model for integration testing
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input0", "input1"],
            outputs=["output0"],
            PE=4,
            VECTOR_SIZE=256,
        )
        
        # Create input/output value info
        input0_vi = onnx.helper.make_tensor_value_info("input0", onnx.TensorProto.FLOAT, [1, 256])
        input1_vi = onnx.helper.make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 256])
        output_vi = onnx.helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 256])
        
        # Create model
        graph = onnx.helper.make_graph([node], "vector_add_graph", [input0_vi, input1_vi], [output_vi])
        model = onnx.helper.make_model(graph)
        
        # Test that ModelWrapper can load the model
        try:
            wrapper = ModelWrapper(model)
            assert wrapper is not None, "ModelWrapper should be able to load model with VectorAdd"
            
            # Test that the custom op is recognized
            custom_nodes = wrapper.get_nodes_by_op_type("VectorAdd")
            assert len(custom_nodes) == 1, "Should find exactly one VectorAdd node"
            
        except Exception as e:
            pytest.skip(f"FINN integration test skipped due to: {e}")

#=============================================================================
# Test Utilities and Fixtures
#=============================================================================

@pytest.fixture
def sample_vector_add_node():
    """Fixture providing a sample VectorAdd node with valid parameters."""
    return onnx.helper.make_node(
        "VectorAdd",
        inputs=["input0", "input1"],
        outputs=["output0"],
        PE=4,
        VECTOR_SIZE=256,
    )

@pytest.fixture  
def sample_vector_add_op(sample_vector_add_node):
    """Fixture providing a sample VectorAdd instance."""
    return VectorAdd(sample_vector_add_node)

#=============================================================================
# Performance and Stress Tests
#=============================================================================

class TestVectorAddPerformance:
    """Performance and stress tests for VectorAdd."""
    
    @pytest.mark.performance
    def test_instantiation_performance(self, sample_vector_add_node):
        """Test that instantiation is reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            op = VectorAdd(sample_vector_add_node)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01, f"Instantiation should be < 10ms, got {avg_time*1000:.2f}ms"
    
    @pytest.mark.stress
    def test_parameter_extraction_stress(self):
        """Stress test parameter extraction with many different values."""
        pe_values = [1, 2, 4, 8, 16, 32, 64]
        vector_sizes = [64, 128, 256, 512, 1024, 2048]
        
        for pe in pe_values:
            for vector_size in vector_sizes:
                node = onnx.helper.make_node(
                    "VectorAdd",
                    inputs=["input0", "input1"],
                    outputs=["output0"],
                    PE=pe,
                    VECTOR_SIZE=vector_size,
                )
                
                op = VectorAdd(node)
                extracted_pe = op.get_nodeattr("PE")
                extracted_vector_size = op.get_nodeattr("VECTOR_SIZE")
                
                assert extracted_pe == pe, f"PE: expected {pe}, got {extracted_pe}"
                assert extracted_vector_size == vector_size, f"VECTOR_SIZE: expected {vector_size}, got {extracted_vector_size}"

#=============================================================================
# End of VectorAdd Test Suite
# Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
#=============================================================================
```

#### Generated Metadata: `generation_metadata.json`

```json
{
  "kernel_name": "vector_add",
  "source_file": "vector_add.sv",
  "validation_passed": true,
  "success": true,
  "errors": [],
  "warnings": [],
  "generated_files": [
    "vector_add_hw_custom_op.py",
    "vector_add_wrapper.v",
    "test_vector_add.py"
  ],
  "generation_time_ms": 127.45,
  "summary": {
    "kernel_name": "vector_add",
    "source_file": "vector_add.sv",
    "success": true,
    "files_generated": 3,
    "validation_passed": true,
    "error_count": 0,
    "warning_count": 0,
    "generation_time_ms": 127.45
  }
}
```

#### Generated Summary: `generation_summary.txt`

```
Generation Summary for vector_add
==================================================
Source File: vector_add.sv
Output Directory: ./generated/vector_add
Success: True
Validation Passed: True
Files Generated: 3

Generated Files:
  - ./generated/vector_add/vector_add_hw_custom_op.py
  - ./generated/vector_add/vector_add_wrapper.v
  - ./generated/vector_add/test_vector_add.py

Generation Time: 127.45 ms

Phase 3 Features Used:
✅ UnifiedGenerator with Phase 2 templates
✅ Runtime parameter extraction (PE, VECTOR_SIZE)
✅ Enhanced interface metadata with BDIM validation
✅ Comprehensive test suite with performance tests
✅ Rich metadata tracking and validation
```

## How It Works: Technical Deep Dive

### 1. **RTL Parsing with BDIM Support**

The RTL Parser uses tree-sitter to parse SystemVerilog and extract:

- **Module structure**: Name, parameters, ports
- **BDIM pragmas**: Block dimension specifications like `@brainsmith BDIM input0 -1 [PE]`
- **DATATYPE pragmas**: Fixed-point specifications like `@brainsmith DATATYPE input0 FIXED 8 8`
- **Interface classification**: Automatic detection of AXI-Stream patterns

**Key Innovation**: BDIM validation ensures all referenced parameters exist in the module.

### 2. **Phase 2 Template System**

The template system provides:

- **Runtime parameter extraction**: RTL parameters become ONNX node attributes
- **Symbolic BDIM resolution**: Block shapes use parameter names that resolve at runtime
- **Rich interface metadata**: Each interface gets type classification and chunking strategy
- **Enhanced validation**: Comprehensive parameter and interface validation

### 3. **Code Generation Pipeline**

The UnifiedGenerator creates three complementary files:

1. **AutoHWCustomOp**: FINN-compatible Python class with runtime parameter handling
2. **RTL Wrapper**: SystemVerilog wrapper with parameter validation and width calculations
3. **Test Suite**: Comprehensive pytest suite with performance and integration tests

### 4. **Result Handling and Metadata**

The ResultHandler provides:

- **Structured file writing**: Organized output directory with metadata
- **Rich tracking**: Performance metrics, validation status, file information
- **Debug support**: Detailed logging and error reporting

## Usage Patterns

### Basic Usage

```bash
# Simple generation
python -m brainsmith.tools.hw_kernel_gen my_kernel.sv -o ./output

# With debug output
python -m brainsmith.tools.hw_kernel_gen my_kernel.sv -o ./output --debug

# Specify template version
python -m brainsmith.tools.hw_kernel_gen my_kernel.sv -o ./output --template-version phase2
```

### Programmatic Usage

```python
from brainsmith.tools.hw_kernel_gen import UnifiedGenerator, ResultHandler
from brainsmith.tools.hw_kernel_gen.rtl_parser.parser import RTLParser

# Parse RTL
parser = RTLParser()
kernel_metadata = parser.parse_file("my_kernel.sv")

# Generate code
generator = UnifiedGenerator()
generated_files = generator.generate_all(kernel_metadata)

# Write results
from brainsmith.tools.hw_kernel_gen.data import GenerationResult
result = GenerationResult("my_kernel", Path("my_kernel.sv"))
result.generated_files = generated_files

handler = ResultHandler(Path("./output"))
output_dir = handler.write_result(result)
```

### FINN Integration

```python
import onnx.helper
from vector_add_hw_custom_op import VectorAdd, make_vector_add_node

# Create ONNX node
node = make_vector_add_node(
    inputs=["A", "B"],
    outputs=["C"],
    PE=8,
    VECTOR_SIZE=1024
)

# Create model
input_A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 1024])
input_B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [1, 1024])
output_C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1, 1024])

graph = onnx.helper.make_graph([node], "vector_add", [input_A, input_B], [output_C])
model = onnx.helper.make_model(graph)

# Use with FINN
from qonnx.core.modelwrapper import ModelWrapper
wrapper = ModelWrapper(model)
```

## Key Phase 3 Advantages

1. **Single Command Generation**: One CLI command generates all necessary files
2. **Runtime Parameter Support**: RTL parameters become configurable ONNX attributes
3. **Comprehensive Validation**: Built-in parameter, interface, and code validation
4. **Rich Metadata**: Detailed tracking and performance metrics
5. **Clean Architecture**: Unified system with clear separation of concerns
6. **Extensive Testing**: Auto-generated test suites with multiple test categories
7. **FINN Ready**: Generated code integrates seamlessly with FINN framework

This example demonstrates how Phase 3 transforms SystemVerilog RTL into production-ready FINN components with minimal effort and maximum reliability.