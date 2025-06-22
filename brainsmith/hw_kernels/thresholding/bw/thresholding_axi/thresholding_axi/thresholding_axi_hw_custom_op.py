############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for thresholding_axi
# Generated from: brainsmith/hw_kernels/thresholding/thresholding_axi_bw.sv
# Generation timestamp: 2025-06-20T22:41:12.375022
#
# PHASE 2: FINN INTEGRATION
# This HWCustomOp follows FINN's standard pattern with simple constructor
# and static interface metadata. Parameters are accessed via get_nodeattr().
############################################################################

from typing import List, Dict, Tuple, Any
import numpy as np
from qonnx.core.datatype import DataType

from brainsmith.dataflow.core import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy


class ThresholdingAxi(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for thresholding_axi kernel.
    
    Generated from RTL: brainsmith/hw_kernels/thresholding/thresholding_axi_bw.sv
    Follows FINN's standard HWCustomOp pattern with static interface metadata.
    
    RTL parameters are defined in get_nodeattr_types().
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize ThresholdingAxi following FINN's standard pattern."""
        super().__init__(onnx_node, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "thresholding_axi"
        self.rtl_source = "brainsmith/hw_kernels/thresholding/thresholding_axi_bw.sv"
    
    @staticmethod
    def get_interface_metadata() -> List[InterfaceMetadata]:
        """
        Return static interface metadata for dataflow interfaces (AXI-Stream only).
        
        Control and configuration interfaces are handled separately and not included here.
        All BDIM parameters have been validated during template generation
        to ensure they reference valid module parameters.
        """
        return [
            InterfaceMetadata(
                name="input",
                interface_type=InterfaceType.INPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="FIXED",
                        min_width=1,
                        max_width=32
                    ),
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="output",
                interface_type=InterfaceType.OUTPUT,
                datatype_constraints=[
                    DatatypeConstraintGroup(
                        base_type="FIXED",
                        min_width=1,
                        max_width=32
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
        # Start with parent class attributes
        my_attrs = {}
        
        # RTL parameters as node attributes (only exposed parameters)
        my_attrs["input_FPARG"] = ("i", True, None)  # Required parameter
        my_attrs["THRESHOLDS_PATH"] = ("i", True, None)  # Required parameter
        my_attrs["USE_AXILITE"] = ("i", True, None)  # Required parameter
        my_attrs["DEPTH_TRIGGER_URAM"] = ("i", True, None)  # Required parameter
        my_attrs["DEPTH_TRIGGER_BRAM"] = ("i", True, None)  # Required parameter
        my_attrs["DEEP_PIPELINE"] = ("i", True, None)  # Required parameter
        
        # Interface datatype attributes (high-level datatype specification)
        my_attrs["input0DataType"] = ('s', False, 'INT8')  # input interface datatype
        my_attrs["output0DataType"] = ('s', False, 'INT8')  # output interface datatype
        
        # Internal datatype attributes (e.g., thresholdDataType, accumulatorDataType)
        my_attrs["thresholdDataType"] = ("s", False, "UINT8")  # Internal threshold datatype
        
        # Hardware-specific attributes from RTL analysis
        my_attrs["runtime_writeable_weights"] = ('i', False, 0, {0, 1})
        my_attrs["numInputVectors"] = ('ints', False, [1])
        
        # Base HWCustomOp attributes
        my_attrs.update({
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            "numInputVectors": ("ints", False, [1]),
        })
        
        # Update with parent class attributes (FINN base classes)
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs
    
    # ===== Automatic Tensor Formatting Methods =====
    # These methods are automatically generated by the enhanced AutoHWCustomOp
    # using dataflow mathematics, eliminating the need for manual implementation
    
    
    
    
    # Note: Legacy methods like calc_simd(), calc_pe(), etc. are automatically
    # derived from the dataflow model parallelism configuration
    
    # Note: Datatype methods handled by AutoHWCustomOp parent class
    # Parent class validates datatypes against constraint groups at runtime
    # Note: Shape calculation methods handled by AutoHWCustomOp parent class
    # Parent class computes shapes from DataflowModel interfaces automatically
    
    # Note: Stream width methods handled by AutoHWCustomOp parent class
    # Parent class calculates stream widths from datatypes and parallelism automatically
    
    # ===== Resource Estimation Methods =====
    
    # Note: Cycle calculation and memory handling done by AutoHWCustomOp parent class
    
    def bram_estimation(self) -> int:
        """Estimate BRAM usage for thresholding_axi."""
        return 1
    
    def lut_estimation(self) -> int:
        """Estimate LUT usage for thresholding_axi."""
        return 2000
    
    def dsp_estimation(self) -> int:
        """Estimate DSP usage for thresholding_axi."""
        return 0
    
    def verify_node(self):
        """Verify kernel-specific constraints."""
        super().verify_node()
        
        # Verify all required parameters are present
        if self.get_nodeattr("input_WIDTH") is None:
            raise ValueError(f"Required parameter 'input_WIDTH' not specified")
        if self.get_nodeattr("output_WIDTH") is None:
            raise ValueError(f"Required parameter 'output_WIDTH' not specified")
        if self.get_nodeattr("T_WIDTH") is None:
            raise ValueError(f"Required parameter 'T_WIDTH' not specified")
        if self.get_nodeattr("input_BDIM") is None:
            raise ValueError(f"Required parameter 'input_BDIM' not specified")
        if self.get_nodeattr("input_SDIM") is None:
            raise ValueError(f"Required parameter 'input_SDIM' not specified")
        if self.get_nodeattr("input_SIGNED") is None:
            raise ValueError(f"Required parameter 'input_SIGNED' not specified")
        if self.get_nodeattr("input_FPARG") is None:
            raise ValueError(f"Required parameter 'input_FPARG' not specified")
        if self.get_nodeattr("output_BIAS") is None:
            raise ValueError(f"Required parameter 'output_BIAS' not specified")
        if self.get_nodeattr("THRESHOLDS_PATH") is None:
            raise ValueError(f"Required parameter 'THRESHOLDS_PATH' not specified")
        if self.get_nodeattr("USE_AXILITE") is None:
            raise ValueError(f"Required parameter 'USE_AXILITE' not specified")
        if self.get_nodeattr("DEPTH_TRIGGER_URAM") is None:
            raise ValueError(f"Required parameter 'DEPTH_TRIGGER_URAM' not specified")
        if self.get_nodeattr("DEPTH_TRIGGER_BRAM") is None:
            raise ValueError(f"Required parameter 'DEPTH_TRIGGER_BRAM' not specified")
        if self.get_nodeattr("DEEP_PIPELINE") is None:
            raise ValueError(f"Required parameter 'DEEP_PIPELINE' not specified")
        
        # Additional thresholding_axi-specific verification
        # TODO: Add kernel-specific constraint checks


# Convenience function for FINN integration
def make_thresholding_axi_node(inputs, outputs, **node_attrs):
    """
    Create ThresholdingAxi ONNX node.
    
    Required algorithm parameters:
    - input_FPARG: int
    - THRESHOLDS_PATH: int
    - USE_AXILITE: int
    - DEPTH_TRIGGER_URAM: int
    - DEPTH_TRIGGER_BRAM: int
    - DEEP_PIPELINE: int
    
    Interface datatype attributes:
    - input0DataType: str = "INT8"  # input interface datatype
    - output0DataType: str = "INT8"  # output interface datatype
    
    Note: RTL-level parameters (width, signed, format, etc.) are automatically derived 
    from interface datatypes by the RTLBackend and should not be specified directly.
    
    Optional parameters (with defaults):
    """
    import onnx.helper
    
    # Validate required algorithm parameters (only exposed parameters)
    required_algorithm_params = [p for p in ['input_WIDTH', 'output_WIDTH', 'T_WIDTH', 'input_BDIM', 'input_SDIM', 'input_SIGNED', 'input_FPARG', 'output_BIAS', 'THRESHOLDS_PATH', 'USE_AXILITE', 'DEPTH_TRIGGER_URAM', 'DEPTH_TRIGGER_BRAM', 'DEEP_PIPELINE'] if p in ['input_FPARG', 'THRESHOLDS_PATH', 'USE_AXILITE', 'DEPTH_TRIGGER_URAM', 'DEPTH_TRIGGER_BRAM', 'DEEP_PIPELINE']]
    missing = [p for p in required_algorithm_params if p not in node_attrs]
    if missing:
        raise ValueError(f"Missing required algorithm parameters: {missing}")
    
    # Note: Interface datatype parameters are handled by RTLBackend, not node attributes
    
    return onnx.helper.make_node(
        "ThresholdingAxi",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )