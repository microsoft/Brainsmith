############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for test_new_format
# Generated from: test_new_pragma_format.sv
# Generation timestamp: 2025-06-16T23:25:42.353960
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


class TestNewFormat(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for test_new_format kernel.
    
    Generated from RTL: test_new_pragma_format.sv
    Follows FINN's standard HWCustomOp pattern with static interface metadata.
    
    RTL parameters are defined in get_nodeattr_types().
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize TestNewFormat following FINN's standard pattern."""
        super().__init__(onnx_node, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "test_new_format"
        self.rtl_source = "test_new_pragma_format.sv"
    
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
                datatype_constraints=[
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="s_axis_input0",
                interface_type=InterfaceType.INPUT,
                datatype_constraints=[
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=['C', 'PE'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="m_axis_output0",
                interface_type=InterfaceType.OUTPUT,
                datatype_constraints=[
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
        
        # RTL parameters as node attributes (excluding datatype-linked parameters)
        my_attrs["INPUT0_BDIM"] = ("i", True, None)  # Required parameter
        my_attrs["INPUT0_SDIM"] = ("i", True, None)  # Required parameter
        my_attrs["OUTPUT0_BDIM"] = ("i", True, None)  # Required parameter
        my_attrs["OUTPUT0_SDIM"] = ("i", True, None)  # Required parameter
        my_attrs["C"] = ("i", True, None)  # Required parameter
        my_attrs["PE"] = ("i", False, 4)  # Optional with default
        
        # Interface datatype attributes (high-level datatype specification)
        my_attrs["s_axis_input0DataType"] = ('s', False, 'INT8')  # input interface datatype
        my_attrs["m_axis_output0DataType"] = ('s', False, 'INT8')  # output interface datatype
        
        # Hardware-specific attributes from RTL analysis
        my_attrs["NumChannels"] = ('i', True, 64)
        my_attrs["inputDataType"] = ('s', True, '')
        my_attrs["outputDataType"] = ('s', True, '')
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
        """Estimate BRAM usage for test_new_format."""
        return 1
    
    def lut_estimation(self) -> int:
        """Estimate LUT usage for test_new_format."""
        return 2000
    
    def dsp_estimation(self) -> int:
        """Estimate DSP usage for test_new_format."""
        return 0
    
    def verify_node(self):
        """Verify kernel-specific constraints."""
        super().verify_node()
        
        # Verify all required parameters are present
        if self.get_nodeattr("INPUT0_WIDTH") is None:
            raise ValueError(f"Required parameter 'INPUT0_WIDTH' not specified")
        if self.get_nodeattr("SIGNED_INPUT0") is None:
            raise ValueError(f"Required parameter 'SIGNED_INPUT0' not specified")
        if self.get_nodeattr("OUTPUT0_WIDTH") is None:
            raise ValueError(f"Required parameter 'OUTPUT0_WIDTH' not specified")
        if self.get_nodeattr("INPUT0_BDIM") is None:
            raise ValueError(f"Required parameter 'INPUT0_BDIM' not specified")
        if self.get_nodeattr("INPUT0_SDIM") is None:
            raise ValueError(f"Required parameter 'INPUT0_SDIM' not specified")
        if self.get_nodeattr("OUTPUT0_BDIM") is None:
            raise ValueError(f"Required parameter 'OUTPUT0_BDIM' not specified")
        if self.get_nodeattr("OUTPUT0_SDIM") is None:
            raise ValueError(f"Required parameter 'OUTPUT0_SDIM' not specified")
        if self.get_nodeattr("C") is None:
            raise ValueError(f"Required parameter 'C' not specified")
        
        # Additional test_new_format-specific verification
        # TODO: Add kernel-specific constraint checks


# Convenience function for FINN integration
def make_test_new_format_node(inputs, outputs, **node_attrs):
    """
    Create TestNewFormat ONNX node.
    
    Required algorithm parameters:
    - INPUT0_BDIM: int
    - INPUT0_SDIM: int
    - OUTPUT0_BDIM: int
    - OUTPUT0_SDIM: int
    - C: int
    
    Interface datatype attributes:
    - s_axis_input0DataType: str = "INT8"  # input interface datatype
    - m_axis_output0DataType: str = "INT8"  # output interface datatype
    
    Note: RTL-level parameters (width, signed, format, etc.) are automatically derived 
    from interface datatypes by the RTLBackend and should not be specified directly.
    
    Optional parameters (with defaults):
    - PE: int = 4
    """
    import onnx.helper
    
    # Validate required algorithm parameters (excluding datatype-linked parameters)
    required_algorithm_params = [p for p in ['INPUT0_WIDTH', 'SIGNED_INPUT0', 'OUTPUT0_WIDTH', 'INPUT0_BDIM', 'INPUT0_SDIM', 'OUTPUT0_BDIM', 'OUTPUT0_SDIM', 'C'] if p not in ['INPUT0_WIDTH', 'OUTPUT0_WIDTH', 'SIGNED_INPUT0', 'SIGNED_OUTPUT0']]
    missing = [p for p in required_algorithm_params if p not in node_attrs]
    if missing:
        raise ValueError(f"Missing required algorithm parameters: {missing}")
    
    # Note: Interface datatype parameters are handled by RTLBackend, not node attributes
    
    return onnx.helper.make_node(
        "TestNewFormat",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )