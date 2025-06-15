############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for multi_input_add
# Generated from: multi_input_test.sv
# Generation timestamp: 2025-06-15T18:28:33.165552
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


class MultiInputAdd(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for multi_input_add kernel.
    
    Generated from RTL: multi_input_test.sv
    Follows FINN's standard HWCustomOp pattern with static interface metadata.
    
    RTL parameters are defined in get_nodeattr_types().
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize MultiInputAdd following FINN's standard pattern."""
        super().__init__(onnx_node, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "multi_input_add"
        self.rtl_source = "multi_input_test.sv"
    
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
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="s_axis_input1",
                interface_type=InterfaceType.INPUT,
                datatype_constraints=[
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
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
        my_attrs["ALGORITHM_PARAM"] = ("i", True, None)  # Required parameter
        my_attrs["PE"] = ("i", False, 1)  # Optional with default
        
        # Note: Interface datatype parameters (width, signed, etc.) are handled by RTLBackend
        # and are not exposed as HWCustomOp node attributes
        
        # Hardware-specific attributes from RTL analysis
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
    
    # Note: Datatype methods handled by AutoHWCustomOp parent class
    # Parent class validates datatypes against constraint groups at runtime
    # Note: Shape calculation methods handled by AutoHWCustomOp parent class
    # Parent class computes shapes from DataflowModel interfaces automatically
    
    # Note: Stream width methods handled by AutoHWCustomOp parent class
    # Parent class calculates stream widths from datatypes and parallelism automatically
    
    # ===== Resource Estimation Methods =====
    
    # Note: Cycle calculation and memory handling done by AutoHWCustomOp parent class
    
    def bram_estimation(self) -> int:
        """Estimate BRAM usage for multi_input_add."""
        return 1
    
    def lut_estimation(self) -> int:
        """Estimate LUT usage for multi_input_add."""
        return 2000
    
    def dsp_estimation(self) -> int:
        """Estimate DSP usage for multi_input_add."""
        return 0
    
    def verify_node(self):
        """Verify kernel-specific constraints."""
        super().verify_node()
        
        # Verify all required parameters are present
        if self.get_nodeattr("INPUT0_WIDTH") is None:
            raise ValueError(f"Required parameter 'INPUT0_WIDTH' not specified")
        if self.get_nodeattr("SIGNED_INPUT0") is None:
            raise ValueError(f"Required parameter 'SIGNED_INPUT0' not specified")
        if self.get_nodeattr("INPUT1_WIDTH") is None:
            raise ValueError(f"Required parameter 'INPUT1_WIDTH' not specified")
        if self.get_nodeattr("SIGNED_INPUT1") is None:
            raise ValueError(f"Required parameter 'SIGNED_INPUT1' not specified")
        if self.get_nodeattr("OUTPUT_WIDTH") is None:
            raise ValueError(f"Required parameter 'OUTPUT_WIDTH' not specified")
        if self.get_nodeattr("SIGNED_OUTPUT") is None:
            raise ValueError(f"Required parameter 'SIGNED_OUTPUT' not specified")
        if self.get_nodeattr("ALGORITHM_PARAM") is None:
            raise ValueError(f"Required parameter 'ALGORITHM_PARAM' not specified")
        
        # Additional multi_input_add-specific verification
        # TODO: Add kernel-specific constraint checks


# Convenience function for FINN integration
def make_multi_input_add_node(inputs, outputs, **node_attrs):
    """
    Create MultiInputAdd ONNX node.
    
    Required algorithm parameters:
    - ALGORITHM_PARAM: int
    
    Note: Interface datatype parameters (width, signed, format, etc.) are handled 
    by the RTLBackend and should not be specified as node attributes.
    
    Optional parameters (with defaults):
    - PE: int = 1
    """
    import onnx.helper
    
    # Validate required algorithm parameters (excluding datatype-linked parameters)
    required_algorithm_params = [p for p in ['INPUT0_WIDTH', 'SIGNED_INPUT0', 'INPUT1_WIDTH', 'SIGNED_INPUT1', 'OUTPUT_WIDTH', 'SIGNED_OUTPUT', 'ALGORITHM_PARAM'] if p not in ['INPUT0_WIDTH', 'INPUT1_WIDTH', 'SIGNED_INPUT1', 'SIGNED_OUTPUT', 'OUTPUT_WIDTH', 'SIGNED_INPUT0']]
    missing = [p for p in required_algorithm_params if p not in node_attrs]
    if missing:
        raise ValueError(f"Missing required algorithm parameters: {missing}")
    
    # Note: Interface datatype parameters are handled by RTLBackend, not node attributes
    
    return onnx.helper.make_node(
        "MultiInputAdd",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )