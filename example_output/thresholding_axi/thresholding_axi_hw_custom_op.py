############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for thresholding_axi
# Generated from: ../examples/thresholding/thresholding_axi.sv
# Generation timestamp: 2025-06-15T18:28:11.493068
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
    
    Generated from RTL: ../examples/thresholding/thresholding_axi.sv
    Follows FINN's standard HWCustomOp pattern with static interface metadata.
    
    RTL parameters are defined in get_nodeattr_types().
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize ThresholdingAxi following FINN's standard pattern."""
        super().__init__(onnx_node, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "thresholding_axi"
        self.rtl_source = "../examples/thresholding/thresholding_axi.sv"
    
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
                name="s_axis",
                interface_type=InterfaceType.INPUT,
                datatype_constraints=[
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="m_axis",
                interface_type=InterfaceType.OUTPUT,
                datatype_constraints=[
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':', ':'],  # Validated symbolic shape
                    rindex=0
                )
            ),
            InterfaceMetadata(
                name="s_axilite",
                interface_type=InterfaceType.CONFIG,
                datatype_constraints=[
                ],
                chunking_strategy=BlockChunkingStrategy(
                    block_shape=[':'],  # Validated symbolic shape
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
        my_attrs["N"] = ("i", True, None)  # Required parameter
        my_attrs["WI"] = ("i", True, None)  # Required parameter
        my_attrs["WT"] = ("i", True, None)  # Required parameter
        my_attrs["C"] = ("i", True, None)  # Required parameter
        my_attrs["PE"] = ("i", False, 1)  # Optional with default
        my_attrs["SIGNED"] = ("i", True, None)  # Required parameter
        my_attrs["FPARG"] = ("i", True, None)  # Required parameter
        my_attrs["BIAS"] = ("i", True, None)  # Required parameter
        my_attrs["THRESHOLDS_PATH"] = ("i", True, None)  # Required parameter
        my_attrs["USE_AXILITE"] = ("i", True, None)  # Required parameter
        my_attrs["DEPTH_TRIGGER_URAM"] = ("i", True, None)  # Required parameter
        my_attrs["DEPTH_TRIGGER_BRAM"] = ("i", True, None)  # Required parameter
        my_attrs["DEEP_PIPELINE"] = ("i", True, None)  # Required parameter
        
        # Note: Interface datatype parameters (width, signed, etc.) are handled by RTLBackend
        # and are not exposed as HWCustomOp node attributes
        
        # Hardware-specific attributes from RTL analysis
        my_attrs["NumChannels"] = ('i', True, 1)
        my_attrs["inputDataType"] = ('s', True, '')
        my_attrs["outputDataType"] = ('s', True, '')
        my_attrs["numSteps"] = ('i', True, 1)
        my_attrs["signed_input"] = ('i', False, 1, {0, 1})
        my_attrs["ActVal"] = ('i', False, 0)
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
        if self.get_nodeattr("N") is None:
            raise ValueError(f"Required parameter 'N' not specified")
        if self.get_nodeattr("WI") is None:
            raise ValueError(f"Required parameter 'WI' not specified")
        if self.get_nodeattr("WT") is None:
            raise ValueError(f"Required parameter 'WT' not specified")
        if self.get_nodeattr("C") is None:
            raise ValueError(f"Required parameter 'C' not specified")
        if self.get_nodeattr("SIGNED") is None:
            raise ValueError(f"Required parameter 'SIGNED' not specified")
        if self.get_nodeattr("FPARG") is None:
            raise ValueError(f"Required parameter 'FPARG' not specified")
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
    - N: int
    - WI: int
    - WT: int
    - C: int
    - SIGNED: int
    - FPARG: int
    - THRESHOLDS_PATH: int
    - USE_AXILITE: int
    - DEPTH_TRIGGER_URAM: int
    - DEPTH_TRIGGER_BRAM: int
    - DEEP_PIPELINE: int
    
    Note: Interface datatype parameters (width, signed, format, etc.) are handled 
    by the RTLBackend and should not be specified as node attributes.
    
    Optional parameters (with defaults):
    - PE: int = 1
    - BIAS: int = 0
    """
    import onnx.helper
    
    # Validate required algorithm parameters (excluding datatype-linked parameters)
    required_algorithm_params = [p for p in ['N', 'WI', 'WT', 'C', 'SIGNED', 'FPARG', 'THRESHOLDS_PATH', 'USE_AXILITE', 'DEPTH_TRIGGER_URAM', 'DEPTH_TRIGGER_BRAM', 'DEEP_PIPELINE'] if p not in []]
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