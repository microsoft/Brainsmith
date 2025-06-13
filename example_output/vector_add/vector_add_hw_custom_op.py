############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for vector_add
# Generated from: example_vector_add.sv
# Generation timestamp: 2025-06-12T22:42:07.089987
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
    
    Generated from RTL: example_vector_add.sv
    Uses validated symbolic BDIM shapes resolved at runtime.
    
    RTL Parameters:
    - PE: Optional (default=4)    - VECTOR_SIZE: Required    """
    
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
        self.rtl_source = "example_vector_add.sv"
    
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
                allowed_datatypes=[
                ],
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
        attrs["VECTOR_SIZE"] = ("i", True, None)  # Required parameter
        
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
        """Returns FINN DataType of input 0 (input0)."""
        return DataType[self.get_nodeattr("inputDataType")]
    
    def get_input_datatype(self, ind=1):
        """Returns FINN DataType of input 1 (input1)."""
        return DataType[self.get_nodeattr("inputDataType")]
    
    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output 0 (output0)."""
        return DataType[self.get_nodeattr("outputDataType")]
    
    # ===== Shape Calculation Methods =====
    
    def get_normal_input_shape(self, ind=0):
        """Calculate normal (non-folded) input shape."""
        
                vecs = list(self.get_nodeattr("numInputVectors"))
                return tuple(vecs + [1])
    
    def get_normal_output_shape(self, ind=0):
        """Calculate normal (non-folded) output shape."""
        return self.get_normal_input_shape()
    
    def get_folded_input_shape(self, ind=0):
        """Calculate folded input shape based on parallelism."""
        
                vecs = list(self.get_nodeattr("numInputVectors"))
                return tuple(vecs + [1, 1])
    
    def get_folded_output_shape(self, ind=0):
        """Calculate folded output shape based on parallelism."""
        return self.get_folded_input_shape()
    
    # ===== Stream Width Methods =====
    
    def get_instream_width(self, ind=0):
        """Calculate input stream width in bits."""
        
                i_bits = self.get_input_datatype().bitwidth()
                pe = self.get_nodeattr("PE") if self.get_nodeattr("PE") else 4
                return i_bits * pe
    
    def get_outstream_width(self, ind=0):
        """Calculate output stream width in bits."""
        
                o_bits = self.get_output_datatype().bitwidth()
                pe = self.get_nodeattr("PE") if self.get_nodeattr("PE") else 4
                return o_bits * pe
    
    
    # ===== Resource Estimation Methods =====
    
    def get_exp_cycles(self):
        """Calculate expected cycles for operation."""
        return np.prod(self.get_folded_output_shape()[:-1])
    
    
    def bram_estimation(self) -> int:
        """Estimate BRAM usage for vector_add."""
        return 1
    
    def lut_estimation(self) -> int:
        """Estimate LUT usage for vector_add."""
        return 2000
    
    def dsp_estimation(self) -> int:
        """Estimate DSP usage for vector_add."""
        return 0
    


# Convenience function for FINN integration
def make_vector_add_node(inputs, outputs, **node_attrs):
    """
    Create VectorAdd ONNX node.
    
    Required parameters:
    - VECTOR_SIZE: int
    
    Optional parameters (with defaults):
    - PE: int = 4
    """
    import onnx.helper
    
    # Validate required parameters
    required = ['VECTOR_SIZE']
    missing = [p for p in required if p not in node_attrs]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    
    return onnx.helper.make_node(
        "VectorAdd",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )