############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for thresholding_axi
# Generated from: brainsmith/hw_kernels/thresholding/thresholding_axi_bw.sv
# Generation timestamp: 2025-07-01T22:30:36.570438
#
# This HWCustomOp uses the modern AutoHWCustomOp base class with explicit
# parameter definitions and no runtime CodegenBinding dependencies.
############################################################################

from typing import List, Dict, Tuple, Any
import numpy as np
from qonnx.core.datatype import DataType

from brainsmith.core.finn.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.core.dataflow import (
    KernelDefinition,
    InputDefinition,
    OutputDefinition,
    RelationType
)
from brainsmith.core.dataflow.qonnx_types import DatatypeConstraintGroup


class ThresholdingAxi(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for thresholding_axi kernel.
    
    Generated from RTL: brainsmith/hw_kernels/thresholding/thresholding_axi_bw.sv
    Uses AutoHWCustomOp for automatic FINN method implementation.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize ThresholdingAxi with KernelDefinition."""
        kernel_def = self._create_kernel_definition()
        super().__init__(onnx_node, kernel_def, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "thresholding_axi"
        self.rtl_source = "brainsmith/hw_kernels/thresholding/thresholding_axi_bw.sv"
    
    def get_nodeattr_types(self):
        """Define interface datatypes and BDIM/SDIM parameters from SHAPE pragmas."""
        return {
            # Interface datatype attributes (required by FINN)
            "inputDataType": ('s', True, ''),
            "outputDataType": ('s', True, ''),
            
            # BDIM/SDIM parameters from SHAPE pragmas
            "CHANNELS": ('i', True, 0),  # BDIM: input
            "PE": ('i', True, 0),  # SDIM: input
            
        }
    
    def _create_kernel_definition(self) -> KernelDefinition:
        """Create simplified KernelDefinition with interface definitions only."""
        kernel_def = KernelDefinition(name="thresholding_axi")
        
        # Add input definitions
        input_def = InputDefinition(
            name="input",
            datatype_constraints=[
                DatatypeConstraintGroup(
                    base_type="ANY",
                    min_width=1,
                    max_width=32
                ),
            ],
            block_tiling=["CHANNELS"],
            stream_tiling=["PE"]
        )
        kernel_def.add_input(input_def)
        
        # Add weight input definitions
        
        # Add output definitions
        output_def = OutputDefinition(
            name="output",
            datatype_constraints=[
                DatatypeConstraintGroup(
                    base_type="ANY",
                    min_width=1,
                    max_width=32
                ),
            ],
        )
        kernel_def.add_output(output_def)
        
        # Add relationships
        return kernel_def
    
    def _extract_input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract input specifications from ONNX context.
        
        Returns:
            Dictionary mapping input names to (shape, datatype) tuples
        """
        specs = {}
        
        # input interface
        # Get shape from ONNX graph context - input shapes must be inferred from graph
        # For now, use a default shape that will be overridden by the ONNX graph
        input_shape_0 = [1, 64]  # Default shape - will be replaced by actual ONNX tensor shape
        input_dtype_0 = DataType[self.get_nodeattr("inputDataType")]
        specs["input"] = (tuple(input_shape_0), input_dtype_0)
        
        return specs
    
    def _extract_output_specs(self) -> Dict[str, Tuple[Tuple[int, ...], DataType]]:
        """
        Extract output specifications from ONNX context.
        
        Returns:
            Dictionary mapping output names to (shape, datatype) tuples
        """
        specs = {}
        
        # output interface
        # For thresholding, output shape equals input shape
        output_shape_0 = [1, 64]  # Default - equals input shape for thresholding
        output_dtype_0 = DataType[self.get_nodeattr("outputDataType")]
        specs["output"] = (tuple(output_shape_0), output_dtype_0)
        
        return specs
    


# Convenience function for FINN integration
def make_thresholding_axi_node(inputs, outputs, **node_attrs):
    """
    Create ThresholdingAxi ONNX node.
    
    Interface datatype attributes (required):
    - inputDataType: str (required)
    - outputDataType: str (required)
    
    BDIM/SDIM parameters from SHAPE pragmas:
    - CHANNELS: int (required)  # BDIM: input
    - PE: int (required)  # SDIM: input
    
    """
    import onnx.helper
    
    # Verify required interface datatypes are specified
    if "inputDataType" not in node_attrs:
        raise ValueError("Required attribute 'inputDataType' not specified")
    if "outputDataType" not in node_attrs:
        raise ValueError("Required attribute 'outputDataType' not specified")
    
    return onnx.helper.make_node(
        "ThresholdingAxi",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )