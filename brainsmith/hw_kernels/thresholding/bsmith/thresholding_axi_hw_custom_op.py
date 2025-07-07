############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for thresholding_axi
# Generated from: brainsmith/hw_kernels/thresholding/thresholding_axi_bw.sv
# Generation timestamp: 2025-07-07T16:13:11.737561
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
        # Get parent attributes first (includes exec_mode, backend, etc.)
        attrs = super().get_nodeattr_types()
        
        # Add kernel-specific attributes
        attrs.update({
            # Interface datatype attributes (required by FINN)
            "inputDataType": ('s', True, ''),
            "outputDataType": ('s', True, ''),
            
            # BDIM/SDIM parameters from SHAPE pragmas
            "CHANNELS": ('i', True, 0),  # BDIM: input
            "PE": ('i', True, 0),  # SDIM: input
            
        })
        
        return attrs
    
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
    
    
    def execute_node(self, context, graph):
        """
        Execute the hardware kernel in simulation.
        
        TODO: Implement this method for your specific kernel.
        This should handle both 'cppsim' and 'rtlsim' execution modes.
        
        For reference implementation, see:
        deps/finn/src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py
        """
        raise NotImplementedError(
            f"execute_node() not implemented for {self.__class__.__name__}. "
            "Please implement this method to support simulation."
        )
    
    def bram_estimation(self):
        """
        Estimate BRAM usage for this kernel.
        
        TODO: Implement based on your kernel's memory requirements.
        Return the number of BRAM blocks needed.
        
        For kernels without memory requirements, return 0.
        For kernels with weights/parameters, calculate based on:
        - Weight tensor dimensions
        - Parallelism factors (PE)
        - Memory packing efficiency
        """
        raise NotImplementedError(
            f"bram_estimation() not implemented for {self.__class__.__name__}. "
            "Please implement this method to provide resource estimates."
        )
    
    def uram_estimation(self):
        """
        Estimate URAM usage for this kernel.
        
        TODO: Implement based on your kernel's memory requirements.
        Return the number of URAM blocks needed.
        
        For kernels without memory requirements, return 0.
        For kernels with large weight tensors, consider URAM usage.
        """
        raise NotImplementedError(
            f"uram_estimation() not implemented for {self.__class__.__name__}. "
            "Please implement this method to provide resource estimates."
        )
    
    def lut_estimation(self):
        """
        Estimate LUT usage for this kernel.
        
        TODO: Implement based on your kernel's logic requirements.
        Return the number of LUTs needed.
        
        Consider:
        - Computational complexity
        - Data path width
        - Control logic overhead
        """
        raise NotImplementedError(
            f"lut_estimation() not implemented for {self.__class__.__name__}. "
            "Please implement this method to provide resource estimates."
        )


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