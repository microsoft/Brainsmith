# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    
    Generated from RTL: brainsmith/kernels/thresholding/thresholding_axi_bw.sv
    Uses direct KernelMetadata access with AutoHWCustomOp base class.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize ThresholdingAxi with KernelDefinition."""
        kernel_def = self._create_kernel_definition()
        super().__init__(onnx_node, kernel_def, **kwargs)
    
    def get_nodeattr_types(self):
        """
        Define all node attributes for thresholding_axi.
        """
        attrs = super().get_nodeattr_types()
        
        kernel_attrs = {
            "inputDataType": ('s', True, ""),
            "weightDataType": ('s', True, ""),
            "outputDataType": ('s', True, ""),
            "CHANNELS": ('i', True, 0),
            "PE": ('i', True, 0),
            "runtime_writeable_weights": ('b', False, True),
            # Backend selection attribute
            "preferred_impl_style": ('s', False, "rtl"),
        }
        attrs.update(kernel_attrs)
        
        return attrs
    
    def _create_kernel_definition(self) -> KernelDefinition:
        """
        Create KernelDefinition for thresholding_axi.
        
        Creates KernelDefinition using direct metadata access.
        """
        kernel_def = KernelDefinition("thresholding_axi")
        
        # All input definitions (regular inputs and AXI-Stream weights)
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
            stream_tiling=["PE"],
        )
        kernel_def.add_input(input_def)
        
        # AXI-Lite weight interfaces as input definitions
        input_def = InputDefinition(
            name="weight",
            datatype_constraints=[
            ],
            is_weight=True
        )
        kernel_def.add_input(input_def)
        
        # Output definitions
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
        
        # Add relationships (if they exist on KernelMetadata)
        
        return kernel_def

    ############################################################################
    # ======================= MANUALLY IMPLEMENT FUNCTIONS BELOW ===============
    # Add custom helper methods, execution logic, and resource estimation logic
    # here. This section is intentionally left for manual implementation.
    ############################################################################
        
    def execute_node(self, context, graph):
        """
        Execute the hardware kernel in simulation.
        
        TODO: Implement this method for your specific kernel.
        This should handle both 'cppsim' and 'rtlsim' execution modes.
        
        For reference implementation, see:
        # TAFK TODO
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


# Kernel metadata for reference
"""
thresholding_axi Kernel Specification:

Core Functionality:
- Module: thresholding_axi
- Source: brainsmith/kernels/thresholding/thresholding_axi_bw.sv

Interfaces:
- Input: input (RTL: input)
- Output: output (RTL: output)

Interface Attributes:
- inputDataType: Input interface datatype selection
- outputDataType: Output interface datatype selection  
- weightDataType: Weight interface datatype selection (AXI-Lite)

Shape Parameters:
BDIM Parameters:
- CHANNELS: int (block dimension parameter)
SDIM Parameters:
- PE: int (stream dimension parameter)

Configuration:
- runtime_writeable_weights: bool = True (supports runtime weight updates)
"""