############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for thresholding_axi
# Generated from: thresholding_axi.sv
# Generation timestamp: 2025-06-10T03:22:48.593307
#
# RUNTIME-CONFIGURABLE HARDWARE COMPONENT
# This HWCustomOp uses runtime dimension extraction from ModelWrapper.
# NEVER set static num_tensors, tDim, or stream_dims values in generated code.
############################################################################

from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType
from brainsmith.dataflow.core.tensor_chunking import index_chunking, default_chunking, last_dim_chunking


class ThresholdingAxiHWCustomOp(AutoHWCustomOp):
    """
    Slim auto-generated HWCustomOp for thresholding_axi kernel.
    
    Generated from RTL: thresholding_axi.sv
    Uses enhanced TDIM pragma integration for automatic chunking strategies.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize ThresholdingAxiHWCustomOp with interface metadata and chunking strategies."""
        
        # Define interface metadata with enhanced TDIM pragma integration
        # Only include AXI_STREAM interfaces in dataflow model (AXI_LITE will be handled separately)
        self._interface_metadata = [
            InterfaceMetadata(
                name="s_axis",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        finn_type="UINT8",
                        bit_width=8,
                        signed=False
                    ),
                ],
                chunking_strategy=default_chunking()            ),
            InterfaceMetadata(
                name="m_axis",
                interface_type=DataflowInterfaceType.OUTPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        finn_type="UINT8",
                        bit_width=8,
                        signed=False
                    ),
                ],
                chunking_strategy=default_chunking()            ),
        ]
        
        # Initialize parent with interface metadata
        super().__init__(onnx_node, interface_metadata=self._interface_metadata, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "thresholding_axi"
        self.rtl_source = "thresholding_axi.sv"
    
    def get_nodeattr_types(self):
        """Define kernel-specific node attributes."""
        attrs = super().get_nodeattr_types()
        
        # Add RTL parameters as node attributes
        kernel_attrs = {
            "N": ("i", False, 0),
            "WI": ("i", False, 0),
            "WT": ("i", False, 0),
            "C": ("i", False, 1),
            "PE": ("i", False, 1),
            "SIGNED": ("i", False, 1),
            "FPARG": ("i", False, 0),
            "BIAS": ("i", False, 0),
            "THRESHOLDS_PATH": ("i", False, ""),
            "USE_AXILITE": ("i", False, 0),
            "DEPTH_TRIGGER_URAM": ("i", False, 0),
            "DEPTH_TRIGGER_BRAM": ("i", False, 0),
            "DEEP_PIPELINE": ("i", False, 0),
        }
        
        attrs.update(kernel_attrs)
        return attrs
    
    def get_kernel_interface_specs(self):
        """Return interface specifications for kernel integration."""
        return {
            "s_axis": {
                "type": "input",
                "interface_type": "AXI_STREAM",
                "chunking_strategy": "default_chunking",
            },
            "m_axis": {
                "type": "output",
                "interface_type": "AXI_STREAM",
                "chunking_strategy": "default_chunking",
            },
        }
    
    


# Convenience function for FINN integration
def make_thresholding_axi_node(inputs, outputs, **node_attrs):
    """Create ThresholdingAxiHWCustomOp ONNX node with enhanced TDIM pragma support."""
    import onnx.helper
    
    return onnx.helper.make_node(
        "ThresholdingAxiHWCustomOp",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )