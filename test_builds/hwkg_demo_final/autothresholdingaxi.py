############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for thresholding_axi
# Generated from: thresholding_axi.sv
# Generation timestamp: 2025-06-10T05:52:41.946737
#
# RUNTIME-CONFIGURABLE HARDWARE COMPONENT
# This HWCustomOp uses runtime dimension extraction from ModelWrapper.
# NEVER set static num_tensors, tDim, or stream_dims values in generated code.
############################################################################

import numpy as np
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType
from brainsmith.dataflow.core.tensor_chunking import default_chunking


class AutoThresholdingAxi(AutoHWCustomOp):
    """
    Slim auto-generated HWCustomOp for thresholding_axi kernel.
    
    Generated from RTL: thresholding_axi.sv
    Uses enhanced TDIM pragma integration for automatic chunking strategies.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize AutoThresholdingAxi with interface metadata and chunking strategies."""
        
        # Define interface metadata with data layout-based chunking
        # Chunking is determined automatically from ONNX tensor layout (NCHW, NHWC, NLC, etc.)
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
                chunking_strategy=default_chunking()  # Data layout-based chunking determined at runtime
            ),
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
                chunking_strategy=default_chunking()  # Data layout-based chunking determined at runtime
            ),
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
    
    def determine_chunking_from_layout(self, interface_name, tensor_shape, onnx_layout):
        """
        Determine qDim and tDim from ONNX tensor layout.
        
        Based on Interface-Wise Dataflow Modeling specification:
        - [N, C]: qDim=1, tDim=C
        - [N, C, H, W]: qDim=C, tDim=H*W  
        - [N, H, W, C]: qDim=H*W, tDim=C
        - [N, L, C]: qDim=L, tDim=C
        - [N, C, L]: qDim=C, tDim=L
        - [N, L, h, d]: qDim=L, tDim=h*d
        
        For weight interfaces:
        - 1D weights: qDim=1, tDim=length
        - 2D weights: qDim=second_dim, tDim=first_dim
        """
        if onnx_layout == "[N, C]":
            N, C = tensor_shape
            return {"qDim": 1, "tDim": C, "chunk_dimension": None}
        elif onnx_layout == "[N, C, H, W]":
            N, C, H, W = tensor_shape
            return {"qDim": C, "tDim": H * W, "chunk_dimension": 1}
        elif onnx_layout == "[N, H, W, C]":
            N, H, W, C = tensor_shape
            return {"qDim": H * W, "tDim": C, "chunk_dimension": 2}
        elif onnx_layout == "[N, L, C]":
            N, L, C = tensor_shape
            return {"qDim": L, "tDim": C, "chunk_dimension": 1}
        elif onnx_layout == "[N, C, L]":
            N, C, L = tensor_shape
            return {"qDim": C, "tDim": L, "chunk_dimension": 1}
        elif onnx_layout == "[N, L, h, d]":
            N, L, h, d = tensor_shape
            return {"qDim": L, "tDim": h * d, "chunk_dimension": 1}
        else:
            # Default fallback - treat as single chunk
            return {"qDim": 1, "tDim": np.prod(tensor_shape[1:]), "chunk_dimension": None}
    
    def get_kernel_interface_specs(self):
        """Return interface specifications for kernel integration."""
        return {
            "s_axis": {
                "type": "input",
                "interface_type": "AXI_STREAM",
                "chunking_strategy": "data_layout_chunking",  # Determined from ONNX tensor layout
                "layout_detection": "automatic",  # qDim/tDim calculated from [N,C,H,W], [N,L,C], etc.
            },
            "m_axis": {
                "type": "output",
                "interface_type": "AXI_STREAM",
                "chunking_strategy": "data_layout_chunking",  # Determined from ONNX tensor layout
                "layout_detection": "automatic",  # qDim/tDim calculated from [N,C,H,W], [N,L,C], etc.
            },
        }
    
    # ===== Kernel-Specific Resource Estimation =====
    
    def bram_estimation(self) -> int:
        """Estimate BRAM usage for thresholding_axi."""
        # TODO: Implement based on thresholding_axi architecture
        # Minimal usage for compute-only kernel
        return 1
    
    def lut_estimation(self) -> int:
        """Estimate LUT usage for thresholding_axi."""
        # TODO: Implement based on thresholding_axi logic complexity
        parallelism = self._get_current_parallelism()
        base_luts = 3000
        return base_luts * sum(parallelism.iPar.values())
    
    def dsp_estimation(self) -> int:
        """Estimate DSP usage for thresholding_axi."""
        # Non-arithmetic kernel typically doesn't use DSPs
        return 0
    
    def verify_node(self):
        """Verify kernel-specific constraints."""
        super().verify_node()
        
        # Add thresholding_axi-specific verification


# Convenience function for FINN integration
def make_thresholding_axi_node(inputs, outputs, **node_attrs):
    """Create AutoThresholdingAxi ONNX node with enhanced TDIM pragma support."""
    import onnx.helper
    
    return onnx.helper.make_node(
        "AutoThresholdingAxi",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )