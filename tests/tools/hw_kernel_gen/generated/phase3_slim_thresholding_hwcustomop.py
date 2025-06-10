############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for phase3_slim_thresholding
# Generated from: unknown.sv
# Generation timestamp: 2025-06-10T02:34:42.983996
#
# RUNTIME-CONFIGURABLE HARDWARE COMPONENT
# This HWCustomOp uses runtime dimension extraction from ModelWrapper.
# NEVER set static num_tensors, tDim, or stream_dims values in generated code.
############################################################################

from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType
from brainsmith.dataflow.core.tensor_chunking import index_chunking, default_chunking, last_dim_chunking


class Phase3SlimThresholdingHWCustomOp(AutoHWCustomOp):
    """
    Slim auto-generated HWCustomOp for phase3_slim_thresholding kernel.
    
    Generated from RTL: unknown.sv
    Uses enhanced TDIM pragma integration for automatic chunking strategies.
    """
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize Phase3SlimThresholdingHWCustomOp with interface metadata and chunking strategies."""
        
        # Define interface metadata with enhanced TDIM pragma integration
        # Only include AXI_STREAM interfaces in dataflow model (AXI_LITE will be handled separately)
        self._interface_metadata = [
            InterfaceMetadata(
                name="s_axis_tdata",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        finn_type="UINT8",
                        bit_width=8,
                        signed=false
                    ),
                ],
                chunking_strategy=index_chunking(-1, "['PE']")            ),
            InterfaceMetadata(
                name="m_axis_tdata",
                interface_type=DataflowInterfaceType.OUTPUT,
                allowed_datatypes=[
                    DataTypeConstraint(
                        finn_type="UINT8",
                        bit_width=8,
                        signed=false
                    ),
                ],
                chunking_strategy=index_chunking(-1, "['PE']")            ),
        ]
        
        # Initialize parent with interface metadata
        super().__init__(onnx_node, interface_metadata=self._interface_metadata, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "phase3_slim_thresholding"
        self.rtl_source = "unknown.sv"
    
    def get_nodeattr_types(self):
        """Define kernel-specific node attributes."""
        attrs = super().get_nodeattr_types()
        
        # Add RTL parameters as node attributes
        kernel_attrs = {
            "PE": ("i", False, 4),
            "SIMD": ("i", False, 8),
            "THRESHOLD_PARAMS": ("i", False, 32),
        }
        
        attrs.update(kernel_attrs)
        return attrs
    
    def get_kernel_interface_specs(self):
        """Return interface specifications for kernel integration."""
        return {
            "s_axis_tdata": {
                "type": "input",
                "interface_type": "AXI_STREAM",
                "chunking_strategy": "index_chunking",
                "chunk_index": -1,
                "chunk_sizes": "['PE']",  # Parameterized - will be resolved at runtime
            },
            "m_axis_tdata": {
                "type": "output",
                "interface_type": "AXI_STREAM",
                "chunking_strategy": "index_chunking",
                "chunk_index": -1,
                "chunk_sizes": "['PE']",  # Parameterized - will be resolved at runtime
            },
        }
    
    # ===== Kernel-Specific Resource Estimation =====
    
    def bram_estimation(self) -> int:
        """Estimate BRAM usage for phase3_slim_thresholding."""
        # TODO: Implement based on phase3_slim_thresholding architecture
        # Minimal usage for compute-only kernel
        return 1
    
    def lut_estimation(self) -> int:
        """Estimate LUT usage for phase3_slim_thresholding."""
        # TODO: Implement based on phase3_slim_thresholding logic complexity
        parallelism = self._get_current_parallelism()
        base_luts = 1500
        return base_luts * sum(parallelism.iPar.values())
    
    def dsp_estimation(self) -> int:
        """Estimate DSP usage for phase3_slim_thresholding."""
        # Non-arithmetic kernel typically doesn't use DSPs
        return 0
    
    def verify_node(self):
        """Verify kernel-specific constraints."""
        super().verify_node()
        
        # Add phase3_slim_thresholding-specific verification


# Convenience function for FINN integration
def make_phase3_slim_thresholding_node(inputs, outputs, **node_attrs):
    """Create Phase3SlimThresholdingHWCustomOp ONNX node with enhanced TDIM pragma support."""
    import onnx.helper
    
    return onnx.helper.make_node(
        "Phase3SlimThresholdingHWCustomOp",
        inputs=inputs,
        outputs=outputs,
        domain="finn.custom_op.fpgadataflow",
        **node_attrs
    )