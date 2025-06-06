############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for thresholding_axi
# Generated from: /tmp/tmp646p9xsl/thresholding_enhanced.sv
# Generation timestamp: 2025-06-06T02:05:27.610875
############################################################################

import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional
import math

# Import AutoHWCustomOp base class and dataflow components
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.dataflow_interface import (
    DataflowInterface, DataflowInterfaceType, DataflowDataType, DataTypeConstraint
)
from brainsmith.dataflow.core.dataflow_model import DataflowModel
from brainsmith.dataflow.core.validation import ConstraintValidator

# FINN imports (for compatibility)
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType


class AutoThresholdingAxi(AutoHWCustomOp):
    """
    Auto-generated HWCustomOp for thresholding_axi kernel.
    
    This class inherits from AutoHWCustomOp which provides standardized
    implementations for all common HWCustomOp methods including:
    - Datatype handling (get_input_datatype, get_output_datatype)
    - Shape inference (get_normal_*_shape, get_folded_*_shape)
    - Stream width calculations (get_instream_width, get_outstream_width)
    - Cycle calculations (get_exp_cycles)
    - Parallelism optimization
    
    Only kernel-specific resource estimation methods need to be implemented.
    
    Generated from RTL: /tmp/tmp646p9xsl/thresholding_enhanced.sv
    
    Interfaces:
    - ap: control (UINT32)
    - s_axis: input (UINT8)
    - m_axis: output (UINT1)
    - s_axilite: config (UINT32)
    """
    
    def __init__(self, onnx_node, **kwargs):
        """
        Initialize AutoThresholdingAxi with dataflow model.
        
        Args:
            onnx_node: ONNX node to wrap
            **kwargs: Additional arguments passed to parent
        """
        # Build dataflow interfaces from generated specifications
        dataflow_interfaces = self._build_dataflow_interfaces()
        
        # Create dataflow model with unified computational framework
        dataflow_model = DataflowModel(
            dataflow_interfaces, 
            self._get_kernel_parameters()
        )
        
        # Initialize parent with dataflow components
        super().__init__(onnx_node, dataflow_interfaces, dataflow_model, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "thresholding_axi"
        self.rtl_source = "/tmp/tmp646p9xsl/thresholding_enhanced.sv"
        
    def _build_dataflow_interfaces(self) -> List[DataflowInterface]:
        """Build dataflow interfaces from template specifications."""
        interfaces = []
        
        # ap interface
        interfaces.append(DataflowInterface(
            name="ap",
            interface_type=DataflowInterfaceType.CONTROL,
            qDim=[1],
            tDim=[1],
            sDim=[1],
            dtype=DataflowDataType(
                base_type="UINT",
                bitwidth=32,
                signed=false,
                finn_type="UINT32"
            ),
            allowed_datatypes=[
                DataTypeConstraint(
                    base_types=['UINT'],
                    min_bitwidth=8,
                    max_bitwidth=32,
                    signed_allowed=false,
                    unsigned_allowed=true
                ),
            ],
            axi_metadata={},
            constraints=[],  # Populated by validation framework
            pragma_metadata={}
        ))
        # s_axis interface
        interfaces.append(DataflowInterface(
            name="s_axis",
            interface_type=DataflowInterfaceType.INPUT,
            qDim=[1, 1],
            tDim=[1, 32],
            sDim=[1, 1],
            dtype=DataflowDataType(
                base_type="UINT",
                bitwidth=8,
                signed=false,
                finn_type="UINT8"
            ),
            allowed_datatypes=[
                DataTypeConstraint(
                    base_types=['UINT'],
                    min_bitwidth=8,
                    max_bitwidth=8,
                    signed_allowed=true,
                    unsigned_allowed=true
                ),
            ],
            axi_metadata={},
            constraints=[],  # Populated by validation framework
            pragma_metadata={'datatype_pragma_applied': True}
        ))
        # m_axis interface
        interfaces.append(DataflowInterface(
            name="m_axis",
            interface_type=DataflowInterfaceType.OUTPUT,
            qDim=[1, 1],
            tDim=[1, 32],
            sDim=[1, 1],
            dtype=DataflowDataType(
                base_type="UINT",
                bitwidth=1,
                signed=false,
                finn_type="UINT1"
            ),
            allowed_datatypes=[
                DataTypeConstraint(
                    base_types=['UINT'],
                    min_bitwidth=1,
                    max_bitwidth=1,
                    signed_allowed=true,
                    unsigned_allowed=true
                ),
            ],
            axi_metadata={},
            constraints=[],  # Populated by validation framework
            pragma_metadata={'datatype_pragma_applied': True}
        ))
        # s_axilite interface
        interfaces.append(DataflowInterface(
            name="s_axilite",
            interface_type=DataflowInterfaceType.CONFIG,
            qDim=[1],
            tDim=[32],
            sDim=[1],
            dtype=DataflowDataType(
                base_type="UINT",
                bitwidth=32,
                signed=false,
                finn_type="UINT32"
            ),
            allowed_datatypes=[
                DataTypeConstraint(
                    base_types=['UINT'],
                    min_bitwidth=8,
                    max_bitwidth=32,
                    signed_allowed=false,
                    unsigned_allowed=true
                ),
            ],
            axi_metadata={},
            constraints=[],  # Populated by validation framework
            pragma_metadata={}
        ))
        
        return interfaces
    
    def _get_kernel_parameters(self) -> Dict[str, Any]:
        """Get kernel-specific parameters."""
        return {
            "N": 1,
            "WI": 8,
            "WT": 8,
            "C": 32,
            "PE": 1,
            "SIGNED": 1,
            "FPARG": 0,
            "BIAS": 0,
            "THRESHOLDS_PATH": "",
            "USE_AXILITE": 1,
            "DEPTH_TRIGGER_URAM": 0,
            "DEPTH_TRIGGER_BRAM": 0,
            "DEEP_PIPELINE": 0,
        }
    
    def get_nodeattr_types(self):
        """
        Define node attributes including kernel-specific parameters.
        
        Most attributes are handled by AutoHWCustomOp. This method
        adds any kernel-specific attributes.
        """
        # Get base attributes from parent
        attrs = super().get_nodeattr_types()
        
        # Add kernel-specific attributes
        kernel_attrs = {
              "N": ("i", False, 1),
              "WI": ("i", False, 8),
              "WT": ("i", False, 8),
              "C": ("i", False, 32),
              "SIGNED": ("i", False, 1),
              "FPARG": ("i", False, 0),
              "BIAS": ("i", False, 0),
              "THRESHOLDS_PATH": ("i", False, ""),
              "USE_AXILITE": ("i", False, 1),
              "DEPTH_TRIGGER_URAM": ("i", False, 0),
              "DEPTH_TRIGGER_BRAM": ("i", False, 0),
              "DEEP_PIPELINE": ("i", False, 0),
        }
        
        attrs.update(kernel_attrs)
        return attrs
    
    # ===== Resource Estimation Methods (Kernel-Specific) =====
    
    def bram_estimation(self) -> int:
        """
        Estimate BRAM usage for thresholding_axi.
        
        This method must be implemented based on the specific memory
        requirements of the thresholding_axi kernel architecture.
        
        Helper methods available from base class:
        - self._get_weight_memory_summary(): Weight storage requirements
        - self._get_activation_buffer_summary(): Activation buffering needs
        - self._get_current_parallelism(): Current parallelism configuration
        """
        # Get memory summaries from base class
        weight_summary = self._get_weight_memory_summary()
        activation_summary = self._get_activation_buffer_summary()
        parallelism = self._get_current_parallelism()
        
        # TODO: Implement based on thresholding_axi architecture
        # Example implementation:
        # No weight interfaces - minimal BRAM usage
        return 1  # Minimum for control/buffering
    
    def lut_estimation(self) -> int:
        """
        Estimate LUT usage for thresholding_axi.
        
        Must be implemented based on the specific logic requirements.
        """
        parallelism = self._get_current_parallelism()
        
        # TODO: Implement based on thresholding_axi architecture
        # Placeholder estimation
        base_luts = 1000  # Base control logic
        luts_per_s_axis_parallel = 50
        base_luts += parallelism.iPar.get("s_axis", 1) * luts_per_s_axis_parallel
        
        return base_luts
    
    def dsp_estimation(self) -> int:
        """
        Estimate DSP usage for thresholding_axi.
        
        Must be implemented based on arithmetic operations required.
        """
        parallelism = self._get_current_parallelism()
        
        # TODO: Implement based on thresholding_axi architecture
        # Placeholder estimation
        # May not use DSPs
        return 0
    
    def uram_estimation(self) -> int:
        """
        Estimate UltraRAM usage for thresholding_axi.
        
        Override if kernel uses UltraRAM for large storage.
        """
        # Most kernels don't use URAM
        # Override this method if thresholding_axi does
        return 0
    
    # ===== Optional Overrides =====
    
    def verify_node(self):
        """
        Verify node configuration with kernel-specific checks.
        
        Base class handles standard dataflow validation.
        Override to add kernel-specific verification.
        """
        # Call parent verification first
        super().verify_node()
        
        # Add kernel-specific verification
        # Example: Verify thresholding-specific constraints
        c = self.get_nodeattr("C")
        pe = self.get_nodeattr("PE") 
        if c % pe != 0:
            raise ValueError(f"C ({c}) must be divisible by PE ({pe})")
        
        # Add any other kernel-specific checks here
    
    def execute_node(self, context, graph):
        """
        Execute node for simulation.
        
        Base class provides standard execution based on dataflow model.
        Override only if kernel needs custom execution behavior.
        """
        # For most kernels, base class execution is sufficient
        return super().execute_node(context, graph)
    
    def generate_params(self, model, path):
        """
        Generate parameters for RTL instantiation.
        
        Base class handles standard parameter generation.
        Override to customize parameter formatting.
        """
        # Get base parameters
        params = super().generate_params(model, path)
        
        # Add any kernel-specific parameter processing
        # Ensure all RTL parameters are included
        if "N" not in params:
            params["N"] = self.get_nodeattr("N")
        if "WI" not in params:
            params["WI"] = self.get_nodeattr("WI")
        if "WT" not in params:
            params["WT"] = self.get_nodeattr("WT")
        if "C" not in params:
            params["C"] = self.get_nodeattr("C")
        if "PE" not in params:
            params["PE"] = self.get_nodeattr("PE")
        if "SIGNED" not in params:
            params["SIGNED"] = self.get_nodeattr("SIGNED")
        if "FPARG" not in params:
            params["FPARG"] = self.get_nodeattr("FPARG")
        if "BIAS" not in params:
            params["BIAS"] = self.get_nodeattr("BIAS")
        if "THRESHOLDS_PATH" not in params:
            params["THRESHOLDS_PATH"] = self.get_nodeattr("THRESHOLDS_PATH")
        if "USE_AXILITE" not in params:
            params["USE_AXILITE"] = self.get_nodeattr("USE_AXILITE")
        if "DEPTH_TRIGGER_URAM" not in params:
            params["DEPTH_TRIGGER_URAM"] = self.get_nodeattr("DEPTH_TRIGGER_URAM")
        if "DEPTH_TRIGGER_BRAM" not in params:
            params["DEPTH_TRIGGER_BRAM"] = self.get_nodeattr("DEPTH_TRIGGER_BRAM")
        if "DEEP_PIPELINE" not in params:
            params["DEEP_PIPELINE"] = self.get_nodeattr("DEEP_PIPELINE")
        
        return params


# Optional: Create convenience function for FINN integration
def make_thresholding_axi_customop(W, pe=1, simd=1, **kwargs):
    """
    Convenience function to create AutoThresholdingAxi node.
    
    This follows FINN conventions for creating custom operations.
    """
    # This would create the ONNX node and wrap it
    # Implementation depends on FINN's current API
    pass