############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Auto-generated HWCustomOp for thresholding_axi
# Generated from: /tmp/tmp3962npoo/thresholding_enhanced.sv
# Generation timestamp: 2025-06-06T02:33:26.678819
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
    
    Generated from RTL: /tmp/tmp3962npoo/thresholding_enhanced.sv
    
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
        # Initialize parent - AutoHWCustomOp will handle dataflow integration
        # Runtime values (tDim, sDim, dtype) will be set by FINN via onnx.helper.make_node
        super().__init__(onnx_node, **kwargs)
        
        # Set kernel-specific attributes
        self.kernel_name = "thresholding_axi"
        self.rtl_source = "/tmp/tmp3962npoo/thresholding_enhanced.sv"
        
    def get_kernel_interface_specs(self) -> List[Dict[str, Any]]:
        """
        Define kernel-level interface specifications for this hardware kernel.
        
        These specifications define the interface types and constraints, but do not
        include runtime values like tDim, sDim, or dtype - those are set by FINN
        when creating the ONNX node instance based on the actual model data.
        
        Returns:
            List of interface specifications for thresholding_axi
        """
        return [
            {
                "name": "ap",
                "interface_type": "CONTROL",
                "direction": "control",
                "allowed_datatypes": [
                    {
                        "base_types": ['UINT'],
                        "min_bitwidth": 8,
                        "max_bitwidth": 32,
                        "signed_allowed": False,
                        "unsigned_allowed": True
                    },
                ],
                "pragma_metadata": {},
                "axi_protocol": "global_control"
            },
            {
                "name": "s_axis",
                "interface_type": "INPUT",
                "direction": "input",
                "allowed_datatypes": [
                    {
                        "base_types": ['UINT'],
                        "min_bitwidth": 8,
                        "max_bitwidth": 8,
                        "signed_allowed": True,
                        "unsigned_allowed": True
                    },
                ],
                "pragma_metadata": {'datatype_pragma_applied': True},
                "axi_protocol": "axi_stream"
            },
            {
                "name": "m_axis",
                "interface_type": "OUTPUT",
                "direction": "output",
                "allowed_datatypes": [
                    {
                        "base_types": ['UINT'],
                        "min_bitwidth": 1,
                        "max_bitwidth": 1,
                        "signed_allowed": True,
                        "unsigned_allowed": True
                    },
                ],
                "pragma_metadata": {'datatype_pragma_applied': True},
                "axi_protocol": "axi_stream"
            },
            {
                "name": "s_axilite",
                "interface_type": "CONFIG",
                "direction": "control",
                "allowed_datatypes": [
                    {
                        "base_types": ['UINT'],
                        "min_bitwidth": 8,
                        "max_bitwidth": 32,
                        "signed_allowed": False,
                        "unsigned_allowed": True
                    },
                ],
                "pragma_metadata": {},
                "axi_protocol": "axi_lite"
            },
        ]
    
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