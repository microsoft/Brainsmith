############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Transform to convert MultiThreshold nodes to ThresholdingAxi AutoHWCustomOp.

Matches the behavior of InferThresholdingLayer but targets the auto-generated
ThresholdingAxi RTL implementation.
"""

from typing import Dict, Any
import numpy as np

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.datatype import DataType

from .infer_auto_hw_custom_op import InferAutoHWCustomOp


class InferThresholdingAxi(InferAutoHWCustomOp):
    """Convert MultiThreshold nodes to ThresholdingAxi AutoHWCustomOp."""
    
    def __init__(self, target_domain: str = "rtl"):
        """
        Initialize transformation for ThresholdingAxi.
        
        Args:
            target_domain: Always "rtl" for ThresholdingAxi
        """
        super().__init__("rtl")  # ThresholdingAxi is RTL-only
    
    def matches_node(self, model: ModelWrapper, node: NodeProto) -> bool:
        """Check if node is a MultiThreshold that should be converted."""
        return node.op_type == "MultiThreshold"
    
    def get_auto_hw_custom_op_name(self) -> str:
        """Return the AutoHWCustomOp class name."""
        return "ThresholdingAxi"
    
    def get_domain_base(self) -> str:
        """Return the domain base for thresholding kernels."""
        return "brainsmith.kernels.thresholding"
    
    def get_target_domain(self) -> str:
        """Override to return base domain without specialization suffix."""
        # Return just the base domain to allow SpecializeLayers to work
        return self.get_domain_base()
    
    def can_convert_node(self, model: ModelWrapper, node: NodeProto) -> bool:
        """Validate that MultiThreshold node can be converted to ThresholdingAxi."""
        
        # Check integer datatypes
        if not self.validate_integer_datatypes(model, node):
            return False
        
        # Must have threshold tensor as second input
        if len(node.input) < 2:
            return False
            
        # Threshold tensor must have initializer (weights)
        threshold_info = self.get_initializer_info(model, node.input[1])
        if not threshold_info["has_initializer"]:
            return False
        
        # Check that threshold tensor has valid shape
        threshold_shape = threshold_info["shape"]
        if not threshold_shape or len(threshold_shape) < 2:
            return False
        
        # Check scale and bias compatibility
        try:
            mt_inst = getCustomOp(node)
            scale = mt_inst.get_nodeattr("out_scale")
            bias = mt_inst.get_nodeattr("out_bias")
            
            # ThresholdingAxi expects scale=1.0 and integer bias
            if scale != 1.0:
                return False
            if not isinstance(bias, (int, float)) or int(bias) != bias:
                return False
                
            # Check output datatype compatibility
            odt = model.get_tensor_datatype(node.output[0])
            # For signed outputs, bias must be negative (except BIPOLAR)
            if odt != DataType["BIPOLAR"] and odt.signed() and int(bias) >= 0:
                return False
                
        except Exception:
            return False
            
        return True
    
    def extract_node_attributes(self, model: ModelWrapper, node: NodeProto) -> Dict[str, Any]:
        """Extract attributes needed for ThresholdingAxi AutoHWCustomOp."""
        
        # Extract threshold tensor information
        threshold_shape = model.get_tensor_shape(node.input[1])
        threshold_init = model.get_initializer(node.input[1])
        num_steps = threshold_shape[1] if len(threshold_shape) > 1 else 1
        
        # Extract channels from threshold shape to ensure consistency
        # This prevents mismatch between NumChannels attribute and actual threshold data
        channels = threshold_shape[0] if threshold_shape else 1
        
        # Validate input shape compatibility
        input_shape = model.get_tensor_shape(node.input[0])
        if input_shape:
            input_channels = self.extract_channels_from_shape(model, node.input[0], channels_last=True)
            if input_channels != channels:
                import warnings
                warnings.warn(
                    f"Input channels ({input_channels}) != threshold channels ({channels}). "
                    f"Using threshold channels ({channels}) for CHANNELS attribute to prevent index errors."
                )
        
        # Get bias from original MultiThreshold node
        mt_inst = getCustomOp(node)
        bias = int(mt_inst.get_nodeattr("out_bias"))
        
        # Get threshold datatype
        threshold_dt = model.get_tensor_datatype(node.input[1])
        
        # Calculate RTL-specific parameters
        # input_FPARG: fractional bits for input (0 for integer types)
        input_fparg = 0
        
        # THRESHOLDS_PATH: empty for now (will be filled during codegen)
        thresholds_path = ""
        
        # Memory depth triggers - use reasonable defaults
        # These control when to use URAM vs BRAM
        depth_trigger_uram = 1024  # Use URAM for depth > 1024
        depth_trigger_bram = 256   # Use BRAM for depth > 256
        
        # Deep pipeline for better timing (1 = enabled)
        deep_pipeline = 1
        
        # Config interface width (threshold datatype width)
        config_width = threshold_dt.bitwidth()
        
        # Build attributes dictionary
        attrs = {
            # Shape parameters
            "CHANNELS": channels,
            "PE": 1,  # Default parallelization
            
            # RTL algorithm parameters
            "input_FPARG": input_fparg,
            "BIAS": bias,
            "THRESHOLDS_PATH": thresholds_path,
            "DEPTH_TRIGGER_URAM": depth_trigger_uram,
            "DEPTH_TRIGGER_BRAM": depth_trigger_bram,
            "DEEP_PIPELINE": deep_pipeline,
            
            # Config interface parameters
            "width": config_width,
            "USE_AXILITE": 1,  # Enable AXI-Lite config interface
            
            # Threshold datatype (for config interface)
            "thresholdDataType": threshold_dt.name,
            
            # Additional metadata
            "numSteps": num_steps,
            "ActVal": bias,  # Compatibility with base class
            "NumChannels": channels,  # Explicitly set for FINN compatibility
        }
        
        return attrs
    
    def create_basic_finn_attributes(self, model: ModelWrapper, node: NodeProto) -> Dict[str, Any]:
        """Override to include threshold datatype."""
        attrs = super().create_basic_finn_attributes(model, node)
        
        # Add threshold datatype if we have a threshold input
        if len(node.input) > 1:
            attrs["thresholdDataType"] = model.get_tensor_datatype(node.input[1]).name
            
        return attrs