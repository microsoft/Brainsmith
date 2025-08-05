############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Transform to convert MultiThreshold nodes to AutoHWCustomOp thresholding kernels.

Converts FINN MultiThreshold operations to our auto-generated ThresholdingAxi
AutoHWCustomOp implementations.
"""

from typing import Dict, Any

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from .infer_auto_hw_custom_op import InferAutoHWCustomOp


class InferAutoThresholding(InferAutoHWCustomOp):
    """Convert MultiThreshold nodes to AutoHWCustomOp thresholding kernels."""
    
    def __init__(self, target_domain: str = "rtl"):
        """
        Args:
            target_domain: "hls" or "rtl" for implementation selection.
                          Defaults to "rtl" for thresholding example.
        """
        super().__init__(target_domain)
    
    def matches_node(self, model: ModelWrapper, node: NodeProto) -> bool:
        """Check if node is a MultiThreshold that should be converted."""
        return node.op_type == "MultiThreshold"
    
    def get_auto_hw_custom_op_name(self) -> str:
        """Return the AutoHWCustomOp class name."""
        return "ThresholdingAxi"
    
    def get_domain_base(self) -> str:
        """Return the domain base for auto-generated thresholding kernels."""
        return "brainsmith.kernels.auto_thresholding"
    
    def can_convert_node(self, model: ModelWrapper, node: NodeProto) -> bool:
        """Validate that MultiThreshold node can be converted to AutoHWCustomOp."""
        
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
        
        # Check compatible scale/bias for our AutoHWCustomOp
        try:
            mt_inst = getCustomOp(node)
            scale = mt_inst.get_nodeattr("out_scale")
            bias = mt_inst.get_nodeattr("out_bias")
            
            # Our AutoHWCustomOp expects scale=1.0 and integer bias
            return scale == 1.0 and isinstance(bias, (int, float))
        except Exception:
            return False
    
    def extract_node_attributes(self, model: ModelWrapper, node: NodeProto) -> Dict[str, Any]:
        """Extract minimal attributes needed for ThresholdingAxi AutoHWCustomOp."""
        
        # Extract channels from input tensor (channels-last assumption)
        channels = self.extract_channels_from_shape(model, node.input[0], channels_last=True)
        
        # Extract threshold levels from weight tensor shape
        threshold_shape = model.get_tensor_shape(node.input[1])
        levels = threshold_shape[1] if len(threshold_shape) > 1 else 1
        
        # Get bias from original MultiThreshold node
        mt_inst = getCustomOp(node)
        bias = int(mt_inst.get_nodeattr("out_bias"))
        
        # Return minimal parameters for AutoHWCustomOp - no optimization
        return {
            "CHANNELS": channels,
            "PE": 1,  # Default parallelization, no optimization
            "LEVELS": levels,
            "ActVal": bias
        }