############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Tensor context for capturing ONNX graph information.

This module provides lightweight structures to capture just the tensor
information needed from ModelWrapper, avoiding the need to pass around
the entire ModelWrapper object.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper


@dataclass(frozen=True)
class TensorInfo:
    """Information about a single tensor.
    
    Captures the minimal tensor information needed for model creation
    without holding references to the full ONNX graph.
    """
    
    name: str
    shape: Tuple[int, ...]
    datatype: DataType


@dataclass(frozen=True) 
class TensorContext:
    """Minimal tensor information needed from ModelWrapper.
    
    This context captures all the tensor-specific information needed
    to complete Phase 2 of model creation, without requiring the full
    ModelWrapper to be passed around.
    """
    
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    
    @staticmethod
    def from_model_wrapper(node, model: ModelWrapper) -> 'TensorContext':
        """Extract context from ModelWrapper.
        
        Args:
            node: ONNX node
            model: ModelWrapper instance
            
        Returns:
            TensorContext with all tensor information
        """
        inputs = []
        for tensor_name in node.input:
            inputs.append(TensorInfo(
                name=tensor_name,
                shape=tuple(model.get_tensor_shape(tensor_name)),
                datatype=model.get_tensor_datatype(tensor_name)
            ))
        
        outputs = []
        for tensor_name in node.output:
            outputs.append(TensorInfo(
                name=tensor_name,
                shape=tuple(model.get_tensor_shape(tensor_name)),
                datatype=model.get_tensor_datatype(tensor_name)
            ))
        
        return TensorContext(inputs=inputs, outputs=outputs)
    
    def get_input_info(self, position: int) -> Optional[TensorInfo]:
        """Get input tensor info by position."""
        if 0 <= position < len(self.inputs):
            return self.inputs[position]
        return None
    
    def get_output_info(self, position: int) -> Optional[TensorInfo]:
        """Get output tensor info by position."""
        if 0 <= position < len(self.outputs):
            return self.outputs[position]
        return None