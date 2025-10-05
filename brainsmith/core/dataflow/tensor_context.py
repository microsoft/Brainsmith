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
    """Tensor information for a node, preserving ONNX positional order.

    inputs[i] corresponds to node.input[i]
    outputs[i] corresponds to node.output[i]

    Optional/missing inputs are represented as None.
    This preserves the positional correspondence between schema and ONNX node.
    """

    inputs: List[Optional[TensorInfo]]
    outputs: List[Optional[TensorInfo]]

    @staticmethod
    def from_model_wrapper(node, model: ModelWrapper) -> 'TensorContext':
        """Extract tensor context from ModelWrapper, preserving positions.

        Args:
            node: ONNX node
            model: ModelWrapper instance

        Returns:
            TensorContext where inputs[i] corresponds to node.input[i].
            Optional/missing inputs (empty strings in ONNX) are stored as None.
        """
        inputs = []
        for tensor_name in node.input:
            if tensor_name:  # Present input
                inputs.append(TensorInfo(
                    name=tensor_name,
                    shape=tuple(model.get_tensor_shape(tensor_name)),
                    datatype=model.get_tensor_datatype(tensor_name)
                ))
            else:  # Optional input omitted (empty string in ONNX)
                inputs.append(None)

        outputs = []
        for tensor_name in node.output:
            # Outputs are always present (no optional outputs in ONNX)
            outputs.append(TensorInfo(
                name=tensor_name,
                shape=tuple(model.get_tensor_shape(tensor_name)),
                datatype=model.get_tensor_datatype(tensor_name)
            ))

        return TensorContext(inputs=inputs, outputs=outputs)