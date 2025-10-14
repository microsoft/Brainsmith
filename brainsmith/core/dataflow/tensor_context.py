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

TensorContext can be serialized to/from ONNX node metadata_props for
persistence across save/load cycles.
"""

import json
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

    Optional inputs (empty strings in ONNX) are represented as None.
    """

    inputs: List[Optional[TensorInfo]]
    outputs: List[TensorInfo]  # Outputs are never optional in ONNX

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

    def to_json(self) -> str:
        """Serialize TensorContext to JSON string for ONNX metadata storage.

        Returns:
            JSON string representation suitable for node.metadata_props
        """
        data = {
            'inputs': [
                {
                    'name': inp.name,
                    'shape': list(inp.shape),
                    'datatype': inp.datatype.name,
                }
                for inp in self.inputs if inp is not None
            ],
            'outputs': [
                {
                    'name': out.name,
                    'shape': list(out.shape),
                    'datatype': out.datatype.name,
                }
                for out in self.outputs
            ],
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'TensorContext':
        """Deserialize TensorContext from JSON string.

        Args:
            json_str: JSON string from node.metadata_props

        Returns:
            TensorContext instance

        Raises:
            ValueError: If JSON is malformed or contains invalid data
        """
        data = json.loads(json_str)

        inputs = []
        for inp_data in data['inputs']:
            inputs.append(TensorInfo(
                name=inp_data['name'],
                shape=tuple(inp_data['shape']),
                datatype=DataType[inp_data['datatype']],
            ))

        outputs = []
        for out_data in data['outputs']:
            outputs.append(TensorInfo(
                name=out_data['name'],
                shape=tuple(out_data['shape']),
                datatype=DataType[out_data['datatype']],
            ))

        return cls(inputs=inputs, outputs=outputs)

    @classmethod
    def from_node_metadata(cls, node) -> Optional['TensorContext']:
        """Read TensorContext from node.metadata_props if present.

        Args:
            node: ONNX NodeProto

        Returns:
            TensorContext if found in metadata, None otherwise
        """
        from qonnx.util.basic import get_by_name

        metadata = get_by_name(
            node.metadata_props,
            'ai.brainsmith.tensor_context',
            'key'
        )
        if metadata:
            return cls.from_json(metadata.value)
        return None

    def attach_to_node(self, node) -> None:
        """Attach this TensorContext to node.metadata_props.

        This persists the tensor context so it survives save/load cycles.
        Uses the key 'ai.brainsmith.tensor_context' following ONNX conventions
        for namespaced metadata.

        Args:
            node: ONNX NodeProto to attach metadata to
        """
        from onnx import StringStringEntryProto
        from qonnx.util.basic import get_by_name

        # Remove existing tensor context if present
        existing = get_by_name(
            node.metadata_props,
            'ai.brainsmith.tensor_context',
            'key'
        )
        if existing:
            node.metadata_props.remove(existing)

        # Add new metadata
        metadata = StringStringEntryProto()
        metadata.key = 'ai.brainsmith.tensor_context'
        metadata.value = self.to_json()
        node.metadata_props.append(metadata)