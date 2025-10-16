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
class TensorContext:
    """Tensor shape information for a node, preserving ONNX positional order.

    Stores only tensor shapes. Datatypes are stored in nodeattrs following
    the schema-driven design.

    input_shapes[i] corresponds to node.input[i]
    output_shapes[i] corresponds to node.output[i]

    Optional inputs (empty strings in ONNX) are represented as None.
    """

    input_shapes: Tuple[Optional[Tuple[int, ...]], ...]
    output_shapes: Tuple[Tuple[int, ...], ...]  # Outputs are never optional in ONNX

    @staticmethod
    def from_model_wrapper(node, model: ModelWrapper) -> 'TensorContext':
        """Extract tensor shapes from ModelWrapper, preserving positions.

        Datatypes are NOT extracted - they are stored in nodeattrs following
        the schema-driven design.

        Args:
            node: ONNX node
            model: ModelWrapper instance

        Returns:
            TensorContext where input_shapes[i] corresponds to node.input[i].
            Optional/missing inputs (empty strings in ONNX) are stored as None.
        """
        input_shapes = []
        for tensor_name in node.input:
            if tensor_name:  # Present input
                input_shapes.append(tuple(model.get_tensor_shape(tensor_name)))
            else:  # Optional input omitted (empty string in ONNX)
                input_shapes.append(None)

        output_shapes = []
        for tensor_name in node.output:
            # Outputs are always present (no optional outputs in ONNX)
            output_shapes.append(tuple(model.get_tensor_shape(tensor_name)))

        return TensorContext(
            input_shapes=tuple(input_shapes),
            output_shapes=tuple(output_shapes)
        )

    def to_json(self) -> str:
        """Serialize TensorContext to JSON string for ONNX metadata storage.

        Only stores shapes. Datatypes are stored in nodeattrs.

        Returns:
            JSON string representation suitable for node.metadata_props
        """
        data = {
            'input_shapes': [
                list(shape) if shape is not None else None
                for shape in self.input_shapes
            ],
            'output_shapes': [
                list(shape) for shape in self.output_shapes
            ],
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'TensorContext':
        """Deserialize TensorContext from JSON string.

        Only restores shapes. Datatypes are read from nodeattrs.

        Args:
            json_str: JSON string from node.metadata_props

        Returns:
            TensorContext instance

        Raises:
            ValueError: If JSON is malformed or contains invalid data
        """
        data = json.loads(json_str)

        input_shapes = tuple(
            tuple(shape) if shape is not None else None
            for shape in data['input_shapes']
        )

        output_shapes = tuple(
            tuple(shape) for shape in data['output_shapes']
        )

        return cls(input_shapes=input_shapes, output_shapes=output_shapes)

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