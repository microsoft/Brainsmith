############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Pure transformation system for ONNX → HW kernel conversion.

This module provides:
- TransformationResult: Result container for infer_from() methods
"""

from dataclasses import dataclass, field
from typing import Any

from onnx import NodeProto


@dataclass(frozen=True)
class TransformationResult:
    """Result of ONNX node to hardware kernel transformation.

    Attributes:
        nodes_to_insert: HW nodes to insert into graph
        nodes_to_remove: ONNX nodes to remove from graph
        metadata: Optional transformation metadata
    """

    nodes_to_insert: list[NodeProto]
    nodes_to_remove: list[NodeProto]
    metadata: dict[str, Any] = field(default_factory=dict)
