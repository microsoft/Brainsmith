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
from typing import List, Dict, Any
from onnx import NodeProto


@dataclass(frozen=True)
class TransformationResult:
    """Result of ONNX → HW transformation.

    Simple data container returned by KernelOp.infer_from() methods.
    Contains nodes to insert/remove and layout tracking.

    Validation is handled by schema constraints during build_design_space().
    """

    nodes_to_insert: List[NodeProto]
    """HW nodes to insert into graph."""

    nodes_to_remove: List[NodeProto]
    """ONNX nodes to remove from graph."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional metadata about transformation.

    Useful for debugging, logging, or tracking transformation decisions.

    Example: {
        "schema_name": "LayerNorm",
        "layout_conversions": ["input"],
    }
    """
