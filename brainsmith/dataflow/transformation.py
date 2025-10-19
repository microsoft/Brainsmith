############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Pure transformation system for ONNX → HW kernel conversion.

This module provides:
- TransformationResult: Result container (NO verification - schema handles that)
- transform_onnx_to_kernel(): Pure transformation function
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from onnx import NodeProto
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformationResult:
    """Result of ONNX → HW transformation.

    Simple data container for transformation results. No validation logic -
    validation is handled by schema.can_transform() before transformation.
    """

    nodes_to_insert: List[NodeProto]
    """HW nodes to insert into graph."""

    nodes_to_remove: List[NodeProto]
    """ONNX nodes to remove from graph."""

    actual_layouts: Dict[str, str]
    """Actual layouts produced (interface_name -> layout).

    Example: {"input": "NHWC", "output": "NHWC"}
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional metadata about transformation.

    Useful for debugging, logging, or tracking transformation decisions.

    Example: {
        "schema_name": "LayerNorm",
        "layout_conversions": ["input"],
    }
    """


# =============================================================================
# Pure Transformation Function (Perfect System)
# =============================================================================


def transform_onnx_to_kernel(
    schema,
    node: NodeProto,
    model,
    insert_index: int,
    kernel_class_name: str
) -> TransformationResult:
    """Transform ONNX node to kernel using schema specification.

    Pure function: (schema, node, model, ...) -> result

    This is THE default transformation algorithm. It operates on a complete
    schema that includes both structure and transformation requirements.

    For new-style schemas (using InputInterface/OutputInterface), layout
    requirements are embedded in interfaces. For old-style schemas (using
    InputSchema/OutputSchema), layout requirements come from separate
    TransformationSpec (transitional support).

    Args:
        schema: Complete kernel schema with transformation requirements (KernelSchema)
        node: ONNX NodeProto to transform
        model: ModelWrapper for graph access
        insert_index: Where to insert new nodes
        kernel_class_name: Name for created HW node (usually kernel class name)

    Returns:
        TransformationResult with created nodes and layouts

    Raises:
        ValueError: If transformation cannot be performed
    """
    from .inference import InferenceHelper
    from .schemas import InputSchema, OutputSchema

    helper = InferenceHelper(model, domain=schema.domain)

    # ============= PHASE 1: TRANSFORM INPUTS =============
    converted_inputs = []
    actual_layouts = {}

    for interface, tensor_name in zip(schema.inputs, node.input):
        # Check if interface has embedded layout requirement
        if isinstance(interface, InputSchema) and interface.required_layout:
            # New style: layout embedded in interface
            converted = helper.ensure_layout(
                tensor_name,
                interface.required_layout,
                insert_index
            )
            converted_inputs.append(converted)
            actual_layouts[interface.name] = interface.required_layout
        else:
            # No layout requirement, use as-is
            converted_inputs.append(tensor_name)
            current_layout = helper.get_layout(tensor_name)
            actual_layouts[interface.name] = current_layout or "UNKNOWN"

    # ============= PHASE 2: BUILD ATTRIBUTES =============
    attributes = dict(schema.initial_parallelization)

    # Map ONNX attributes to kernel parameters
    for onnx_attr, kernel_param in schema.attribute_mapping.items():
        try:
            value = helper.get_node_attr_value(node, onnx_attr)
            attributes[kernel_param] = value
        except (AttributeError, StopIteration):
            # Attribute not found - check if it has a default
            if kernel_param in schema.kernel_params:
                param_type, required, default = schema.kernel_params[kernel_param]
                if required:
                    logger.warning(
                        f"Required ONNX attribute '{onnx_attr}' not found on node '{node.name}'"
                    )
                # Use default if not required
                if not required:
                    attributes[kernel_param] = default

    # ============= PHASE 3: CREATE HW NODE =============
    hw_node = helper.make_node(
        kernel_class_name,
        inputs=converted_inputs,
        outputs=list(node.output),
        attributes=attributes,
        name_prefix=f"{kernel_class_name}_{node.name}"
    )

    # ============= PHASE 4: TRACK OUTPUT LAYOUTS =============
    for interface, tensor_name in zip(schema.outputs, node.output):
        # Check if interface has embedded layout requirement
        if isinstance(interface, OutputSchema) and interface.required_layout:
            # Explicit output layout
            actual_layouts[interface.name] = interface.required_layout
        elif isinstance(interface, OutputSchema) and interface.preserves_input_layout:
            # Preserve first input's layout (common case)
            if len(schema.inputs) > 0:
                first_input_layout = actual_layouts[schema.inputs[0].name]
                actual_layouts[interface.name] = first_input_layout
            else:
                actual_layouts[interface.name] = "UNKNOWN"
        else:
            # Query current layout
            current_layout = helper.get_layout(tensor_name)
            actual_layouts[interface.name] = current_layout or "UNKNOWN"

    # ============= PHASE 5: CREATE RESULT =============
    result = TransformationResult(
        nodes_to_insert=[hw_node],
        nodes_to_remove=[node],
        actual_layouts=actual_layouts,
        metadata={
            "schema_name": schema.name,
            "transformation_algorithm": "default",
            "layout_conversions": [
                inp.name for inp in schema.inputs
                if isinstance(inp, InputSchema) and inp.required_layout
            ],
        }
    )

    return result
