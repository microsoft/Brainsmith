############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Generic kernel inference transform for automatic HW layer inference.

This module provides a unified transform that discovers and delegates to
kernel-specific inference logic, replacing individual Infer* transforms
with a single extensible mechanism.

Example usage:
    from brainsmith.transforms.infer_kernels import InferKernels

    # Infer all registered kernels
    model = model.transform(InferKernels())

    # Infer only specific kernels
    model = model.transform(InferKernels(
        kernel_filter=lambda name, cls: name in ["AddStreams", "Softmax"]
    ))
"""

import logging
from typing import Optional, Callable

from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.core.modelwrapper import ModelWrapper

logger = logging.getLogger(__name__)


class InferKernels(Transformation):
    """Generic transform that infers HW layers from ONNX nodes.

    Discovers registered kernels that support inference and delegates to
    their inference methods. This replaces individual Infer* transforms
    with a unified approach.

    The transform iterates through ONNX nodes and for each node:
    1. Checks all registered kernels to see if they can infer from this node
    2. Delegates to the first matching kernel's infer_from() method
    3. Applies the returned graph modifications
    4. Re-runs shape/datatype inference

    Attributes:
        kernel_filter: Optional function to filter which kernels to try
            Signature: (kernel_name: str, kernel_cls: type) -> bool

    Example:
        # Infer all kernels
        model = model.transform(InferKernels())

        # Infer only AddStreams
        model = model.transform(InferKernels(
            kernel_filter=lambda name, cls: name == "AddStreams"
        ))

        # Infer all except MVAU
        model = model.transform(InferKernels(
            kernel_filter=lambda name, cls: name != "MVAU"
        ))
    """

    def __init__(self, kernel_filter: Optional[Callable[[str, type], bool]] = None):
        """Initialize transform with optional kernel filter.

        Args:
            kernel_filter: Optional function to filter which kernels to try.
                          Returns True to include kernel, False to exclude.
                          Default: include all kernels
        """
        super().__init__()
        self.kernel_filter = kernel_filter or (lambda name, cls: True)

    def apply(self, model: ModelWrapper):
        """Apply kernel inference to the model.

        Discovers kernels with inference support and attempts to infer
        hardware layers from ONNX nodes.

        Args:
            model: QONNX ModelWrapper to transform

        Returns:
            Tuple of (transformed_model, graph_modified_flag)
        """
        from brainsmith.core.plugins import list_kernels, get_kernel

        graph = model.graph
        graph_modified = False
        node_ind = 0

        # Discover all kernels (KernelOp subclasses MUST define inference patterns)
        from brainsmith.dataflow import KernelOp

        all_kernel_names = list_kernels()
        inference_kernels = {}
        for name in all_kernel_names:
            cls = get_kernel(name)
            # Only KernelOp subclasses have inference patterns
            if not issubclass(cls, KernelOp):
                continue

            # Filter kernels and check for non-empty source_ops
            if self.kernel_filter(name, cls):
                pattern = cls.get_inference_pattern()
                # Only include kernels with actual ONNX sources
                if pattern.source_ops:
                    inference_kernels[name] = cls

        if not inference_kernels:
            logger.warning("No kernels with ONNX inference patterns found")
            return (model, False)

        logger.info(f"Found {len(inference_kernels)} kernels with ONNX inference: "
                   f"{', '.join(inference_kernels.keys())}")

        # Iterate through nodes (copy list since we'll modify it)
        for node in list(graph.node):
            node_ind += 1

            # Try each kernel in registration order
            for kernel_name, kernel_cls in inference_kernels.items():
                try:
                    # Check if this kernel can infer from this node
                    if not kernel_cls.can_infer_from(node, model):
                        continue

                    logger.debug(f"Inferring {kernel_name} from {node.op_type} node {node.name}")

                    # Delegate to kernel-specific inference
                    result = kernel_cls.infer_from(node, model, node_ind)

                    # Apply graph modifications
                    for i, new_node in enumerate(result.nodes_to_insert):
                        graph.node.insert(node_ind + i, new_node)
                        logger.debug(f"  Inserted {new_node.op_type} node {new_node.name}")

                    for old_node in result.nodes_to_remove:
                        graph.node.remove(old_node)
                        logger.debug(f"  Removed {old_node.op_type} node {old_node.name}")

                    # Log metadata if present
                    if result.metadata:
                        logger.debug(f"  Metadata: {result.metadata}")

                    graph_modified = True

                    # Only one kernel per node - break after first match
                    break

                except Exception as e:
                    logger.warning(
                        f"Failed to infer {kernel_name} from {node.op_type} node {node.name}: {e}",
                        exc_info=True
                    )
                    # Continue trying other kernels

        # Re-run shape/datatype inference if graph was modified
        if graph_modified:
            logger.info("Graph modified, running InferShapes and InferDataTypes")
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)
