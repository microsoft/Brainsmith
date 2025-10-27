# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SpecializeKernels: Brainsmith implementation of layer specialization.

Replaces FINN's SpecializeLayers with Brainsmith-aware version that:
- For Brainsmith nodes: Sets implementation attribute, keeps domain stable
- For FINN nodes: Mutates domain (backward compatibility)

This eliminates brittle import structure (brainsmith/kernels/hls/__init__.py)
and enables co-located kernel files.
"""

import warnings
from onnx import helper
from qonnx.custom_op.registry import hasCustomOp
from qonnx.transformation.base import Transformation

from finn.util.basic import getHWCustomOp


class SpecializeKernels(Transformation):
    """Specialize hardware nodes to implementation backends.

    Brainsmith behavior (stable domain):
        - Domain: brainsmith.kernels (unchanged)
        - Op type: ChannelwiseOp (unchanged)
        - Attributes: implementation="vitis_hls", backend="fpgadataflow"

    FINN behavior (domain mutation, backward compat):
        - Domain: finn.custom_op.fpgadataflow → finn.custom_op.fpgadataflow.hls
        - Op type: ChannelwiseOp → ChannelwiseOp_hls
        - Attributes: backend="fpgadataflow" (existing)
    """

    def __init__(self, fpgapart):
        super().__init__()
        self.fpgapart = fpgapart

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        for node in graph.node:
            # Skip nodes that are not hardware layers
            if not self._is_hw_node(node):
                continue

            node_ind += 1

            if self._is_brainsmith_node(node):
                # Brainsmith: stable domain, implementation attribute
                self._specialize_brainsmith_node(node, model)
            else:
                # FINN: domain mutation (backward compatibility)
                new_node = self._specialize_finn_node(node, model)
                graph.node.insert(node_ind, new_node)
                graph.node.remove(node)
                graph_modified = True

        return (model, graph_modified)

    def _is_hw_node(self, node):
        """Check if node is a hardware layer."""
        return (
            node.domain.endswith(".custom_op.fpgadataflow") or
            (
                node.domain.startswith("brainsmith.kernels") and
                not (node.domain.endswith(".hls") or node.domain.endswith(".rtl"))
            )
        )

    def _is_brainsmith_node(self, node):
        """Check if node is a Brainsmith kernel."""
        return node.domain.startswith("brainsmith.kernels")

    def _specialize_brainsmith_node(self, node, model):
        """Specialize Brainsmith node: stable domain, implementation attribute.

        Sets:
        - implementation attribute (e.g., "vitis_hls")
        - backend="fpgadataflow" (FINN compatibility)

        Does NOT mutate domain or op_type.
        """
        op = getHWCustomOp(node, model)

        # Determine implementation
        impl = self._select_brainsmith_implementation(node, model)

        # Set attributes (NO domain mutation)
        op.set_nodeattr("implementation", impl)
        op.set_nodeattr("backend", "fpgadataflow")  # FINN compatibility

    def _select_brainsmith_implementation(self, node, model):
        """Select implementation for Brainsmith node.

        Simple heuristic for now:
        1. Check preferred_impl_style
        2. Default to "vitis_hls" (most common)

        Future: Check available backends, backend viability, scoring, etc.
        """
        op = getHWCustomOp(node, model)
        preferred = op.get_nodeattr("preferred_impl_style")

        if preferred == "hls":
            return "vitis_hls"
        elif preferred == "rtl":
            return "verilog"
        else:
            # Default to HLS
            return "vitis_hls"

    def _specialize_finn_node(self, node, model):
        """Specialize FINN node: domain mutation (backward compatibility).

        Delegates to FINN's _determine_impl_style() logic, then creates
        new node with mutated domain and op_type.

        Returns:
            New ONNX node with mutated domain/op_type
        """
        # Import FINN's helpers here to avoid circular imports
        from finn.transformation.fpgadataflow.specialize_layers import _determine_impl_style

        impl_style = _determine_impl_style(node, self.fpgapart, model)
        optype = node.op_type + "_" + impl_style

        # Create new node with mutated domain
        new_node = helper.make_node(
            optype,
            node.input,
            node.output,
            domain=f"{node.domain}.{impl_style}",
        )

        # Copy all attributes except preferred_impl_style
        for attribute in node.attribute:
            if attribute.name != "preferred_impl_style":
                new_node.attribute.append(attribute)

        return new_node
