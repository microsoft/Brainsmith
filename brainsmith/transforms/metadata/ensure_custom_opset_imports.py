# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Ensure custom opset imports are present in the ONNX model.
"""
from typing import Tuple
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from onnx import helper
from brainsmith.core.plugins import transform


@transform(
    name="EnsureCustomOpsetImports",
    stage="metadata",
    description="Ensures all custom domains have corresponding opset imports",
    author="brainsmith",
    version="1.0.0"
)
class EnsureCustomOpsetImports(Transformation):
    """
    Ensures all custom node domains have corresponding opset imports.
    
    This transform scans all nodes in the graph and ensures that any
    custom domains (non-empty domains) have corresponding opset imports.
    This is crucial for FINN hardware layers which use custom domains
    like 'brainsmith.kernels' and 'finn.custom_op.fpgadataflow'.
    """
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        """Apply the transformation."""
        # Collect all custom domains used in the graph
        custom_domains = set()
        for node in model.graph.node:
            if node.domain and node.domain != "":
                custom_domains.add(node.domain)
        
        # Check existing opset imports
        existing_domains = {op.domain for op in model.model.opset_import}
        
        # Add missing opset imports
        graph_modified = False
        for domain in custom_domains:
            if domain not in existing_domains:
                # Add opset import for this domain
                opset = helper.make_opsetid(domain, 1)
                model.model.opset_import.append(opset)
                graph_modified = True
        
        return (model, graph_modified)