"""Test transforms that can be used both standalone and within steps."""

import logging
import onnx
from typing import Any
from qonnx.transformation.base import Transformation
# Note: Old registry decorators removed - new plugin system doesn't use decorators

logger = logging.getLogger(__name__)


# Standalone transforms that inherit from Transformation base class
# These have apply(self, model) signature

class TestAddMetadata(Transformation):
    """Transform that adds metadata to the model."""
    
    def __init__(self, metadata_key="test_key", metadata_value="test_value"):
        super().__init__()
        self.metadata_key = metadata_key
        self.metadata_value = metadata_value
    
    def apply(self, model):
        """Add metadata to model."""
        # For QONNX ModelWrapper
        if hasattr(model, 'model'):
            onnx_model = model.model
        else:
            onnx_model = model
            
        # Add metadata
        if hasattr(onnx_model, 'metadata_props'):
            onnx_model.metadata_props.append(
                onnx_model.metadata_props.add()
            )
            onnx_model.metadata_props[-1].key = self.metadata_key
            onnx_model.metadata_props[-1].value = self.metadata_value
            
        return model, True  # Return (model, modified) tuple


class TestNodeCounter(Transformation):
    """Transform that counts nodes by op type."""
    
    def apply(self, model):
        """Count nodes in model."""
        node_counts = {}
        
        for node in model.graph.node:
            op_type = node.op_type
            node_counts[op_type] = node_counts.get(op_type, 0) + 1
        
        logger.info(f"Node counts: {node_counts}")
        
        # Add count as metadata
        if hasattr(model, 'model'):
            onnx_model = model.model
        else:
            onnx_model = model
            
        if hasattr(onnx_model, 'metadata_props'):
            onnx_model.metadata_props.append(
                onnx_model.metadata_props.add()
            )
            onnx_model.metadata_props[-1].key = "node_count"
            onnx_model.metadata_props[-1].value = str(sum(node_counts.values()))
        
        return model, False  # No graph modification


class TestAttributeAdder(Transformation):
    """Transform that adds attributes to specific nodes."""
    
    def __init__(self, target_op_type="Add", attribute_name="test_attr", attribute_value=42):
        super().__init__()
        self.target_op_type = target_op_type
        self.attribute_name = attribute_name
        self.attribute_value = attribute_value
    
    def apply(self, model):
        """Add attributes to matching nodes."""
        modified = False
        
        for node in model.graph.node:
            if node.op_type == self.target_op_type:
                # Check if attribute already exists
                attr_names = [attr.name for attr in node.attribute]
                if self.attribute_name not in attr_names:
                    attr = node.attribute.add()
                    attr.name = self.attribute_name
                    attr.type = onnx.AttributeProto.INT
                    attr.i = self.attribute_value
                    modified = True
        
        return model, modified


# Steps that use transforms (have model, cfg signature)
# These are registered as steps and can be used in blueprints

def test_apply_custom_transforms(model: Any, cfg: Any) -> Any:
    """Step that applies custom test transforms."""
    logger.info("Applying custom test transforms")
    
    # Check if we should add attributes based on config
    add_attributes = getattr(cfg, 'add_attributes', True)
    target_op = getattr(cfg, 'target_op', 'Relu')
    
    if add_attributes:
        # Apply the TestAttributeAdder transform
        from brainsmith import import_transform
        transform_class = import_transform('TestAttributeAdder')
        transform = transform_class(
            target_op_type=target_op,
            attribute_name="custom_test",
            attribute_value=123
        )
        model = model.transform(transform)
    
    # Always apply metadata transform
    from brainsmith import import_transform
    metadata_class = import_transform('TestAddMetadata')
    metadata_transform = metadata_class(
        metadata_key="custom_transforms_applied",
        metadata_value="true"
    )
    model = model.transform(metadata_transform)
    
    return model


def test_transform_chain(model: Any, cfg: Any) -> Any:
    """Step that chains multiple transforms."""
    logger.info("Executing transform chain")
    
    # Chain several transforms
    # 1. Count nodes
    from brainsmith import import_transform
    counter_class = import_transform('TestNodeCounter')
    model = model.transform(counter_class())
    
    # 2. Add metadata
    metadata_class = import_transform('TestAddMetadata')
    model = model.transform(metadata_class(
        metadata_key="chain_step",
        metadata_value="executed"
    ))
    
    # 3. Apply standard transforms
    from brainsmith._internal.io.transform_utils import apply_transforms
    model = apply_transforms(model, [
        'InferShapes',
        'FoldConstants'
    ])
    
    return model


# Simple transforms from plugins.py (not QONNX-style)
# These have apply(self, model) -> model signature

