"""Test transforms that can be used both standalone and within steps."""

import logging
import onnx
from typing import Any
from qonnx.transformation.base import Transformation
from brainsmith.core.plugins import transform, step

logger = logging.getLogger(__name__)


# Standalone transforms that inherit from Transformation base class
# These have apply(self, model) signature

@transform(
    name="TestAddMetadata",
    category="test",
    description="Adds test metadata to model"
)
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


@transform(
    name="TestNodeCounter", 
    category="test",
    description="Counts and logs nodes in model"
)
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


@transform(
    name="TestAttributeAdder",
    category="test", 
    description="Adds attributes to nodes matching criteria"
)
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

@step(
    name="test_apply_custom_transforms",
    category="test",
    description="Apply custom transforms to test their usage"
)
def test_apply_custom_transforms(model: Any, cfg: Any) -> Any:
    """Step that applies custom test transforms."""
    logger.info("Applying custom test transforms")
    
    # Check if we should add attributes based on config
    add_attributes = getattr(cfg, 'add_attributes', True)
    target_op = getattr(cfg, 'target_op', 'Relu')
    
    if add_attributes:
        # Apply the TestAttributeAdder transform
        from brainsmith.core.plugins import get_transform
        transform_class = get_transform('TestAttributeAdder')
        transform = transform_class(
            target_op_type=target_op,
            attribute_name="custom_test",
            attribute_value=123
        )
        model = model.transform(transform)
    
    # Always apply metadata transform
    from brainsmith.core.plugins import get_transform
    metadata_class = get_transform('TestAddMetadata')
    metadata_transform = metadata_class(
        metadata_key="custom_transforms_applied",
        metadata_value="true"
    )
    model = model.transform(metadata_transform)
    
    return model


@step(
    name="test_transform_chain", 
    category="test",
    description="Chain multiple transforms together"
)
def test_transform_chain(model: Any, cfg: Any) -> Any:
    """Step that chains multiple transforms."""
    logger.info("Executing transform chain")
    
    # Chain several transforms
    # 1. Count nodes
    from brainsmith.core.plugins import get_transform
    counter_class = get_transform('TestNodeCounter')
    model = model.transform(counter_class())
    
    # 2. Add metadata
    metadata_class = get_transform('TestAddMetadata')
    model = model.transform(metadata_class(
        metadata_key="chain_step",
        metadata_value="executed"
    ))
    
    # 3. Apply standard transforms
    from brainsmith.utils import apply_transforms
    model = apply_transforms(model, [
        'InferShapes',
        'FoldConstants'
    ])
    
    return model


# Simple transforms from plugins.py (not QONNX-style)
# These have apply(self, model) -> model signature

@transform(name="test_transform")
class TestTransformPlugin:
    """A simple test transform that adds a node attribute."""
    
    def apply(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply transform to model."""
        # Add a custom attribute to the first node
        if model.graph.node:
            node = model.graph.node[0]
            attr = node.attribute.add()
            attr.name = "test_transform_applied"
            attr.type = onnx.AttributeProto.INT
            attr.i = 1
        return model


@transform(name="test_transform_with_metadata", test_metadata="value")
class TestTransformWithMetadataPlugin:
    """Transform with custom metadata."""
    
    def apply(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply transform to model."""
        return model


@transform(name="failing_transform")
class FailingTransform:
    """Transform that always fails for error testing."""
    
    def apply(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply transform to model."""
        raise RuntimeError("This transform always fails")


# Helper function to create transforms dynamically
def create_model_producing_transform(output_path: str) -> Any:
    """Create a transform that produces an ONNX model at specified path."""
    
    @transform(name="model_producing_transform")
    class ModelProducingTransform:
        def __init__(self):
            self.output_path = output_path
            
        def apply(self, model: onnx.ModelProto) -> onnx.ModelProto:
            """Save model to intermediate_models directory."""
            import os
            from pathlib import Path
            
            # Create intermediate_models directory
            intermediate_dir = Path(self.output_path) / "intermediate_models"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = intermediate_dir / "produced_model.onnx"
            onnx.save(model, str(model_path))
            
            return model
    
    return ModelProducingTransform()


# Steps that use transforms
# These have (model, cfg) signature

@step(
    name="test_apply_custom_transforms",
    category="test",
    description="Step that applies custom test transforms"
)
def test_apply_custom_transforms_step(model: Any, cfg: Any) -> Any:
    """Step that demonstrates using custom transforms."""
    logger.info("Applying custom test transforms")
    
    # Use transforms directly
    transform1 = TestAddMetadata(
        metadata_key="test_step_executed",
        metadata_value="test_apply_custom_transforms"
    )
    model = model.transform(transform1)
    
    # Use transform through plugin system
    from brainsmith.core.plugins import get_transform
    NodeCounter = get_transform("TestNodeCounter")
    model = model.transform(NodeCounter())
    
    # Conditional transform based on config
    if getattr(cfg, 'add_attributes', False):
        AttributeAdder = get_transform("TestAttributeAdder")
        model = model.transform(AttributeAdder(
            target_op_type=getattr(cfg, 'target_op', 'Add'),
            attribute_name="custom_attr",
            attribute_value=123
        ))
    
    return model


@step(
    name="test_transform_chain",
    category="test",
    description="Step that chains multiple transforms"
)  
def test_transform_chain_step(model: Any, cfg: Any) -> Any:
    """Step demonstrating transform chaining."""
    from brainsmith.utils import apply_transforms_with_params
    
    # Chain transforms with parameters
    transforms = [
        ('TestAddMetadata', {'metadata_key': 'chain_start', 'metadata_value': 'true'}),
        ('TestNodeCounter', {}),
        ('TestAddMetadata', {'metadata_key': 'chain_end', 'metadata_value': 'true'}),
    ]
    
    model = apply_transforms_with_params(model, transforms)
    
    # Also apply standard transforms
    from brainsmith.utils import apply_transforms
    model = apply_transforms(model, ['GiveUniqueNodeNames', 'InferShapes'])
    
    return model