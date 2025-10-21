"""Test steps for use in blueprint testing."""

import logging
from typing import Any, Dict
# Note: Old registry decorators removed - new plugin system doesn't use decorators
from brainsmith.primitives.utils import apply_transforms

logger = logging.getLogger(__name__)


def test_identity_step(model: Any, cfg: Any) -> Any:
    """Test step that simply returns the model unchanged."""
    logger.info("Executing test_identity_step")
    
    # Mark that this step was executed in the model metadata
    if hasattr(model, 'model'):
        # Add custom metadata to track execution
        model.model.metadata_props.append(
            model.model.metadata_props.add()
        )
        model.model.metadata_props[-1].key = "test_identity_step_executed"
        model.model.metadata_props[-1].value = "true"
    
    return model


def test_transform_sequence_step(model: Any, cfg: Any) -> Any:
    """Test step that applies multiple transforms in sequence."""
    logger.info("Executing test_transform_sequence")
    
    # Apply some basic transforms
    model = apply_transforms(model, [
        'GiveUniqueNodeNames',
        'InferShapes',
        'InferDataTypes'
    ])
    
    return model


def test_config_aware_step(model: Any, cfg: Any) -> Any:
    """Test step that reads and uses configuration."""
    logger.info(f"Executing test_config_aware_step with config: {cfg}")
    
    # Example: Use a config value if present
    test_value = getattr(cfg, 'test_value', 'default')
    logger.info(f"Test value from config: {test_value}")
    
    # Add metadata about config usage
    if hasattr(model, 'model'):
        model.model.metadata_props.append(
            model.model.metadata_props.add()
        )
        model.model.metadata_props[-1].key = "test_config_value"
        model.model.metadata_props[-1].value = str(test_value)
    
    return model


def test_failing_step(model: Any, cfg: Any) -> Any:
    """Test step that always raises an exception."""
    raise RuntimeError("This test step always fails as designed")


def test_conditional_step(model: Any, cfg: Any) -> Any:
    """Test step that behaves differently based on config."""
    logger.info("Executing test_conditional_step")
    
    # Check for a flag in config
    should_apply_transforms = getattr(cfg, 'apply_transforms', False)
    
    if should_apply_transforms:
        logger.info("Applying transforms based on config flag")
        model = apply_transforms(model, [
            'FoldConstants',
            'RemoveUnusedTensors'
        ])
    else:
        logger.info("Skipping transforms based on config flag")
    
    return model


def test_custom_transform_step(model: Any, cfg: Any) -> Any:
    """Test step demonstrating custom transform logic."""
    logger.info("Executing test_custom_transform_step")
    
    # Example of custom transform logic without creating a full Transform class
    # Add a marker attribute to the first node if any exist
    if model.graph.node:
        first_node = model.graph.node[0]
        # Check if attribute already exists
        attr_names = [attr.name for attr in first_node.attribute]
        if "test_custom_transform_applied" not in attr_names:
            attr = first_node.attribute.add()
            attr.name = "test_custom_transform_applied"
            attr.type = 1  # INT type
            attr.i = 1
    
    return model


def test_debug_output_step(model: Any, cfg: Any) -> Any:
    """Test step that demonstrates debug output capabilities."""
    logger.info("Executing test_debug_output_step")
    
    # Get output directory from config if available
    output_dir = getattr(cfg, 'output_dir', '/tmp')
    debug_mode = getattr(cfg, 'debug_mode', False)
    
    if debug_mode:
        import os
        debug_path = os.path.join(output_dir, "test_debug")
        logger.info(f"Debug mode enabled, saving to {debug_path}")
        
        # Apply transforms with debug output
        from brainsmith.primitives.utils import apply_transforms
        model = apply_transforms(
            model, 
            ['GiveReadableTensorNames', 'SortGraph'],
            debug_path=debug_path
        )
    
    return model


# Steps with blueprint/context signature (from plugins.py)
# These are used in design space and test plugins

