"""Real test step components for DSE integration testing.

These steps are used in blueprint parsing and DSE execution tests.
"""

import logging
from typing import Any
from brainsmith.registry import step

logger = logging.getLogger(__name__)


# Complex test steps for DSE integration tests

@step(name='test_identity_step')
def test_identity_step(model: Any, cfg: Any) -> Any:
    """Test step that simply returns the model unchanged with metadata tracking."""
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


@step(name='test_transform_sequence_step')
def test_transform_sequence_step(model: Any, cfg: Any) -> Any:
    """Test step that applies multiple QONNX transforms in sequence."""
    from qonnx.transformation.general import GiveUniqueNodeNames
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.infer_datatypes import InferDataTypes

    logger.info("Executing test_transform_sequence")

    # Apply some basic transforms
    for transform in [GiveUniqueNodeNames(), InferShapes(), InferDataTypes()]:
        model = model.transform(transform)

    return model


@step(name='test_config_aware_step')
def test_config_aware_step(model: Any, cfg: Any) -> Any:
    """Test step that reads and uses configuration values."""
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


@step(name='test_failing_step')
def test_failing_step(model: Any, cfg: Any) -> Any:
    """Test step that always raises an exception for error handling tests."""
    raise RuntimeError("This test step always fails as designed")


@step(name='test_conditional_step')
def test_conditional_step(model: Any, cfg: Any) -> Any:
    """Test step that behaves differently based on config flags."""
    from qonnx.transformation.fold_constants import FoldConstants
    from qonnx.transformation.general import RemoveUnusedTensors

    logger.info("Executing test_conditional_step")

    # Check for a flag in config
    should_apply_transforms = getattr(cfg, 'apply_transforms', False)

    if should_apply_transforms:
        logger.info("Applying transforms based on config flag")
        for transform in [FoldConstants(), RemoveUnusedTensors()]:
            model = model.transform(transform)
    else:
        logger.info("Skipping transforms based on config flag")

    return model


@step(name='test_custom_transform_step')
def test_custom_transform_step(model: Any, cfg: Any) -> Any:
    """Test step demonstrating custom transform logic without Transform class."""
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


@step(name='test_debug_output_step')
def test_debug_output_step(model: Any, cfg: Any) -> Any:
    """Test step that demonstrates debug output capabilities."""
    from qonnx.transformation.general import GiveReadableTensorNames, SortGraph

    logger.info("Executing test_debug_output_step")

    # Get output directory from config if available
    output_dir = getattr(cfg, 'output_dir', '/tmp')
    debug_mode = getattr(cfg, 'debug_mode', False)

    if debug_mode:
        import os
        debug_path = os.path.join(output_dir, "test_debug")
        logger.info(f"Debug mode enabled, saving to {debug_path}")

        # Apply transforms
        for transform in [GiveReadableTensorNames(), SortGraph()]:
            model = model.transform(transform)

    return model


# Simple test steps for blueprint branching tests
# These are minimal steps used in design space exploration tests

@step(name='test_step')
def test_step(model: Any, cfg: Any) -> Any:
    """Minimal test step for basic blueprint testing."""
    logger.info("Executing test_step")
    return model


@step(name='test_step1')
def test_step1(model: Any, cfg: Any) -> Any:
    """Minimal test step 1 for blueprint branching tests."""
    logger.info("Executing test_step1")
    return model


@step(name='test_step2')
def test_step2(model: Any, cfg: Any) -> Any:
    """Minimal test step 2 for blueprint branching tests."""
    logger.info("Executing test_step2")
    return model


@step(name='test_step3')
def test_step3(model: Any, cfg: Any) -> Any:
    """Minimal test step 3 for blueprint sequential testing."""
    logger.info("Executing test_step3")
    return model
