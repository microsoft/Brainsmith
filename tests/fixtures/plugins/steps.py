"""Test steps for use in blueprint testing."""

import logging
from typing import Any, Dict
from brainsmith.core.plugins import step, get_transform
from brainsmith._internal.io.transform_utils import apply_transforms

logger = logging.getLogger(__name__)


@step(
    name="test_identity_step",
    category="test",
    description="Test step that returns model unchanged"
)
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


@step(
    name="test_transform_sequence",
    category="test", 
    description="Test step that applies a sequence of transforms"
)
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


@step(
    name="test_config_aware_step",
    category="test",
    description="Test step that uses configuration values"
)
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


@step(
    name="test_failing_step",
    category="test",
    description="Test step that always fails"
)
def test_failing_step(model: Any, cfg: Any) -> Any:
    """Test step that always raises an exception."""
    raise RuntimeError("This test step always fails as designed")


@step(
    name="test_conditional_step",
    category="test",
    description="Test step with conditional behavior"
)
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


@step(
    name="test_custom_transform_step",
    category="test",
    description="Test step that applies a custom inline transform"  
)
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


@step(
    name="test_debug_output_step",
    category="test",
    description="Test step that saves debug output"
)
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
        from brainsmith._internal.io.transform_utils import apply_transforms
        model = apply_transforms(
            model, 
            ['GiveReadableTensorNames', 'SortGraph'],
            debug_path=debug_path
        )
    
    return model


# Steps with blueprint/context signature (from plugins.py)
# These are used in design space and test plugins

@step(name="test_step")
def plugin_test_step(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """A simple test step."""
    # Mark that this step was executed
    if "executed_steps" not in context:
        context["executed_steps"] = []
    context["executed_steps"].append("test_step")


@step(name="test_step1")
def plugin_test_step1(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """First test step."""
    if "executed_steps" not in context:
        context["executed_steps"] = []
    context["executed_steps"].append("test_step1")


@step(name="test_step2")
def plugin_test_step2(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Second test step."""
    if "executed_steps" not in context:
        context["executed_steps"] = []
    context["executed_steps"].append("test_step2")


@step(name="test_step2a")
def plugin_test_step2a(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Branch variant 2a."""
    if "executed_steps" not in context:
        context["executed_steps"] = []
    context["executed_steps"].append("test_step2a")


@step(name="test_step2b") 
def plugin_test_step2b(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Branch variant 2b."""
    if "executed_steps" not in context:
        context["executed_steps"] = []
    context["executed_steps"].append("test_step2b")


@step(name="test_step3")
def plugin_test_step3(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Third test step."""
    if "executed_steps" not in context:
        context["executed_steps"] = []
    context["executed_steps"].append("test_step3")


@step(name="infer_kernels")
def infer_kernels_test(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Test implementation of kernel inference step."""
    # This step should receive kernel_backends from design space
    kernel_backends = context.get("kernel_backends", {})
    
    if "executed_steps" not in context:
        context["executed_steps"] = []
    context["executed_steps"].append("infer_kernels")
    
    # Store the kernel backends that were passed
    context["inferred_kernels"] = kernel_backends


@step(name="export_to_build")
def export_to_build_test(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Test implementation of export step."""
    if "executed_steps" not in context:
        context["executed_steps"] = []
    context["executed_steps"].append("export_to_build")


@step(name="failing_step")
def failing_step(blueprint: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Step that always fails for error testing."""
    raise RuntimeError("This step always fails")


# Steps needed for end-to-end tests (with brainsmith: prefix)
@step(name="brainsmith:TestStep1")
def brainsmith_test_step1(model: Any, cfg: Any) -> Any:
    """Test step 1 - marks model as processed."""
    return model


@step(name="brainsmith:TestStep2")
def brainsmith_test_step2(model: Any, cfg: Any) -> Any:
    """Test step 2 - another pass-through."""
    return model


@step(name="brainsmith:TestStep3")
def brainsmith_test_step3(model: Any, cfg: Any) -> Any:
    """Test step 3 - final pass-through."""
    return model


@step(name="brainsmith:BranchA")
def brainsmith_branch_a(model: Any, cfg: Any) -> Any:
    """Branch A variant."""
    return model


@step(name="brainsmith:BranchB") 
def brainsmith_branch_b(model: Any, cfg: Any) -> Any:
    """Branch B variant."""
    return model


@step(name="brainsmith:PrepStep")
def brainsmith_prep_step(model: Any, cfg: Any) -> Any:
    """Preparation step."""
    return model


@step(name="brainsmith:OptA")
def brainsmith_opt_a(model: Any, cfg: Any) -> Any:
    """Optimization variant A."""
    return model


@step(name="brainsmith:OptB")
def brainsmith_opt_b(model: Any, cfg: Any) -> Any:
    """Optimization variant B."""
    return model


@step(name="brainsmith:OptC")
def brainsmith_opt_c(model: Any, cfg: Any) -> Any:
    """Optimization variant C."""
    return model


@step(name="brainsmith:FinalStep")
def brainsmith_final_step(model: Any, cfg: Any) -> Any:
    """Final processing step."""
    return model