"""Integration tests for the blueprint parser system."""

import pytest
from pathlib import Path

from brainsmith.dse import parse_blueprint, DesignSpace, DSEConfig
from brainsmith.dse._types import OutputType
from tests.utils.blueprint_helpers import (
    create_minimal_blueprint,
    create_full_blueprint,
    create_extends_blueprint,
    create_inheritance_parent,
    create_inheritance_grandparent,
    create_base_steps_blueprint,
    create_step_insert_after_blueprint,
    create_step_insert_start_blueprint,
    create_step_insert_end_blueprint,
    create_step_replace_blueprint,
    create_step_remove_blueprint,
    create_branch_points_blueprint,
    create_step_range_blueprint,
    create_blueprint_file
)


class TestBasicParsing:
    """Test suite for basic blueprint parsing."""
    
    def test_parse_minimal_blueprint(self, tmp_path):
        """Test parsing a minimal valid blueprint."""
        blueprint_path = create_minimal_blueprint(
            tmp_path, 
            name="test_minimal",
            steps=["test_step1", "test_step2", "test_step3"]
        )
        
        design_space, blueprint_config = parse_blueprint(str(blueprint_path), "test_model.onnx")
        
        assert design_space.model_path == "test_model.onnx"
        assert design_space.steps == ["test_step1", "test_step2", "test_step3"]
        assert blueprint_config.clock_ns == 5.0
    
    def test_extract_blueprint_config(self, tmp_path):
        """Test extraction of BlueprintConfig from blueprint."""
        blueprint_path = create_full_blueprint(
            tmp_path,
            name="test_config",
            description="Test configuration extraction", 
            clock_ns=3.5,
            steps=["test_step1"]
        )
        
        design_space, blueprint_config = parse_blueprint(str(blueprint_path), "test_model.onnx")

        assert blueprint_config.clock_ns == 3.5
        assert blueprint_config.output == OutputType.BITFILE
        assert blueprint_config.board == "V80"
        assert blueprint_config.save_intermediate_models is True

        # Test default values
        minimal_path = create_minimal_blueprint(
            tmp_path,
            name="minimal",
            steps=["test_step"]
        )

        _, blueprint_config2 = parse_blueprint(str(minimal_path), "test_model.onnx")
        assert blueprint_config2.output == OutputType.ESTIMATES  # default
        assert blueprint_config2.save_intermediate_models is False  # default
    
    def test_parse_design_space(self, tmp_path):
        """Test parsing steps and kernels from design space."""
        design_space_template = """
name: {name}
clock_ns: {clock_ns}
design_space:
  steps: {steps}
  kernels: {kernels}
"""
        blueprint_path = create_blueprint_file(
            tmp_path,
            design_space_template,
            name="test_design",
            steps=["test_step1", "test_step2", "test_step3"],
            kernels=["TestKernel", "TestKernelWithBackends"]
        )
        
        design_space, blueprint_config = parse_blueprint(str(blueprint_path), "test_model.onnx")
        
        assert design_space.steps == ["test_step1", "test_step2", "test_step3"]
        assert len(design_space.kernel_backends) == 2
        kernel_names = [kb[0] for kb in design_space.kernel_backends]
        assert "TestKernel" in kernel_names
        assert "TestKernelWithBackends" in kernel_names


class TestBlueprintInheritance:
    """Test suite for blueprint inheritance."""
    
    def test_single_level_inheritance(self, tmp_path):
        """Test single level blueprint inheritance."""
        # Create parent blueprint
        parent_path = create_inheritance_parent(
            tmp_path,
            name="parent",
            steps=["test_step1", "test_step2"]
        )
        
        # Create child blueprint
        child_path = create_extends_blueprint(
            tmp_path,
            name="child",
            extends="parent.yaml",
            clock_ns=3.0,
            steps=["test_step2", "test_step3"]
        )
        
        design_space, blueprint_config = parse_blueprint(str(child_path), "test_model.onnx")
        
        assert blueprint_config.clock_ns == 3.0  # Child override
        assert blueprint_config.board == "V70"  # Parent value
        assert design_space.steps == ["test_step2", "test_step3"]  # Child override
        kernel_names = [kb[0] for kb in design_space.kernel_backends]
        assert "TestKernel" in kernel_names  # Parent value inherited
    
    def test_multi_level_inheritance(self, tmp_path):
        """Test multi-level inheritance chain."""
        # Create grandparent
        gp_path = create_inheritance_grandparent(
            tmp_path,
            name="grandparent",
            clock_ns=10.0,
            steps=["test_step1"]
        )
        
        # Create parent extending grandparent
        parent_path = create_extends_blueprint(
            tmp_path,
            name="parent",
            extends="grandparent.yaml",
            clock_ns=7.0,
            steps=["test_step2"]
        )
        
        # Create child extending parent
        child_path = create_extends_blueprint(
            tmp_path,
            name="child",
            extends="parent.yaml",
            clock_ns=7.0,  # Will inherit parent's value
            steps=["test_step3"]
        )
        
        # Add output override to child
        child_content = child_path.read_text()
        child_content = child_content.replace("extends: parent.yaml\n", "extends: parent.yaml\noutput: bitfile\n")
        child_path.write_text(child_content)
        
        # Parse child blueprint
        design_space, blueprint_config = parse_blueprint(str(child_path), "test_model.onnx")
        
        # Verify deep merging
        assert blueprint_config.clock_ns == 7.0  # Parent's override
        assert blueprint_config.board == "V100"  # Grandparent's value
        assert blueprint_config.output == OutputType.BITFILE  # Child's override
        
        # Test override precedence
        assert design_space.steps == ["test_step3"]  # Most recent wins
    
    def test_inheritance_with_lists(self, tmp_path):
        """Test inheritance behavior with list fields."""
        # Parent has steps list
        parent_path = create_inheritance_parent(
            tmp_path,
            name="parent",
            steps=["test_step", "test_step1", "test_step2"],
            kernels=["TestKernel", "TestKernelWithBackends"]
        )
        
        # Child modifies steps
        child_path = create_extends_blueprint(
            tmp_path,
            name="child",
            extends="parent.yaml",
            steps=["test_step1", "test_step2"]
        )
        
        # Parse child
        design_space, blueprint_config = parse_blueprint(str(child_path), "test_model.onnx")
        
        # Verify list handling
        assert design_space.steps == ["test_step1", "test_step2"]  # Replaced
        # Check kernel_backends
        kernel_names = [kb[0] for kb in design_space.kernel_backends]
        assert "TestKernel" in kernel_names  # Inherited
        assert "TestKernelWithBackends" in kernel_names  # Inherited


class TestStepOperations:
    """Test suite for step operation features."""
    
    def test_insert_operations(self, tmp_path):
        """Test insert operations for steps."""
        # Create base blueprint
        base_path = create_base_steps_blueprint(tmp_path)
        
        # Test insert after
        after_path = create_step_insert_after_blueprint(tmp_path)
        design_space, blueprint_config = parse_blueprint(str(after_path), "test_model.onnx")
        assert design_space.steps == ["test_step", "infer_kernels", "test_step1", "test_step2"]
        
        # Test insert at_start
        start_path = create_step_insert_start_blueprint(tmp_path)
        design_space, blueprint_config = parse_blueprint(str(start_path), "test_model.onnx")
        assert design_space.steps == ["export_to_build", "test_step", "test_step1", "test_step2"]
        
        # Test insert at_end
        end_path = create_step_insert_end_blueprint(tmp_path)
        design_space, blueprint_config = parse_blueprint(str(end_path), "test_model.onnx")
        assert design_space.steps == ["test_step", "test_step1", "test_step2", "infer_kernels"]
    
    def test_replace_operation(self, tmp_path):
        """Test replace operation for steps."""
        # Create base blueprint
        base_path = create_base_steps_blueprint(tmp_path)
        
        # Test replace
        replace_path = create_step_replace_blueprint(tmp_path)
        design_space, blueprint_config = parse_blueprint(str(replace_path), "test_model.onnx")
        
        # Verify order preservation
        assert design_space.steps == ["test_step", "infer_kernels", "test_step2"]
    
    def test_remove_operation(self, tmp_path):
        """Test remove operation for steps."""
        # Create base blueprint
        base_path = create_base_steps_blueprint(tmp_path)
        
        # Test remove
        remove_path = create_step_remove_blueprint(tmp_path)
        design_space, blueprint_config = parse_blueprint(str(remove_path), "test_model.onnx")
        
        # Verify removal
        assert design_space.steps == ["test_step", "test_step2"]
    
    def test_branch_points_with_skip(self, tmp_path):
        """Test branch points with skip operator."""
        blueprint_path = create_branch_points_blueprint(tmp_path)
        design_space, blueprint_config = parse_blueprint(str(blueprint_path), "test_model.onnx")
        
        # Verify skip handling
        assert design_space.steps[0] == "test_step"
        assert isinstance(design_space.steps[1], list)
        assert "~" in design_space.steps[1]  # Skip preserved
        assert design_space.steps[2] == "infer_kernels"
        
        # Note: Nested branches are not supported by the parser
        # The parser will raise an error if it encounters nested lists


class TestDesignSpaceValidation:
    """Test suite for design space validation."""
    
    def test_validate_step_plugins_exist(self, tmp_path):
        """Test validation of step plugin existence."""
        # Use valid plugin names
        valid_path = create_minimal_blueprint(
            tmp_path,
            name="valid",
            steps=["test_step1", "test_step2", "test_step3"]
        )
        
        design_space, blueprint_config = parse_blueprint(str(valid_path), "test_model.onnx")
        
        # Verify successful validation
        assert len(design_space.steps) == 3
        assert design_space.steps == ["test_step1", "test_step2", "test_step3"]
    
    def test_combination_limit_check(self, tmp_path):
        """Test design space combination limit checking."""
        # Create large design space with nested branching
        large_design_template = """
name: {name}
clock_ns: {clock_ns}
design_space:
  steps:
    - test_step
    - [test_step1, test_step2, infer_kernels, export_to_build]
    - [test_step, test_step1, test_step2]
    - [infer_kernels, export_to_build]
    - test_step1
"""
        large_path = create_blueprint_file(
            tmp_path,
            large_design_template,
            name="large"
        )
        
        design_space, blueprint_config = parse_blueprint(str(large_path), "test_model.onnx")
        
        # Verify combination calculation
        # Should have 4 * 3 * 2 = 24 combinations
        # This should be within limits
        assert len(design_space.steps) == 5
    
    def test_kernel_backend_resolution(self, tmp_path):
        """Test kernel backend specification and resolution."""
        # Specify kernel with backends using complex structure
        kernel_template = """
name: {name}
clock_ns: {clock_ns}
design_space:
  steps: {steps}
  kernels:
    - TestKernel
    - TestKernelWithBackends: [TestKernelWithBackends_hls]
"""
        kernel_path = create_blueprint_file(
            tmp_path,
            kernel_template,
            name="kernel_test",
            steps=["test_step"]
        )
        
        design_space, blueprint_config = parse_blueprint(str(kernel_path), "test_model.onnx")
        
        # Verify backend discovery
        kernel_names = [kb[0] for kb in design_space.kernel_backends]
        assert "TestKernel" in kernel_names
        assert "TestKernelWithBackends" in kernel_names

        # Note: This is placeholder functionality
        # Backend resolution will be updated when FINN integration is complete


class TestStepRangeConfiguration:
    """Test suite for step range control in blueprints."""

    def test_parse_start_step_from_blueprint(self, tmp_path):
        """Test parsing start_step from blueprint."""
        blueprint_path = create_step_range_blueprint(
            tmp_path,
            name="test_start",
            start_step="test_step1",
            steps=["test_step", "test_step1", "test_step2", "test_step3"]
        )

        design_space, blueprint_config = parse_blueprint(str(blueprint_path), "model.onnx")
        assert blueprint_config.start_step == "test_step1"

    def test_parse_stop_step_from_blueprint(self, tmp_path):
        """Test parsing stop_step from blueprint."""
        blueprint_path = create_step_range_blueprint(
            tmp_path,
            name="test_stop",
            stop_step="test_step2",
            steps=["test_step", "test_step1", "test_step2", "test_step3"]
        )

        design_space, blueprint_config = parse_blueprint(str(blueprint_path), "model.onnx")
        assert blueprint_config.stop_step == "test_step2"

    def test_parse_both_start_and_stop(self, tmp_path):
        """Test parsing both start_step and stop_step."""
        blueprint_path = create_step_range_blueprint(
            tmp_path,
            name="test_both",
            start_step="test_step1",
            stop_step="test_step2",
            steps=["test_step", "test_step1", "test_step2", "test_step3"]
        )

        design_space, blueprint_config = parse_blueprint(str(blueprint_path), "model.onnx")
        assert blueprint_config.start_step == "test_step1"
        assert blueprint_config.stop_step == "test_step2"

    def test_default_values_when_not_specified(self, tmp_path):
        """Test that start_step and stop_step default to None."""
        blueprint_path = create_minimal_blueprint(
            tmp_path,
            name="test_defaults",
            steps=["test_step1", "test_step2"]
        )

        design_space, blueprint_config = parse_blueprint(str(blueprint_path), "model.onnx")
        assert blueprint_config.start_step is None
        assert blueprint_config.stop_step is None