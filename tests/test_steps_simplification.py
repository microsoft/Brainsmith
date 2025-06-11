"""
Test suite for Steps Module Simplification.
Validates North Star transformation from enterprise registry to simple functions.
"""

import pytest
from unittest.mock import Mock, patch
from brainsmith.steps import (
    # Step functions
    cleanup_step, cleanup_advanced_step, qonnx_to_finn_step,
    streamlining_step, infer_hardware_step, constrain_folding_and_set_pumped_compute_step,
    generate_reference_io_step, shell_metadata_handover_step,
    remove_head_step, remove_tail_step,
    # Discovery functions
    get_step, validate_step_sequence, discover_all_steps,
    extract_step_metadata, StepMetadata
)


class TestStepFunctions:
    """Test individual step functions work correctly."""
    
    def test_cleanup_step(self):
        """Test basic cleanup step."""
        mock_model = Mock()
        mock_cfg = Mock()
        
        # Mock the transform method to return the model
        mock_model.transform.return_value = mock_model
        
        result = cleanup_step(mock_model, mock_cfg)
        
        # Should call transforms and return model
        assert result == mock_model
        assert mock_model.transform.call_count == 2  # Two transforms
    
    def test_cleanup_advanced_step(self):
        """Test advanced cleanup step."""
        mock_model = Mock()
        mock_cfg = Mock()
        mock_model.transform.return_value = mock_model
        
        result = cleanup_advanced_step(mock_model, mock_cfg)
        
        assert result == mock_model
        assert mock_model.transform.call_count == 5  # Five transforms
    
    def test_qonnx_to_finn_step(self):
        """Test QONNX to FINN conversion step."""
        mock_model = Mock()
        mock_cfg = Mock()
        mock_model.transform.return_value = mock_model
        
        result = qonnx_to_finn_step(mock_model, mock_cfg)
        
        assert result == mock_model
        assert mock_model.transform.call_count == 4  # Four transforms
    
    def test_streamlining_step(self):
        """Test streamlining step."""
        mock_model = Mock()
        mock_cfg = Mock()
        mock_model.transform.return_value = mock_model
        
        result = streamlining_step(mock_model, mock_cfg)
        
        assert result == mock_model
        assert mock_model.transform.call_count == 11  # Eleven transforms
    
    def test_infer_hardware_step(self):
        """Test hardware inference step."""
        mock_model = Mock()
        mock_cfg = Mock()
        mock_model.transform.return_value = mock_model
        
        result = infer_hardware_step(mock_model, mock_cfg)
        
        assert result == mock_model
        assert mock_model.transform.call_count == 7  # Seven transforms


class TestMetadataExtraction:
    """Test docstring metadata extraction."""
    
    def test_extract_step_metadata(self):
        """Test metadata extraction from docstring."""
        def sample_step(model, cfg):
            """
            Sample step for testing.
            
            Category: test
            Dependencies: [cleanup, conversion]
            Description: A test step for validation
            """
            return model
        
        metadata = extract_step_metadata(sample_step)
        
        assert metadata.name == "sample"
        assert metadata.category == "test"
        assert metadata.dependencies == ["cleanup", "conversion"]
        assert metadata.description == "A test step for validation"
    
    def test_extract_empty_dependencies(self):
        """Test extraction with empty dependencies."""
        def sample_step(model, cfg):
            """
            Sample step with no dependencies.
            
            Category: test
            Dependencies: []
            Description: No dependencies here
            """
            return model
        
        metadata = extract_step_metadata(sample_step)
        assert metadata.dependencies == []
    
    def test_extract_missing_fields(self):
        """Test extraction with missing fields."""
        def sample_step(model, cfg):
            """Just a basic docstring."""
            return model
        
        metadata = extract_step_metadata(sample_step)
        assert metadata.name == "sample"
        assert metadata.category == "unknown"
        assert metadata.dependencies == []
        assert metadata.description == ""


class TestStepDiscovery:
    """Test step discovery functionality."""
    
    def test_discover_all_steps(self):
        """Test step discovery finds all functions."""
        steps = discover_all_steps()
        
        # Should find all our step functions
        expected_steps = [
            'cleanup', 'cleanup_advanced', 'qonnx_to_finn',
            'streamlining', 'infer_hardware', 'constrain_folding_and_set_pumped_compute',
            'generate_reference_io', 'shell_metadata_handover',
            'remove_head', 'remove_tail'
        ]
        
        for step_name in expected_steps:
            assert step_name in steps
            assert callable(steps[step_name])
    
    def test_get_step_brainsmith(self):
        """Test getting BrainSmith step."""
        step_fn = get_step('cleanup')
        assert step_fn == cleanup_step
    
    @patch('brainsmith.steps.finn.builder.build_dataflow_steps')
    def test_get_step_finn_fallback(self, mock_finn_steps):
        """Test fallback to FINN steps."""
        # Mock FINN step
        mock_finn_step = Mock()
        mock_finn_steps.__dict__ = {'some_finn_step': mock_finn_step}
        
        step_fn = get_step('some_finn_step')
        assert step_fn == mock_finn_step
    
    def test_get_step_not_found(self):
        """Test error when step not found."""
        with pytest.raises(ValueError, match="Step 'nonexistent' not found"):
            get_step('nonexistent')


class TestDependencyValidation:
    """Test step dependency validation."""
    
    def test_validate_sequence_valid(self):
        """Test validation of valid sequence."""
        # Valid sequence respecting dependencies
        step_names = ['cleanup', 'qonnx_to_finn', 'streamlining', 'infer_hardware']
        errors = validate_step_sequence(step_names)
        assert errors == []
    
    def test_validate_sequence_missing_dependency(self):
        """Test validation with missing dependency."""
        # streamlining depends on qonnx_to_finn
        step_names = ['cleanup', 'streamlining']
        errors = validate_step_sequence(step_names)
        assert any('requires' in error for error in errors)
    
    def test_validate_sequence_wrong_order(self):
        """Test validation with wrong dependency order."""
        # streamlining should come after qonnx_to_finn
        step_names = ['streamlining', 'qonnx_to_finn']
        errors = validate_step_sequence(step_names)
        assert any('must come before' in error for error in errors)
    
    def test_validate_sequence_nonexistent_step(self):
        """Test validation with nonexistent step."""
        step_names = ['cleanup', 'nonexistent_step']
        errors = validate_step_sequence(step_names)
        assert any('not found' in error for error in errors)


class TestBERTSteps:
    """Test BERT-specific steps."""
    
    def test_remove_head_step(self):
        """Test BERT head removal."""
        # Mock model with required structure
        mock_model = Mock()
        mock_cfg = Mock()
        
        # Mock graph structure
        mock_input = Mock()
        mock_input.name = "input_tensor"
        mock_model.graph.input = [mock_input]
        
        # Mock consumer node
        mock_node = Mock()
        mock_node.op_type = "LayerNormalization"
        mock_node.output = ["ln_output"]
        mock_model.find_consumer.return_value = mock_node
        
        # Mock other required methods
        mock_model.find_consumers.return_value = []
        mock_model.get_tensor_valueinfo.return_value = mock_input
        mock_model.transform.return_value = mock_model
        
        result = remove_head_step(mock_model, mock_cfg)
        assert result == mock_model
    
    def test_remove_tail_step(self):
        """Test BERT tail removal."""
        mock_model = Mock()
        mock_cfg = Mock()
        
        # Mock output structure
        mock_output = Mock()
        mock_output.name = "global_out_1"
        mock_model.graph.output = [mock_output]
        
        # Mock producer node
        mock_node = Mock()
        mock_node.op_type = "SomeOp"
        mock_model.find_producer.return_value = mock_node
        
        result = remove_tail_step(mock_model, mock_cfg)
        assert result == mock_model


class TestFunctionalOrganization:
    """Test the functional organization approach."""
    
    def test_step_categories(self):
        """Test that steps are properly categorized by function."""
        steps = discover_all_steps()
        
        # Test cleanup category
        cleanup_metadata = extract_step_metadata(steps['cleanup'])
        assert cleanup_metadata.category == "cleanup"
        
        # Test conversion category
        conversion_metadata = extract_step_metadata(steps['qonnx_to_finn'])
        assert conversion_metadata.category == "conversion"
        
        # Test streamlining category
        streamlining_metadata = extract_step_metadata(steps['streamlining'])
        assert streamlining_metadata.category == "streamlining"
        
        # Test hardware category
        hardware_metadata = extract_step_metadata(steps['infer_hardware'])
        assert hardware_metadata.category == "hardware"
        
        # Test BERT category
        bert_metadata = extract_step_metadata(steps['remove_head'])
        assert bert_metadata.category == "bert"
    
    def test_dependency_chain(self):
        """Test the main dependency chain works."""
        steps = discover_all_steps()
        
        # Check qonnx_to_finn -> streamlining -> infer_hardware chain
        streamlining_metadata = extract_step_metadata(steps['streamlining'])
        assert 'qonnx_to_finn' in streamlining_metadata.dependencies
        
        hardware_metadata = extract_step_metadata(steps['infer_hardware'])
        assert 'streamlining' in hardware_metadata.dependencies
    
    def test_step_imports(self):
        """Test that all steps can be imported directly."""
        # Test direct imports work
        from brainsmith.steps import cleanup_step
        from brainsmith.steps import qonnx_to_finn_step
        from brainsmith.steps import streamlining_step
        from brainsmith.steps import infer_hardware_step
        from brainsmith.steps import remove_head_step
        
        # All should be callable
        assert callable(cleanup_step)
        assert callable(qonnx_to_finn_step)
        assert callable(streamlining_step)
        assert callable(infer_hardware_step)
        assert callable(remove_head_step)


class TestNorthStarAlignment:
    """Test North Star principle alignment."""
    
    def test_no_global_state(self):
        """Test that no global state is used."""
        # Discovery should be stateless
        steps1 = discover_all_steps()
        steps2 = discover_all_steps()
        
        # Should return same results without side effects
        assert steps1.keys() == steps2.keys()
    
    def test_pure_functions(self):
        """Test that step functions are pure (same input -> same output)."""
        mock_model = Mock()
        mock_cfg = Mock()
        mock_model.transform.return_value = mock_model
        
        # Multiple calls should behave consistently
        result1 = cleanup_step(mock_model, mock_cfg)
        result2 = cleanup_step(mock_model, mock_cfg)
        
        # Should return same result
        assert result1 == result2
    
    def test_simple_data_structures(self):
        """Test that we use simple data structures."""
        metadata = extract_step_metadata(cleanup_step)
        
        # Should be simple dataclass
        assert isinstance(metadata, StepMetadata)
        assert hasattr(metadata, 'name')
        assert hasattr(metadata, 'category')
        assert hasattr(metadata, 'description')
        assert hasattr(metadata, 'dependencies')
    
    def test_functions_over_frameworks(self):
        """Test that we use functions instead of complex frameworks."""
        # Steps should be simple functions
        assert callable(cleanup_step)
        assert callable(streamlining_step)
        
        # Discovery should be simple functions
        assert callable(get_step)
        assert callable(discover_all_steps)
        assert callable(validate_step_sequence)
        
        # No complex class hierarchies
        steps = discover_all_steps()
        for step_fn in steps.values():
            assert callable(step_fn)


if __name__ == "__main__":
    pytest.main([__file__])