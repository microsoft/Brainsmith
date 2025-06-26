"""
Unit tests for StepResolver.
"""

import pytest
from brainsmith.core_v3.phase3.step_resolver import StepResolver, InputType, OutputType


class TestStepResolver:
    """Test StepResolver class."""
    
    @pytest.fixture
    def resolver(self):
        """Create a StepResolver instance for testing."""
        return StepResolver()
    
    @pytest.fixture
    def custom_steps(self):
        """Sample custom step list for testing."""
        return [
            "custom_cleanup",
            "custom_convert", 
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_hw_codegen",
            "step_create_stitched_ip"
        ]
    
    def test_resolver_creation(self, resolver):
        """Test creating StepResolver."""
        assert resolver is not None
        assert len(resolver.get_standard_steps()) > 0
        assert len(resolver.get_supported_input_types()) > 0
        assert len(resolver.get_supported_output_types()) > 0
    
    def test_resolve_step_name_valid(self, resolver, custom_steps):
        """Test resolving valid step names."""
        step = resolver.resolve_step_specification("step_create_dataflow_partition", custom_steps)
        assert step == "step_create_dataflow_partition"
    
    def test_resolve_step_name_invalid(self, resolver, custom_steps):
        """Test resolving invalid step names."""
        with pytest.raises(ValueError, match="Step 'invalid_step' not found"):
            resolver.resolve_step_specification("invalid_step", custom_steps)
    
    def test_resolve_step_index_valid(self, resolver, custom_steps):
        """Test resolving valid step indices."""
        step = resolver.resolve_step_specification(2, custom_steps)
        assert step == 2
        
        step = resolver.resolve_step_specification(0, custom_steps)
        assert step == 0
        
        step = resolver.resolve_step_specification(len(custom_steps) - 1, custom_steps)
        assert step == len(custom_steps) - 1
    
    def test_resolve_step_index_out_of_range(self, resolver, custom_steps):
        """Test resolving out-of-range step indices."""
        with pytest.raises(ValueError, match="Step index .* out of range"):
            resolver.resolve_step_specification(100, custom_steps)
        
        with pytest.raises(ValueError, match="Step index .* out of range"):
            resolver.resolve_step_specification(-1, custom_steps)
    
    def test_resolve_step_none(self, resolver, custom_steps):
        """Test resolving None step specification."""
        step = resolver.resolve_step_specification(None, custom_steps)
        assert step is None
    
    def test_resolve_step_invalid_type(self, resolver, custom_steps):
        """Test resolving invalid step specification types."""
        with pytest.raises(ValueError, match="Invalid step specification"):
            resolver.resolve_step_specification([], custom_steps)
        
        with pytest.raises(ValueError, match="Invalid step specification"):
            resolver.resolve_step_specification({}, custom_steps)
    
    def test_resolve_semantic_input_types(self, resolver):
        """Test resolving semantic input types."""
        step = resolver.resolve_step_specification(None, semantic_type=InputType.ONNX)
        assert step == "custom_step_cleanup"
        
        step = resolver.resolve_step_specification(None, semantic_type=InputType.HWGRAPH)
        assert step == "step_create_dataflow_partition"
        
        step = resolver.resolve_step_specification(None, semantic_type="onnx")
        assert step == "custom_step_cleanup"
    
    def test_resolve_semantic_output_types(self, resolver):
        """Test resolving semantic output types."""
        step = resolver.resolve_step_specification(None, semantic_type=OutputType.FINN)
        assert step == "custom_streamlining_step"
        
        step = resolver.resolve_step_specification(None, semantic_type=OutputType.RTL)
        assert step == "step_hw_codegen"
        
        step = resolver.resolve_step_specification(None, semantic_type="rtl")
        assert step == "step_hw_codegen"
    
    def test_resolve_semantic_invalid_type(self, resolver):
        """Test resolving invalid semantic types."""
        with pytest.raises(ValueError, match="Invalid semantic type"):
            resolver.resolve_step_specification(None, semantic_type="invalid_type")
    
    def test_resolve_step_range_basic(self, resolver, custom_steps):
        """Test resolving basic step ranges."""
        start, stop = resolver.resolve_step_range(
            start_step="custom_convert",
            stop_step="step_hw_codegen",
            step_list=custom_steps
        )
        assert start == "custom_convert"
        assert stop == "step_hw_codegen"
    
    def test_resolve_step_range_indices(self, resolver, custom_steps):
        """Test resolving step ranges with indices."""
        start, stop = resolver.resolve_step_range(
            start_step=1,
            stop_step=4,
            step_list=custom_steps
        )
        assert start == 1
        assert stop == 4
    
    def test_resolve_step_range_semantic_types(self, resolver):
        """Test resolving step ranges with semantic types."""
        start, stop = resolver.resolve_step_range(
            input_type=InputType.HWGRAPH,
            output_type=OutputType.RTL
        )
        assert start == "step_create_dataflow_partition"
        assert stop == "step_hw_codegen"
    
    def test_resolve_step_range_mixed(self, resolver, custom_steps):
        """Test resolving step ranges with mixed specifications."""
        start, stop = resolver.resolve_step_range(
            start_step="custom_convert",
            stop_step=4,
            step_list=custom_steps
        )
        assert start == "custom_convert"
        assert stop == 4
    
    def test_resolve_step_range_invalid_order_indices(self, resolver, custom_steps):
        """Test resolving step ranges with invalid ordering (indices)."""
        with pytest.raises(ValueError, match="Start step index .* must be <= stop step index"):
            resolver.resolve_step_range(
                start_step=4,
                stop_step=1,
                step_list=custom_steps
            )
    
    def test_resolve_step_range_invalid_order_names(self, resolver, custom_steps):
        """Test resolving step ranges with invalid ordering (names)."""
        with pytest.raises(ValueError, match="Start step .* must come before stop step"):
            resolver.resolve_step_range(
                start_step="step_hw_codegen",
                stop_step="custom_convert",
                step_list=custom_steps
            )
    
    def test_resolve_step_range_none_values(self, resolver, custom_steps):
        """Test resolving step ranges with None values."""
        start, stop = resolver.resolve_step_range(
            start_step=None,
            stop_step=None,
            step_list=custom_steps
        )
        assert start is None
        assert stop is None
        
        start, stop = resolver.resolve_step_range(
            start_step="custom_convert",
            stop_step=None,
            step_list=custom_steps
        )
        assert start == "custom_convert"
        assert stop is None
    
    def test_get_step_slice_names(self, resolver, custom_steps):
        """Test getting step slices with step names."""
        sliced = resolver.get_step_slice(
            custom_steps,
            start_step="custom_convert",
            stop_step="step_specialize_layers"
        )
        expected = ["custom_convert", "step_create_dataflow_partition", "step_specialize_layers"]
        assert sliced == expected
    
    def test_get_step_slice_indices(self, resolver, custom_steps):
        """Test getting step slices with indices."""
        sliced = resolver.get_step_slice(
            custom_steps,
            start_step=1,
            stop_step=3
        )
        expected = ["custom_convert", "step_create_dataflow_partition", "step_specialize_layers"]
        assert sliced == expected
    
    def test_get_step_slice_start_only(self, resolver, custom_steps):
        """Test getting step slices with start only."""
        sliced = resolver.get_step_slice(
            custom_steps,
            start_step="step_create_dataflow_partition",
            stop_step=None
        )
        expected = ["step_create_dataflow_partition", "step_specialize_layers", "step_hw_codegen", "step_create_stitched_ip"]
        assert sliced == expected
    
    def test_get_step_slice_stop_only(self, resolver, custom_steps):
        """Test getting step slices with stop only."""
        sliced = resolver.get_step_slice(
            custom_steps,
            start_step=None,
            stop_step="step_specialize_layers"
        )
        expected = ["custom_cleanup", "custom_convert", "step_create_dataflow_partition", "step_specialize_layers"]
        assert sliced == expected
    
    def test_get_step_slice_full_range(self, resolver, custom_steps):
        """Test getting step slices with full range."""
        sliced = resolver.get_step_slice(
            custom_steps,
            start_step=None,
            stop_step=None
        )
        assert sliced == custom_steps
    
    def test_get_step_slice_single_step(self, resolver, custom_steps):
        """Test getting step slices with single step."""
        sliced = resolver.get_step_slice(
            custom_steps,
            start_step="step_specialize_layers",
            stop_step="step_specialize_layers"
        )
        expected = ["step_specialize_layers"]
        assert sliced == expected
    
    def test_input_output_type_enums(self):
        """Test InputType and OutputType enums."""
        # Test InputType values
        assert InputType.ONNX.value == "onnx"
        assert InputType.QONNX.value == "qonnx"
        assert InputType.FINN.value == "finn"
        assert InputType.HWGRAPH.value == "hwgraph"
        
        # Test OutputType values
        assert OutputType.QONNX.value == "qonnx"
        assert OutputType.FINN.value == "finn"
        assert OutputType.HWGRAPH.value == "hwgraph"
        assert OutputType.RTL.value == "rtl"
        assert OutputType.IP.value == "ip"
        assert OutputType.BITSTREAM.value == "bitstream"
    
    def test_standard_steps_not_empty(self, resolver):
        """Test that standard steps list is not empty."""
        steps = resolver.get_standard_steps()
        assert len(steps) > 0
        assert all(isinstance(step, str) for step in steps)
    
    def test_supported_types_coverage(self, resolver):
        """Test that all supported types have mappings."""
        input_types = resolver.get_supported_input_types()
        output_types = resolver.get_supported_output_types()
        
        # All input types should resolve to steps
        for input_type in input_types:
            step = resolver.resolve_step_specification(None, semantic_type=input_type)
            assert step is not None
            assert isinstance(step, str)
        
        # All output types should resolve to steps  
        for output_type in output_types:
            step = resolver.resolve_step_specification(None, semantic_type=output_type)
            assert step is not None
            assert isinstance(step, str)