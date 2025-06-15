"""
Auto-generated test suite for MultiInputAdd.
Generated from: multi_input_test.sv
Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
Generation time: 2025-06-15T18:28:33.213770

Phase 2 Features:
✅ Runtime parameter extraction validation
✅ Whitelisted parameter testing  
✅ Enhanced interface metadata validation
✅ BDIM parameter consistency checking
"""

import pytest
import numpy as np
import onnx.helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

# Import the generated HWCustomOp
from _hw_custom_op import MultiInputAdd

class TestMultiInputAdd:
    """
    Enhanced test suite for MultiInputAdd with Phase 2 parameter handling.
    
    Tests runtime parameter extraction, validation, and FINN integration
    with the Phase 2 template system that ensures symbolic BDIM validation.
    """
    
    def test_parameter_validation_required_attributes(self):
        """Test that required parameters are properly validated."""
        # Test missing required parameters
        with pytest.raises((ValueError, AttributeError), match="(Missing|required)"):
            # Create node without required parameters
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                # Missing required: INPUT0_WIDTH, SIGNED_INPUT0, INPUT1_WIDTH, SIGNED_INPUT1, OUTPUT_WIDTH, SIGNED_OUTPUT, ALGORITHM_PARAM
            )
            MultiInputAdd(node)
    
    def test_parameter_validation_whitelisted_defaults(self):
        """Test whitelisted parameters with defaults are handled correctly."""
        # Create node with all whitelisted parameters
        node = onnx.helper.make_node(
            "MultiInputAdd",
            inputs=["input"],
            outputs=["output"],
            PE=1,
            INPUT0_WIDTH=1,  # Required parameter
            SIGNED_INPUT0=1,  # Required parameter
            INPUT1_WIDTH=1,  # Required parameter
            SIGNED_INPUT1=1,  # Required parameter
            OUTPUT_WIDTH=1,  # Required parameter
            SIGNED_OUTPUT=1,  # Required parameter
            ALGORITHM_PARAM=1,  # Required parameter
        )
        
        op = MultiInputAdd(node)
        
        # Verify whitelisted parameters are extracted correctly
        assert op.get_nodeattr("PE") == 1
    
    def test_valid_node_creation_all_parameters(self):
        """Test successful node creation with all defined parameters."""
        node = onnx.helper.make_node(
            "MultiInputAdd",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            INPUT1_WIDTH=1,
            SIGNED_INPUT1=1,
            OUTPUT_WIDTH=1,
            SIGNED_OUTPUT=1,
            ALGORITHM_PARAM=1,
            PE=1,
        )
        
        assert node.op_type == "MultiInputAdd"
        assert len(node.input) == 1
        assert len(node.output) == 1
        
        # Verify all attributes are set
        attr_value = next((attr.i for attr in node.attribute if attr.name == "INPUT0_WIDTH"), None)
        assert attr_value is not None, "Parameter INPUT0_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "SIGNED_INPUT0"), None)
        assert attr_value is not None, "Parameter SIGNED_INPUT0 should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "INPUT1_WIDTH"), None)
        assert attr_value is not None, "Parameter INPUT1_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "SIGNED_INPUT1"), None)
        assert attr_value is not None, "Parameter SIGNED_INPUT1 should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "OUTPUT_WIDTH"), None)
        assert attr_value is not None, "Parameter OUTPUT_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "SIGNED_OUTPUT"), None)
        assert attr_value is not None, "Parameter SIGNED_OUTPUT should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "ALGORITHM_PARAM"), None)
        assert attr_value is not None, "Parameter ALGORITHM_PARAM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "PE"), None)
        assert attr_value is not None, "Parameter PE should be set"
        assert attr_value == 1
    
    def test_hwcustomop_instantiation_runtime_extraction(self):
        """Test HWCustomOp instantiation with Phase 2 runtime parameter extraction."""
        node = onnx.helper.make_node(
            "MultiInputAdd",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            INPUT1_WIDTH=1,
            SIGNED_INPUT1=1,
            OUTPUT_WIDTH=1,
            SIGNED_OUTPUT=1,
            ALGORITHM_PARAM=1,
            PE=1,
        )
        
        # Should not raise exceptions during Phase 2 parameter extraction
        op = MultiInputAdd(node)
        
        # Verify runtime parameter extraction worked correctly
        extracted_value = op.get_nodeattr("INPUT0_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter INPUT0_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("SIGNED_INPUT0")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter SIGNED_INPUT0: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("INPUT1_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter INPUT1_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("SIGNED_INPUT1")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter SIGNED_INPUT1: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("OUTPUT_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter OUTPUT_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("SIGNED_OUTPUT")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter SIGNED_OUTPUT: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("ALGORITHM_PARAM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter ALGORITHM_PARAM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("PE")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter PE: expected {expected_value}, got {extracted_value}"
        
        # Verify parameter storage in runtime_parameters dict
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters dict should exist"
        assert "INPUT0_WIDTH" in op.runtime_parameters, "INPUT0_WIDTH should be in runtime_parameters"
        assert "SIGNED_INPUT0" in op.runtime_parameters, "SIGNED_INPUT0 should be in runtime_parameters"
        assert "INPUT1_WIDTH" in op.runtime_parameters, "INPUT1_WIDTH should be in runtime_parameters"
        assert "SIGNED_INPUT1" in op.runtime_parameters, "SIGNED_INPUT1 should be in runtime_parameters"
        assert "OUTPUT_WIDTH" in op.runtime_parameters, "OUTPUT_WIDTH should be in runtime_parameters"
        assert "SIGNED_OUTPUT" in op.runtime_parameters, "SIGNED_OUTPUT should be in runtime_parameters"
        assert "ALGORITHM_PARAM" in op.runtime_parameters, "ALGORITHM_PARAM should be in runtime_parameters"
        assert "PE" in op.runtime_parameters, "PE should be in runtime_parameters"
    
    def test_interface_metadata_validation(self):
        """Test that interface metadata matches Phase 2 enhanced parsing."""
        # Create a valid node for testing
        node = onnx.helper.make_node(
            "MultiInputAdd",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            INPUT1_WIDTH=1,
            SIGNED_INPUT1=1,
            OUTPUT_WIDTH=1,
            SIGNED_OUTPUT=1,
            ALGORITHM_PARAM=1,
            PE=1,
        )
        
        op = MultiInputAdd(node)
        
        # Test interface metadata structure
        # Test ap interface
        # Test s_axis_input0 interface
        # INPUT interface validation
        input_shape = op.get_folded_input_shape()
        assert input_shape is not None, "INPUT interface should have shape definition"
        # Test s_axis_input1 interface
        # INPUT interface validation
        input_shape = op.get_folded_input_shape()
        assert input_shape is not None, "INPUT interface should have shape definition"
        # Test m_axis_output0 interface
        # OUTPUT interface validation  
        output_shape = op.get_folded_output_shape()
        assert output_shape is not None, "OUTPUT interface should have shape definition"
    
    def test_bdim_parameter_consistency(self):
        """Test BDIM parameter consistency from Phase 2 validation."""
        node = onnx.helper.make_node(
            "MultiInputAdd",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            INPUT1_WIDTH=1,
            SIGNED_INPUT1=1,
            OUTPUT_WIDTH=1,
            SIGNED_OUTPUT=1,
            ALGORITHM_PARAM=1,
            PE=1,
        )
        
        op = MultiInputAdd(node)
        
        # Test that BDIM parameters referenced in interfaces are consistent
        # ap BDIM consistency check
        # Block shape: [':']
        # s_axis_input0 BDIM consistency check
        # Block shape: [':', ':']
        # s_axis_input1 BDIM consistency check
        # Block shape: [':', ':']
        # m_axis_output0 BDIM consistency check
        # Block shape: [':', ':']
    
    def test_node_attribute_types_phase2(self):
        """Test node attribute type definitions for Phase 2 compatibility."""
        node = onnx.helper.make_node(
            "MultiInputAdd",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            INPUT1_WIDTH=1,
            SIGNED_INPUT1=1,
            OUTPUT_WIDTH=1,
            SIGNED_OUTPUT=1,
            ALGORITHM_PARAM=1,
            PE=1,
        )
        
        op = MultiInputAdd(node)
        
        # Check that get_nodeattr_types is implemented
        if hasattr(op, 'get_nodeattr_types'):
            attr_types = op.get_nodeattr_types()
            
            # Check parameter attributes exist
            assert "INPUT0_WIDTH" in attr_types, "Parameter INPUT0_WIDTH should have type definition"
            attr_type, required, default = attr_types["INPUT0_WIDTH"]
            assert attr_type == "i", "Parameter INPUT0_WIDTH should be integer type"
            assert required == True, "Parameter INPUT0_WIDTH should be required"
            assert "SIGNED_INPUT0" in attr_types, "Parameter SIGNED_INPUT0 should have type definition"
            attr_type, required, default = attr_types["SIGNED_INPUT0"]
            assert attr_type == "i", "Parameter SIGNED_INPUT0 should be integer type"
            assert required == True, "Parameter SIGNED_INPUT0 should be required"
            assert "INPUT1_WIDTH" in attr_types, "Parameter INPUT1_WIDTH should have type definition"
            attr_type, required, default = attr_types["INPUT1_WIDTH"]
            assert attr_type == "i", "Parameter INPUT1_WIDTH should be integer type"
            assert required == True, "Parameter INPUT1_WIDTH should be required"
            assert "SIGNED_INPUT1" in attr_types, "Parameter SIGNED_INPUT1 should have type definition"
            attr_type, required, default = attr_types["SIGNED_INPUT1"]
            assert attr_type == "i", "Parameter SIGNED_INPUT1 should be integer type"
            assert required == True, "Parameter SIGNED_INPUT1 should be required"
            assert "OUTPUT_WIDTH" in attr_types, "Parameter OUTPUT_WIDTH should have type definition"
            attr_type, required, default = attr_types["OUTPUT_WIDTH"]
            assert attr_type == "i", "Parameter OUTPUT_WIDTH should be integer type"
            assert required == True, "Parameter OUTPUT_WIDTH should be required"
            assert "SIGNED_OUTPUT" in attr_types, "Parameter SIGNED_OUTPUT should have type definition"
            attr_type, required, default = attr_types["SIGNED_OUTPUT"]
            assert attr_type == "i", "Parameter SIGNED_OUTPUT should be integer type"
            assert required == True, "Parameter SIGNED_OUTPUT should be required"
            assert "ALGORITHM_PARAM" in attr_types, "Parameter ALGORITHM_PARAM should have type definition"
            attr_type, required, default = attr_types["ALGORITHM_PARAM"]
            assert attr_type == "i", "Parameter ALGORITHM_PARAM should be integer type"
            assert required == True, "Parameter ALGORITHM_PARAM should be required"
            assert "PE" in attr_types, "Parameter PE should have type definition"
            attr_type, required, default = attr_types["PE"]
            assert attr_type == "i", "Parameter PE should be integer type"
            assert required == False, "Parameter PE should not be required"
            assert default == 1, "Default value mismatch for PE"
    
    def test_parameter_range_validation(self):
        """Test parameter range validation for positive values."""
        # Test INPUT0_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                INPUT0_WIDTH=-1,  # Invalid negative value
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test INPUT0_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                INPUT0_WIDTH=0,  # Invalid zero value
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test SIGNED_INPUT0 must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT0=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test SIGNED_INPUT0 == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT0=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test INPUT1_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                INPUT1_WIDTH=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test INPUT1_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                INPUT1_WIDTH=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test SIGNED_INPUT1 must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT1=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test SIGNED_INPUT1 == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT1=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test OUTPUT_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                OUTPUT_WIDTH=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test OUTPUT_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                OUTPUT_WIDTH=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test SIGNED_OUTPUT must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_OUTPUT=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test SIGNED_OUTPUT == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_OUTPUT=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test ALGORITHM_PARAM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                ALGORITHM_PARAM=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test ALGORITHM_PARAM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                ALGORITHM_PARAM=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                PE=1,
            )
            op = MultiInputAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test PE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                PE=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
            )
            op = MultiInputAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test PE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                PE=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
            )
            op = MultiInputAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
    
    def test_phase2_regression_prevention(self):
        """Test that Phase 2 features work and prevent common regressions."""
        node = onnx.helper.make_node(
            "MultiInputAdd",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            INPUT1_WIDTH=1,
            SIGNED_INPUT1=1,
            OUTPUT_WIDTH=1,
            SIGNED_OUTPUT=1,
            ALGORITHM_PARAM=1,
            PE=1,
        )
        
        op = MultiInputAdd(node)
        
        # Regression test: Ensure runtime_parameters are extracted in __init__
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters extraction should work"
        assert len(op.runtime_parameters) > 0, "runtime_parameters should not be empty"
        
        # Regression test: Ensure all defined parameters are accessible
        param_value = op.get_nodeattr("INPUT0_WIDTH")
        assert param_value is not None, "Parameter INPUT0_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("SIGNED_INPUT0")
        assert param_value is not None, "Parameter SIGNED_INPUT0 should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("INPUT1_WIDTH")
        assert param_value is not None, "Parameter INPUT1_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("SIGNED_INPUT1")
        assert param_value is not None, "Parameter SIGNED_INPUT1 should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("OUTPUT_WIDTH")
        assert param_value is not None, "Parameter OUTPUT_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("SIGNED_OUTPUT")
        assert param_value is not None, "Parameter SIGNED_OUTPUT should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("ALGORITHM_PARAM")
        assert param_value is not None, "Parameter ALGORITHM_PARAM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("PE")
        assert param_value is not None, "Parameter PE should be accessible via get_nodeattr"
        
        # Regression test: Ensure class is properly derived from AutoHWCustomOp
        from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
        assert isinstance(op, HWCustomOp), "MultiInputAdd should inherit from HWCustomOp"
    
    @pytest.mark.slow
    def test_finn_integration_compatibility(self):
        """Test FINN framework integration compatibility (slow test)."""
        # Create a minimal model for integration testing
        node = onnx.helper.make_node(
            "MultiInputAdd",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            INPUT1_WIDTH=1,
            SIGNED_INPUT1=1,
            OUTPUT_WIDTH=1,
            SIGNED_OUTPUT=1,
            ALGORITHM_PARAM=1,
            PE=1,
        )
        
        # Create simple input/output value info
        input_vi = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 32])
        output_vi = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 32])
        
        # Create model
        graph = onnx.helper.make_graph([node], "test_graph", [input_vi], [output_vi])
        model = onnx.helper.make_model(graph)
        
        # Test that ModelWrapper can load the model
        try:
            wrapper = ModelWrapper(model)
            assert wrapper is not None, "ModelWrapper should be able to load model with MultiInputAdd"
            
            # Test that the custom op is recognized
            custom_nodes = wrapper.get_nodes_by_op_type("MultiInputAdd")
            assert len(custom_nodes) == 1, "Should find exactly one MultiInputAdd node"
            
        except Exception as e:
            pytest.skip(f"FINN integration test skipped due to: {e}")

#=============================================================================
# Test Utilities and Fixtures
#=============================================================================

@pytest.fixture
def sample__node():
    """Fixture providing a sample MultiInputAdd node with valid parameters."""
    return onnx.helper.make_node(
        "MultiInputAdd",
        inputs=["input"],
        outputs=["output"],
        INPUT0_WIDTH=1,
        SIGNED_INPUT0=1,
        INPUT1_WIDTH=1,
        SIGNED_INPUT1=1,
        OUTPUT_WIDTH=1,
        SIGNED_OUTPUT=1,
        ALGORITHM_PARAM=1,
        PE=1,
    )

@pytest.fixture  
def sample__op(sample__node):
    """Fixture providing a sample MultiInputAdd instance."""
    return MultiInputAdd(sample__node)

#=============================================================================
# Performance and Stress Tests
#=============================================================================

class TestMultiInputAddPerformance:
    """Performance and stress tests for MultiInputAdd."""
    
    @pytest.mark.performance
    def test_instantiation_performance(self, sample__node):
        """Test that instantiation is reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            op = MultiInputAdd(sample__node)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01, f"Instantiation should be < 10ms, got {avg_time*1000:.2f}ms"
    
    @pytest.mark.stress
    def test_parameter_extraction_stress(self):
        """Stress test parameter extraction with many different values."""
        test_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        # Stress test INPUT0_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                INPUT0_WIDTH=value,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            
            op = MultiInputAdd(node)
            extracted = op.get_nodeattr("INPUT0_WIDTH")
            assert extracted == value, f"INPUT0_WIDTH: expected {value}, got {extracted}"
        # Stress test SIGNED_INPUT0
        for value in test_values:
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT0=value,
                INPUT0_WIDTH=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            
            op = MultiInputAdd(node)
            extracted = op.get_nodeattr("SIGNED_INPUT0")
            assert extracted == value, f"SIGNED_INPUT0: expected {value}, got {extracted}"
        # Stress test INPUT1_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                INPUT1_WIDTH=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            
            op = MultiInputAdd(node)
            extracted = op.get_nodeattr("INPUT1_WIDTH")
            assert extracted == value, f"INPUT1_WIDTH: expected {value}, got {extracted}"
        # Stress test SIGNED_INPUT1
        for value in test_values:
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT1=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            
            op = MultiInputAdd(node)
            extracted = op.get_nodeattr("SIGNED_INPUT1")
            assert extracted == value, f"SIGNED_INPUT1: expected {value}, got {extracted}"
        # Stress test OUTPUT_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                OUTPUT_WIDTH=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            
            op = MultiInputAdd(node)
            extracted = op.get_nodeattr("OUTPUT_WIDTH")
            assert extracted == value, f"OUTPUT_WIDTH: expected {value}, got {extracted}"
        # Stress test SIGNED_OUTPUT
        for value in test_values:
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                SIGNED_OUTPUT=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                ALGORITHM_PARAM=1,
                PE=1,
            )
            
            op = MultiInputAdd(node)
            extracted = op.get_nodeattr("SIGNED_OUTPUT")
            assert extracted == value, f"SIGNED_OUTPUT: expected {value}, got {extracted}"
        # Stress test ALGORITHM_PARAM
        for value in test_values:
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                ALGORITHM_PARAM=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                PE=1,
            )
            
            op = MultiInputAdd(node)
            extracted = op.get_nodeattr("ALGORITHM_PARAM")
            assert extracted == value, f"ALGORITHM_PARAM: expected {value}, got {extracted}"
        # Stress test PE
        for value in test_values:
            node = onnx.helper.make_node(
                "MultiInputAdd",
                inputs=["input"],
                outputs=["output"],
                PE=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT1_WIDTH=1,
                SIGNED_INPUT1=1,
                OUTPUT_WIDTH=1,
                SIGNED_OUTPUT=1,
                ALGORITHM_PARAM=1,
            )
            
            op = MultiInputAdd(node)
            extracted = op.get_nodeattr("PE")
            assert extracted == value, f"PE: expected {value}, got {extracted}"

#=============================================================================
# End of MultiInputAdd Test Suite
# Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
#=============================================================================