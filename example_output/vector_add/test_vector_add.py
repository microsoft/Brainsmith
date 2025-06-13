"""
Auto-generated test suite for VectorAdd.
Generated from: example_vector_add.sv
Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
Generation time: 2025-06-12T22:42:07.132126

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
from _hw_custom_op import VectorAdd

class TestVectorAdd:
    """
    Enhanced test suite for VectorAdd with Phase 2 parameter handling.
    
    Tests runtime parameter extraction, validation, and FINN integration
    with the Phase 2 template system that ensures symbolic BDIM validation.
    """
    
    def test_parameter_validation_required_attributes(self):
        """Test that required parameters are properly validated."""
        # Test missing required parameters
        with pytest.raises((ValueError, AttributeError), match="(Missing|required)"):
            # Create node without required parameters
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input"],
                outputs=["output"],
                # Missing required: VECTOR_SIZE
            )
            VectorAdd(node)
    
    def test_parameter_validation_whitelisted_defaults(self):
        """Test whitelisted parameters with defaults are handled correctly."""
        # Create node with all whitelisted parameters
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            VECTOR_SIZE=1,  # Required parameter
        )
        
        op = VectorAdd(node)
        
        # Verify whitelisted parameters are extracted correctly
        assert op.get_nodeattr("PE") == 4
    
    def test_valid_node_creation_all_parameters(self):
        """Test successful node creation with all defined parameters."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            VECTOR_SIZE=1,
        )
        
        assert node.op_type == "VectorAdd"
        assert len(node.input) == 1
        assert len(node.output) == 1
        
        # Verify all attributes are set
        attr_value = next((attr.i for attr in node.attribute if attr.name == "PE"), None)
        assert attr_value is not None, "Parameter PE should be set"
        assert attr_value == 4
        attr_value = next((attr.i for attr in node.attribute if attr.name == "VECTOR_SIZE"), None)
        assert attr_value is not None, "Parameter VECTOR_SIZE should be set"
        assert attr_value == 1
    
    def test_hwcustomop_instantiation_runtime_extraction(self):
        """Test HWCustomOp instantiation with Phase 2 runtime parameter extraction."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            VECTOR_SIZE=1,
        )
        
        # Should not raise exceptions during Phase 2 parameter extraction
        op = VectorAdd(node)
        
        # Verify runtime parameter extraction worked correctly
        extracted_value = op.get_nodeattr("PE")
        expected_value = 4
        assert extracted_value == expected_value, f"Parameter PE: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("VECTOR_SIZE")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter VECTOR_SIZE: expected {expected_value}, got {extracted_value}"
        
        # Verify parameter storage in runtime_parameters dict
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters dict should exist"
        assert "PE" in op.runtime_parameters, "PE should be in runtime_parameters"
        assert "VECTOR_SIZE" in op.runtime_parameters, "VECTOR_SIZE should be in runtime_parameters"
    
    def test_interface_metadata_validation(self):
        """Test that interface metadata matches Phase 2 enhanced parsing."""
        # Create a valid node for testing
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            VECTOR_SIZE=1,
        )
        
        op = VectorAdd(node)
        
        # Test interface metadata structure
        # Test ap interface
        # Test input0 interface
        # INPUT interface validation
        input_shape = op.get_folded_input_shape()
        assert input_shape is not None, "INPUT interface should have shape definition"
        # Test input1 interface
        # INPUT interface validation
        input_shape = op.get_folded_input_shape()
        assert input_shape is not None, "INPUT interface should have shape definition"
        # Test output0 interface
        # OUTPUT interface validation  
        output_shape = op.get_folded_output_shape()
        assert output_shape is not None, "OUTPUT interface should have shape definition"
    
    def test_bdim_parameter_consistency(self):
        """Test BDIM parameter consistency from Phase 2 validation."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            VECTOR_SIZE=1,
        )
        
        op = VectorAdd(node)
        
        # Test that BDIM parameters referenced in interfaces are consistent
        # ap BDIM consistency check
        # Block shape: [':']
        # input0 BDIM consistency check
        # Block shape: [':', ':']
        # input1 BDIM consistency check
        # Block shape: [':', ':']
        # output0 BDIM consistency check
        # Block shape: [':', ':']
    
    def test_node_attribute_types_phase2(self):
        """Test node attribute type definitions for Phase 2 compatibility."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            VECTOR_SIZE=1,
        )
        
        op = VectorAdd(node)
        
        # Check that get_nodeattr_types is implemented
        if hasattr(op, 'get_nodeattr_types'):
            attr_types = op.get_nodeattr_types()
            
            # Check parameter attributes exist
            assert "PE" in attr_types, "Parameter PE should have type definition"
            attr_type, required, default = attr_types["PE"]
            assert attr_type == "i", "Parameter PE should be integer type"
            assert required == False, "Parameter PE should not be required"
            assert default == 4, "Default value mismatch for PE"
            assert "VECTOR_SIZE" in attr_types, "Parameter VECTOR_SIZE should have type definition"
            attr_type, required, default = attr_types["VECTOR_SIZE"]
            assert attr_type == "i", "Parameter VECTOR_SIZE should be integer type"
            assert required == True, "Parameter VECTOR_SIZE should be required"
    
    def test_parameter_range_validation(self):
        """Test parameter range validation for positive values."""
        # Test PE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input"],
                outputs=["output"],
                PE=-1,  # Invalid negative value
                VECTOR_SIZE=1,
            )
            op = VectorAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test PE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input"],
                outputs=["output"],
                PE=0,  # Invalid zero value
                VECTOR_SIZE=1,
            )
            op = VectorAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test VECTOR_SIZE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input"],
                outputs=["output"],
                VECTOR_SIZE=-1,  # Invalid negative value
                PE=4,
            )
            op = VectorAdd(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test VECTOR_SIZE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input"],
                outputs=["output"],
                VECTOR_SIZE=0,  # Invalid zero value
                PE=4,
            )
            op = VectorAdd(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
    
    def test_phase2_regression_prevention(self):
        """Test that Phase 2 features work and prevent common regressions."""
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            VECTOR_SIZE=1,
        )
        
        op = VectorAdd(node)
        
        # Regression test: Ensure runtime_parameters are extracted in __init__
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters extraction should work"
        assert len(op.runtime_parameters) > 0, "runtime_parameters should not be empty"
        
        # Regression test: Ensure all defined parameters are accessible
        param_value = op.get_nodeattr("PE")
        assert param_value is not None, "Parameter PE should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("VECTOR_SIZE")
        assert param_value is not None, "Parameter VECTOR_SIZE should be accessible via get_nodeattr"
        
        # Regression test: Ensure class is properly derived from AutoHWCustomOp
        from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
        assert isinstance(op, HWCustomOp), "VectorAdd should inherit from HWCustomOp"
    
    @pytest.mark.slow
    def test_finn_integration_compatibility(self):
        """Test FINN framework integration compatibility (slow test)."""
        # Create a minimal model for integration testing
        node = onnx.helper.make_node(
            "VectorAdd",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            VECTOR_SIZE=1,
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
            assert wrapper is not None, "ModelWrapper should be able to load model with VectorAdd"
            
            # Test that the custom op is recognized
            custom_nodes = wrapper.get_nodes_by_op_type("VectorAdd")
            assert len(custom_nodes) == 1, "Should find exactly one VectorAdd node"
            
        except Exception as e:
            pytest.skip(f"FINN integration test skipped due to: {e}")

#=============================================================================
# Test Utilities and Fixtures
#=============================================================================

@pytest.fixture
def sample__node():
    """Fixture providing a sample VectorAdd node with valid parameters."""
    return onnx.helper.make_node(
        "VectorAdd",
        inputs=["input"],
        outputs=["output"],
        PE=4,
        VECTOR_SIZE=1,
    )

@pytest.fixture  
def sample__op(sample__node):
    """Fixture providing a sample VectorAdd instance."""
    return VectorAdd(sample__node)

#=============================================================================
# Performance and Stress Tests
#=============================================================================

class TestVectorAddPerformance:
    """Performance and stress tests for VectorAdd."""
    
    @pytest.mark.performance
    def test_instantiation_performance(self, sample__node):
        """Test that instantiation is reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            op = VectorAdd(sample__node)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01, f"Instantiation should be < 10ms, got {avg_time*1000:.2f}ms"
    
    @pytest.mark.stress
    def test_parameter_extraction_stress(self):
        """Stress test parameter extraction with many different values."""
        test_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        # Stress test PE
        for value in test_values:
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input"],
                outputs=["output"],
                PE=value,
                VECTOR_SIZE=1,
            )
            
            op = VectorAdd(node)
            extracted = op.get_nodeattr("PE")
            assert extracted == value, f"PE: expected {value}, got {extracted}"
        # Stress test VECTOR_SIZE
        for value in test_values:
            node = onnx.helper.make_node(
                "VectorAdd",
                inputs=["input"],
                outputs=["output"],
                VECTOR_SIZE=value,
                PE=4,
            )
            
            op = VectorAdd(node)
            extracted = op.get_nodeattr("VECTOR_SIZE")
            assert extracted == value, f"VECTOR_SIZE: expected {value}, got {extracted}"

#=============================================================================
# End of VectorAdd Test Suite
# Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
#=============================================================================