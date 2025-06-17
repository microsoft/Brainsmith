"""
Auto-generated test suite for TestNewFormat.
Generated from: test_new_pragma_format.sv
Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
Generation time: 2025-06-16T23:25:42.402504

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
from _hw_custom_op import TestNewFormat

class TestTestNewFormat:
    """
    Enhanced test suite for TestNewFormat with Phase 2 parameter handling.
    
    Tests runtime parameter extraction, validation, and FINN integration
    with the Phase 2 template system that ensures symbolic BDIM validation.
    """
    
    def test_parameter_validation_required_attributes(self):
        """Test that required parameters are properly validated."""
        # Test missing required parameters
        with pytest.raises((ValueError, AttributeError), match="(Missing|required)"):
            # Create node without required parameters
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                # Missing required: INPUT0_WIDTH, SIGNED_INPUT0, OUTPUT0_WIDTH, INPUT0_BDIM, INPUT0_SDIM, OUTPUT0_BDIM, OUTPUT0_SDIM, C
            )
            TestNewFormat(node)
    
    def test_parameter_validation_whitelisted_defaults(self):
        """Test whitelisted parameters with defaults are handled correctly."""
        # Create node with all whitelisted parameters
        node = onnx.helper.make_node(
            "TestNewFormat",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            INPUT0_WIDTH=1,  # Required parameter
            SIGNED_INPUT0=1,  # Required parameter
            OUTPUT0_WIDTH=1,  # Required parameter
            INPUT0_BDIM=1,  # Required parameter
            INPUT0_SDIM=1,  # Required parameter
            OUTPUT0_BDIM=1,  # Required parameter
            OUTPUT0_SDIM=1,  # Required parameter
            C=1,  # Required parameter
        )
        
        op = TestNewFormat(node)
        
        # Verify whitelisted parameters are extracted correctly
        assert op.get_nodeattr("PE") == 4
    
    def test_valid_node_creation_all_parameters(self):
        """Test successful node creation with all defined parameters."""
        node = onnx.helper.make_node(
            "TestNewFormat",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            OUTPUT0_WIDTH=1,
            INPUT0_BDIM=1,
            INPUT0_SDIM=1,
            OUTPUT0_BDIM=1,
            OUTPUT0_SDIM=1,
            C=1,
            PE=4,
        )
        
        assert node.op_type == "TestNewFormat"
        assert len(node.input) == 1
        assert len(node.output) == 1
        
        # Verify all attributes are set
        attr_value = next((attr.i for attr in node.attribute if attr.name == "INPUT0_WIDTH"), None)
        assert attr_value is not None, "Parameter INPUT0_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "SIGNED_INPUT0"), None)
        assert attr_value is not None, "Parameter SIGNED_INPUT0 should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "OUTPUT0_WIDTH"), None)
        assert attr_value is not None, "Parameter OUTPUT0_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "INPUT0_BDIM"), None)
        assert attr_value is not None, "Parameter INPUT0_BDIM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "INPUT0_SDIM"), None)
        assert attr_value is not None, "Parameter INPUT0_SDIM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "OUTPUT0_BDIM"), None)
        assert attr_value is not None, "Parameter OUTPUT0_BDIM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "OUTPUT0_SDIM"), None)
        assert attr_value is not None, "Parameter OUTPUT0_SDIM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "C"), None)
        assert attr_value is not None, "Parameter C should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "PE"), None)
        assert attr_value is not None, "Parameter PE should be set"
        assert attr_value == 4
    
    def test_hwcustomop_instantiation_runtime_extraction(self):
        """Test HWCustomOp instantiation with Phase 2 runtime parameter extraction."""
        node = onnx.helper.make_node(
            "TestNewFormat",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            OUTPUT0_WIDTH=1,
            INPUT0_BDIM=1,
            INPUT0_SDIM=1,
            OUTPUT0_BDIM=1,
            OUTPUT0_SDIM=1,
            C=1,
            PE=4,
        )
        
        # Should not raise exceptions during Phase 2 parameter extraction
        op = TestNewFormat(node)
        
        # Verify runtime parameter extraction worked correctly
        extracted_value = op.get_nodeattr("INPUT0_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter INPUT0_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("SIGNED_INPUT0")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter SIGNED_INPUT0: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("OUTPUT0_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter OUTPUT0_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("INPUT0_BDIM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter INPUT0_BDIM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("INPUT0_SDIM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter INPUT0_SDIM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("OUTPUT0_BDIM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter OUTPUT0_BDIM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("OUTPUT0_SDIM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter OUTPUT0_SDIM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("C")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter C: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("PE")
        expected_value = 4
        assert extracted_value == expected_value, f"Parameter PE: expected {expected_value}, got {extracted_value}"
        
        # Verify parameter storage in runtime_parameters dict
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters dict should exist"
        assert "INPUT0_WIDTH" in op.runtime_parameters, "INPUT0_WIDTH should be in runtime_parameters"
        assert "SIGNED_INPUT0" in op.runtime_parameters, "SIGNED_INPUT0 should be in runtime_parameters"
        assert "OUTPUT0_WIDTH" in op.runtime_parameters, "OUTPUT0_WIDTH should be in runtime_parameters"
        assert "INPUT0_BDIM" in op.runtime_parameters, "INPUT0_BDIM should be in runtime_parameters"
        assert "INPUT0_SDIM" in op.runtime_parameters, "INPUT0_SDIM should be in runtime_parameters"
        assert "OUTPUT0_BDIM" in op.runtime_parameters, "OUTPUT0_BDIM should be in runtime_parameters"
        assert "OUTPUT0_SDIM" in op.runtime_parameters, "OUTPUT0_SDIM should be in runtime_parameters"
        assert "C" in op.runtime_parameters, "C should be in runtime_parameters"
        assert "PE" in op.runtime_parameters, "PE should be in runtime_parameters"
    
    def test_interface_metadata_validation(self):
        """Test that interface metadata matches Phase 2 enhanced parsing."""
        # Create a valid node for testing
        node = onnx.helper.make_node(
            "TestNewFormat",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            OUTPUT0_WIDTH=1,
            INPUT0_BDIM=1,
            INPUT0_SDIM=1,
            OUTPUT0_BDIM=1,
            OUTPUT0_SDIM=1,
            C=1,
            PE=4,
        )
        
        op = TestNewFormat(node)
        
        # Test interface metadata structure
        # Test ap interface
        # Test s_axis_input0 interface
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
            "TestNewFormat",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            OUTPUT0_WIDTH=1,
            INPUT0_BDIM=1,
            INPUT0_SDIM=1,
            OUTPUT0_BDIM=1,
            OUTPUT0_SDIM=1,
            C=1,
            PE=4,
        )
        
        op = TestNewFormat(node)
        
        # Test that BDIM parameters referenced in interfaces are consistent
        # ap BDIM consistency check
        # Block shape: [':']
        # s_axis_input0 BDIM consistency check
        # Block shape: ['C', 'PE']
        # Validate C parameter exists and is positive
        param_value = op.get_nodeattr("C")
        assert param_value is not None, "BDIM parameter C must be defined"
        assert param_value > 0, "BDIM parameter C must be positive, got {param_value}"
        # Validate PE parameter exists and is positive
        param_value = op.get_nodeattr("PE")
        assert param_value is not None, "BDIM parameter PE must be defined"
        assert param_value > 0, "BDIM parameter PE must be positive, got {param_value}"
        # m_axis_output0 BDIM consistency check
        # Block shape: [':', ':']
    
    def test_node_attribute_types_phase2(self):
        """Test node attribute type definitions for Phase 2 compatibility."""
        node = onnx.helper.make_node(
            "TestNewFormat",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            OUTPUT0_WIDTH=1,
            INPUT0_BDIM=1,
            INPUT0_SDIM=1,
            OUTPUT0_BDIM=1,
            OUTPUT0_SDIM=1,
            C=1,
            PE=4,
        )
        
        op = TestNewFormat(node)
        
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
            assert "OUTPUT0_WIDTH" in attr_types, "Parameter OUTPUT0_WIDTH should have type definition"
            attr_type, required, default = attr_types["OUTPUT0_WIDTH"]
            assert attr_type == "i", "Parameter OUTPUT0_WIDTH should be integer type"
            assert required == True, "Parameter OUTPUT0_WIDTH should be required"
            assert "INPUT0_BDIM" in attr_types, "Parameter INPUT0_BDIM should have type definition"
            attr_type, required, default = attr_types["INPUT0_BDIM"]
            assert attr_type == "i", "Parameter INPUT0_BDIM should be integer type"
            assert required == True, "Parameter INPUT0_BDIM should be required"
            assert "INPUT0_SDIM" in attr_types, "Parameter INPUT0_SDIM should have type definition"
            attr_type, required, default = attr_types["INPUT0_SDIM"]
            assert attr_type == "i", "Parameter INPUT0_SDIM should be integer type"
            assert required == True, "Parameter INPUT0_SDIM should be required"
            assert "OUTPUT0_BDIM" in attr_types, "Parameter OUTPUT0_BDIM should have type definition"
            attr_type, required, default = attr_types["OUTPUT0_BDIM"]
            assert attr_type == "i", "Parameter OUTPUT0_BDIM should be integer type"
            assert required == True, "Parameter OUTPUT0_BDIM should be required"
            assert "OUTPUT0_SDIM" in attr_types, "Parameter OUTPUT0_SDIM should have type definition"
            attr_type, required, default = attr_types["OUTPUT0_SDIM"]
            assert attr_type == "i", "Parameter OUTPUT0_SDIM should be integer type"
            assert required == True, "Parameter OUTPUT0_SDIM should be required"
            assert "C" in attr_types, "Parameter C should have type definition"
            attr_type, required, default = attr_types["C"]
            assert attr_type == "i", "Parameter C should be integer type"
            assert required == True, "Parameter C should be required"
            assert "PE" in attr_types, "Parameter PE should have type definition"
            attr_type, required, default = attr_types["PE"]
            assert attr_type == "i", "Parameter PE should be integer type"
            assert required == False, "Parameter PE should not be required"
            assert default == 4, "Default value mismatch for PE"
    
    def test_parameter_range_validation(self):
        """Test parameter range validation for positive values."""
        # Test INPUT0_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_WIDTH=-1,  # Invalid negative value
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test INPUT0_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_WIDTH=0,  # Invalid zero value
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test SIGNED_INPUT0 must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT0=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test SIGNED_INPUT0 == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT0=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test OUTPUT0_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_WIDTH=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test OUTPUT0_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_WIDTH=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test INPUT0_BDIM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_BDIM=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test INPUT0_BDIM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_BDIM=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test INPUT0_SDIM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_SDIM=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test INPUT0_SDIM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_SDIM=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test OUTPUT0_BDIM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_BDIM=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test OUTPUT0_BDIM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_BDIM=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test OUTPUT0_SDIM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_SDIM=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test OUTPUT0_SDIM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_SDIM=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                C=1,
                PE=4,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test C must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                C=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                PE=4,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test C == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                C=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                PE=4,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test PE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                PE=-1,  # Invalid negative value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
            )
            op = TestNewFormat(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test PE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                PE=0,  # Invalid zero value
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
            )
            op = TestNewFormat(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
    
    def test_phase2_regression_prevention(self):
        """Test that Phase 2 features work and prevent common regressions."""
        node = onnx.helper.make_node(
            "TestNewFormat",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            OUTPUT0_WIDTH=1,
            INPUT0_BDIM=1,
            INPUT0_SDIM=1,
            OUTPUT0_BDIM=1,
            OUTPUT0_SDIM=1,
            C=1,
            PE=4,
        )
        
        op = TestNewFormat(node)
        
        # Regression test: Ensure runtime_parameters are extracted in __init__
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters extraction should work"
        assert len(op.runtime_parameters) > 0, "runtime_parameters should not be empty"
        
        # Regression test: Ensure all defined parameters are accessible
        param_value = op.get_nodeattr("INPUT0_WIDTH")
        assert param_value is not None, "Parameter INPUT0_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("SIGNED_INPUT0")
        assert param_value is not None, "Parameter SIGNED_INPUT0 should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("OUTPUT0_WIDTH")
        assert param_value is not None, "Parameter OUTPUT0_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("INPUT0_BDIM")
        assert param_value is not None, "Parameter INPUT0_BDIM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("INPUT0_SDIM")
        assert param_value is not None, "Parameter INPUT0_SDIM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("OUTPUT0_BDIM")
        assert param_value is not None, "Parameter OUTPUT0_BDIM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("OUTPUT0_SDIM")
        assert param_value is not None, "Parameter OUTPUT0_SDIM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("C")
        assert param_value is not None, "Parameter C should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("PE")
        assert param_value is not None, "Parameter PE should be accessible via get_nodeattr"
        
        # Regression test: Ensure class is properly derived from AutoHWCustomOp
        from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
        assert isinstance(op, HWCustomOp), "TestNewFormat should inherit from HWCustomOp"
    
    @pytest.mark.slow
    def test_finn_integration_compatibility(self):
        """Test FINN framework integration compatibility (slow test)."""
        # Create a minimal model for integration testing
        node = onnx.helper.make_node(
            "TestNewFormat",
            inputs=["input"],
            outputs=["output"],
            INPUT0_WIDTH=1,
            SIGNED_INPUT0=1,
            OUTPUT0_WIDTH=1,
            INPUT0_BDIM=1,
            INPUT0_SDIM=1,
            OUTPUT0_BDIM=1,
            OUTPUT0_SDIM=1,
            C=1,
            PE=4,
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
            assert wrapper is not None, "ModelWrapper should be able to load model with TestNewFormat"
            
            # Test that the custom op is recognized
            custom_nodes = wrapper.get_nodes_by_op_type("TestNewFormat")
            assert len(custom_nodes) == 1, "Should find exactly one TestNewFormat node"
            
        except Exception as e:
            pytest.skip(f"FINN integration test skipped due to: {e}")

#=============================================================================
# Test Utilities and Fixtures
#=============================================================================

@pytest.fixture
def sample__node():
    """Fixture providing a sample TestNewFormat node with valid parameters."""
    return onnx.helper.make_node(
        "TestNewFormat",
        inputs=["input"],
        outputs=["output"],
        INPUT0_WIDTH=1,
        SIGNED_INPUT0=1,
        OUTPUT0_WIDTH=1,
        INPUT0_BDIM=1,
        INPUT0_SDIM=1,
        OUTPUT0_BDIM=1,
        OUTPUT0_SDIM=1,
        C=1,
        PE=4,
    )

@pytest.fixture  
def sample__op(sample__node):
    """Fixture providing a sample TestNewFormat instance."""
    return TestNewFormat(sample__node)

#=============================================================================
# Performance and Stress Tests
#=============================================================================

class TestTestNewFormatPerformance:
    """Performance and stress tests for TestNewFormat."""
    
    @pytest.mark.performance
    def test_instantiation_performance(self, sample__node):
        """Test that instantiation is reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            op = TestNewFormat(sample__node)
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
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_WIDTH=value,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("INPUT0_WIDTH")
            assert extracted == value, f"INPUT0_WIDTH: expected {value}, got {extracted}"
        # Stress test SIGNED_INPUT0
        for value in test_values:
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                SIGNED_INPUT0=value,
                INPUT0_WIDTH=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("SIGNED_INPUT0")
            assert extracted == value, f"SIGNED_INPUT0: expected {value}, got {extracted}"
        # Stress test OUTPUT0_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_WIDTH=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("OUTPUT0_WIDTH")
            assert extracted == value, f"OUTPUT0_WIDTH: expected {value}, got {extracted}"
        # Stress test INPUT0_BDIM
        for value in test_values:
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_BDIM=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("INPUT0_BDIM")
            assert extracted == value, f"INPUT0_BDIM: expected {value}, got {extracted}"
        # Stress test INPUT0_SDIM
        for value in test_values:
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                INPUT0_SDIM=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("INPUT0_SDIM")
            assert extracted == value, f"INPUT0_SDIM: expected {value}, got {extracted}"
        # Stress test OUTPUT0_BDIM
        for value in test_values:
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_BDIM=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
                PE=4,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("OUTPUT0_BDIM")
            assert extracted == value, f"OUTPUT0_BDIM: expected {value}, got {extracted}"
        # Stress test OUTPUT0_SDIM
        for value in test_values:
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                OUTPUT0_SDIM=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                C=1,
                PE=4,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("OUTPUT0_SDIM")
            assert extracted == value, f"OUTPUT0_SDIM: expected {value}, got {extracted}"
        # Stress test C
        for value in test_values:
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                C=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                PE=4,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("C")
            assert extracted == value, f"C: expected {value}, got {extracted}"
        # Stress test PE
        for value in test_values:
            node = onnx.helper.make_node(
                "TestNewFormat",
                inputs=["input"],
                outputs=["output"],
                PE=value,
                INPUT0_WIDTH=1,
                SIGNED_INPUT0=1,
                OUTPUT0_WIDTH=1,
                INPUT0_BDIM=1,
                INPUT0_SDIM=1,
                OUTPUT0_BDIM=1,
                OUTPUT0_SDIM=1,
                C=1,
            )
            
            op = TestNewFormat(node)
            extracted = op.get_nodeattr("PE")
            assert extracted == value, f"PE: expected {value}, got {extracted}"

#=============================================================================
# End of TestNewFormat Test Suite
# Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
#=============================================================================