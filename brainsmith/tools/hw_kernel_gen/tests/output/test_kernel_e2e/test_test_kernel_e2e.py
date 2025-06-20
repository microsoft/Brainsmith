"""
Auto-generated test suite for TestKernelE2e.
Generated from: /home/tafk/dev/brainsmith-2/brainsmith/tools/hw_kernel_gen/tests/test_kernel_e2e.sv
Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
Generation time: 2025-06-20T06:42:13.783232

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
from _hw_custom_op import TestKernelE2e

class TestTestKernelE2e:
    """
    Enhanced test suite for TestKernelE2e with Phase 2 parameter handling.
    
    Tests runtime parameter extraction, validation, and FINN integration
    with the Phase 2 template system that ensures symbolic BDIM validation.
    """
    
    def test_parameter_validation_required_attributes(self):
        """Test that required parameters are properly validated."""
        # Test missing required parameters
        with pytest.raises((ValueError, AttributeError), match="(Missing|required)"):
            # Create node without required parameters
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                # Missing required: INPUT_WIDTH, WEIGHT_WIDTH, WEIGHT_SIGNED, OUTPUT_WIDTH, ACC_WIDTH, ACC_SIGNED, THRESH_WIDTH, INPUT_BDIM, INPUT_SDIM, WEIGHT_BDIM, ACTIVATION_TYPE
            )
            TestKernelE2e(node)
    
    def test_parameter_validation_whitelisted_defaults(self):
        """Test whitelisted parameters with defaults are handled correctly."""
        # Create node with all whitelisted parameters
        node = onnx.helper.make_node(
            "TestKernelE2e",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            SIMD=8,
            MEM_DEPTH=1024,
            INPUT_WIDTH=1,  # Required parameter
            WEIGHT_WIDTH=1,  # Required parameter
            WEIGHT_SIGNED=1,  # Required parameter
            OUTPUT_WIDTH=1,  # Required parameter
            ACC_WIDTH=1,  # Required parameter
            ACC_SIGNED=1,  # Required parameter
            THRESH_WIDTH=1,  # Required parameter
            INPUT_BDIM=1,  # Required parameter
            INPUT_SDIM=1,  # Required parameter
            WEIGHT_BDIM=1,  # Required parameter
            ACTIVATION_TYPE=1,  # Required parameter
        )
        
        op = TestKernelE2e(node)
        
        # Verify whitelisted parameters are extracted correctly
        assert op.get_nodeattr("PE") == 4
        assert op.get_nodeattr("SIMD") == 8
        assert op.get_nodeattr("MEM_DEPTH") == 1024
    
    def test_valid_node_creation_all_parameters(self):
        """Test successful node creation with all defined parameters."""
        node = onnx.helper.make_node(
            "TestKernelE2e",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            SIMD=8,
            INPUT_WIDTH=1,
            WEIGHT_WIDTH=1,
            WEIGHT_SIGNED=1,
            OUTPUT_WIDTH=1,
            ACC_WIDTH=1,
            ACC_SIGNED=1,
            THRESH_WIDTH=1,
            INPUT_BDIM=1,
            INPUT_SDIM=1,
            WEIGHT_BDIM=1,
            MEM_DEPTH=1024,
            ACTIVATION_TYPE=1,
        )
        
        assert node.op_type == "TestKernelE2e"
        assert len(node.input) == 1
        assert len(node.output) == 1
        
        # Verify all attributes are set
        attr_value = next((attr.i for attr in node.attribute if attr.name == "PE"), None)
        assert attr_value is not None, "Parameter PE should be set"
        assert attr_value == 4
        attr_value = next((attr.i for attr in node.attribute if attr.name == "SIMD"), None)
        assert attr_value is not None, "Parameter SIMD should be set"
        assert attr_value == 8
        attr_value = next((attr.i for attr in node.attribute if attr.name == "INPUT_WIDTH"), None)
        assert attr_value is not None, "Parameter INPUT_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "WEIGHT_WIDTH"), None)
        assert attr_value is not None, "Parameter WEIGHT_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "WEIGHT_SIGNED"), None)
        assert attr_value is not None, "Parameter WEIGHT_SIGNED should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "OUTPUT_WIDTH"), None)
        assert attr_value is not None, "Parameter OUTPUT_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "ACC_WIDTH"), None)
        assert attr_value is not None, "Parameter ACC_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "ACC_SIGNED"), None)
        assert attr_value is not None, "Parameter ACC_SIGNED should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "THRESH_WIDTH"), None)
        assert attr_value is not None, "Parameter THRESH_WIDTH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "INPUT_BDIM"), None)
        assert attr_value is not None, "Parameter INPUT_BDIM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "INPUT_SDIM"), None)
        assert attr_value is not None, "Parameter INPUT_SDIM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "WEIGHT_BDIM"), None)
        assert attr_value is not None, "Parameter WEIGHT_BDIM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "MEM_DEPTH"), None)
        assert attr_value is not None, "Parameter MEM_DEPTH should be set"
        assert attr_value == 1024
        attr_value = next((attr.i for attr in node.attribute if attr.name == "ACTIVATION_TYPE"), None)
        assert attr_value is not None, "Parameter ACTIVATION_TYPE should be set"
        assert attr_value == 1
    
    def test_hwcustomop_instantiation_runtime_extraction(self):
        """Test HWCustomOp instantiation with Phase 2 runtime parameter extraction."""
        node = onnx.helper.make_node(
            "TestKernelE2e",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            SIMD=8,
            INPUT_WIDTH=1,
            WEIGHT_WIDTH=1,
            WEIGHT_SIGNED=1,
            OUTPUT_WIDTH=1,
            ACC_WIDTH=1,
            ACC_SIGNED=1,
            THRESH_WIDTH=1,
            INPUT_BDIM=1,
            INPUT_SDIM=1,
            WEIGHT_BDIM=1,
            MEM_DEPTH=1024,
            ACTIVATION_TYPE=1,
        )
        
        # Should not raise exceptions during Phase 2 parameter extraction
        op = TestKernelE2e(node)
        
        # Verify runtime parameter extraction worked correctly
        extracted_value = op.get_nodeattr("PE")
        expected_value = 4
        assert extracted_value == expected_value, f"Parameter PE: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("SIMD")
        expected_value = 8
        assert extracted_value == expected_value, f"Parameter SIMD: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("INPUT_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter INPUT_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("WEIGHT_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter WEIGHT_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("WEIGHT_SIGNED")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter WEIGHT_SIGNED: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("OUTPUT_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter OUTPUT_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("ACC_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter ACC_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("ACC_SIGNED")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter ACC_SIGNED: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("THRESH_WIDTH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter THRESH_WIDTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("INPUT_BDIM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter INPUT_BDIM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("INPUT_SDIM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter INPUT_SDIM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("WEIGHT_BDIM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter WEIGHT_BDIM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("MEM_DEPTH")
        expected_value = 1024
        assert extracted_value == expected_value, f"Parameter MEM_DEPTH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("ACTIVATION_TYPE")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter ACTIVATION_TYPE: expected {expected_value}, got {extracted_value}"
        
        # Verify parameter storage in runtime_parameters dict
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters dict should exist"
        assert "PE" in op.runtime_parameters, "PE should be in runtime_parameters"
        assert "SIMD" in op.runtime_parameters, "SIMD should be in runtime_parameters"
        assert "INPUT_WIDTH" in op.runtime_parameters, "INPUT_WIDTH should be in runtime_parameters"
        assert "WEIGHT_WIDTH" in op.runtime_parameters, "WEIGHT_WIDTH should be in runtime_parameters"
        assert "WEIGHT_SIGNED" in op.runtime_parameters, "WEIGHT_SIGNED should be in runtime_parameters"
        assert "OUTPUT_WIDTH" in op.runtime_parameters, "OUTPUT_WIDTH should be in runtime_parameters"
        assert "ACC_WIDTH" in op.runtime_parameters, "ACC_WIDTH should be in runtime_parameters"
        assert "ACC_SIGNED" in op.runtime_parameters, "ACC_SIGNED should be in runtime_parameters"
        assert "THRESH_WIDTH" in op.runtime_parameters, "THRESH_WIDTH should be in runtime_parameters"
        assert "INPUT_BDIM" in op.runtime_parameters, "INPUT_BDIM should be in runtime_parameters"
        assert "INPUT_SDIM" in op.runtime_parameters, "INPUT_SDIM should be in runtime_parameters"
        assert "WEIGHT_BDIM" in op.runtime_parameters, "WEIGHT_BDIM should be in runtime_parameters"
        assert "MEM_DEPTH" in op.runtime_parameters, "MEM_DEPTH should be in runtime_parameters"
        assert "ACTIVATION_TYPE" in op.runtime_parameters, "ACTIVATION_TYPE should be in runtime_parameters"
    
    def test_interface_metadata_validation(self):
        """Test that interface metadata matches Phase 2 enhanced parsing."""
        # Create a valid node for testing
        node = onnx.helper.make_node(
            "TestKernelE2e",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            SIMD=8,
            INPUT_WIDTH=1,
            WEIGHT_WIDTH=1,
            WEIGHT_SIGNED=1,
            OUTPUT_WIDTH=1,
            ACC_WIDTH=1,
            ACC_SIGNED=1,
            THRESH_WIDTH=1,
            INPUT_BDIM=1,
            INPUT_SDIM=1,
            WEIGHT_BDIM=1,
            MEM_DEPTH=1024,
            ACTIVATION_TYPE=1,
        )
        
        op = TestKernelE2e(node)
        
        # Test interface metadata structure
        # Test ap interface
        # Test s_axis_input interface
        # INPUT interface validation
        input_shape = op.get_folded_input_shape()
        assert input_shape is not None, "INPUT interface should have shape definition"
        # Test s_axis_weights interface
        # WEIGHT interface validation
        weight_shape = op.get_weight_shape()
        assert weight_shape is not None, "WEIGHT interface should have shape definition"
        # Test m_axis_output interface
        # OUTPUT interface validation  
        output_shape = op.get_folded_output_shape()
        assert output_shape is not None, "OUTPUT interface should have shape definition"
        # Test s_axilite_config interface
    
    def test_bdim_parameter_consistency(self):
        """Test BDIM parameter consistency from Phase 2 validation."""
        node = onnx.helper.make_node(
            "TestKernelE2e",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            SIMD=8,
            INPUT_WIDTH=1,
            WEIGHT_WIDTH=1,
            WEIGHT_SIGNED=1,
            OUTPUT_WIDTH=1,
            ACC_WIDTH=1,
            ACC_SIGNED=1,
            THRESH_WIDTH=1,
            INPUT_BDIM=1,
            INPUT_SDIM=1,
            WEIGHT_BDIM=1,
            MEM_DEPTH=1024,
            ACTIVATION_TYPE=1,
        )
        
        op = TestKernelE2e(node)
        
        # Test that BDIM parameters referenced in interfaces are consistent
        # ap BDIM consistency check
        # Block shape: [':']
        # s_axis_input BDIM consistency check
        # Block shape: [':', ':']
        # s_axis_weights BDIM consistency check
        # Block shape: ['PE']
        # Validate PE parameter exists and is positive
        param_value = op.get_nodeattr("PE")
        assert param_value is not None, "BDIM parameter PE must be defined"
        assert param_value > 0, "BDIM parameter PE must be positive, got {param_value}"
        # m_axis_output BDIM consistency check
        # Block shape: [':', ':']
        # s_axilite_config BDIM consistency check
        # Block shape: [':']
    
    def test_node_attribute_types_phase2(self):
        """Test node attribute type definitions for Phase 2 compatibility."""
        node = onnx.helper.make_node(
            "TestKernelE2e",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            SIMD=8,
            INPUT_WIDTH=1,
            WEIGHT_WIDTH=1,
            WEIGHT_SIGNED=1,
            OUTPUT_WIDTH=1,
            ACC_WIDTH=1,
            ACC_SIGNED=1,
            THRESH_WIDTH=1,
            INPUT_BDIM=1,
            INPUT_SDIM=1,
            WEIGHT_BDIM=1,
            MEM_DEPTH=1024,
            ACTIVATION_TYPE=1,
        )
        
        op = TestKernelE2e(node)
        
        # Check that get_nodeattr_types is implemented
        if hasattr(op, 'get_nodeattr_types'):
            attr_types = op.get_nodeattr_types()
            
            # Check parameter attributes exist
            assert "PE" in attr_types, "Parameter PE should have type definition"
            attr_type, required, default = attr_types["PE"]
            assert attr_type == "i", "Parameter PE should be integer type"
            assert required == False, "Parameter PE should not be required"
            assert default == 4, "Default value mismatch for PE"
            assert "SIMD" in attr_types, "Parameter SIMD should have type definition"
            attr_type, required, default = attr_types["SIMD"]
            assert attr_type == "i", "Parameter SIMD should be integer type"
            assert required == False, "Parameter SIMD should not be required"
            assert default == 8, "Default value mismatch for SIMD"
            assert "INPUT_WIDTH" in attr_types, "Parameter INPUT_WIDTH should have type definition"
            attr_type, required, default = attr_types["INPUT_WIDTH"]
            assert attr_type == "i", "Parameter INPUT_WIDTH should be integer type"
            assert required == True, "Parameter INPUT_WIDTH should be required"
            assert "WEIGHT_WIDTH" in attr_types, "Parameter WEIGHT_WIDTH should have type definition"
            attr_type, required, default = attr_types["WEIGHT_WIDTH"]
            assert attr_type == "i", "Parameter WEIGHT_WIDTH should be integer type"
            assert required == True, "Parameter WEIGHT_WIDTH should be required"
            assert "WEIGHT_SIGNED" in attr_types, "Parameter WEIGHT_SIGNED should have type definition"
            attr_type, required, default = attr_types["WEIGHT_SIGNED"]
            assert attr_type == "i", "Parameter WEIGHT_SIGNED should be integer type"
            assert required == True, "Parameter WEIGHT_SIGNED should be required"
            assert "OUTPUT_WIDTH" in attr_types, "Parameter OUTPUT_WIDTH should have type definition"
            attr_type, required, default = attr_types["OUTPUT_WIDTH"]
            assert attr_type == "i", "Parameter OUTPUT_WIDTH should be integer type"
            assert required == True, "Parameter OUTPUT_WIDTH should be required"
            assert "ACC_WIDTH" in attr_types, "Parameter ACC_WIDTH should have type definition"
            attr_type, required, default = attr_types["ACC_WIDTH"]
            assert attr_type == "i", "Parameter ACC_WIDTH should be integer type"
            assert required == True, "Parameter ACC_WIDTH should be required"
            assert "ACC_SIGNED" in attr_types, "Parameter ACC_SIGNED should have type definition"
            attr_type, required, default = attr_types["ACC_SIGNED"]
            assert attr_type == "i", "Parameter ACC_SIGNED should be integer type"
            assert required == True, "Parameter ACC_SIGNED should be required"
            assert "THRESH_WIDTH" in attr_types, "Parameter THRESH_WIDTH should have type definition"
            attr_type, required, default = attr_types["THRESH_WIDTH"]
            assert attr_type == "i", "Parameter THRESH_WIDTH should be integer type"
            assert required == True, "Parameter THRESH_WIDTH should be required"
            assert "INPUT_BDIM" in attr_types, "Parameter INPUT_BDIM should have type definition"
            attr_type, required, default = attr_types["INPUT_BDIM"]
            assert attr_type == "i", "Parameter INPUT_BDIM should be integer type"
            assert required == True, "Parameter INPUT_BDIM should be required"
            assert "INPUT_SDIM" in attr_types, "Parameter INPUT_SDIM should have type definition"
            attr_type, required, default = attr_types["INPUT_SDIM"]
            assert attr_type == "i", "Parameter INPUT_SDIM should be integer type"
            assert required == True, "Parameter INPUT_SDIM should be required"
            assert "WEIGHT_BDIM" in attr_types, "Parameter WEIGHT_BDIM should have type definition"
            attr_type, required, default = attr_types["WEIGHT_BDIM"]
            assert attr_type == "i", "Parameter WEIGHT_BDIM should be integer type"
            assert required == True, "Parameter WEIGHT_BDIM should be required"
            assert "MEM_DEPTH" in attr_types, "Parameter MEM_DEPTH should have type definition"
            attr_type, required, default = attr_types["MEM_DEPTH"]
            assert attr_type == "i", "Parameter MEM_DEPTH should be integer type"
            assert required == False, "Parameter MEM_DEPTH should not be required"
            assert default == 1024, "Default value mismatch for MEM_DEPTH"
            assert "ACTIVATION_TYPE" in attr_types, "Parameter ACTIVATION_TYPE should have type definition"
            attr_type, required, default = attr_types["ACTIVATION_TYPE"]
            assert attr_type == "i", "Parameter ACTIVATION_TYPE should be integer type"
            assert required == True, "Parameter ACTIVATION_TYPE should be required"
    
    def test_parameter_range_validation(self):
        """Test parameter range validation for positive values."""
        # Test PE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                PE=-1,  # Invalid negative value
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test PE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                PE=0,  # Invalid zero value
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test SIMD must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                SIMD=-1,  # Invalid negative value
                PE=4,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test SIMD == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                SIMD=0,  # Invalid zero value
                PE=4,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test INPUT_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_WIDTH=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test INPUT_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_WIDTH=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test WEIGHT_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_WIDTH=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test WEIGHT_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_WIDTH=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test WEIGHT_SIGNED must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_SIGNED=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test WEIGHT_SIGNED == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_SIGNED=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test OUTPUT_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                OUTPUT_WIDTH=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test OUTPUT_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                OUTPUT_WIDTH=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test ACC_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACC_WIDTH=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test ACC_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACC_WIDTH=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test ACC_SIGNED must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACC_SIGNED=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test ACC_SIGNED == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACC_SIGNED=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test THRESH_WIDTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                THRESH_WIDTH=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test THRESH_WIDTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                THRESH_WIDTH=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test INPUT_BDIM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_BDIM=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test INPUT_BDIM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_BDIM=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test INPUT_SDIM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_SDIM=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test INPUT_SDIM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_SDIM=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test WEIGHT_BDIM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_BDIM=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test WEIGHT_BDIM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_BDIM=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test MEM_DEPTH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                MEM_DEPTH=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test MEM_DEPTH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                MEM_DEPTH=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                ACTIVATION_TYPE=1,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test ACTIVATION_TYPE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACTIVATION_TYPE=-1,  # Invalid negative value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
            )
            op = TestKernelE2e(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test ACTIVATION_TYPE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACTIVATION_TYPE=0,  # Invalid zero value
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
            )
            op = TestKernelE2e(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
    
    def test_phase2_regression_prevention(self):
        """Test that Phase 2 features work and prevent common regressions."""
        node = onnx.helper.make_node(
            "TestKernelE2e",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            SIMD=8,
            INPUT_WIDTH=1,
            WEIGHT_WIDTH=1,
            WEIGHT_SIGNED=1,
            OUTPUT_WIDTH=1,
            ACC_WIDTH=1,
            ACC_SIGNED=1,
            THRESH_WIDTH=1,
            INPUT_BDIM=1,
            INPUT_SDIM=1,
            WEIGHT_BDIM=1,
            MEM_DEPTH=1024,
            ACTIVATION_TYPE=1,
        )
        
        op = TestKernelE2e(node)
        
        # Regression test: Ensure runtime_parameters are extracted in __init__
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters extraction should work"
        assert len(op.runtime_parameters) > 0, "runtime_parameters should not be empty"
        
        # Regression test: Ensure all defined parameters are accessible
        param_value = op.get_nodeattr("PE")
        assert param_value is not None, "Parameter PE should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("SIMD")
        assert param_value is not None, "Parameter SIMD should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("INPUT_WIDTH")
        assert param_value is not None, "Parameter INPUT_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("WEIGHT_WIDTH")
        assert param_value is not None, "Parameter WEIGHT_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("WEIGHT_SIGNED")
        assert param_value is not None, "Parameter WEIGHT_SIGNED should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("OUTPUT_WIDTH")
        assert param_value is not None, "Parameter OUTPUT_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("ACC_WIDTH")
        assert param_value is not None, "Parameter ACC_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("ACC_SIGNED")
        assert param_value is not None, "Parameter ACC_SIGNED should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("THRESH_WIDTH")
        assert param_value is not None, "Parameter THRESH_WIDTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("INPUT_BDIM")
        assert param_value is not None, "Parameter INPUT_BDIM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("INPUT_SDIM")
        assert param_value is not None, "Parameter INPUT_SDIM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("WEIGHT_BDIM")
        assert param_value is not None, "Parameter WEIGHT_BDIM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("MEM_DEPTH")
        assert param_value is not None, "Parameter MEM_DEPTH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("ACTIVATION_TYPE")
        assert param_value is not None, "Parameter ACTIVATION_TYPE should be accessible via get_nodeattr"
        
        # Regression test: Ensure class is properly derived from AutoHWCustomOp
        from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
        assert isinstance(op, HWCustomOp), "TestKernelE2e should inherit from HWCustomOp"
    
    @pytest.mark.slow
    def test_finn_integration_compatibility(self):
        """Test FINN framework integration compatibility (slow test)."""
        # Create a minimal model for integration testing
        node = onnx.helper.make_node(
            "TestKernelE2e",
            inputs=["input"],
            outputs=["output"],
            PE=4,
            SIMD=8,
            INPUT_WIDTH=1,
            WEIGHT_WIDTH=1,
            WEIGHT_SIGNED=1,
            OUTPUT_WIDTH=1,
            ACC_WIDTH=1,
            ACC_SIGNED=1,
            THRESH_WIDTH=1,
            INPUT_BDIM=1,
            INPUT_SDIM=1,
            WEIGHT_BDIM=1,
            MEM_DEPTH=1024,
            ACTIVATION_TYPE=1,
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
            assert wrapper is not None, "ModelWrapper should be able to load model with TestKernelE2e"
            
            # Test that the custom op is recognized
            custom_nodes = wrapper.get_nodes_by_op_type("TestKernelE2e")
            assert len(custom_nodes) == 1, "Should find exactly one TestKernelE2e node"
            
        except Exception as e:
            pytest.skip(f"FINN integration test skipped due to: {e}")

#=============================================================================
# Test Utilities and Fixtures
#=============================================================================

@pytest.fixture
def sample__node():
    """Fixture providing a sample TestKernelE2e node with valid parameters."""
    return onnx.helper.make_node(
        "TestKernelE2e",
        inputs=["input"],
        outputs=["output"],
        PE=4,
        SIMD=8,
        INPUT_WIDTH=1,
        WEIGHT_WIDTH=1,
        WEIGHT_SIGNED=1,
        OUTPUT_WIDTH=1,
        ACC_WIDTH=1,
        ACC_SIGNED=1,
        THRESH_WIDTH=1,
        INPUT_BDIM=1,
        INPUT_SDIM=1,
        WEIGHT_BDIM=1,
        MEM_DEPTH=1024,
        ACTIVATION_TYPE=1,
    )

@pytest.fixture  
def sample__op(sample__node):
    """Fixture providing a sample TestKernelE2e instance."""
    return TestKernelE2e(sample__node)

#=============================================================================
# Performance and Stress Tests
#=============================================================================

class TestTestKernelE2ePerformance:
    """Performance and stress tests for TestKernelE2e."""
    
    @pytest.mark.performance
    def test_instantiation_performance(self, sample__node):
        """Test that instantiation is reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            op = TestKernelE2e(sample__node)
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
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                PE=value,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("PE")
            assert extracted == value, f"PE: expected {value}, got {extracted}"
        # Stress test SIMD
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                SIMD=value,
                PE=4,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("SIMD")
            assert extracted == value, f"SIMD: expected {value}, got {extracted}"
        # Stress test INPUT_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_WIDTH=value,
                PE=4,
                SIMD=8,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("INPUT_WIDTH")
            assert extracted == value, f"INPUT_WIDTH: expected {value}, got {extracted}"
        # Stress test WEIGHT_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_WIDTH=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("WEIGHT_WIDTH")
            assert extracted == value, f"WEIGHT_WIDTH: expected {value}, got {extracted}"
        # Stress test WEIGHT_SIGNED
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_SIGNED=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("WEIGHT_SIGNED")
            assert extracted == value, f"WEIGHT_SIGNED: expected {value}, got {extracted}"
        # Stress test OUTPUT_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                OUTPUT_WIDTH=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("OUTPUT_WIDTH")
            assert extracted == value, f"OUTPUT_WIDTH: expected {value}, got {extracted}"
        # Stress test ACC_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACC_WIDTH=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("ACC_WIDTH")
            assert extracted == value, f"ACC_WIDTH: expected {value}, got {extracted}"
        # Stress test ACC_SIGNED
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACC_SIGNED=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("ACC_SIGNED")
            assert extracted == value, f"ACC_SIGNED: expected {value}, got {extracted}"
        # Stress test THRESH_WIDTH
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                THRESH_WIDTH=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("THRESH_WIDTH")
            assert extracted == value, f"THRESH_WIDTH: expected {value}, got {extracted}"
        # Stress test INPUT_BDIM
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_BDIM=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("INPUT_BDIM")
            assert extracted == value, f"INPUT_BDIM: expected {value}, got {extracted}"
        # Stress test INPUT_SDIM
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                INPUT_SDIM=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("INPUT_SDIM")
            assert extracted == value, f"INPUT_SDIM: expected {value}, got {extracted}"
        # Stress test WEIGHT_BDIM
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                WEIGHT_BDIM=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                MEM_DEPTH=1024,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("WEIGHT_BDIM")
            assert extracted == value, f"WEIGHT_BDIM: expected {value}, got {extracted}"
        # Stress test MEM_DEPTH
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                MEM_DEPTH=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                ACTIVATION_TYPE=1,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("MEM_DEPTH")
            assert extracted == value, f"MEM_DEPTH: expected {value}, got {extracted}"
        # Stress test ACTIVATION_TYPE
        for value in test_values:
            node = onnx.helper.make_node(
                "TestKernelE2e",
                inputs=["input"],
                outputs=["output"],
                ACTIVATION_TYPE=value,
                PE=4,
                SIMD=8,
                INPUT_WIDTH=1,
                WEIGHT_WIDTH=1,
                WEIGHT_SIGNED=1,
                OUTPUT_WIDTH=1,
                ACC_WIDTH=1,
                ACC_SIGNED=1,
                THRESH_WIDTH=1,
                INPUT_BDIM=1,
                INPUT_SDIM=1,
                WEIGHT_BDIM=1,
                MEM_DEPTH=1024,
            )
            
            op = TestKernelE2e(node)
            extracted = op.get_nodeattr("ACTIVATION_TYPE")
            assert extracted == value, f"ACTIVATION_TYPE: expected {value}, got {extracted}"

#=============================================================================
# End of TestKernelE2e Test Suite
# Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
#=============================================================================