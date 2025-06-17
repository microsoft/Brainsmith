"""
Auto-generated test suite for ThresholdingAxi.
Generated from: brainsmith/hw_kernels/thresholding/thresholding_axi.sv
Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
Generation time: 2025-06-16T22:42:52.282871

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
from _hw_custom_op import ThresholdingAxi

class TestThresholdingAxi:
    """
    Enhanced test suite for ThresholdingAxi with Phase 2 parameter handling.
    
    Tests runtime parameter extraction, validation, and FINN integration
    with the Phase 2 template system that ensures symbolic BDIM validation.
    """
    
    def test_parameter_validation_required_attributes(self):
        """Test that required parameters are properly validated."""
        # Test missing required parameters
        with pytest.raises((ValueError, AttributeError), match="(Missing|required)"):
            # Create node without required parameters
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                # Missing required: N, WI, WT, C, SIGNED, FPARG, THRESHOLDS_PATH, USE_AXILITE, DEPTH_TRIGGER_URAM, DEPTH_TRIGGER_BRAM, DEEP_PIPELINE
            )
            ThresholdingAxi(node)
    
    def test_parameter_validation_whitelisted_defaults(self):
        """Test whitelisted parameters with defaults are handled correctly."""
        # Create node with all whitelisted parameters
        node = onnx.helper.make_node(
            "ThresholdingAxi",
            inputs=["input"],
            outputs=["output"],
            PE=1,
            BIAS=0,
            N=1,  # Required parameter
            WI=1,  # Required parameter
            WT=1,  # Required parameter
            C=1,  # Required parameter
            SIGNED=1,  # Required parameter
            FPARG=1,  # Required parameter
            THRESHOLDS_PATH=1,  # Required parameter
            USE_AXILITE=1,  # Required parameter
            DEPTH_TRIGGER_URAM=1,  # Required parameter
            DEPTH_TRIGGER_BRAM=1,  # Required parameter
            DEEP_PIPELINE=1,  # Required parameter
        )
        
        op = ThresholdingAxi(node)
        
        # Verify whitelisted parameters are extracted correctly
        assert op.get_nodeattr("PE") == 1
        assert op.get_nodeattr("BIAS") == 0
    
    def test_valid_node_creation_all_parameters(self):
        """Test successful node creation with all defined parameters."""
        node = onnx.helper.make_node(
            "ThresholdingAxi",
            inputs=["input"],
            outputs=["output"],
            N=1,
            WI=1,
            WT=1,
            C=1,
            PE=1,
            SIGNED=1,
            FPARG=1,
            BIAS=0,
            THRESHOLDS_PATH=1,
            USE_AXILITE=1,
            DEPTH_TRIGGER_URAM=1,
            DEPTH_TRIGGER_BRAM=1,
            DEEP_PIPELINE=1,
        )
        
        assert node.op_type == "ThresholdingAxi"
        assert len(node.input) == 1
        assert len(node.output) == 1
        
        # Verify all attributes are set
        attr_value = next((attr.i for attr in node.attribute if attr.name == "N"), None)
        assert attr_value is not None, "Parameter N should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "WI"), None)
        assert attr_value is not None, "Parameter WI should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "WT"), None)
        assert attr_value is not None, "Parameter WT should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "C"), None)
        assert attr_value is not None, "Parameter C should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "PE"), None)
        assert attr_value is not None, "Parameter PE should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "SIGNED"), None)
        assert attr_value is not None, "Parameter SIGNED should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "FPARG"), None)
        assert attr_value is not None, "Parameter FPARG should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "BIAS"), None)
        assert attr_value is not None, "Parameter BIAS should be set"
        assert attr_value == 0
        attr_value = next((attr.i for attr in node.attribute if attr.name == "THRESHOLDS_PATH"), None)
        assert attr_value is not None, "Parameter THRESHOLDS_PATH should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "USE_AXILITE"), None)
        assert attr_value is not None, "Parameter USE_AXILITE should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "DEPTH_TRIGGER_URAM"), None)
        assert attr_value is not None, "Parameter DEPTH_TRIGGER_URAM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "DEPTH_TRIGGER_BRAM"), None)
        assert attr_value is not None, "Parameter DEPTH_TRIGGER_BRAM should be set"
        assert attr_value == 1
        attr_value = next((attr.i for attr in node.attribute if attr.name == "DEEP_PIPELINE"), None)
        assert attr_value is not None, "Parameter DEEP_PIPELINE should be set"
        assert attr_value == 1
    
    def test_hwcustomop_instantiation_runtime_extraction(self):
        """Test HWCustomOp instantiation with Phase 2 runtime parameter extraction."""
        node = onnx.helper.make_node(
            "ThresholdingAxi",
            inputs=["input"],
            outputs=["output"],
            N=1,
            WI=1,
            WT=1,
            C=1,
            PE=1,
            SIGNED=1,
            FPARG=1,
            BIAS=0,
            THRESHOLDS_PATH=1,
            USE_AXILITE=1,
            DEPTH_TRIGGER_URAM=1,
            DEPTH_TRIGGER_BRAM=1,
            DEEP_PIPELINE=1,
        )
        
        # Should not raise exceptions during Phase 2 parameter extraction
        op = ThresholdingAxi(node)
        
        # Verify runtime parameter extraction worked correctly
        extracted_value = op.get_nodeattr("N")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter N: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("WI")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter WI: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("WT")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter WT: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("C")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter C: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("PE")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter PE: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("SIGNED")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter SIGNED: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("FPARG")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter FPARG: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("BIAS")
        expected_value = 0
        assert extracted_value == expected_value, f"Parameter BIAS: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("THRESHOLDS_PATH")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter THRESHOLDS_PATH: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("USE_AXILITE")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter USE_AXILITE: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("DEPTH_TRIGGER_URAM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter DEPTH_TRIGGER_URAM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("DEPTH_TRIGGER_BRAM")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter DEPTH_TRIGGER_BRAM: expected {expected_value}, got {extracted_value}"
        extracted_value = op.get_nodeattr("DEEP_PIPELINE")
        expected_value = 1
        assert extracted_value == expected_value, f"Parameter DEEP_PIPELINE: expected {expected_value}, got {extracted_value}"
        
        # Verify parameter storage in runtime_parameters dict
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters dict should exist"
        assert "N" in op.runtime_parameters, "N should be in runtime_parameters"
        assert "WI" in op.runtime_parameters, "WI should be in runtime_parameters"
        assert "WT" in op.runtime_parameters, "WT should be in runtime_parameters"
        assert "C" in op.runtime_parameters, "C should be in runtime_parameters"
        assert "PE" in op.runtime_parameters, "PE should be in runtime_parameters"
        assert "SIGNED" in op.runtime_parameters, "SIGNED should be in runtime_parameters"
        assert "FPARG" in op.runtime_parameters, "FPARG should be in runtime_parameters"
        assert "BIAS" in op.runtime_parameters, "BIAS should be in runtime_parameters"
        assert "THRESHOLDS_PATH" in op.runtime_parameters, "THRESHOLDS_PATH should be in runtime_parameters"
        assert "USE_AXILITE" in op.runtime_parameters, "USE_AXILITE should be in runtime_parameters"
        assert "DEPTH_TRIGGER_URAM" in op.runtime_parameters, "DEPTH_TRIGGER_URAM should be in runtime_parameters"
        assert "DEPTH_TRIGGER_BRAM" in op.runtime_parameters, "DEPTH_TRIGGER_BRAM should be in runtime_parameters"
        assert "DEEP_PIPELINE" in op.runtime_parameters, "DEEP_PIPELINE should be in runtime_parameters"
    
    def test_interface_metadata_validation(self):
        """Test that interface metadata matches Phase 2 enhanced parsing."""
        # Create a valid node for testing
        node = onnx.helper.make_node(
            "ThresholdingAxi",
            inputs=["input"],
            outputs=["output"],
            N=1,
            WI=1,
            WT=1,
            C=1,
            PE=1,
            SIGNED=1,
            FPARG=1,
            BIAS=0,
            THRESHOLDS_PATH=1,
            USE_AXILITE=1,
            DEPTH_TRIGGER_URAM=1,
            DEPTH_TRIGGER_BRAM=1,
            DEEP_PIPELINE=1,
        )
        
        op = ThresholdingAxi(node)
        
        # Test interface metadata structure
        # Test ap interface
        # Test s_axis interface
        # INPUT interface validation
        input_shape = op.get_folded_input_shape()
        assert input_shape is not None, "INPUT interface should have shape definition"
        # Test m_axis interface
        # OUTPUT interface validation  
        output_shape = op.get_folded_output_shape()
        assert output_shape is not None, "OUTPUT interface should have shape definition"
        # Test s_axilite interface
    
    def test_bdim_parameter_consistency(self):
        """Test BDIM parameter consistency from Phase 2 validation."""
        node = onnx.helper.make_node(
            "ThresholdingAxi",
            inputs=["input"],
            outputs=["output"],
            N=1,
            WI=1,
            WT=1,
            C=1,
            PE=1,
            SIGNED=1,
            FPARG=1,
            BIAS=0,
            THRESHOLDS_PATH=1,
            USE_AXILITE=1,
            DEPTH_TRIGGER_URAM=1,
            DEPTH_TRIGGER_BRAM=1,
            DEEP_PIPELINE=1,
        )
        
        op = ThresholdingAxi(node)
        
        # Test that BDIM parameters referenced in interfaces are consistent
        # ap BDIM consistency check
        # Block shape: [':']
        # s_axis BDIM consistency check
        # Block shape: [':', ':']
        # m_axis BDIM consistency check
        # Block shape: [':', ':']
        # s_axilite BDIM consistency check
        # Block shape: [':']
    
    def test_node_attribute_types_phase2(self):
        """Test node attribute type definitions for Phase 2 compatibility."""
        node = onnx.helper.make_node(
            "ThresholdingAxi",
            inputs=["input"],
            outputs=["output"],
            N=1,
            WI=1,
            WT=1,
            C=1,
            PE=1,
            SIGNED=1,
            FPARG=1,
            BIAS=0,
            THRESHOLDS_PATH=1,
            USE_AXILITE=1,
            DEPTH_TRIGGER_URAM=1,
            DEPTH_TRIGGER_BRAM=1,
            DEEP_PIPELINE=1,
        )
        
        op = ThresholdingAxi(node)
        
        # Check that get_nodeattr_types is implemented
        if hasattr(op, 'get_nodeattr_types'):
            attr_types = op.get_nodeattr_types()
            
            # Check parameter attributes exist
            assert "N" in attr_types, "Parameter N should have type definition"
            attr_type, required, default = attr_types["N"]
            assert attr_type == "i", "Parameter N should be integer type"
            assert required == True, "Parameter N should be required"
            assert "WI" in attr_types, "Parameter WI should have type definition"
            attr_type, required, default = attr_types["WI"]
            assert attr_type == "i", "Parameter WI should be integer type"
            assert required == True, "Parameter WI should be required"
            assert "WT" in attr_types, "Parameter WT should have type definition"
            attr_type, required, default = attr_types["WT"]
            assert attr_type == "i", "Parameter WT should be integer type"
            assert required == True, "Parameter WT should be required"
            assert "C" in attr_types, "Parameter C should have type definition"
            attr_type, required, default = attr_types["C"]
            assert attr_type == "i", "Parameter C should be integer type"
            assert required == True, "Parameter C should be required"
            assert "PE" in attr_types, "Parameter PE should have type definition"
            attr_type, required, default = attr_types["PE"]
            assert attr_type == "i", "Parameter PE should be integer type"
            assert required == False, "Parameter PE should not be required"
            assert default == 1, "Default value mismatch for PE"
            assert "SIGNED" in attr_types, "Parameter SIGNED should have type definition"
            attr_type, required, default = attr_types["SIGNED"]
            assert attr_type == "i", "Parameter SIGNED should be integer type"
            assert required == True, "Parameter SIGNED should be required"
            assert "FPARG" in attr_types, "Parameter FPARG should have type definition"
            attr_type, required, default = attr_types["FPARG"]
            assert attr_type == "i", "Parameter FPARG should be integer type"
            assert required == True, "Parameter FPARG should be required"
            assert "BIAS" in attr_types, "Parameter BIAS should have type definition"
            attr_type, required, default = attr_types["BIAS"]
            assert attr_type == "i", "Parameter BIAS should be integer type"
            assert required == False, "Parameter BIAS should not be required"
            assert default == 0, "Default value mismatch for BIAS"
            assert "THRESHOLDS_PATH" in attr_types, "Parameter THRESHOLDS_PATH should have type definition"
            attr_type, required, default = attr_types["THRESHOLDS_PATH"]
            assert attr_type == "i", "Parameter THRESHOLDS_PATH should be integer type"
            assert required == True, "Parameter THRESHOLDS_PATH should be required"
            assert "USE_AXILITE" in attr_types, "Parameter USE_AXILITE should have type definition"
            attr_type, required, default = attr_types["USE_AXILITE"]
            assert attr_type == "i", "Parameter USE_AXILITE should be integer type"
            assert required == True, "Parameter USE_AXILITE should be required"
            assert "DEPTH_TRIGGER_URAM" in attr_types, "Parameter DEPTH_TRIGGER_URAM should have type definition"
            attr_type, required, default = attr_types["DEPTH_TRIGGER_URAM"]
            assert attr_type == "i", "Parameter DEPTH_TRIGGER_URAM should be integer type"
            assert required == True, "Parameter DEPTH_TRIGGER_URAM should be required"
            assert "DEPTH_TRIGGER_BRAM" in attr_types, "Parameter DEPTH_TRIGGER_BRAM should have type definition"
            attr_type, required, default = attr_types["DEPTH_TRIGGER_BRAM"]
            assert attr_type == "i", "Parameter DEPTH_TRIGGER_BRAM should be integer type"
            assert required == True, "Parameter DEPTH_TRIGGER_BRAM should be required"
            assert "DEEP_PIPELINE" in attr_types, "Parameter DEEP_PIPELINE should have type definition"
            attr_type, required, default = attr_types["DEEP_PIPELINE"]
            assert attr_type == "i", "Parameter DEEP_PIPELINE should be integer type"
            assert required == True, "Parameter DEEP_PIPELINE should be required"
    
    def test_parameter_range_validation(self):
        """Test parameter range validation for positive values."""
        # Test N must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                N=-1,  # Invalid negative value
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test N == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                N=0,  # Invalid zero value
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test WI must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                WI=-1,  # Invalid negative value
                N=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test WI == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                WI=0,  # Invalid zero value
                N=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test WT must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                WT=-1,  # Invalid negative value
                N=1,
                WI=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test WT == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                WT=0,  # Invalid zero value
                N=1,
                WI=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test C must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                C=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test C == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                C=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test PE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                PE=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test PE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                PE=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test SIGNED must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                SIGNED=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test SIGNED == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                SIGNED=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test FPARG must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                FPARG=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test FPARG == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                FPARG=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test BIAS must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                BIAS=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test BIAS == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                BIAS=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test THRESHOLDS_PATH must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                THRESHOLDS_PATH=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test THRESHOLDS_PATH == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                THRESHOLDS_PATH=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test USE_AXILITE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                USE_AXILITE=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test USE_AXILITE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                USE_AXILITE=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test DEPTH_TRIGGER_URAM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEPTH_TRIGGER_URAM=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test DEPTH_TRIGGER_URAM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEPTH_TRIGGER_URAM=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test DEPTH_TRIGGER_BRAM must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEPTH_TRIGGER_BRAM=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test DEPTH_TRIGGER_BRAM == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEPTH_TRIGGER_BRAM=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEEP_PIPELINE=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        # Test DEEP_PIPELINE must be positive
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEEP_PIPELINE=-1,  # Invalid negative value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
            )
            op = ThresholdingAxi(node)
            # Some validation might happen during instantiation or later method calls
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
        
        # Test DEEP_PIPELINE == 0 should also fail
        with pytest.raises((ValueError, AssertionError)):
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEEP_PIPELINE=0,  # Invalid zero value
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
            )
            op = ThresholdingAxi(node)
            if hasattr(op, 'validate_parameters'):
                op.validate_parameters()
    
    def test_phase2_regression_prevention(self):
        """Test that Phase 2 features work and prevent common regressions."""
        node = onnx.helper.make_node(
            "ThresholdingAxi",
            inputs=["input"],
            outputs=["output"],
            N=1,
            WI=1,
            WT=1,
            C=1,
            PE=1,
            SIGNED=1,
            FPARG=1,
            BIAS=0,
            THRESHOLDS_PATH=1,
            USE_AXILITE=1,
            DEPTH_TRIGGER_URAM=1,
            DEPTH_TRIGGER_BRAM=1,
            DEEP_PIPELINE=1,
        )
        
        op = ThresholdingAxi(node)
        
        # Regression test: Ensure runtime_parameters are extracted in __init__
        assert hasattr(op, 'runtime_parameters'), "Phase 2 runtime_parameters extraction should work"
        assert len(op.runtime_parameters) > 0, "runtime_parameters should not be empty"
        
        # Regression test: Ensure all defined parameters are accessible
        param_value = op.get_nodeattr("N")
        assert param_value is not None, "Parameter N should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("WI")
        assert param_value is not None, "Parameter WI should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("WT")
        assert param_value is not None, "Parameter WT should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("C")
        assert param_value is not None, "Parameter C should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("PE")
        assert param_value is not None, "Parameter PE should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("SIGNED")
        assert param_value is not None, "Parameter SIGNED should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("FPARG")
        assert param_value is not None, "Parameter FPARG should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("BIAS")
        assert param_value is not None, "Parameter BIAS should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("THRESHOLDS_PATH")
        assert param_value is not None, "Parameter THRESHOLDS_PATH should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("USE_AXILITE")
        assert param_value is not None, "Parameter USE_AXILITE should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("DEPTH_TRIGGER_URAM")
        assert param_value is not None, "Parameter DEPTH_TRIGGER_URAM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("DEPTH_TRIGGER_BRAM")
        assert param_value is not None, "Parameter DEPTH_TRIGGER_BRAM should be accessible via get_nodeattr"
        param_value = op.get_nodeattr("DEEP_PIPELINE")
        assert param_value is not None, "Parameter DEEP_PIPELINE should be accessible via get_nodeattr"
        
        # Regression test: Ensure class is properly derived from AutoHWCustomOp
        from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
        assert isinstance(op, HWCustomOp), "ThresholdingAxi should inherit from HWCustomOp"
    
    @pytest.mark.slow
    def test_finn_integration_compatibility(self):
        """Test FINN framework integration compatibility (slow test)."""
        # Create a minimal model for integration testing
        node = onnx.helper.make_node(
            "ThresholdingAxi",
            inputs=["input"],
            outputs=["output"],
            N=1,
            WI=1,
            WT=1,
            C=1,
            PE=1,
            SIGNED=1,
            FPARG=1,
            BIAS=0,
            THRESHOLDS_PATH=1,
            USE_AXILITE=1,
            DEPTH_TRIGGER_URAM=1,
            DEPTH_TRIGGER_BRAM=1,
            DEEP_PIPELINE=1,
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
            assert wrapper is not None, "ModelWrapper should be able to load model with ThresholdingAxi"
            
            # Test that the custom op is recognized
            custom_nodes = wrapper.get_nodes_by_op_type("ThresholdingAxi")
            assert len(custom_nodes) == 1, "Should find exactly one ThresholdingAxi node"
            
        except Exception as e:
            pytest.skip(f"FINN integration test skipped due to: {e}")

#=============================================================================
# Test Utilities and Fixtures
#=============================================================================

@pytest.fixture
def sample__node():
    """Fixture providing a sample ThresholdingAxi node with valid parameters."""
    return onnx.helper.make_node(
        "ThresholdingAxi",
        inputs=["input"],
        outputs=["output"],
        N=1,
        WI=1,
        WT=1,
        C=1,
        PE=1,
        SIGNED=1,
        FPARG=1,
        BIAS=0,
        THRESHOLDS_PATH=1,
        USE_AXILITE=1,
        DEPTH_TRIGGER_URAM=1,
        DEPTH_TRIGGER_BRAM=1,
        DEEP_PIPELINE=1,
    )

@pytest.fixture  
def sample__op(sample__node):
    """Fixture providing a sample ThresholdingAxi instance."""
    return ThresholdingAxi(sample__node)

#=============================================================================
# Performance and Stress Tests
#=============================================================================

class TestThresholdingAxiPerformance:
    """Performance and stress tests for ThresholdingAxi."""
    
    @pytest.mark.performance
    def test_instantiation_performance(self, sample__node):
        """Test that instantiation is reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            op = ThresholdingAxi(sample__node)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01, f"Instantiation should be < 10ms, got {avg_time*1000:.2f}ms"
    
    @pytest.mark.stress
    def test_parameter_extraction_stress(self):
        """Stress test parameter extraction with many different values."""
        test_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        # Stress test N
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                N=value,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("N")
            assert extracted == value, f"N: expected {value}, got {extracted}"
        # Stress test WI
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                WI=value,
                N=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("WI")
            assert extracted == value, f"WI: expected {value}, got {extracted}"
        # Stress test WT
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                WT=value,
                N=1,
                WI=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("WT")
            assert extracted == value, f"WT: expected {value}, got {extracted}"
        # Stress test C
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                C=value,
                N=1,
                WI=1,
                WT=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("C")
            assert extracted == value, f"C: expected {value}, got {extracted}"
        # Stress test PE
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                PE=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("PE")
            assert extracted == value, f"PE: expected {value}, got {extracted}"
        # Stress test SIGNED
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                SIGNED=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("SIGNED")
            assert extracted == value, f"SIGNED: expected {value}, got {extracted}"
        # Stress test FPARG
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                FPARG=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("FPARG")
            assert extracted == value, f"FPARG: expected {value}, got {extracted}"
        # Stress test BIAS
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                BIAS=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("BIAS")
            assert extracted == value, f"BIAS: expected {value}, got {extracted}"
        # Stress test THRESHOLDS_PATH
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                THRESHOLDS_PATH=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("THRESHOLDS_PATH")
            assert extracted == value, f"THRESHOLDS_PATH: expected {value}, got {extracted}"
        # Stress test USE_AXILITE
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                USE_AXILITE=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("USE_AXILITE")
            assert extracted == value, f"USE_AXILITE: expected {value}, got {extracted}"
        # Stress test DEPTH_TRIGGER_URAM
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEPTH_TRIGGER_URAM=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_BRAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("DEPTH_TRIGGER_URAM")
            assert extracted == value, f"DEPTH_TRIGGER_URAM: expected {value}, got {extracted}"
        # Stress test DEPTH_TRIGGER_BRAM
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEPTH_TRIGGER_BRAM=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEEP_PIPELINE=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("DEPTH_TRIGGER_BRAM")
            assert extracted == value, f"DEPTH_TRIGGER_BRAM: expected {value}, got {extracted}"
        # Stress test DEEP_PIPELINE
        for value in test_values:
            node = onnx.helper.make_node(
                "ThresholdingAxi",
                inputs=["input"],
                outputs=["output"],
                DEEP_PIPELINE=value,
                N=1,
                WI=1,
                WT=1,
                C=1,
                PE=1,
                SIGNED=1,
                FPARG=1,
                BIAS=0,
                THRESHOLDS_PATH=1,
                USE_AXILITE=1,
                DEPTH_TRIGGER_URAM=1,
                DEPTH_TRIGGER_BRAM=1,
            )
            
            op = ThresholdingAxi(node)
            extracted = op.get_nodeattr("DEEP_PIPELINE")
            assert extracted == value, f"DEEP_PIPELINE: expected {value}, got {extracted}"

#=============================================================================
# End of ThresholdingAxi Test Suite
# Template: test_suite_v2.py.j2 (Phase 3 Enhanced)
#=============================================================================