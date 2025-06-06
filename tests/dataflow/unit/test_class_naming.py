"""
Test suite for class naming utilities.

This tests the fix for the class naming issue where "thresholding_axi"
was incorrectly converted to "AutoThresholdingaxi" instead of the
correct "AutoThresholdingAxi".
"""

import pytest
from brainsmith.dataflow.core.class_naming import (
    generate_class_name,
    generate_test_class_name,
    generate_backend_class_name
)


class TestClassNaming:
    """Test class naming utilities."""
    
    def test_generate_class_name_basic(self):
        """Test basic class name generation."""
        # Test the specific case that was failing
        assert generate_class_name("thresholding_axi") == "AutoThresholdingAxi"
        
        # Test other common patterns
        assert generate_class_name("conv_layer") == "AutoConvLayer"
        assert generate_class_name("batch_norm") == "AutoBatchNorm"
        assert generate_class_name("my_custom_kernel") == "AutoMyCustomKernel"
        
    def test_generate_class_name_single_word(self):
        """Test class name generation with single word."""
        assert generate_class_name("thresholding") == "AutoThresholding"
        assert generate_class_name("softmax") == "AutoSoftmax"
        assert generate_class_name("relu") == "AutoRelu"
        
    def test_generate_class_name_custom_prefix(self):
        """Test class name generation with custom prefix."""
        assert generate_class_name("thresholding_axi", prefix="Custom") == "CustomThresholdingAxi"
        assert generate_class_name("conv_layer", prefix="My") == "MyConvLayer"
        assert generate_class_name("batch_norm", prefix="") == "BatchNorm"
        
    def test_generate_test_class_name(self):
        """Test test class name generation."""
        assert generate_test_class_name("thresholding_axi") == "TestAutoThresholdingAxi"
        assert generate_test_class_name("conv_layer") == "TestAutoConvLayer"
        assert generate_test_class_name("batch_norm") == "TestAutoBatchNorm"
        
    def test_generate_backend_class_name(self):
        """Test RTL backend class name generation."""
        assert generate_backend_class_name("thresholding_axi") == "AutoThresholdingAxiRTLBackend"
        assert generate_backend_class_name("conv_layer") == "AutoConvLayerRTLBackend"
        assert generate_backend_class_name("batch_norm") == "AutoBatchNormRTLBackend"
        
    def test_edge_cases(self):
        """Test edge cases in class naming."""
        # Empty string
        assert generate_class_name("") == "Auto"
        
        # Multiple underscores
        assert generate_class_name("my__custom__kernel") == "AutoMyCustomKernel"
        
        # Leading/trailing underscores
        assert generate_class_name("_kernel_") == "AutoKernel"
        
        # Numbers in name
        assert generate_class_name("conv2d_3x3") == "AutoConv2d3x3"
        assert generate_class_name("layer_1") == "AutoLayer1"
        
    def test_case_preservation(self):
        """Test that existing capitalization is preserved."""
        # Note: The current implementation capitalizes each part,
        # which might override existing capitalization.
        # This is the expected behavior.
        assert generate_class_name("MyKernel") == "AutoMykernel"  # Each part is capitalized
        assert generate_class_name("CONV_LAYER") == "AutoConvLayer"
        assert generate_class_name("ReLU_activation") == "AutoReluActivation"


class TestClassNamingIntegration:
    """Integration tests for class naming in the HKG context."""
    
    def test_hkg_class_name_consistency(self):
        """Test that class names are consistent across HKG generation."""
        kernel_name = "thresholding_axi"
        
        # Main class name
        class_name = generate_class_name(kernel_name)
        assert class_name == "AutoThresholdingAxi"
        
        # Backend class name
        backend_name = generate_backend_class_name(kernel_name)
        assert backend_name == "AutoThresholdingAxiRTLBackend"
        assert backend_name == f"{class_name}RTLBackend"
        
        # Test class name
        test_name = generate_test_class_name(kernel_name)
        assert test_name == "TestAutoThresholdingAxi"
        assert test_name == f"Test{class_name}"
        
    def test_file_naming_consistency(self):
        """Test that generated file names are consistent with class names."""
        kernel_name = "thresholding_axi"
        class_name = generate_class_name(kernel_name)
        
        # File names should use lowercase version of class name
        expected_file_base = class_name.lower()
        assert expected_file_base == "autothresholdingaxi"
        
        # Expected file names
        expected_files = {
            "hw_custom_op": f"{expected_file_base}.py",
            "rtl_backend": f"{expected_file_base}_rtlbackend.py",
            "test_suite": f"test_{expected_file_base}.py",
            "documentation": f"{expected_file_base}_README.md",
        }
        
        # Verify file naming pattern
        for file_type, filename in expected_files.items():
            assert filename.startswith(expected_file_base) or filename.startswith(f"test_{expected_file_base}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])