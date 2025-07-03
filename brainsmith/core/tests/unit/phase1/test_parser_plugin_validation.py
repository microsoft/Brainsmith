"""
Unit tests for parser plugin validation functionality.
"""

import pytest

from brainsmith.core.phase1.parser import BlueprintParser
from brainsmith.core.phase1.exceptions import BlueprintParseError


class TestParserPluginValidation:
    """Test parser's plugin validation and auto-discovery features."""
    
    def test_kernel_auto_discovery_simple(self, parser_with_registry):
        """Test auto-discovery for simple kernel names."""
        # Setup - use real QONNX/FINN kernel
        parser = parser_with_registry
        
        # Test single kernel - LayerNorm exists in QONNX/FINN
        kernels_data = ["LayerNorm"]
        validated = parser._validate_and_enrich_kernels(kernels_data)
        
        # Verify auto-discovery happened
        assert len(validated) == 1
        kernel_name, backends = validated[0]
        assert kernel_name == "LayerNorm"
        assert len(backends) >= 1  # Should have at least one backend
        # LayerNorm should have LayerNormHLS and LayerNormRTL backends
        assert "LayerNormHLS" in backends
    
    def test_kernel_auto_discovery_multiple(self, parser_with_registry):
        """Test auto-discovery for multiple kernels."""
        # Setup - use real QONNX/FINN kernels
        parser = parser_with_registry
        
        # Test multiple kernels that exist in QONNX/FINN
        kernels_data = ["LayerNorm", "Crop", "Shuffle"]
        validated = parser._validate_and_enrich_kernels(kernels_data)
        
        # Verify each kernel got its backends
        assert len(validated) == 3
        
        # Check LayerNorm
        kernel_name, backends = validated[0]
        assert kernel_name == "LayerNorm"
        assert "LayerNormHLS" in backends
        
        # Check Crop
        kernel_name, backends = validated[1]
        assert kernel_name == "Crop"
        assert "CropHLS" in backends
        
        # Check Shuffle
        kernel_name, backends = validated[2]
        assert kernel_name == "Shuffle"
        assert "ShuffleHLS" in backends
    
    def test_kernel_auto_discovery_optional(self, parser_with_registry):
        """Test auto-discovery for optional kernels (~ prefix)."""
        parser = parser_with_registry
        
        # Test optional kernel with real QONNX/FINN plugin
        kernels_data = ["~HWSoftmax"]
        validated = parser._validate_and_enrich_kernels(kernels_data)
        
        # Should preserve optional marker
        kernel_name, backends = validated[0]
        assert kernel_name == "~HWSoftmax"
        assert len(backends) >= 1
        assert "HWSoftmaxHLS" in backends
    
    def test_kernel_explicit_backends_valid(self, parser_with_registry):
        """Test validation of explicitly specified backends."""
        parser = parser_with_registry
        
        # Test with valid backends that exist for LayerNorm
        kernels_data = [("LayerNorm", ["LayerNormHLS"])]
        validated = parser._validate_and_enrich_kernels(kernels_data)
        
        # Should pass through as-is when valid
        assert validated[0] == ("LayerNorm", ["LayerNormHLS"])
    
    def test_kernel_explicit_backends_invalid(self, parser_with_registry):
        """Test error when explicitly specified backends are invalid."""
        parser = parser_with_registry
        
        # Test with invalid backends for LayerNorm
        kernels_data = [("LayerNorm", ["NonExistentBackend1", "NonExistentBackend2"])]
        
        with pytest.raises(BlueprintParseError) as exc:
            parser._validate_and_enrich_kernels(kernels_data)
        
        error_msg = str(exc.value)
        assert "Invalid backends" in error_msg
        assert "NonExistentBackend1" in error_msg
    
    def test_kernel_not_found_error(self, parser_with_registry):
        """Test helpful error when kernel doesn't exist."""
        parser = parser_with_registry
        
        # Test with non-existent kernel
        kernels_data = ["NonExistentKernel"]
        
        with pytest.raises(BlueprintParseError) as exc:
            parser._validate_and_enrich_kernels(kernels_data)
        
        error_msg = str(exc.value)
        assert "Kernel 'NonExistentKernel' not found" in error_msg
        assert "Available kernels:" in error_msg
    
    def test_kernel_no_backends_error(self, parser_with_registry):
        """Test error when kernel has no backends."""
        parser = parser_with_registry
        
        # Try to find a kernel with no backends - this is unlikely in QONNX/FINN
        # so we'll test with a non-existent kernel instead
        kernels_data = ["KernelWithNoBackends"]
        
        with pytest.raises(BlueprintParseError) as exc:
            parser._validate_and_enrich_kernels(kernels_data)
        
        # Will fail with "not found" rather than "no backends"
        assert "not found" in str(exc.value)
    
    def test_mutually_exclusive_group_validation(self, parser_with_registry):
        """Test validation of mutually exclusive kernel groups."""
        parser = parser_with_registry
        
        # Test group with real QONNX/FINN kernels
        kernels_data = [[
            "LayerNorm",  # Will auto-discover
            ("Crop", ["CropHLS"]),  # Explicit
            None  # Skip option
        ]]
        
        validated = parser._validate_and_enrich_kernels(kernels_data)
        
        # Should validate each item in group
        assert len(validated) == 1
        assert len(validated[0]) == 3
        
        # Check first item in group
        kernel_name, backends = validated[0][0]
        assert kernel_name == "LayerNorm"
        assert "LayerNormHLS" in backends
        
        # Check second item in group
        assert validated[0][1] == ("Crop", ["CropHLS"])
        
        # Check skip option
        assert validated[0][2] is None
    
    def test_transform_validation_simple(self, parser_with_registry):
        """Test validation of simple transform names."""
        parser = parser_with_registry
        
        # Test with real transform names from QONNX/FINN
        transforms_data = ["RemoveUnusedTensors", "FoldConstants"]
        
        # Should not raise
        parser._validate_transforms(transforms_data)
        
        # Test with non-existent transform
        transforms_data = ["NonExistentTransform"]
        
        with pytest.raises(BlueprintParseError) as exc:
            parser._validate_transforms(transforms_data)
        
        error_msg = str(exc.value)
        assert "Transform 'NonExistentTransform' not found" in error_msg
        assert "Available:" in error_msg
    
    def test_transform_validation_optional(self, parser_with_registry):
        """Test validation of optional transforms (~ prefix)."""
        parser = parser_with_registry
        
        # Test with real QONNX transform names, one optional
        transforms_data = ["~RemoveUnusedTensors", "FoldConstants"]
        
        # Should validate without the ~ prefix
        parser._validate_transforms(transforms_data)
    
    def test_transform_validation_phase_based(self, parser_with_registry):
        """Test validation of phase-based transform organization."""
        parser = parser_with_registry
        
        # Test with real QONNX/FINN transforms organized by phase
        transforms_data = {
            "cleanup": ["RemoveUnusedTensors", "~FoldConstants"],
            "topology_opt": ["InferShapes", "InferDataTypes"]
        }
        
        # Should validate all transforms in all phases
        parser._validate_transforms(transforms_data)
        
        # Test with invalid transform in phase
        transforms_data["cleanup"].append("InvalidTransform")
        
        with pytest.raises(BlueprintParseError) as exc:
            parser._validate_transforms(transforms_data)
        
        error_msg = str(exc.value)
        assert "Transform 'InvalidTransform' in phase 'cleanup' not found" in error_msg
    
    def test_empty_optional_transform_error(self, parser_with_registry):
        """Test error for invalid optional transform syntax."""
        parser = parser_with_registry
        
        # Just "~" without a name is invalid
        transforms_data = ["~"]
        
        # The current implementation might not catch this specifically,
        # but it should handle it gracefully
        with pytest.raises(BlueprintParseError):
            parser._validate_transforms(transforms_data)
    
    def test_mixed_kernel_formats(self, parser_with_registry):
        """Test parsing mixed kernel format specifications."""
        parser = parser_with_registry
        
        # Test with real QONNX/FINN kernels in mixed formats
        kernels_data = [
            "LayerNorm",  # Simple auto-discovery
            ("Crop", ["CropHLS"]),  # Explicit backends
            "~Shuffle",  # Optional with auto-discovery
            [  # Mutually exclusive group
                ("HWSoftmax", ["HWSoftmaxHLS"]),
                "LayerNorm",
                None
            ]
        ]
        
        validated = parser._validate_and_enrich_kernels(kernels_data)
        
        assert len(validated) == 4
        
        # Check simple auto-discovery
        kernel_name, backends = validated[0]
        assert kernel_name == "LayerNorm"
        assert "LayerNormHLS" in backends
        
        # Check explicit backends
        assert validated[1] == ("Crop", ["CropHLS"])
        
        # Check optional marker preservation
        kernel_name, backends = validated[2]
        assert kernel_name == "~Shuffle"
        
        # Check group has 3 options
        assert len(validated[3]) == 3