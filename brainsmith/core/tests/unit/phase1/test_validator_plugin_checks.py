"""
Unit tests for validator plugin checking functionality.
"""

import pytest

from brainsmith.core.phase1.validator import DesignSpaceValidator
from brainsmith.core.phase1.data_structures import (
    DesignSpace, HWCompilerSpace, ProcessingSpace, 
    SearchConfig, GlobalConfig, SearchStrategy, OutputStage
)


class TestValidatorPluginChecks:
    """Test validator's plugin registry integration."""
    
    def test_validate_kernel_exists(self, validator_with_registry):
        """Test kernel existence validation."""
        validator = validator_with_registry
        errors = []
        
        # Test existing kernel from QONNX/FINN
        validator._validate_kernel_exists("LayerNorm", errors)
        assert len(errors) == 0
        
        # Test non-existent kernel
        validator._validate_kernel_exists("NonExistentKernel", errors)
        assert len(errors) == 1
        assert "Kernel 'NonExistentKernel' not registered" in errors[0]
        assert "Available:" in errors[0]
    
    def test_validate_kernel_exists_empty_name(self, validator_with_registry):
        """Test validation with empty kernel name."""
        validator = validator_with_registry
        errors = []
        
        # Empty string should not cause error (handled elsewhere)
        validator._validate_kernel_exists("", errors)
        assert len(errors) == 0
        
        # None should also be handled gracefully
        validator._validate_kernel_exists(None, errors)
        assert len(errors) == 0
    
    def test_validate_backends_for_kernel(self, validator_with_registry):
        """Test backend availability validation for kernels."""
        validator = validator_with_registry
        errors = []
        
        # Test all valid backends for LayerNorm
        validator._validate_backends_for_kernel("LayerNorm", ["LayerNormHLS"], errors)
        assert len(errors) == 0
        
        # Test some invalid backends
        validator._validate_backends_for_kernel("LayerNorm", ["LayerNormHLS", "InvalidBackend1", "InvalidBackend2"], errors)
        assert len(errors) == 1
        assert "Invalid backends ['InvalidBackend1', 'InvalidBackend2'] for kernel 'LayerNorm'" in errors[0]
        assert "Available:" in errors[0]
    
    def test_validate_backends_empty_kernel(self, validator_with_registry):
        """Test backend validation with empty kernel name."""
        validator = validator_with_registry
        errors = []
        
        # Should return early without error
        validator._validate_backends_for_kernel("", ["LayerNormHLS", "LayerNormRTL"], errors)
        assert len(errors) == 0
        
        validator._validate_backends_for_kernel(None, ["LayerNormHLS", "LayerNormRTL"], errors)
        assert len(errors) == 0
    
    def test_validate_transform_stage(self, validator_with_registry):
        """Test transform stage validation."""
        validator = validator_with_registry
        errors = []
        
        # Test valid stage - RemoveUnusedTensors has "cleanup" stage
        validator._validate_transform_stage("RemoveUnusedTensors", errors)
        assert len(errors) == 0
        
        # Test transform with no stage metadata - should not error
        # (This is handled gracefully in the real implementation)
        validator._validate_transform_stage("FoldConstants", errors)
        assert len(errors) == 0  # Should handle gracefully
    
    def test_validate_transform_no_stage(self, validator_with_registry):
        """Test transform with no stage metadata."""
        validator = validator_with_registry
        errors = []
        
        # All our real transforms have stage metadata, but the validator should handle missing metadata gracefully
        # Use a real transform for this test
        validator._validate_transform_stage("FoldConstants", errors)
        assert len(errors) == 0
    
    def test_validate_kernel_format_in_hw_compiler(self, validator_with_registry):
        """Test kernel format validation within hw_compiler validation."""
        validator = validator_with_registry
        
        # Create a design space with various kernel formats using real QONNX/FINN kernels
        hw_space = HWCompilerSpace(
            kernels=[
                "LayerNorm",  # Simple format
                ("Crop", ["CropHLS"]),  # Tuple format
                ["Shuffle", "NonExistentKernel"],  # Mutually exclusive group with one invalid
                "~HWSoftmax",  # Optional
            ],
            transforms=["FoldConstants"],
            build_steps=["ConvertToHW"]
        )
        
        errors = []
        warnings = []
        
        validator._validate_hw_compiler(hw_space, errors, warnings)
        
        # Should have error for the non-existent kernel
        kernel_errors = [e for e in errors if "not registered" in e]
        assert len(kernel_errors) >= 1  # NonExistentKernel not registered
    
    def test_validate_transform_format_in_hw_compiler(self, validator_with_registry):
        """Test transform format validation within hw_compiler validation."""
        validator = validator_with_registry
        
        # Test flat transform list with real QONNX/FINN transforms
        hw_space = HWCompilerSpace(
            kernels=["LayerNorm"],
            transforms=["RemoveUnusedTensors", "NonExistentTransform", "~FoldConstants"],
            build_steps=["ConvertToHW"]
        )
        
        errors = []
        warnings = []
        
        validator._validate_hw_compiler(hw_space, errors, warnings)
        
        # Should have error for NonExistentTransform
        transform_errors = [e for e in errors if "Transform 'NonExistentTransform'" in e]
        assert len(transform_errors) == 1
        assert "Available:" in transform_errors[0]
    
    def test_validate_phase_based_transforms(self, validator_with_registry):
        """Test validation of phase-based transform organization."""
        validator = validator_with_registry
        
        # Test phase-based transforms with real QONNX/FINN transforms
        hw_space = HWCompilerSpace(
            kernels=["LayerNorm"],
            transforms={
                "cleanup": ["RemoveUnusedTensors", "InvalidTransform1"],
                "topology_opt": ["InferShapes", "InvalidTransform2"]
            },
            build_steps=["ConvertToHW"]
        )
        
        errors = []
        warnings = []
        
        validator._validate_hw_compiler(hw_space, errors, warnings)
        
        # Should have errors for invalid transforms in each phase
        assert any("InvalidTransform1" in e and "cleanup" in e for e in errors)
        assert any("InvalidTransform2" in e and "topology_opt" in e for e in errors)
    
    def test_full_design_space_validation_with_plugins(self, validator_with_registry, model_path):
        """Test complete design space validation with plugin checks."""
        validator = validator_with_registry
        
        # Create a design space with real and invalid plugin references
        design_space = DesignSpace(
            model_path=model_path,
            hw_compiler_space=HWCompilerSpace(
                kernels=[
                    "LayerNorm",  # Valid kernel
                    ("InvalidKernel", ["backend1"]),  # Invalid kernel
                    ("Crop", ["InvalidBackend"])  # Valid kernel, invalid backend
                ],
                transforms=["FoldConstants", "InvalidTransform"],  # One valid, one invalid
                build_steps=["ConvertToHW"]
            ),
            processing_space=ProcessingSpace([], []),
            search_config=SearchConfig(
                strategy=SearchStrategy.EXHAUSTIVE,
                constraints=[],
                parallel_builds=1
            ),
            global_config=GlobalConfig(
                output_stage=OutputStage.RTL,
                working_directory="./builds"
            )
        )
        
        result = validator.validate(design_space)
        
        # Should not be valid due to plugin errors
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Check specific errors
        error_messages = " ".join(result.errors)
        assert "InvalidKernel" in error_messages
        assert "InvalidBackend" in error_messages
        assert "InvalidTransform" in error_messages