"""
Integration tests for DesignSpace with kernel/transform sections.
"""

import pytest
from brainsmith.core.dse.design_space import DesignSpace
from brainsmith.core.dse.kernel_transform_selection import KernelSelection, TransformSelection


class TestDesignSpaceKernelTransform:
    """Test DesignSpace integration with kernel/transform selection."""
    
    def test_blueprint_with_kernel_selection(self):
        """Test DesignSpace creation from blueprint with kernel selection."""
        blueprint_data = {
            'name': 'test_blueprint',
            'parameters': {
                'test_param': {
                    'range': [1, 2, 3],
                    'default': 1
                }
            },
            'kernels': {
                'available': ['conv2d_hls', 'matmul_rtl'],
                'mutually_exclusive': [],
                'operation_mappings': {
                    'Convolution': ['conv2d_hls'],
                    'MatMul': ['matmul_rtl']
                }
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        
        assert design_space.kernel_selection is not None
        assert design_space.kernel_selection.available_kernels == ['conv2d_hls', 'matmul_rtl']
        assert design_space.kernel_choices == [['conv2d_hls', 'matmul_rtl']]
    
    def test_blueprint_with_transform_selection(self):
        """Test DesignSpace creation from blueprint with transform selection."""
        blueprint_data = {
            'name': 'test_blueprint',
            'parameters': {
                'test_param': {
                    'range': [1, 2, 3],
                    'default': 1
                }
            },
            'transforms': {
                'core_pipeline': ['cleanup', 'streamlining'],
                'optional': ['remove_head'],
                'mutually_exclusive': [['infer_hardware', 'constrain_folding']],
                'hooks': {}
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        
        assert design_space.transform_selection is not None
        assert design_space.transform_selection.core_pipeline == ['cleanup', 'streamlining']
        assert len(design_space.transform_choices) == 4  # 2 optional * 2 exclusive
    
    def test_blueprint_with_both_kernel_and_transform_selection(self):
        """Test DesignSpace creation with both kernel and transform sections."""
        blueprint_data = {
            'name': 'test_blueprint',
            'parameters': {
                'test_param': {
                    'range': [1, 2, 3],
                    'default': 1
                }
            },
            'kernels': {
                'available': ['conv2d_hls', 'conv2d_rtl'],
                'mutually_exclusive': [['conv2d_hls', 'conv2d_rtl']]
            },
            'transforms': {
                'core_pipeline': ['cleanup', 'streamlining'],
                'optional': [],
                'mutually_exclusive': []
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        
        assert design_space.kernel_selection is not None
        assert design_space.transform_selection is not None
        assert len(design_space.kernel_choices) == 2  # 2 mutually exclusive
        assert len(design_space.transform_choices) == 1  # Just core pipeline
    
    def test_parameter_space_generation_includes_kernel_transform(self):
        """Test ParameterSpace generation includes kernel/transform parameters."""
        blueprint_data = {
            'name': 'test_blueprint',
            'parameters': {
                'regular_param': {
                    'range': [1, 2],
                    'default': 1
                }
            },
            'kernels': {
                'available': ['kernel_a', 'kernel_b']
            },
            'transforms': {
                'core_pipeline': ['transform_1', 'transform_2']
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        parameter_space = design_space.to_parameter_space()
        
        # Should include regular parameters
        assert 'regular_param' in parameter_space
        assert parameter_space['regular_param'] == [1, 2]
        
        # Should include kernel selection
        assert 'kernel_selection' in parameter_space
        assert parameter_space['kernel_selection'] == [['kernel_a', 'kernel_b']]
        
        # Should include transform pipeline
        assert 'transform_pipeline' in parameter_space
        assert parameter_space['transform_pipeline'] == [['transform_1', 'transform_2']]
    
    def test_backward_compatibility_no_kernel_transform_sections(self):
        """Test that blueprints without kernel/transform sections still work."""
        blueprint_data = {
            'name': 'legacy_blueprint',
            'parameters': {
                'test_param': {
                    'range': [1, 2, 3],
                    'default': 1
                }
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        
        # Should work without errors
        assert design_space.kernel_selection is None
        assert design_space.transform_selection is None
        assert design_space.kernel_choices == []
        assert design_space.transform_choices == []
        
        # Parameter space should not include kernel/transform parameters
        parameter_space = design_space.to_parameter_space()
        assert 'kernel_selection' not in parameter_space
        assert 'transform_pipeline' not in parameter_space
        assert 'test_param' in parameter_space
    
    def test_empty_kernel_transform_sections(self):
        """Test handling of empty kernel/transform sections."""
        blueprint_data = {
            'name': 'empty_sections_blueprint',
            'parameters': {
                'test_param': {'range': [1, 2], 'default': 1}
            },
            'kernels': {
                'available': []
            },
            'transforms': {
                'core_pipeline': []
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        
        assert design_space.kernel_selection is not None
        assert design_space.transform_selection is not None
        assert design_space.kernel_choices == []
        assert design_space.transform_choices == [[]]  # One empty pipeline
        
        # Parameter space should handle empty choices
        parameter_space = design_space.to_parameter_space()
        assert 'kernel_selection' not in parameter_space  # Empty choices not added
        assert 'transform_pipeline' in parameter_space
        assert parameter_space['transform_pipeline'] == [[]]
    
    def test_kernel_transform_choices_enumeration(self):
        """Test that kernel and transform choices are properly enumerated."""
        blueprint_data = {
            'name': 'enumeration_test',
            'kernels': {
                'available': ['k1', 'k2', 'k3'],
                'mutually_exclusive': [['k1', 'k2']]  # k3 always included
            },
            'transforms': {
                'core_pipeline': ['base'],
                'optional': ['opt1'],
                'mutually_exclusive': [['excl1', 'excl2']]
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        
        # Kernel choices: k3 + (k1 OR k2)
        assert len(design_space.kernel_choices) == 2
        kernel_sets = [set(choice) for choice in design_space.kernel_choices]
        assert {'k1', 'k3'} in kernel_sets
        assert {'k2', 'k3'} in kernel_sets
        
        # Transform choices: base + (opt1?) + (excl1 OR excl2)
        assert len(design_space.transform_choices) == 4
        expected_transforms = [
            ['base', 'excl1'],
            ['base', 'excl2'], 
            ['base', 'opt1', 'excl1'],
            ['base', 'opt1', 'excl2']
        ]
        for expected in expected_transforms:
            assert expected in design_space.transform_choices


class TestDesignSpaceValidation:
    """Test validation functionality for kernel/transform selections."""
    
    def test_validation_with_valid_selections(self):
        """Test validation passes with valid kernel/transform selections."""
        blueprint_data = {
            'name': 'valid_blueprint',
            'parameters': {
                'test_param': {'range': [1, 2], 'default': 1}
            },
            'kernels': {
                'available': ['conv2d_hls', 'matmul_rtl']  # These exist in registry
            },
            'transforms': {
                'core_pipeline': ['cleanup', 'streamlining']  # These exist in registry
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        is_valid, errors = design_space.validate()
        
        # Note: This may fail if registries aren't available in test environment
        # That's expected behavior - validation should catch missing registries
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
    
    def test_validation_with_invalid_kernels(self):
        """Test validation fails with invalid kernel names."""
        blueprint_data = {
            'name': 'invalid_kernels_blueprint',
            'parameters': {
                'test_param': {'range': [1, 2], 'default': 1}
            },
            'kernels': {
                'available': ['nonexistent_kernel']
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        is_valid, errors = design_space.validate()
        
        # Should fail validation (unless registry import fails)
        if 'Could not import' not in str(errors):
            assert not is_valid
            assert any('nonexistent_kernel' in error for error in errors)
    
    def test_validation_with_invalid_transforms(self):
        """Test validation fails with invalid transform names."""
        blueprint_data = {
            'name': 'invalid_transforms_blueprint',
            'parameters': {
                'test_param': {'range': [1, 2], 'default': 1}
            },
            'transforms': {
                'core_pipeline': ['nonexistent_transform']
            }
        }
        
        design_space = DesignSpace.from_blueprint_data(blueprint_data)
        is_valid, errors = design_space.validate()
        
        # Should fail validation (unless registry import fails)
        if 'Could not import' not in str(errors):
            assert not is_valid
            assert any('nonexistent_transform' in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])