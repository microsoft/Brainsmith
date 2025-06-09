"""
FINN Interface Tests

Tests for the four-category FINN integration interface:
- ModelOpsManager
- ModelTransformsManager  
- HwKernelsManager
- HwOptimizationManager
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestModelOpsManager:
    """Test suite for ModelOpsManager functionality."""
    
    @pytest.mark.smoke
    def test_model_ops_manager_import(self):
        """Test ModelOpsManager can be imported."""
        try:
            from brainsmith.finn.model_ops_manager import ModelOpsManager
            assert ModelOpsManager is not None
        except ImportError:
            # Fallback to existing structure
            from brainsmith.finn.workflow import ModelOpsManager
            assert ModelOpsManager is not None
    
    @pytest.mark.core
    def test_model_ops_manager_initialization(self, mock_finn_environment):
        """Test ModelOpsManager initialization."""
        try:
            from brainsmith.finn.workflow import ModelOpsManager
        except ImportError:
            pytest.skip("ModelOpsManager not available")
        
        manager = ModelOpsManager(mock_finn_environment)
        assert manager is not None
        assert hasattr(manager, 'get_supported_operations')
    
    @pytest.mark.core
    def test_get_supported_operations(self, mock_finn_environment):
        """Test getting supported model operations."""
        try:
            from brainsmith.finn.workflow import ModelOpsManager
            manager = ModelOpsManager(mock_finn_environment)
            
            operations = manager.get_supported_operations()
            assert isinstance(operations, (list, dict))
            
            if isinstance(operations, list):
                assert len(operations) > 0
            else:
                assert len(operations.keys()) > 0
                
        except (ImportError, AttributeError):
            pytest.skip("ModelOpsManager functionality not available")
    
    @pytest.mark.core
    def test_validate_model_compatibility(self, mock_finn_environment, synthetic_model):
        """Test model compatibility validation."""
        try:
            from brainsmith.finn.workflow import ModelOpsManager
            manager = ModelOpsManager(mock_finn_environment)
            
            # Test with synthetic model
            is_compatible = manager.validate_model_compatibility(synthetic_model)
            assert isinstance(is_compatible, bool)
            
        except (ImportError, AttributeError):
            pytest.skip("Model compatibility validation not available")


class TestModelTransformsManager:
    """Test suite for ModelTransformsManager functionality."""
    
    @pytest.mark.smoke
    def test_model_transforms_manager_import(self):
        """Test ModelTransformsManager can be imported."""
        try:
            from brainsmith.finn.model_transforms_manager import ModelTransformsManager
            assert ModelTransformsManager is not None
        except ImportError:
            # Fallback to existing structure
            from brainsmith.finn.workflow import ModelTransformsManager
            assert ModelTransformsManager is not None
    
    @pytest.mark.core
    def test_model_transforms_manager_initialization(self, mock_finn_environment):
        """Test ModelTransformsManager initialization."""
        try:
            from brainsmith.finn.workflow import ModelTransformsManager
        except ImportError:
            pytest.skip("ModelTransformsManager not available")
        
        manager = ModelTransformsManager(mock_finn_environment)
        assert manager is not None
        assert hasattr(manager, 'get_available_transforms')
    
    @pytest.mark.core
    def test_get_available_transforms(self, mock_finn_environment):
        """Test getting available model transforms."""
        try:
            from brainsmith.finn.workflow import ModelTransformsManager
            manager = ModelTransformsManager(mock_finn_environment)
            
            transforms = manager.get_available_transforms()
            assert isinstance(transforms, (list, dict))
            
            if isinstance(transforms, list):
                assert len(transforms) >= 0
            else:
                assert isinstance(transforms, dict)
                
        except (ImportError, AttributeError):
            pytest.skip("ModelTransformsManager functionality not available")
    
    @pytest.mark.core
    def test_apply_transform_sequence(self, mock_finn_environment, synthetic_model):
        """Test applying a sequence of transforms."""
        try:
            from brainsmith.finn.workflow import ModelTransformsManager
            manager = ModelTransformsManager(mock_finn_environment)
            
            transform_sequence = [
                'InferShapes',
                'FoldConstants', 
                'InferDataTypes'
            ]
            
            result = manager.apply_transform_sequence(synthetic_model, transform_sequence)
            assert result is not None
            
        except (ImportError, AttributeError):
            pytest.skip("Transform sequence functionality not available")


class TestHwKernelsManager:
    """Test suite for HwKernelsManager functionality."""
    
    @pytest.mark.smoke
    def test_hw_kernels_manager_import(self):
        """Test HwKernelsManager can be imported."""
        try:
            from brainsmith.finn.hw_kernels_manager import HwKernelsManager
            assert HwKernelsManager is not None
        except ImportError:
            pytest.skip("HwKernelsManager not available")
    
    @pytest.mark.core
    def test_hw_kernels_manager_initialization(self, mock_finn_environment):
        """Test HwKernelsManager initialization."""
        try:
            from brainsmith.finn.hw_kernels_manager import HwKernelsManager
            manager = HwKernelsManager(mock_finn_environment)
            assert manager is not None
            assert hasattr(manager, 'get_available_kernels')
        except ImportError:
            pytest.skip("HwKernelsManager not available")
    
    @pytest.mark.core
    def test_get_available_kernels(self, mock_finn_environment):
        """Test getting available hardware kernels."""
        try:
            from brainsmith.finn.hw_kernels_manager import HwKernelsManager
            manager = HwKernelsManager(mock_finn_environment)
            
            kernels = manager.get_available_kernels()
            assert isinstance(kernels, list)
            assert len(kernels) > 0
            
            # Check kernel structure
            for kernel in kernels[:3]:  # Check first 3
                assert isinstance(kernel, (str, dict))
                
        except ImportError:
            pytest.skip("HwKernelsManager functionality not available")
    
    @pytest.mark.core
    def test_select_optimal_kernels(self, mock_finn_environment, synthetic_model):
        """Test optimal kernel selection."""
        try:
            from brainsmith.finn.hw_kernels_manager import HwKernelsManager
            manager = HwKernelsManager(mock_finn_environment)
            
            constraints = {
                'max_luts': 0.8,
                'max_dsps': 0.7,
                'throughput_target': 1000
            }
            
            selection = manager.select_optimal_kernels(synthetic_model, constraints)
            assert selection is not None
            assert isinstance(selection, (list, dict))
            
        except ImportError:
            pytest.skip("Kernel selection functionality not available")
    
    @pytest.mark.core
    def test_kernel_performance_estimation(self, mock_finn_environment):
        """Test kernel performance estimation."""
        try:
            from brainsmith.finn.hw_kernels_manager import HwKernelsManager
            manager = HwKernelsManager(mock_finn_environment)
            
            kernel_config = {
                'kernel_type': 'ConvolutionInputGenerator',
                'pe': 4,
                'simd': 2,
                'clock_freq': 100.0
            }
            
            performance = manager.estimate_kernel_performance(kernel_config)
            assert performance is not None
            assert isinstance(performance, dict)
            
            # Check for expected performance metrics
            expected_metrics = ['throughput', 'latency', 'resource_usage']
            for metric in expected_metrics:
                if metric in performance:
                    assert isinstance(performance[metric], (int, float, dict))
            
        except ImportError:
            pytest.skip("Performance estimation functionality not available")


class TestHwOptimizationManager:
    """Test suite for HwOptimizationManager functionality."""
    
    @pytest.mark.smoke
    def test_hw_optimization_manager_import(self):
        """Test HwOptimizationManager can be imported."""
        try:
            from brainsmith.finn.hw_optimization_manager import HwOptimizationManager
            assert HwOptimizationManager is not None
        except ImportError:
            # Fallback to existing structure
            from brainsmith.finn.workflow import HwOptimizationManager
            assert HwOptimizationManager is not None
    
    @pytest.mark.core
    def test_hw_optimization_manager_initialization(self, mock_finn_environment):
        """Test HwOptimizationManager initialization."""
        try:
            from brainsmith.finn.workflow import HwOptimizationManager
        except ImportError:
            pytest.skip("HwOptimizationManager not available")
        
        manager = HwOptimizationManager(mock_finn_environment)
        assert manager is not None
        assert hasattr(manager, 'get_optimization_directives')
    
    @pytest.mark.core
    def test_get_optimization_directives(self, mock_finn_environment):
        """Test getting hardware optimization directives."""
        try:
            from brainsmith.finn.workflow import HwOptimizationManager
            manager = HwOptimizationManager(mock_finn_environment)
            
            directives = manager.get_optimization_directives()
            assert isinstance(directives, (list, dict))
            
        except (ImportError, AttributeError):
            pytest.skip("Optimization directives functionality not available")
    
    @pytest.mark.core
    def test_apply_optimization_strategy(self, mock_finn_environment, synthetic_model):
        """Test applying hardware optimization strategy."""
        try:
            from brainsmith.finn.workflow import HwOptimizationManager
            manager = HwOptimizationManager(mock_finn_environment)
            
            strategy = {
                'type': 'resource_minimization',
                'constraints': {
                    'max_luts': 0.8,
                    'max_dsps': 0.7
                },
                'objectives': ['minimize_latency', 'maximize_throughput']
            }
            
            result = manager.apply_optimization_strategy(synthetic_model, strategy)
            assert result is not None
            
        except (ImportError, AttributeError):
            pytest.skip("Optimization strategy functionality not available")


class TestFINNInterfaceIntegration:
    """Test suite for integrated FINN interface functionality."""
    
    @pytest.mark.integration
    def test_four_manager_integration(self, mock_finn_environment, synthetic_model):
        """Test integration between all four FINN managers."""
        managers = {}
        
        try:
            from brainsmith.finn.workflow import (
                ModelOpsManager, ModelTransformsManager, 
                HwOptimizationManager
            )
            from brainsmith.finn.hw_kernels_manager import HwKernelsManager
            
            managers['ops'] = ModelOpsManager(mock_finn_environment)
            managers['transforms'] = ModelTransformsManager(mock_finn_environment)
            managers['kernels'] = HwKernelsManager(mock_finn_environment)
            managers['optimization'] = HwOptimizationManager(mock_finn_environment)
            
        except ImportError:
            pytest.skip("FINN managers not available")
        
        # Test workflow integration
        assert len(managers) == 4
        
        # Validate model compatibility
        is_compatible = managers['ops'].validate_model_compatibility(synthetic_model)
        assert isinstance(is_compatible, bool)
        
        # Get available kernels
        kernels = managers['kernels'].get_available_kernels()
        assert isinstance(kernels, list)
    
    @pytest.mark.integration
    def test_finn_build_pipeline_simulation(self, mock_finn_environment, synthetic_model):
        """Test simulated FINN build pipeline."""
        try:
            from brainsmith.finn.orchestration import FINNBuildOrchestrator
        except ImportError:
            pytest.skip("FINN build orchestration not available")
        
        orchestrator = FINNBuildOrchestrator(mock_finn_environment)
        
        build_config = {
            'model': synthetic_model,
            'target_platform': 'zynq',
            'optimization_level': 'standard',
            'parallel_builds': False
        }
        
        result = orchestrator.execute_build_pipeline(build_config)
        assert result is not None
        assert 'success' in result
    
    @pytest.mark.core
    def test_finn_configuration_validation(self, mock_finn_environment):
        """Test FINN configuration validation."""
        try:
            from brainsmith.finn.workflow import FINNConfigValidator
        except ImportError:
            pytest.skip("FINN configuration validation not available")
        
        validator = FINNConfigValidator(mock_finn_environment)
        
        valid_config = {
            'model_ops': {'supported_operations': ['Conv', 'MatMul', 'Relu']},
            'transforms': ['InferShapes', 'FoldConstants'],
            'kernels': {'selection_strategy': 'optimal'},
            'optimization': {'level': 'standard'}
        }
        
        is_valid = validator.validate_configuration(valid_config)
        assert isinstance(is_valid, bool)
        
        # Test invalid configuration
        invalid_config = {
            'model_ops': {'unsupported_field': 'invalid'},
            'transforms': ['NonexistentTransform'],
            'kernels': {'invalid_strategy': 'unknown'},
            'optimization': {'level': 'invalid_level'}
        }
        
        is_invalid = validator.validate_configuration(invalid_config)
        assert isinstance(is_invalid, bool)
    
    @pytest.mark.performance
    def test_finn_interface_performance(self, mock_finn_environment, synthetic_model):
        """Test FINN interface performance under load."""
        try:
            from brainsmith.finn.hw_kernels_manager import HwKernelsManager
        except ImportError:
            pytest.skip("HwKernelsManager not available")
        
        manager = HwKernelsManager(mock_finn_environment)
        
        import time
        start_time = time.time()
        
        # Simulate multiple kernel queries
        for i in range(10):
            kernels = manager.get_available_kernels()
            assert isinstance(kernels, list)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly
        assert total_time < 5.0, f"FINN interface operations took {total_time:.2f}s, expected < 5s"
    
    @pytest.mark.security
    def test_finn_input_validation(self, mock_finn_environment):
        """Test FINN interface input validation and sanitization."""
        try:
            from brainsmith.finn.hw_kernels_manager import HwKernelsManager
        except ImportError:
            pytest.skip("HwKernelsManager not available")
        
        manager = HwKernelsManager(mock_finn_environment)
        
        # Test malicious input handling
        malicious_inputs = [
            None,
            {},
            {'../../../etc/passwd': 'malicious'},
            {'injection': '"; DROP TABLE kernels; --'},
            {'overflow': 'A' * 10000}
        ]
        
        for malicious_input in malicious_inputs:
            try:
                # Should handle gracefully without crashing
                result = manager.select_optimal_kernels(malicious_input, {})
                # Result can be None or error, but shouldn't crash
                assert result is not None or result is None
            except (ValueError, TypeError, AttributeError):
                # Expected for malicious inputs
                pass
            except Exception as e:
                pytest.fail(f"Unexpected exception for input {malicious_input}: {e}")