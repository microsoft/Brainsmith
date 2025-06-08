"""
Test suite for FINN Interface

Tests FINN integration layer with legacy DataflowBuildConfig support
and future 4-hook interface placeholder.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Import components to test
try:
    from brainsmith.core.finn_interface import FINNInterface, FINNHooksPlaceholder
except ImportError as e:
    print(f"Warning: Could not import FINN interface components: {e}")
    FINNInterface = None
    FINNHooksPlaceholder = None


class TestFINNHooksPlaceholder(unittest.TestCase):
    """Test cases for FINNHooksPlaceholder."""
    
    def setUp(self):
        """Set up test fixtures."""
        if FINNHooksPlaceholder is None:
            self.skipTest("FINNHooksPlaceholder not available")
        
        self.hooks = FINNHooksPlaceholder()
    
    def test_hooks_initialization(self):
        """Test that hooks placeholder initializes correctly."""
        self.assertIsNotNone(self.hooks)
        self.assertIsNone(self.hooks.preprocessing_hook)
        self.assertIsNone(self.hooks.transformation_hook)
        self.assertIsNone(self.hooks.optimization_hook)
        self.assertIsNone(self.hooks.generation_hook)
        self.assertIsInstance(self.hooks.hook_config, dict)
    
    def test_is_available(self):
        """Test that 4-hook interface is not available yet."""
        self.assertFalse(self.hooks.is_available())
    
    def test_prepare_for_future_interface(self):
        """Test preparation of configuration for future interface."""
        design_point = {
            'preprocessing': {'param1': 'value1'},
            'transforms': {'param2': 'value2'},
            'hw_optimization': {'param3': 'value3'},
            'generation': {'param4': 'value4'}
        }
        
        config = self.hooks.prepare_for_future_interface(design_point)
        
        self.assertIsInstance(config, dict)
        expected_keys = ['preprocessing_config', 'transformation_config', 'optimization_config', 'generation_config']
        for key in expected_keys:
            self.assertIn(key, config)
        
        # Verify mapping
        self.assertEqual(config['preprocessing_config'], design_point['preprocessing'])
        self.assertEqual(config['transformation_config'], design_point['transforms'])
        self.assertEqual(config['optimization_config'], design_point['hw_optimization'])
        self.assertEqual(config['generation_config'], design_point['generation'])
    
    def test_validate_hook_config(self):
        """Test hook configuration validation."""
        is_valid, errors = self.hooks.validate_hook_config()
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(errors, list)
        
        # Should be valid with empty config
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_custom_hook_config(self):
        """Test hooks with custom configuration."""
        custom_config = {
            'preprocessing': {'enabled': True},
            'transformation': {'enabled': True},
            'optimization': {'enabled': False},
            'generation': {'enabled': True}
        }
        
        hooks = FINNHooksPlaceholder(hook_config=custom_config)
        self.assertEqual(hooks.hook_config, custom_config)


class TestFINNInterface(unittest.TestCase):
    """Test cases for FINNInterface."""
    
    def setUp(self):
        """Set up test fixtures."""
        if FINNInterface is None or FINNHooksPlaceholder is None:
            self.skipTest("FINN interface components not available")
        
        self.legacy_config = {
            'fpga_part': 'xcvu9p-flga2104-2-i',
            'auto_fifo_depths': True,
            'generate_outputs': ['estimate', 'bitfile']
        }
        self.hooks = FINNHooksPlaceholder()
        self.finn_interface = FINNInterface(self.legacy_config, self.hooks)
    
    def test_interface_initialization(self):
        """Test that FINN interface initializes correctly."""
        self.assertIsNotNone(self.finn_interface)
        self.assertEqual(self.finn_interface.legacy_config, self.legacy_config)
        self.assertEqual(self.finn_interface.future_hooks, self.hooks)
        self.assertTrue(self.finn_interface.use_legacy)  # Should always be True for now
    
    def test_get_interface_status(self):
        """Test interface status reporting."""
        status = self.finn_interface.get_interface_status()
        
        self.assertIsInstance(status, dict)
        expected_keys = ['using_legacy', 'future_hooks_available', 'legacy_config_keys', 'interface_ready']
        for key in expected_keys:
            self.assertIn(key, status)
        
        self.assertTrue(status['using_legacy'])
        self.assertFalse(status['future_hooks_available'])
        self.assertTrue(status['interface_ready'])
        self.assertEqual(status['legacy_config_keys'], list(self.legacy_config.keys()))
    
    def test_generate_implementation_existing(self):
        """Test implementation generation using existing interface."""
        model_path = "test_model.onnx"
        design_point = {
            'kernels': {'simd': 4, 'pe': 4},
            'transforms': {'streamlining': True},
            'hw_optimization': {'strategy': 'random'},
            'finn_config': {'auto_fifo_depths': True}
        }
        
        # Mock the actual FINN build to avoid import dependencies
        with patch.object(self.finn_interface, '_execute_existing_finn_build') as mock_build:
            mock_build.return_value = {
                'rtl_files': ['test.v'],
                'hls_files': ['test.cpp'],
                'synthesis_results': {'status': 'success'},
                'throughput': 1000,
                'latency': 10
            }
            
            result = self.finn_interface.generate_implementation_existing(model_path, design_point)
            
            self.assertIsInstance(result, dict)
            expected_keys = ['rtl_files', 'hls_files', 'synthesis_results', 'performance_metrics', 'interface_type', 'status']
            for key in expected_keys:
                self.assertIn(key, result)
            
            self.assertEqual(result['interface_type'], 'legacy_dataflow_build_config')
            self.assertEqual(result['status'], 'success')
            
            # Verify mock was called
            mock_build.assert_called_once()
    
    def test_create_legacy_build_config(self):
        """Test creation of legacy build configuration."""
        design_point = {
            'kernels': {'simd': 8, 'pe': 8},
            'transforms': {'folding': True},
            'hw_optimization': {'budget': 100},
            'finn_config': {'generate_outputs': ['estimate']}
        }
        
        # Mock the method directly instead of trying to patch the import
        with patch.object(self.finn_interface, '_create_mock_build_config') as mock_create:
            mock_config = {'mock_config': True}
            mock_create.return_value = mock_config
            
            config = self.finn_interface._create_legacy_build_config(design_point)
            
            # Should return the mocked config
            self.assertEqual(config, mock_config)
            mock_create.assert_called_once()
    
    def test_parameter_mapping(self):
        """Test mapping of design point parameters to legacy format."""
        design_point = {
            'kernels': {'param1': 'value1'},
            'transforms': {'param2': 'value2'},
            'hw_optimization': {'param3': 'value3'},
            'finn_config': {'param4': 'value4'}
        }
        
        mapped_params = self.finn_interface._map_design_point_to_legacy_config(design_point)
        
        self.assertIsInstance(mapped_params, dict)
        
        # Check that all parameter categories are mapped
        expected_mappings = ['kernel_params', 'transform_params', 'optimization_params']
        for mapping in expected_mappings:
            self.assertIn(mapping, mapped_params)
        
        # Check FINN config is directly included
        self.assertEqual(mapped_params['param4'], 'value4')
    
    def test_performance_metrics_extraction(self):
        """Test extraction of performance metrics from build results."""
        build_results = {
            'throughput': 2000,
            'latency': 5,
            'clock_freq': 200,
            'efficiency': 0.85,
            'performance_analysis': {'additional_metric': 123}
        }
        
        metrics = self.finn_interface._extract_performance_metrics(build_results)
        
        self.assertIsInstance(metrics, dict)
        expected_keys = ['throughput_ops_sec', 'latency_ms', 'clock_frequency_mhz', 'resource_efficiency']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        self.assertEqual(metrics['throughput_ops_sec'], 2000)
        self.assertEqual(metrics['latency_ms'], 5)
        self.assertEqual(metrics['clock_frequency_mhz'], 200)
        self.assertEqual(metrics['resource_efficiency'], 0.85)
        self.assertIn('additional_metric', metrics)
    
    def test_error_handling_build_failure(self):
        """Test error handling when FINN build fails."""
        model_path = "test_model.onnx"
        design_point = {'test': 'config'}
        
        # Mock build failure
        with patch.object(self.finn_interface, '_execute_existing_finn_build') as mock_build:
            mock_build.side_effect = Exception("Build failed")
            
            result = self.finn_interface.generate_implementation_existing(model_path, design_point)
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('error', result)
            self.assertIn('fallback_results', result)
    
    def test_future_interface_placeholder(self):
        """Test future 4-hook interface placeholder functionality."""
        model_path = "test_model.onnx"
        design_point = {
            'preprocessing': {'param1': 'value1'},
            'transforms': {'param2': 'value2'},
            'hw_optimization': {'param3': 'value3'},
            'generation': {'param4': 'value4'}
        }
        
        # Temporarily set interface to use future hooks (for testing)
        self.finn_interface.use_legacy = False
        
        result = self.finn_interface._generate_with_future_interface(model_path, design_point)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['interface_type'], 'future_4_hook_interface')
        self.assertEqual(result['status'], 'placeholder_implementation')
        self.assertIn('hook_results', result)
        
        # Check all hooks were executed
        hook_results = result['hook_results']
        expected_hooks = ['preprocessing_results', 'transformation_results', 'optimization_results', 'generation_results']
        for hook in expected_hooks:
            self.assertIn(hook, hook_results)
        
        # Reset to legacy
        self.finn_interface.use_legacy = True
    
    def test_mock_build_results(self):
        """Test creation of mock build results when FINN not available."""
        model_path = "test_model.onnx"
        build_config = {'mock': 'config'}
        
        result = self.finn_interface._create_mock_build_results(model_path, build_config)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result['mock_results'])
        self.assertEqual(result['model_path'], model_path)
        self.assertEqual(result['status'], 'mock_success')
        self.assertIn('rtl_files', result)
        self.assertIn('hls_files', result)


class TestFINNInterfaceIntegration(unittest.TestCase):
    """Integration tests for FINN interface with various scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        if FINNInterface is None or FINNHooksPlaceholder is None:
            self.skipTest("FINN interface components not available")
    
    def test_interface_with_minimal_config(self):
        """Test interface with minimal configuration."""
        minimal_config = {}
        hooks = FINNHooksPlaceholder()
        interface = FINNInterface(minimal_config, hooks)
        
        self.assertIsNotNone(interface)
        self.assertEqual(interface.legacy_config, minimal_config)
        self.assertTrue(interface.use_legacy)
    
    def test_interface_with_comprehensive_config(self):
        """Test interface with comprehensive configuration."""
        comprehensive_config = {
            'fpga_part': 'xczu7ev-ffvc1156-2-e',
            'auto_fifo_depths': True,
            'generate_outputs': ['estimate', 'bitfile', 'deployment'],
            'target_clk_ns': 5.0,
            'board': 'Pynq-Z1',
            'shell_flow_type': 'vivado_synth'
        }
        
        hooks = FINNHooksPlaceholder(hook_config={
            'preprocessing': {'enabled': True},
            'transformation': {'enabled': True},
            'optimization': {'enabled': True},
            'generation': {'enabled': True}
        })
        
        interface = FINNInterface(comprehensive_config, hooks)
        
        # Test status with comprehensive config
        status = interface.get_interface_status()
        self.assertEqual(len(status['legacy_config_keys']), len(comprehensive_config))
        
        # Test generation with comprehensive config
        design_point = {
            'kernels': {'simd': 16, 'pe': 16},
            'transforms': {'streamlining': True, 'folding': True},
            'hw_optimization': {'strategy': 'bayesian', 'budget': 50},
            'finn_config': {'target_clk_ns': 3.0}
        }
        
        with patch.object(interface, '_execute_existing_finn_build') as mock_build:
            mock_build.return_value = {
                'rtl_files': ['design.v', 'testbench.v'],
                'hls_files': ['kernel.cpp', 'kernel.h'],
                'synthesis_results': {'lut_utilization': 0.75, 'timing_met': True}
            }
            
            result = interface.generate_implementation_existing("model.onnx", design_point)
            
            self.assertEqual(result['status'], 'success')
            self.assertEqual(len(result['rtl_files']), 2)
            self.assertEqual(len(result['hls_files']), 2)
    
    def test_transition_readiness(self):
        """Test readiness for transition to 4-hook interface."""
        hooks = FINNHooksPlaceholder()
        interface = FINNInterface({}, hooks)
        
        # Test that future configuration can be prepared
        design_point = {
            'preprocessing': {'normalize': True},
            'transforms': {'quantize': False},  # No quantization per requirements
            'hw_optimization': {'parallel_strategy': 'data_parallel'},
            'generation': {'target_platform': 'fpga'}
        }
        
        future_config = hooks.prepare_for_future_interface(design_point)
        
        # Verify structure is ready for 4-hook interface
        self.assertIn('preprocessing_config', future_config)
        self.assertIn('transformation_config', future_config)
        self.assertIn('optimization_config', future_config)
        self.assertIn('generation_config', future_config)
        
        # Verify no quantization in transforms (per architectural requirements)
        self.assertFalse(future_config['transformation_config'].get('quantize', True))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)