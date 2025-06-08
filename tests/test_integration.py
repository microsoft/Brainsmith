"""
Integration test suite for Brainsmith Week 1 implementation.

Tests the complete workflow from API calls through orchestration
to result generation using existing components.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import components for integration testing
try:
    from brainsmith.core import (
        DesignSpaceOrchestrator, FINNInterface, WorkflowManager,
        brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
        brainsmith_generate, get_core_status
    )
except ImportError as e:
    print(f"Warning: Could not import core components for integration testing: {e}")
    DesignSpaceOrchestrator = None
    FINNInterface = None
    WorkflowManager = None
    brainsmith_explore = None
    get_core_status = None


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        if brainsmith_explore is None:
            self.skipTest("Core components not available for integration testing")
        
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "integration_model.onnx")
        self.blueprint_path = os.path.join(self.temp_dir, "integration_blueprint.yaml")
        
        # Create mock model file
        with open(self.model_path, 'w') as f:
            f.write("# Mock ONNX model for integration testing")
        
        # Create comprehensive blueprint
        blueprint_content = """
name: "integration_test_blueprint"
description: "Comprehensive blueprint for integration testing"

kernels:
  available:
    - name: "conv2d_hls"
      type: "Conv"
      source: "custom_op.hw_conv2d"
      parameters:
        simd: [1, 2, 4, 8]
        pe: [1, 2, 4, 8]
    - name: "linear_hls"
      type: "MatMul"
      source: "custom_op.hw_linear"
      parameters:
        simd: [1, 4, 8, 16]
        pe: [1, 4, 8, 16]

transforms:
  pipeline:
    - name: "streamlining"
      enabled: true
      source: "steps.streamlining"
    - name: "folding"
      enabled: true
      source: "steps.folding"
    - name: "partitioning"
      enabled: false
      source: "steps.partitioning"

hw_optimization:
  strategies:
    - name: "random_search"
      algorithm: "random"
      budget: 20
      source: "dse.random"
    - name: "grid_search"
      algorithm: "grid"
      budget: 10
      source: "dse.grid"

analysis:
  tools:
    - name: "roofline"
      source: "analysis.roofline"
    - name: "performance_estimation"
      source: "analysis.estimation"

finn_interface:
  legacy_config:
    fpga_part: "xcvu9p-flga2104-2-i"
    auto_fifo_depths: true
    generate_outputs: ["estimate", "bitfile"]
    target_clk_ns: 5.0
    board: "Pynq-Z1"

constraints:
  target_device: "xcvu9p-flga2104-2-i"
  resource_limits:
    lut_utilization: 0.85
    bram_utilization: 0.90
    dsp_utilization: 0.95
  performance_targets:
    min_throughput_ops_sec: 1000
    max_latency_ms: 100

objectives:
  - name: "maximize_throughput"
    weight: 0.7
  - name: "minimize_latency"
    weight: 0.2
  - name: "minimize_resources"
    weight: 0.1

metadata:
  created_for: "integration_testing"
  components_used: "existing_only"
  test_version: "week1"
"""
        
        with open(self.blueprint_path, 'w') as f:
            f.write(blueprint_content)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('brainsmith.core.design_space_orchestrator.DesignSpaceOrchestrator')
    def test_complete_roofline_workflow(self, mock_orchestrator_class):
        """Test complete roofline analysis workflow."""
        # Setup comprehensive mock
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.analysis = {
            'exit_point': 'roofline',
            'method': 'existing_analysis_tools',
            'roofline_results': {
                'compute_intensity': 2.5,
                'performance_bounds': {'max_ops_sec': 5000, 'min_latency_ms': 10},
                'memory_bandwidth_gb_s': 100,
                'analysis_method': 'existing_tools'
            },
            'libraries_used': ['analysis'],
            'components_source': 'existing_only'
        }
        mock_orchestrator.orchestrate_exploration.return_value = mock_result
        mock_orchestrator.get_orchestration_history.return_value = [
            {'exit_point': 'roofline', 'status': 'success', 'timestamp': 'test'}
        ]
        mock_orchestrator.libraries = {'analysis': Mock()}
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Execute complete workflow
        with patch('brainsmith.core.api._load_and_validate_blueprint') as mock_load:
            mock_blueprint = Mock()
            mock_blueprint.name = "integration_test"
            mock_blueprint.model_path = self.model_path
            mock_load.return_value = mock_blueprint
            
            results, analysis = brainsmith_explore(
                self.model_path,
                self.blueprint_path,
                exit_point="roofline",
                output_dir=os.path.join(self.temp_dir, "roofline_output")
            )
        
        # Verify complete workflow
        self.assertIsNotNone(results)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['exit_point'], 'roofline')
        self.assertEqual(analysis['method'], 'existing_tools')
        self.assertIn('roofline_specific', analysis)
        
        # Verify orchestrator was used correctly
        mock_orchestrator.orchestrate_exploration.assert_called_once_with("roofline")
    
    @patch('brainsmith.core.design_space_orchestrator.DesignSpaceOrchestrator')
    def test_complete_dataflow_analysis_workflow(self, mock_orchestrator_class):
        """Test complete dataflow analysis workflow."""
        # Setup comprehensive mock
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.analysis = {
            'exit_point': 'dataflow_analysis',
            'method': 'existing_dataflow_tools',
            'transformed_model': {
                'transforms_applied': ['streamlining', 'folding'],
                'model_size_kb': 150,
                'num_operations': 25
            },
            'kernel_mapping': {
                'conv_layers': 3,
                'linear_layers': 2,
                'kernels_mapped': ['conv2d_hls', 'linear_hls']
            },
            'performance_estimates': {
                'estimated_throughput_ops_sec': 2500,
                'estimated_latency_ms': 20,
                'resource_estimates': {'lut_usage': 0.6, 'bram_usage': 0.4}
            },
            'libraries_used': ['transforms', 'kernels', 'analysis'],
            'components_source': 'existing_only'
        }
        mock_orchestrator.orchestrate_exploration.return_value = mock_result
        mock_orchestrator.get_orchestration_history.return_value = []
        mock_orchestrator.libraries = {
            'transforms': Mock(), 'kernels': Mock(), 'analysis': Mock()
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Execute complete workflow
        with patch('brainsmith.core.api._load_and_validate_blueprint') as mock_load:
            mock_blueprint = Mock()
            mock_blueprint.name = "integration_test"
            mock_load.return_value = mock_blueprint
            
            results, analysis = brainsmith_dataflow_analysis(
                self.model_path,
                self.blueprint_path,
                output_dir=os.path.join(self.temp_dir, "dataflow_output")
            )
        
        # Verify complete workflow
        self.assertIsNotNone(results)
        self.assertEqual(analysis['exit_point'], 'dataflow_analysis')
        self.assertIn('dataflow_specific', analysis)
        
        # Verify dataflow-specific results
        dataflow_data = analysis['dataflow_specific']
        self.assertEqual(dataflow_data['analysis_type'], 'dataflow_estimation')
    
    @patch('brainsmith.core.design_space_orchestrator.DesignSpaceOrchestrator')
    def test_complete_generation_workflow(self, mock_orchestrator_class):
        """Test complete generation workflow."""
        # Setup comprehensive mock
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.analysis = {
            'exit_point': 'dataflow_generation',
            'method': 'existing_finn_generation',
            'optimization_results': {
                'best_point': {
                    'conv2d_simd': 4, 'conv2d_pe': 4,
                    'linear_simd': 8, 'linear_pe': 8
                },
                'all_results': [
                    {'config': 'point1', 'throughput': 2000},
                    {'config': 'point2', 'throughput': 2500}
                ],
                'optimization_time_s': 120
            },
            'generation_results': {
                'rtl_files': ['design.v', 'testbench.v', 'constraints.xdc'],
                'hls_files': ['kernel.cpp', 'kernel.h', 'tb.cpp'],
                'synthesis_results': {
                    'lut_utilization': 0.75,
                    'bram_utilization': 0.60,
                    'timing_met': True,
                    'max_frequency_mhz': 200
                },
                'interface_type': 'legacy_dataflow_build_config'
            },
            'libraries_used': ['transforms', 'kernels', 'hw_optim', 'analysis'],
            'components_source': 'existing_only'
        }
        mock_orchestrator.orchestrate_exploration.return_value = mock_result
        mock_orchestrator.get_orchestration_history.return_value = []
        mock_orchestrator.libraries = {
            'transforms': Mock(), 'kernels': Mock(), 
            'hw_optim': Mock(), 'analysis': Mock()
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Execute complete workflow
        with patch('brainsmith.core.api._load_and_validate_blueprint') as mock_load:
            mock_blueprint = Mock()
            mock_blueprint.name = "integration_test"
            mock_load.return_value = mock_blueprint
            
            results, analysis = brainsmith_generate(
                self.model_path,
                self.blueprint_path,
                output_dir=os.path.join(self.temp_dir, "generation_output")
            )
        
        # Verify complete workflow
        self.assertIsNotNone(results)
        self.assertEqual(analysis['exit_point'], 'dataflow_generation')
        self.assertIn('generation_specific', analysis)
        
        # Verify generation-specific results
        generation_data = analysis['generation_specific']
        self.assertEqual(generation_data['analysis_type'], 'complete_generation')
        self.assertEqual(generation_data['rtl_files_count'], 3)
        self.assertEqual(generation_data['hls_files_count'], 3)


class TestWorkflowProgression(unittest.TestCase):
    """Test progressive workflow execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        if brainsmith_explore is None:
            self.skipTest("Core components not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "progression_model.onnx")
        self.blueprint_path = os.path.join(self.temp_dir, "progression_blueprint.yaml")
        
        with open(self.model_path, 'w') as f:
            f.write("# Model for progression testing")
        
        with open(self.blueprint_path, 'w') as f:
            f.write("name: progression_test\nkernels:\n  available: []")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('brainsmith.core.design_space_orchestrator.DesignSpaceOrchestrator')
    def test_workflow_progression_fast_to_comprehensive(self, mock_orchestrator_class):
        """Test progression from fast to comprehensive analysis."""
        # Setup orchestrator mock to track calls
        mock_orchestrator = Mock()
        mock_orchestrator.get_orchestration_history.return_value = []
        mock_orchestrator.libraries = {
            'analysis': Mock(),
            'transforms': Mock(),
            'kernels': Mock(),
            'hw_optim': Mock()
        }
        
        # Mock different results for different exit points
        def mock_orchestrate(exit_point):
            mock_result = Mock()
            if exit_point == 'roofline':
                mock_result.analysis = {
                    'exit_point': 'roofline',
                    'method': 'existing_analysis_tools',
                    'libraries_used': ['analysis']
                }
            elif exit_point == 'dataflow_analysis':
                mock_result.analysis = {
                    'exit_point': 'dataflow_analysis',
                    'method': 'existing_dataflow_tools',
                    'libraries_used': ['transforms', 'kernels', 'analysis']
                }
            elif exit_point == 'dataflow_generation':
                mock_result.analysis = {
                    'exit_point': 'dataflow_generation',
                    'method': 'existing_finn_generation',
                    'libraries_used': ['transforms', 'kernels', 'hw_optim', 'analysis']
                }
            return mock_result
        
        mock_orchestrator.orchestrate_exploration.side_effect = mock_orchestrate
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Test progression through all exit points
        with patch('brainsmith.core.api._load_and_validate_blueprint') as mock_load:
            mock_blueprint = Mock()
            mock_blueprint.name = "progression_test"
            mock_load.return_value = mock_blueprint
            
            # Fast analysis (roofline)
            results1, analysis1 = brainsmith_roofline(self.model_path, self.blueprint_path)
            
            # Standard analysis (dataflow)
            results2, analysis2 = brainsmith_dataflow_analysis(self.model_path, self.blueprint_path)
            
            # Comprehensive analysis (generation)
            results3, analysis3 = brainsmith_generate(self.model_path, self.blueprint_path)
        
        # Verify progression
        self.assertEqual(analysis1['exit_point'], 'roofline')
        self.assertEqual(analysis2['exit_point'], 'dataflow_analysis')
        self.assertEqual(analysis3['exit_point'], 'dataflow_generation')
        
        # Verify increasing library usage
        libs1 = len(analysis1['libraries_status'])
        libs2 = len(analysis2['libraries_status'])
        libs3 = len(analysis3['libraries_status'])
        
        self.assertLessEqual(libs1, libs2)
        self.assertLessEqual(libs2, libs3)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if DesignSpaceOrchestrator is None:
            self.skipTest("Components not available for integration testing")
    
    def test_orchestrator_finn_interface_integration(self):
        """Test integration between orchestrator and FINN interface."""
        # Create mock blueprint
        mock_blueprint = Mock()
        mock_blueprint.name = "integration_test"
        mock_blueprint.model_path = "test_model.onnx"
        mock_blueprint.get_finn_legacy_config = Mock(return_value={
            'fpga_part': 'xcvu9p-flga2104-2-i'
        })
        
        # Create orchestrator
        orchestrator = DesignSpaceOrchestrator(mock_blueprint)
        
        # Verify FINN interface was created
        self.assertIsNotNone(orchestrator.finn_interface)
        
        # Verify interface status
        status = orchestrator.finn_interface.get_interface_status()
        self.assertTrue(status['using_legacy'])
        self.assertFalse(status['future_hooks_available'])
    
    def test_orchestrator_workflow_manager_integration(self):
        """Test integration between orchestrator and workflow manager."""
        if WorkflowManager is None:
            self.skipTest("WorkflowManager not available")
        
        # Create mock blueprint
        mock_blueprint = Mock()
        mock_blueprint.name = "workflow_integration_test"
        mock_blueprint.model_path = "test_model.onnx"
        mock_blueprint.get_finn_legacy_config = Mock(return_value={})
        
        # Create orchestrator and workflow manager
        orchestrator = DesignSpaceOrchestrator(mock_blueprint)
        workflow_manager = WorkflowManager(orchestrator)
        
        # Test workflow execution
        result = workflow_manager.execute_existing_workflow("fast")
        
        # Verify integration worked
        self.assertIsInstance(result, dict)
        self.assertEqual(result['workflow'], 'fast')
        self.assertIn('result', result)
        self.assertIn('metadata', result)


class TestSystemHealthAndStatus(unittest.TestCase):
    """Test overall system health and status reporting."""
    
    def test_core_status_reporting(self):
        """Test core system status reporting."""
        if get_core_status is None:
            self.skipTest("Core status functions not available")
        
        status = get_core_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('version', status)
        self.assertIn('components', status)
        self.assertIn('api_functions', status)
        self.assertIn('readiness', status)
        
        # Check component availability
        expected_components = [
            'DesignSpaceOrchestrator', 'FINNInterface', 
            'WorkflowManager', 'API', 'CLI', 'LegacySupport'
        ]
        
        for component in expected_components:
            self.assertIn(component, status['components'])
    
    def test_system_readiness_calculation(self):
        """Test system readiness calculation."""
        if get_core_status is None:
            self.skipTest("Core status functions not available")
        
        status = get_core_status()
        
        # Readiness should be between 0 and 1
        self.assertGreaterEqual(status['readiness'], 0.0)
        self.assertLessEqual(status['readiness'], 1.0)
        
        # If readiness is high, essential components should be available
        if status['readiness'] >= 0.8:
            essential_components = ['DesignSpaceOrchestrator', 'FINNInterface', 'API']
            for component in essential_components:
                self.assertTrue(status['components'].get(component, False),
                              f"Essential component {component} not available despite high readiness")


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test error handling and recovery mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        if brainsmith_explore is None:
            self.skipTest("API functions not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "error_test_model.onnx")
        self.blueprint_path = os.path.join(self.temp_dir, "error_test_blueprint.yaml")
        
        with open(self.model_path, 'w') as f:
            f.write("# Error test model")
        with open(self.blueprint_path, 'w') as f:
            f.write("name: error_test")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('brainsmith.core.api.DesignSpaceOrchestrator')
    def test_graceful_error_handling(self, mock_orchestrator_class):
        """Test graceful error handling throughout the system."""
        # Mock orchestrator to raise exception
        mock_orchestrator_class.side_effect = Exception("Orchestrator failed")
        
        # Should handle error gracefully
        with patch('brainsmith.core.api._load_and_validate_blueprint') as mock_load:
            mock_load.return_value = Mock()
            
            results, analysis = brainsmith_explore(self.model_path, self.blueprint_path)
        
        # Should return fallback results instead of crashing
        self.assertIsNotNone(results)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['status'], 'failed')
        self.assertIn('error', analysis)
    
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_blueprint_error_recovery(self, mock_load):
        """Test recovery from blueprint loading errors."""
        mock_load.side_effect = FileNotFoundError("Blueprint not found")
        
        # Should handle blueprint error gracefully
        results, analysis = brainsmith_explore(self.model_path, self.blueprint_path)
        
        self.assertIsNotNone(results)
        self.assertEqual(analysis['status'], 'failed')
        self.assertIn('components_source', analysis)


if __name__ == '__main__':
    # Run integration tests with verbose output
    unittest.main(verbosity=2)