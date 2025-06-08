"""
Test suite for DesignSpaceOrchestrator

Tests the main orchestration engine with hierarchical exit points
and coordination of existing library components.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Import components to test
try:
    from brainsmith.core.design_space_orchestrator import (
        DesignSpaceOrchestrator,
        ExistingKernelLibrary,
        ExistingTransformLibrary, 
        ExistingOptimizationLibrary,
        ExistingAnalysisLibrary
    )
except ImportError as e:
    print(f"Warning: Could not import orchestrator components: {e}")
    DesignSpaceOrchestrator = None


class MockBlueprint:
    """Mock blueprint for testing."""
    
    def __init__(self, name="test_blueprint", model_path="test_model.onnx"):
        self.name = name
        self.model_path = model_path
        self.yaml_data = {
            'name': name,
            'kernels': {'available': []},
            'transforms': {'pipeline': []},
            'hw_optimization': {'strategies': []},
            'analysis': {'tools': []},
            'finn_interface': {'legacy_config': {'fpga_part': 'xcvu9p-flga2104-2-i'}}
        }
    
    def get_finn_legacy_config(self):
        return self.yaml_data.get('finn_interface', {}).get('legacy_config', {})
    
    def get_library_configs(self):
        return {
            'kernels': self.yaml_data.get('kernels', {}),
            'transforms': self.yaml_data.get('transforms', {}),
            'hw_optimization': self.yaml_data.get('hw_optimization', {}),
            'analysis': self.yaml_data.get('analysis', {})
        }
    
    def get_constraints_config(self):
        return self.yaml_data.get('constraints', {})
    
    def get_objectives_config(self):
        return self.yaml_data.get('objectives', [])
    
    def get_target_device(self):
        return self.yaml_data.get('constraints', {}).get('target_device', 'xcvu9p-flga2104-2-i')


class TestDesignSpaceOrchestrator(unittest.TestCase):
    """Test cases for DesignSpaceOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.blueprint = MockBlueprint()
        
        if DesignSpaceOrchestrator is not None:
            self.orchestrator = DesignSpaceOrchestrator(self.blueprint)
        else:
            self.skipTest("DesignSpaceOrchestrator not available")
    
    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly."""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.blueprint, self.blueprint)
        self.assertIsNotNone(self.orchestrator.libraries)
        self.assertIsNotNone(self.orchestrator.finn_interface)
        self.assertEqual(len(self.orchestrator.libraries), 4)
        
        # Check library types
        expected_libraries = ['kernels', 'transforms', 'hw_optim', 'analysis']
        for lib_name in expected_libraries:
            self.assertIn(lib_name, self.orchestrator.libraries)
    
    def test_roofline_exit_point(self):
        """Test roofline analysis exit point."""
        result = self.orchestrator.orchestrate_exploration("roofline")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.analysis['exit_point'], 'roofline')
        self.assertEqual(result.analysis['method'], 'existing_analysis_tools')
        self.assertIn('roofline_results', result.analysis)
        self.assertIn('libraries_used', result.analysis)
        self.assertEqual(result.analysis['libraries_used'], ['analysis'])
    
    def test_dataflow_analysis_exit_point(self):
        """Test dataflow analysis exit point."""
        result = self.orchestrator.orchestrate_exploration("dataflow_analysis")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.analysis['exit_point'], 'dataflow_analysis')
        self.assertEqual(result.analysis['method'], 'existing_dataflow_tools')
        self.assertIn('transformed_model', result.analysis)
        self.assertIn('kernel_mapping', result.analysis)
        self.assertIn('performance_estimates', result.analysis)
        
        expected_libraries = ['transforms', 'kernels', 'analysis']
        self.assertEqual(result.analysis['libraries_used'], expected_libraries)
    
    def test_dataflow_generation_exit_point(self):
        """Test dataflow generation exit point."""
        result = self.orchestrator.orchestrate_exploration("dataflow_generation")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.analysis['exit_point'], 'dataflow_generation')
        self.assertEqual(result.analysis['method'], 'existing_finn_generation')
        self.assertIn('optimization_results', result.analysis)
        self.assertIn('generation_results', result.analysis)
        
        expected_libraries = ['transforms', 'kernels', 'hw_optim', 'analysis']
        self.assertEqual(result.analysis['libraries_used'], expected_libraries)
    
    def test_invalid_exit_point(self):
        """Test that invalid exit points raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.orchestrator.orchestrate_exploration("invalid_exit_point")
        
        self.assertIn("Invalid exit point", str(context.exception))
    
    def test_orchestration_history(self):
        """Test that orchestration history is recorded."""
        initial_history_length = len(self.orchestrator.get_orchestration_history())
        
        # Execute orchestration
        self.orchestrator.orchestrate_exploration("roofline")
        
        # Check history was updated
        history = self.orchestrator.get_orchestration_history()
        self.assertEqual(len(history), initial_history_length + 1)
        
        # Check history entry content
        latest_entry = history[-1]
        self.assertEqual(latest_entry['exit_point'], 'roofline')
        self.assertEqual(latest_entry['status'], 'success')
        self.assertIn('result_summary', latest_entry)
    
    def test_design_space_construction(self):
        """Test design space construction from existing components."""
        design_space = self.orchestrator.construct_design_space_from_existing()
        
        self.assertIsNotNone(design_space)
        self.assertIn('library_spaces', design_space)
        self.assertIn('source', design_space)
        self.assertEqual(design_space['source'], 'existing_components_only')
    
    def test_error_handling_missing_model_path(self):
        """Test error handling when model path is missing."""
        blueprint_no_model = MockBlueprint()
        blueprint_no_model.model_path = None
        
        orchestrator = DesignSpaceOrchestrator(blueprint_no_model)
        result = orchestrator.orchestrate_exploration("roofline")
        
        # Should handle error gracefully
        self.assertIsNotNone(result)
        self.assertEqual(result.analysis['status'], 'failed')
        self.assertIn('error', result.analysis)


class TestExistingLibraries(unittest.TestCase):
    """Test cases for existing library placeholder classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.blueprint = MockBlueprint()
    
    def test_existing_kernel_library(self):
        """Test ExistingKernelLibrary functionality."""
        if 'ExistingKernelLibrary' not in globals() or ExistingKernelLibrary is None:
            self.skipTest("ExistingKernelLibrary not available")
        
        kernel_lib = ExistingKernelLibrary(self.blueprint)
        
        # Test basic functionality
        self.assertIsNotNone(kernel_lib)
        self.assertEqual(kernel_lib.blueprint, self.blueprint)
        
        # Test kernel mapping
        mapping = kernel_lib.map_to_existing_kernels("test_model")
        self.assertIsInstance(mapping, dict)
        self.assertIn('kernel_mapping', mapping)
        
        # Test design space
        design_space = kernel_lib.get_design_space_from_existing()
        self.assertIsInstance(design_space, dict)
    
    def test_existing_transform_library(self):
        """Test ExistingTransformLibrary functionality."""
        if 'ExistingTransformLibrary' not in globals() or ExistingTransformLibrary is None:
            self.skipTest("ExistingTransformLibrary not available")
        
        transform_lib = ExistingTransformLibrary(self.blueprint)
        
        # Test basic functionality
        self.assertIsNotNone(transform_lib)
        self.assertEqual(transform_lib.blueprint, self.blueprint)
        
        # Test transform application
        result = transform_lib.apply_existing_pipeline("test_model.onnx")
        self.assertIsInstance(result, dict)
        self.assertIn('transformed_model', result)
        
        # Test design space
        design_space = transform_lib.get_design_space_from_existing()
        self.assertIsInstance(design_space, dict)
    
    def test_existing_optimization_library(self):
        """Test ExistingOptimizationLibrary functionality."""
        if 'ExistingOptimizationLibrary' not in globals() or ExistingOptimizationLibrary is None:
            self.skipTest("ExistingOptimizationLibrary not available")
        
        optim_lib = ExistingOptimizationLibrary(self.blueprint)
        
        # Test basic functionality
        self.assertIsNotNone(optim_lib)
        self.assertEqual(optim_lib.blueprint, self.blueprint)
        
        # Test optimization
        result = optim_lib.optimize_using_existing_strategies(self.blueprint)
        self.assertIsInstance(result, dict)
        self.assertIn('best_point', result)
        self.assertIn('all_results', result)
        
        # Test design space
        design_space = optim_lib.get_design_space_from_existing()
        self.assertIsInstance(design_space, dict)
    
    def test_existing_analysis_library(self):
        """Test ExistingAnalysisLibrary functionality."""
        if 'ExistingAnalysisLibrary' not in globals() or ExistingAnalysisLibrary is None:
            self.skipTest("ExistingAnalysisLibrary not available")
        
        analysis_lib = ExistingAnalysisLibrary(self.blueprint)
        
        # Test basic functionality
        self.assertIsNotNone(analysis_lib)
        self.assertEqual(analysis_lib.blueprint, self.blueprint)
        
        # Test model analysis
        result = analysis_lib.analyze_model_characteristics("test_model.onnx")
        self.assertIsInstance(result, dict)
        self.assertIn('model_path', result)
        self.assertIn('analysis_method', result)
        
        # Test roofline analysis
        roofline_result = analysis_lib.perform_roofline_analysis("test_model.onnx")
        self.assertIsInstance(roofline_result, dict)
        self.assertIn('roofline_analysis', roofline_result)
        
        # Test performance estimation
        perf_result = analysis_lib.estimate_dataflow_performance({}, "test_model")
        self.assertIsInstance(perf_result, dict)
        self.assertIn('performance_estimates', perf_result)


class TestOrchestratorIntegration(unittest.TestCase):
    """Integration tests for orchestrator with various scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.blueprint = MockBlueprint()
        
        if DesignSpaceOrchestrator is not None:
            self.orchestrator = DesignSpaceOrchestrator(self.blueprint)
        else:
            self.skipTest("DesignSpaceOrchestrator not available")
    
    def test_multiple_exit_points_sequence(self):
        """Test executing multiple exit points in sequence."""
        exit_points = ["roofline", "dataflow_analysis", "dataflow_generation"]
        results = []
        
        for exit_point in exit_points:
            result = self.orchestrator.orchestrate_exploration(exit_point)
            results.append(result)
            
            # Verify each result
            self.assertIsNotNone(result)
            self.assertEqual(result.analysis['exit_point'], exit_point)
        
        # Verify history contains all executions
        history = self.orchestrator.get_orchestration_history()
        self.assertGreaterEqual(len(history), 3)
        
        # Verify progression of complexity
        self.assertEqual(len(results[0].analysis['libraries_used']), 1)  # roofline: analysis only
        self.assertEqual(len(results[1].analysis['libraries_used']), 3)  # dataflow: +transforms, kernels
        self.assertEqual(len(results[2].analysis['libraries_used']), 4)  # generation: +hw_optim
    
    def test_orchestrator_with_custom_blueprint(self):
        """Test orchestrator with custom blueprint configuration."""
        custom_blueprint = MockBlueprint("custom_test", "custom_model.onnx")
        custom_blueprint.yaml_data.update({
            'kernels': {
                'available': [
                    {'name': 'conv2d', 'type': 'Conv'},
                    {'name': 'linear', 'type': 'MatMul'}
                ]
            },
            'transforms': {
                'pipeline': [
                    {'name': 'streamlining', 'enabled': True},
                    {'name': 'folding', 'enabled': True}
                ]
            }
        })
        
        orchestrator = DesignSpaceOrchestrator(custom_blueprint)
        result = orchestrator.orchestrate_exploration("dataflow_analysis")
        
        # Verify custom configuration was used
        self.assertEqual(orchestrator.blueprint.name, "custom_test")
        self.assertEqual(orchestrator.blueprint.model_path, "custom_model.onnx")
        self.assertIsNotNone(result)
    
    def test_error_recovery(self):
        """Test orchestrator error recovery mechanisms."""
        # Test with blueprint that might cause issues
        error_blueprint = MockBlueprint()
        error_blueprint.model_path = None  # This should cause an error
        
        orchestrator = DesignSpaceOrchestrator(error_blueprint)
        
        # Should handle error gracefully
        result = orchestrator.orchestrate_exploration("roofline")
        self.assertIsNotNone(result)
        
        # Check error was recorded in history
        history = orchestrator.get_orchestration_history()
        if history:
            latest_entry = history[-1]
            # Should have some form of error handling
            self.assertIn('status', latest_entry)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)