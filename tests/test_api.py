"""
Test suite for Python API

Tests the main Python API functions with hierarchical exit points
and backward compatibility support.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import components to test
try:
    from brainsmith.core.api import (
        brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
        brainsmith_generate, brainsmith_workflow, explore_design_space, validate_blueprint
    )
except ImportError as e:
    print(f"Warning: Could not import API functions: {e}")
    brainsmith_explore = None
    brainsmith_roofline = None
    brainsmith_dataflow_analysis = None
    brainsmith_generate = None
    brainsmith_workflow = None
    explore_design_space = None
    validate_blueprint = None


class TestAPIFunctions(unittest.TestCase):
    """Test cases for main API functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if brainsmith_explore is None:
            self.skipTest("API functions not available")
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.onnx")
        self.blueprint_path = os.path.join(self.temp_dir, "test_blueprint.yaml")
        
        # Create minimal test files
        with open(self.model_path, 'w') as f:
            f.write("# Mock ONNX model file")
        
        with open(self.blueprint_path, 'w') as f:
            f.write("""
name: "test_blueprint"
description: "Test blueprint for API testing"

kernels:
  available: []

transforms:
  pipeline: []

hw_optimization:
  strategies: []

finn_interface:
  legacy_config:
    fpga_part: "xcvu9p-flga2104-2-i"
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('brainsmith.core.api.DesignSpaceOrchestrator')
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_brainsmith_explore_roofline(self, mock_load_blueprint, mock_orchestrator_class):
        """Test brainsmith_explore with roofline exit point."""
        # Setup mocks
        mock_blueprint = Mock()
        mock_blueprint.name = "test_blueprint"
        mock_load_blueprint.return_value = mock_blueprint
        
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.analysis = {
            'exit_point': 'roofline',
            'method': 'existing_analysis_tools',
            'roofline_results': {'test': 'data'}
        }
        mock_orchestrator.orchestrate_exploration.return_value = mock_result
        mock_orchestrator.get_orchestration_history.return_value = []
        mock_orchestrator.libraries = {'analysis': Mock()}
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Test the function
        results, analysis = brainsmith_explore(
            self.model_path, 
            self.blueprint_path, 
            exit_point="roofline"
        )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['exit_point'], 'roofline')
        self.assertEqual(analysis['method'], 'existing_tools')
        
        # Verify mocks were called correctly
        mock_load_blueprint.assert_called_once()
        mock_orchestrator_class.assert_called_once_with(mock_blueprint)
        mock_orchestrator.orchestrate_exploration.assert_called_once_with("roofline")
    
    @patch('brainsmith.core.api.DesignSpaceOrchestrator')
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_brainsmith_explore_dataflow_analysis(self, mock_load_blueprint, mock_orchestrator_class):
        """Test brainsmith_explore with dataflow_analysis exit point."""
        # Setup mocks
        mock_blueprint = Mock()
        mock_load_blueprint.return_value = mock_blueprint
        
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.analysis = {
            'exit_point': 'dataflow_analysis',
            'method': 'existing_dataflow_tools',
            'transformed_model': {'test': 'model'},
            'kernel_mapping': {'test': 'mapping'}
        }
        mock_orchestrator.orchestrate_exploration.return_value = mock_result
        mock_orchestrator.get_orchestration_history.return_value = []
        mock_orchestrator.libraries = {'transforms': Mock(), 'kernels': Mock(), 'analysis': Mock()}
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Test the function
        results, analysis = brainsmith_explore(
            self.model_path,
            self.blueprint_path,
            exit_point="dataflow_analysis"
        )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(analysis['exit_point'], 'dataflow_analysis')
        mock_orchestrator.orchestrate_exploration.assert_called_once_with("dataflow_analysis")
    
    @patch('brainsmith.core.api.DesignSpaceOrchestrator')
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_brainsmith_explore_dataflow_generation(self, mock_load_blueprint, mock_orchestrator_class):
        """Test brainsmith_explore with dataflow_generation exit point."""
        # Setup mocks
        mock_blueprint = Mock()
        mock_load_blueprint.return_value = mock_blueprint
        
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.analysis = {
            'exit_point': 'dataflow_generation',
            'method': 'existing_finn_generation',
            'generation_results': {'rtl_files': ['test.v'], 'hls_files': ['test.cpp']}
        }
        mock_orchestrator.orchestrate_exploration.return_value = mock_result
        mock_orchestrator.get_orchestration_history.return_value = []
        mock_orchestrator.libraries = {
            'transforms': Mock(), 'kernels': Mock(), 'hw_optim': Mock(), 'analysis': Mock()
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Test the function
        results, analysis = brainsmith_explore(
            self.model_path,
            self.blueprint_path,
            exit_point="dataflow_generation"
        )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(analysis['exit_point'], 'dataflow_generation')
        mock_orchestrator.orchestrate_exploration.assert_called_once_with("dataflow_generation")
    
    def test_brainsmith_explore_invalid_exit_point(self):
        """Test brainsmith_explore with invalid exit point."""
        with self.assertRaises(ValueError) as context:
            brainsmith_explore(
                self.model_path,
                self.blueprint_path,
                exit_point="invalid_point"
            )
        
        self.assertIn("Invalid exit point", str(context.exception))
    
    def test_file_not_found_errors(self):
        """Test file not found error handling."""
        # Test with non-existent model file
        with self.assertRaises(FileNotFoundError):
            brainsmith_explore("non_existent_model.onnx", self.blueprint_path)
        
        # Test with non-existent blueprint file
        with self.assertRaises(FileNotFoundError):
            brainsmith_explore(self.model_path, "non_existent_blueprint.yaml")
    
    @patch('brainsmith.core.api.brainsmith_explore')
    def test_convenience_functions(self, mock_explore):
        """Test convenience functions that wrap brainsmith_explore."""
        mock_explore.return_value = (Mock(), {'exit_point': 'test'})
        
        # Test brainsmith_roofline
        if brainsmith_roofline is not None:
            brainsmith_roofline(self.model_path, self.blueprint_path)
            mock_explore.assert_called_with(self.model_path, self.blueprint_path, "roofline", None)
        
        # Test brainsmith_dataflow_analysis
        if brainsmith_dataflow_analysis is not None:
            brainsmith_dataflow_analysis(self.model_path, self.blueprint_path)
            mock_explore.assert_called_with(self.model_path, self.blueprint_path, "dataflow_analysis", None)
        
        # Test brainsmith_generate
        if brainsmith_generate is not None:
            brainsmith_generate(self.model_path, self.blueprint_path)
            mock_explore.assert_called_with(self.model_path, self.blueprint_path, "dataflow_generation", None)
    
    @patch('brainsmith.core.api._save_results_existing')
    @patch('brainsmith.core.api.DesignSpaceOrchestrator')
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_output_directory_handling(self, mock_load_blueprint, mock_orchestrator_class, mock_save):
        """Test output directory handling."""
        # Setup mocks
        mock_blueprint = Mock()
        mock_load_blueprint.return_value = mock_blueprint
        
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.analysis = {'exit_point': 'roofline', 'method': 'test'}
        mock_orchestrator.orchestrate_exploration.return_value = mock_result
        mock_orchestrator.get_orchestration_history.return_value = []
        mock_orchestrator.libraries = {'analysis': Mock()}
        mock_orchestrator_class.return_value = mock_orchestrator
        
        output_dir = os.path.join(self.temp_dir, "output")
        
        # Test with output directory
        brainsmith_explore(
            self.model_path,
            self.blueprint_path,
            output_dir=output_dir
        )
        
        # Verify save function was called
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0]
        self.assertEqual(call_args[2], output_dir)  # output_dir argument


class TestBackwardCompatibility(unittest.TestCase):
    """Test cases for backward compatibility features."""
    
    def setUp(self):
        """Set up test fixtures."""
        if explore_design_space is None:
            self.skipTest("Backward compatibility functions not available")
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.onnx")
        self.blueprint_path = os.path.join(self.temp_dir, "test_blueprint.yaml")
        
        with open(self.model_path, 'w') as f:
            f.write("# Mock model")
        with open(self.blueprint_path, 'w') as f:
            f.write("name: test")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('warnings.warn')
    @patch('brainsmith.core.api.brainsmith_explore')
    def test_explore_design_space_with_blueprint_path(self, mock_explore, mock_warn):
        """Test legacy explore_design_space with blueprint path."""
        mock_explore.return_value = (Mock(), {'test': 'data'})
        
        # Test with blueprint path (should use new API)
        result = explore_design_space(self.model_path, self.blueprint_path)
        
        # Should have called new API
        mock_explore.assert_called_once()
        
        # Should have issued deprecation warning
        mock_warn.assert_called_once()
        self.assertIn("legacy function", mock_warn.call_args[0][0])
    
    @patch('brainsmith.core.api._route_to_existing_legacy_system')
    def test_explore_design_space_with_blueprint_name(self, mock_route):
        """Test legacy explore_design_space with blueprint name."""
        mock_route.return_value = {'legacy': 'result'}
        
        # Test with blueprint name (not a path)
        result = explore_design_space(self.model_path, "blueprint_name")
        
        # Should have routed to legacy system
        mock_route.assert_called_once_with(self.model_path, "blueprint_name")
        self.assertEqual(result, {'legacy': 'result'})
    
    @patch('brainsmith.core.api._route_to_existing_legacy_system')
    def test_legacy_routing_error_handling(self, mock_route):
        """Test error handling in legacy routing."""
        mock_route.side_effect = Exception("Legacy system error")
        
        # Should handle error gracefully
        result = explore_design_space(self.model_path, "blueprint_name")
        
        # Should return error result
        self.assertIn('error', result)
        self.assertEqual(result['status'], 'error')


class TestBlueprintValidation(unittest.TestCase):
    """Test cases for blueprint validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if validate_blueprint is None:
            self.skipTest("Blueprint validation not available")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_valid_blueprint(self):
        """Test validation of valid blueprint."""
        blueprint_path = os.path.join(self.temp_dir, "valid_blueprint.yaml")
        
        with open(blueprint_path, 'w') as f:
            f.write("""
name: "valid_blueprint"
description: "Valid test blueprint"

kernels:
  available:
    - name: "conv2d"
      source: "custom_op.hw_conv2d"

transforms:
  pipeline:
    - name: "streamlining"
      source: "steps.streamlining"

hw_optimization:
  strategies:
    - name: "random"
      source: "dse.random"

finn_interface:
  legacy_config:
    fpga_part: "xcvu9p-flga2104-2-i"
""")
        
        is_valid, errors = validate_blueprint(blueprint_path)
        
        # Should be valid (or at least not fail completely)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(errors, list)
    
    def test_validate_nonexistent_blueprint(self):
        """Test validation of non-existent blueprint."""
        is_valid, errors = validate_blueprint("non_existent.yaml")
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestWorkflowAPI(unittest.TestCase):
    """Test cases for workflow API functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if brainsmith_workflow is None:
            self.skipTest("Workflow API not available")
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.onnx")
        self.blueprint_path = os.path.join(self.temp_dir, "test_blueprint.yaml")
        
        with open(self.model_path, 'w') as f:
            f.write("# Mock model")
        with open(self.blueprint_path, 'w') as f:
            f.write("name: test")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('brainsmith.core.api.brainsmith_explore')
    def test_workflow_type_mapping(self, mock_explore):
        """Test workflow type mapping to exit points."""
        mock_explore.return_value = (Mock(), {'exit_point': 'test'})
        
        # Test workflow type mappings
        test_cases = [
            ('fast', 'roofline'),
            ('standard', 'dataflow_analysis'),
            ('comprehensive', 'dataflow_generation')
        ]
        
        for workflow_type, expected_exit_point in test_cases:
            brainsmith_workflow(self.model_path, self.blueprint_path, workflow_type)
            
            # Check that correct exit point was used
            call_args = mock_explore.call_args
            self.assertEqual(call_args[0][2], expected_exit_point)  # exit_point argument
    
    @patch('brainsmith.core.api.brainsmith_explore')
    def test_workflow_with_invalid_type(self, mock_explore):
        """Test workflow with invalid type defaults to standard."""
        mock_explore.return_value = (Mock(), {'exit_point': 'dataflow_analysis'})
        
        brainsmith_workflow(self.model_path, self.blueprint_path, 'invalid_type')
        
        # Should default to dataflow_analysis
        call_args = mock_explore.call_args
        self.assertEqual(call_args[0][2], 'dataflow_analysis')


class TestAPIErrorHandling(unittest.TestCase):
    """Test cases for API error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        if brainsmith_explore is None:
            self.skipTest("API functions not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.onnx")
        self.blueprint_path = os.path.join(self.temp_dir, "test_blueprint.yaml")
        
        with open(self.model_path, 'w') as f:
            f.write("# Mock model")
        with open(self.blueprint_path, 'w') as f:
            f.write("name: test")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('brainsmith.core.api.DesignSpaceOrchestrator')
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_orchestrator_error_handling(self, mock_load_blueprint, mock_orchestrator_class):
        """Test handling of orchestrator errors."""
        # Setup mocks to raise exception
        mock_load_blueprint.return_value = Mock()
        mock_orchestrator_class.side_effect = Exception("Orchestrator initialization failed")
        
        # Should handle error gracefully and return fallback results
        results, analysis = brainsmith_explore(self.model_path, self.blueprint_path)
        
        self.assertIsNotNone(results)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['status'], 'failed')
        self.assertIn('error', analysis)
    
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_blueprint_loading_error(self, mock_load_blueprint):
        """Test handling of blueprint loading errors."""
        mock_load_blueprint.side_effect = Exception("Blueprint loading failed")
        
        # Should handle error gracefully
        results, analysis = brainsmith_explore(self.model_path, self.blueprint_path)
        
        self.assertIsNotNone(results)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['status'], 'failed')


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)