"""
Test suite for Legacy Support

Tests legacy compatibility layer that maintains existing functionality
while providing transition to new extensible architecture.
"""

import unittest
import warnings
from unittest.mock import Mock, patch, MagicMock

# Import components to test
try:
    from brainsmith.core.legacy_support import (
        maintain_existing_api_compatibility, route_to_existing_implementation,
        warn_legacy_usage, create_legacy_wrapper, get_legacy_compatibility_report,
        install_legacy_compatibility, LegacyAPIWarning
    )
except ImportError as e:
    print(f"Warning: Could not import legacy support components: {e}")
    maintain_existing_api_compatibility = None
    route_to_existing_implementation = None
    warn_legacy_usage = None
    create_legacy_wrapper = None
    get_legacy_compatibility_report = None
    install_legacy_compatibility = None
    LegacyAPIWarning = None


class TestLegacyCompatibility(unittest.TestCase):
    """Test cases for legacy compatibility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if maintain_existing_api_compatibility is None:
            self.skipTest("Legacy support functions not available")
    
    def test_maintain_existing_api_compatibility(self):
        """Test checking existing API compatibility."""
        # This function checks if existing APIs are still accessible
        result = maintain_existing_api_compatibility()
        
        self.assertIsInstance(result, bool)
        # Should return either True or False, not crash
    
    @patch('brainsmith.core.legacy_support._check_existing_explore_design_space')
    @patch('brainsmith.core.legacy_support._check_existing_blueprint_functions')
    @patch('brainsmith.core.legacy_support._check_existing_dse_functions')
    def test_compatibility_check_components(self, mock_dse, mock_blueprint, mock_explore):
        """Test individual compatibility check components."""
        # Mock all checks to return True
        mock_explore.return_value = True
        mock_blueprint.return_value = True
        mock_dse.return_value = True
        
        result = maintain_existing_api_compatibility()
        
        # Should return True when all components are compatible
        self.assertTrue(result)
        
        # Verify all checks were called
        mock_explore.assert_called_once()
        mock_blueprint.assert_called_once()
        mock_dse.assert_called_once()
    
    @patch('brainsmith.core.legacy_support._check_existing_explore_design_space')
    @patch('brainsmith.core.legacy_support._check_existing_blueprint_functions')
    @patch('brainsmith.core.legacy_support._check_existing_dse_functions')
    def test_compatibility_check_partial_failure(self, mock_dse, mock_blueprint, mock_explore):
        """Test compatibility check with some components missing."""
        # Mock some checks to fail
        mock_explore.return_value = False
        mock_blueprint.return_value = True
        mock_dse.return_value = True
        
        result = maintain_existing_api_compatibility()
        
        # Should return False when any component is incompatible
        self.assertFalse(result)


class TestLegacyRouting(unittest.TestCase):
    """Test cases for legacy function routing."""
    
    def setUp(self):
        """Set up test fixtures."""
        if route_to_existing_implementation is None:
            self.skipTest("Legacy routing functions not available")
    
    @patch('brainsmith.core.legacy_support._route_explore_design_space')
    def test_route_explore_design_space(self, mock_route_explore):
        """Test routing to existing explore_design_space implementation."""
        mock_route_explore.return_value = {'result': 'success'}
        
        result = route_to_existing_implementation(
            'explore_design_space', 
            'model.onnx', 
            'blueprint.yaml'
        )
        
        self.assertEqual(result, {'result': 'success'})
        mock_route_explore.assert_called_once_with('model.onnx', 'blueprint.yaml')
    
    @patch('brainsmith.core.legacy_support._route_get_blueprint')
    def test_route_get_blueprint(self, mock_route_blueprint):
        """Test routing to existing get_blueprint implementation."""
        mock_route_blueprint.return_value = {'blueprint': 'data'}
        
        result = route_to_existing_implementation('get_blueprint', 'blueprint_name')
        
        self.assertEqual(result, {'blueprint': 'data'})
        mock_route_blueprint.assert_called_once_with('blueprint_name')
    
    def test_route_unknown_function(self):
        """Test routing unknown function raises ValueError."""
        with self.assertRaises(ValueError) as context:
            route_to_existing_implementation('unknown_function', 'arg1', 'arg2')
        
        self.assertIn("Unknown legacy function", str(context.exception))
    
    @patch('brainsmith.core.legacy_support._route_explore_design_space')
    def test_route_with_error_handling(self, mock_route_explore):
        """Test routing with error handling."""
        mock_route_explore.side_effect = Exception("Routing failed")
        
        result = route_to_existing_implementation('explore_design_space', 'model.onnx')
        
        # Should return fallback result instead of raising exception
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'error')
        self.assertIn('error', result)


class TestLegacyWarnings(unittest.TestCase):
    """Test cases for legacy warning system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if warn_legacy_usage is None:
            self.skipTest("Legacy warning functions not available")
    
    def test_warn_legacy_usage(self):
        """Test legacy usage warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            warn_legacy_usage("old_function", "new_function")
            
            # Should have issued a warning
            self.assertEqual(len(w), 1)
            if LegacyAPIWarning is not None:
                self.assertTrue(issubclass(w[0].category, LegacyAPIWarning))
            self.assertIn("old_function", str(w[0].message))
            self.assertIn("new_function", str(w[0].message))


class TestLegacyWrapper(unittest.TestCase):
    """Test cases for legacy wrapper creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if create_legacy_wrapper is None:
            self.skipTest("Legacy wrapper functions not available")
    
    def test_create_legacy_wrapper(self):
        """Test creation of legacy wrapper function."""
        # Create a mock new function
        def new_function(arg1, arg2):
            return f"new_result_{arg1}_{arg2}"
        
        # Create legacy wrapper
        legacy_wrapper = create_legacy_wrapper(new_function, "old_func", "new_func")
        
        # Test wrapper functionality
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = legacy_wrapper("test1", "test2")
            
            # Should call new function and return result
            self.assertEqual(result, "new_result_test1_test2")
            
            # Should issue deprecation warning
            self.assertEqual(len(w), 1)
        
        # Check wrapper metadata
        self.assertEqual(legacy_wrapper.__name__, "old_func")
        self.assertIn("Legacy wrapper", legacy_wrapper.__doc__)
    
    def test_legacy_wrapper_error_handling(self):
        """Test legacy wrapper error handling."""
        # Create a mock function that raises exception
        def failing_function():
            raise Exception("New function failed")
        
        # Mock the routing function
        with patch('brainsmith.core.legacy_support.route_to_existing_implementation') as mock_route:
            mock_route.return_value = {'fallback': 'result'}
            
            legacy_wrapper = create_legacy_wrapper(failing_function, "old_func", "new_func")
            
            # Should handle error and route to existing implementation
            result = legacy_wrapper()
            
            self.assertEqual(result, {'fallback': 'result'})
            mock_route.assert_called_once_with("old_func")


class TestCompatibilityReport(unittest.TestCase):
    """Test cases for compatibility reporting."""
    
    def setUp(self):
        """Set up test fixtures."""
        if get_legacy_compatibility_report is None:
            self.skipTest("Compatibility report functions not available")
    
    def test_get_legacy_compatibility_report(self):
        """Test generation of compatibility report."""
        report = get_legacy_compatibility_report()
        
        self.assertIsInstance(report, dict)
        
        # Check required fields
        expected_fields = [
            'timestamp', 'overall_compatibility', 'functions_checked',
            'compatibility_details', 'recommendations'
        ]
        
        for field in expected_fields:
            self.assertIn(field, report)
        
        # Check data types
        self.assertIsInstance(report['overall_compatibility'], bool)
        self.assertIsInstance(report['functions_checked'], list)
        self.assertIsInstance(report['compatibility_details'], dict)
        self.assertIsInstance(report['recommendations'], list)
    
    @patch('brainsmith.core.legacy_support.maintain_existing_api_compatibility')
    def test_compatibility_report_with_mock_results(self, mock_compatibility):
        """Test compatibility report with mocked compatibility results."""
        mock_compatibility.return_value = True
        
        report = get_legacy_compatibility_report()
        
        self.assertTrue(report['overall_compatibility'])
        mock_compatibility.assert_called_once()
    
    @patch('brainsmith.core.legacy_support._check_existing_explore_design_space')
    @patch('brainsmith.core.legacy_support._check_existing_blueprint_functions')
    @patch('brainsmith.core.legacy_support._check_existing_dse_functions')
    def test_compatibility_report_details(self, mock_dse, mock_blueprint, mock_explore):
        """Test detailed compatibility report generation."""
        # Mock specific function checks
        mock_explore.return_value = True
        mock_blueprint.return_value = False
        mock_dse.return_value = True
        
        report = get_legacy_compatibility_report()
        
        # Check compatibility details
        details = report['compatibility_details']
        self.assertIn('explore_design_space', details)
        self.assertIn('blueprint_functions', details)
        self.assertIn('dse_functions', details)
        
        # Check that failures generate recommendations
        if not report['overall_compatibility']:
            self.assertGreater(len(report['recommendations']), 0)


class TestLegacyInstallation(unittest.TestCase):
    """Test cases for legacy compatibility installation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if install_legacy_compatibility is None:
            self.skipTest("Legacy installation functions not available")
    
    @patch('brainsmith.core.legacy_support.brainsmith')
    @patch('brainsmith.core.legacy_support.create_legacy_wrapper')
    def test_install_legacy_compatibility(self, mock_wrapper, mock_brainsmith_module):
        """Test installation of legacy compatibility shims."""
        # Mock brainsmith module
        mock_brainsmith_module.explore_design_space = None  # Not already present
        
        # Mock wrapper creation
        mock_legacy_wrapper = Mock()
        mock_wrapper.return_value = mock_legacy_wrapper
        
        # Test installation
        result = install_legacy_compatibility()
        
        # Should succeed and install wrapper
        self.assertTrue(result)
        
        # Should have created wrapper
        mock_wrapper.assert_called()
    
    @patch('brainsmith.core.legacy_support.brainsmith')
    def test_install_when_already_present(self, mock_brainsmith_module):
        """Test installation when legacy functions already present."""
        # Mock that function already exists
        mock_brainsmith_module.explore_design_space = Mock()
        
        # Should still succeed (just won't overwrite)
        result = install_legacy_compatibility()
        self.assertIsInstance(result, bool)


class TestLegacyIntegration(unittest.TestCase):
    """Integration tests for legacy support system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if maintain_existing_api_compatibility is None:
            self.skipTest("Legacy support not available")
    
    def test_end_to_end_legacy_workflow(self):
        """Test complete legacy support workflow."""
        # 1. Check compatibility
        compatibility = maintain_existing_api_compatibility()
        self.assertIsInstance(compatibility, bool)
        
        # 2. Generate report
        report = get_legacy_compatibility_report()
        self.assertIsInstance(report, dict)
        
        # 3. Install compatibility if needed
        installation_result = install_legacy_compatibility()
        self.assertIsInstance(installation_result, bool)
        
        # All steps should complete without error
    
    @patch('warnings.warn')
    def test_legacy_warning_integration(self, mock_warn):
        """Test integration of warning system with wrapper creation."""
        def new_test_function():
            return "new_result"
        
        # Create wrapper and test warning
        wrapper = create_legacy_wrapper(new_test_function, "test_old", "test_new")
        
        # Call wrapper
        result = wrapper()
        
        # Should work and issue warning
        self.assertEqual(result, "new_result")
        mock_warn.assert_called_once()


class TestLegacyErrorScenarios(unittest.TestCase):
    """Test cases for legacy support error scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        if route_to_existing_implementation is None:
            self.skipTest("Legacy support not available")
    
    def test_routing_with_import_errors(self):
        """Test routing behavior when imports fail."""
        # Test with function that would require imports
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            result = route_to_existing_implementation('explore_design_space', 'test_args')
            
            # Should return fallback result instead of crashing
            self.assertIsInstance(result, dict)
            self.assertIn('fallback', result)
    
    def test_compatibility_check_with_exceptions(self):
        """Test compatibility checking with exceptions."""
        # Mock check functions to raise exceptions
        with patch('brainsmith.core.legacy_support._check_existing_explore_design_space') as mock_check:
            mock_check.side_effect = Exception("Check failed")
            
            # Should handle exception gracefully
            result = maintain_existing_api_compatibility()
            self.assertIsInstance(result, bool)
            # Result may be False due to error, but should not crash


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)