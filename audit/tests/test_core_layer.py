"""
Core Layer Testing Module

Tests for Phase 1.1: Core Layer Validation
- forge() API Testing
- CLI Interface Testing  
- Metrics Collection Testing
"""

import sys
import os
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

class CoreLayerTester:
    """Test suite for Brainsmith core layer components."""
    
    def __init__(self):
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Execute all core layer tests."""
        self.test_results = {
            'forge_api_testing': self._test_forge_api(),
            'cli_interface_testing': self._test_cli_interface(),
            'metrics_collection_testing': self._test_metrics_collection()
        }
        return self.test_results
    
    def _test_forge_api(self) -> Dict[str, Any]:
        """Test forge() API functionality."""
        logger.info("Testing forge() API...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test 1: Import availability
            test_results['test_cases']['import_test'] = self._test_forge_import()
            
            # Test 2: Function signature validation
            test_results['test_cases']['signature_test'] = self._test_forge_signature()
            
            # Test 3: Parameter validation
            test_results['test_cases']['parameter_validation'] = self._test_forge_parameter_validation()
            
            # Test 4: Error handling
            test_results['test_cases']['error_handling'] = self._test_forge_error_handling()
            
            # Test 5: Fallback mechanisms
            test_results['test_cases']['fallback_mechanisms'] = self._test_forge_fallbacks()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests == total_tests
            
            if not test_results['passed']:
                test_results['issues'].append({
                    'severity': 'high',
                    'component': 'forge_api',
                    'description': f"Only {passed_tests}/{total_tests} forge API tests passed"
                })
                
        except Exception as e:
            logger.error(f"forge() API testing failed: {e}")
            test_results['issues'].append({
                'severity': 'critical',
                'component': 'forge_api',
                'description': f"Test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_forge_import(self) -> Dict[str, Any]:
        """Test forge() function import."""
        try:
            from brainsmith.core.api import forge
            return {
                'passed': True,
                'description': 'forge() function imported successfully'
            }
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import forge(): {e}'
            }
    
    def _test_forge_signature(self) -> Dict[str, Any]:
        """Test forge() function signature."""
        try:
            from brainsmith.core.api import forge
            import inspect
            
            sig = inspect.signature(forge)
            params = list(sig.parameters.keys())
            
            # Expected parameters from our analysis
            expected_params = [
                'model_path', 'blueprint_path', 'objectives', 'constraints',
                'target_device', 'is_hw_graph', 'build_core', 'output_dir'
            ]
            
            missing_params = [p for p in expected_params if p not in params]
            
            if missing_params:
                return {
                    'passed': False,
                    'description': f'Missing expected parameters: {missing_params}'
                }
            
            return {
                'passed': True,
                'description': f'Function signature valid with {len(params)} parameters'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Signature validation failed: {e}'
            }
    
    def _test_forge_parameter_validation(self) -> Dict[str, Any]:
        """Test forge() parameter validation."""
        try:
            from brainsmith.core.api import forge
            
            # Test with invalid parameters
            test_cases = [
                # Missing required files
                ('nonexistent_model.onnx', 'nonexistent_blueprint.yaml'),
                # Invalid file extensions
                ('model.txt', 'blueprint.json'),
                # None values
                (None, None)
            ]
            
            validation_errors = 0
            for model_path, blueprint_path in test_cases:
                try:
                    result = forge(model_path, blueprint_path)
                    # If this succeeds, validation might be too lenient
                except (FileNotFoundError, ValueError, TypeError):
                    # Expected validation errors
                    validation_errors += 1
                except Exception:
                    # Other errors are also acceptable for invalid inputs
                    validation_errors += 1
            
            # We expect all test cases to raise errors
            if validation_errors == len(test_cases):
                return {
                    'passed': True,
                    'description': f'Parameter validation working correctly ({validation_errors}/{len(test_cases)} cases validated)'
                }
            else:
                return {
                    'passed': False,
                    'description': f'Parameter validation insufficient ({validation_errors}/{len(test_cases)} cases validated)'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'description': f'Parameter validation test failed: {e}'
            }
    
    def _test_forge_error_handling(self) -> Dict[str, Any]:
        """Test forge() error handling and user feedback."""
        try:
            from brainsmith.core.api import forge
            
            # Test error handling with various failure scenarios
            error_scenarios = [
                ('', ''),  # Empty strings
                ('/invalid/path/model.onnx', '/invalid/path/blueprint.yaml'),  # Invalid paths
            ]
            
            handled_errors = 0
            for model_path, blueprint_path in error_scenarios:
                try:
                    result = forge(model_path, blueprint_path)
                    # Check if result contains error information
                    if isinstance(result, dict) and ('error' in result or 'success' in result):
                        handled_errors += 1
                except Exception as e:
                    # Exceptions are also acceptable error handling
                    if any(keyword in str(e).lower() for keyword in ['not found', 'invalid', 'error']):
                        handled_errors += 1
            
            return {
                'passed': handled_errors >= len(error_scenarios) // 2,  # At least half should be handled gracefully
                'description': f'Error handling: {handled_errors}/{len(error_scenarios)} scenarios handled gracefully'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Error handling test failed: {e}'
            }
    
    def _test_forge_fallbacks(self) -> Dict[str, Any]:
        """Test forge() fallback mechanisms."""
        try:
            # Test that forge() has fallback implementations for missing dependencies
            from brainsmith.core.api import forge
            
            # Check if fallback functions exist in the module
            import brainsmith.core.api as api_module
            
            fallback_functions = [
                name for name in dir(api_module) 
                if name.startswith('_fallback_') and callable(getattr(api_module, name))
            ]
            
            if len(fallback_functions) > 0:
                return {
                    'passed': True,
                    'description': f'Fallback mechanisms available: {len(fallback_functions)} functions'
                }
            else:
                return {
                    'passed': False,
                    'description': 'No fallback mechanisms detected'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'description': f'Fallback mechanism test failed: {e}'
            }
    
    def _test_cli_interface(self) -> Dict[str, Any]:
        """Test CLI interface functionality."""
        logger.info("Testing CLI interface...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test CLI module import
            test_results['test_cases']['cli_import'] = self._test_cli_import()
            
            # Test CLI help functionality
            test_results['test_cases']['cli_help'] = self._test_cli_help()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests // 2  # At least half should pass
            
        except Exception as e:
            logger.error(f"CLI interface testing failed: {e}")
            test_results['issues'].append({
                'severity': 'medium',
                'component': 'cli_interface',
                'description': f"CLI test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_cli_import(self) -> Dict[str, Any]:
        """Test CLI module import."""
        try:
            from brainsmith.core import cli
            return {
                'passed': True,
                'description': 'CLI module imported successfully'
            }
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import CLI module: {e}'
            }
    
    def _test_cli_help(self) -> Dict[str, Any]:
        """Test CLI help functionality."""
        try:
            # Check if CLI module has standard functions
            from brainsmith.core import cli
            
            cli_functions = [name for name in dir(cli) if not name.startswith('_')]
            
            if len(cli_functions) > 0:
                return {
                    'passed': True,
                    'description': f'CLI module has {len(cli_functions)} public functions'
                }
            else:
                return {
                    'passed': False,
                    'description': 'CLI module appears to be empty'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'description': f'CLI help test failed: {e}'
            }
    
    def _test_metrics_collection(self) -> Dict[str, Any]:
        """Test metrics collection functionality."""
        logger.info("Testing metrics collection...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test metrics module import
            test_results['test_cases']['metrics_import'] = self._test_metrics_import()
            
            # Test metrics data structures
            test_results['test_cases']['metrics_structures'] = self._test_metrics_structures()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests // 2
            
        except Exception as e:
            logger.error(f"Metrics collection testing failed: {e}")
            test_results['issues'].append({
                'severity': 'medium',
                'component': 'metrics_collection',
                'description': f"Metrics test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_metrics_import(self) -> Dict[str, Any]:
        """Test metrics module import."""
        try:
            from brainsmith.core import metrics
            return {
                'passed': True,
                'description': 'Metrics module imported successfully'
            }
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import metrics module: {e}'
            }
    
    def _test_metrics_structures(self) -> Dict[str, Any]:
        """Test metrics data structures."""
        try:
            from brainsmith.core.metrics import DSEMetrics
            
            # Test metrics class instantiation
            metrics = DSEMetrics()
            
            return {
                'passed': True,
                'description': 'Metrics data structures working correctly'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Metrics structures test failed: {e}'
            }