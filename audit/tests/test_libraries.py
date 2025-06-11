"""
Libraries Layer Testing Module

Tests for Phase 1.3: Libraries Layer Validation
- Kernels Library Testing
- Transforms Library Testing
- Analysis Library Testing
- Automation Library Testing
- Blueprints Library Testing
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

class LibrariesTester:
    """Test suite for Brainsmith libraries layer components."""
    
    def __init__(self):
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Execute all libraries layer tests."""
        self.test_results = {
            'kernels_library_testing': self._test_kernels_library(),
            'transforms_library_testing': self._test_transforms_library(),
            'analysis_library_testing': self._test_analysis_library(),
            'automation_library_testing': self._test_automation_library(),
            'blueprints_library_testing': self._test_blueprints_library()
        }
        return self.test_results
    
    def _test_kernels_library(self) -> Dict[str, Any]:
        """Test kernels library functionality."""
        logger.info("Testing Kernels Library...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test kernels registry
            test_results['test_cases']['kernels_registry'] = self._test_kernels_registry()
            
            # Test custom operations
            test_results['test_cases']['custom_operations'] = self._test_custom_operations()
            
            # Test kernel packages
            test_results['test_cases']['kernel_packages'] = self._test_kernel_packages()
            
            # Test contrib directory
            test_results['test_cases']['kernels_contrib'] = self._test_kernels_contrib()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.7
            
            if not test_results['passed']:
                test_results['issues'].append({
                    'severity': 'high',
                    'component': 'kernels_library',
                    'description': f"Only {passed_tests}/{total_tests} kernels library tests passed"
                })
                
        except Exception as e:
            logger.error(f"Kernels library testing failed: {e}")
            test_results['issues'].append({
                'severity': 'critical',
                'component': 'kernels_library',
                'description': f"Kernels test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_kernels_registry(self) -> Dict[str, Any]:
        """Test kernels registry functionality."""
        try:
            from brainsmith.libraries.kernels.registry import discover_all_kernels, get_kernel_by_name
            
            # Test kernel discovery
            kernels = discover_all_kernels()
            
            return {
                'passed': True,
                'description': f'Kernels registry discovered {len(kernels)} kernels'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import kernels registry: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Kernels registry test failed: {e}'
            }
    
    def _test_custom_operations(self) -> Dict[str, Any]:
        """Test custom operations functionality."""
        try:
            from brainsmith.libraries.kernels.custom_ops.fpgadataflow.layernorm import LayerNorm
            
            # Test LayerNorm operation import
            return {
                'passed': True,
                'description': 'Custom operations (LayerNorm) imported successfully'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import custom operations: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Custom operations test failed: {e}'
            }
    
    def _test_kernel_packages(self) -> Dict[str, Any]:
        """Test kernel packages functionality."""
        try:
            # Check if kernel package directories exist
            kernels_dir = Path(__file__).parent.parent.parent / 'brainsmith/libraries/kernels'
            
            kernel_packages = []
            for item in kernels_dir.iterdir():
                if item.is_dir() and not item.name.startswith('__') and item.name not in ['contrib', 'custom_ops', 'hw_sources']:
                    kernel_packages.append(item.name)
            
            return {
                'passed': len(kernel_packages) > 0,
                'description': f'Found {len(kernel_packages)} kernel packages: {kernel_packages}'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Kernel packages test failed: {e}'
            }
    
    def _test_kernels_contrib(self) -> Dict[str, Any]:
        """Test kernels contrib directory."""
        try:
            contrib_dir = Path(__file__).parent.parent.parent / 'brainsmith/libraries/kernels/contrib'
            
            if contrib_dir.exists():
                return {
                    'passed': True,
                    'description': 'Kernels contrib directory exists'
                }
            else:
                return {
                    'passed': False,
                    'description': 'Kernels contrib directory not found'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'description': f'Kernels contrib test failed: {e}'
            }
    
    def _test_transforms_library(self) -> Dict[str, Any]:
        """Test transforms library functionality."""
        logger.info("Testing Transforms Library...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test transforms registry
            test_results['test_cases']['transforms_registry'] = self._test_transforms_registry()
            
            # Test transform steps
            test_results['test_cases']['transform_steps'] = self._test_transform_steps()
            
            # Test transform operations
            test_results['test_cases']['transform_operations'] = self._test_transform_operations()
            
            # Test contrib directory
            test_results['test_cases']['transforms_contrib'] = self._test_transforms_contrib()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.7
            
        except Exception as e:
            logger.error(f"Transforms library testing failed: {e}")
            test_results['issues'].append({
                'severity': 'medium',
                'component': 'transforms_library',
                'description': f"Transforms test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_transforms_registry(self) -> Dict[str, Any]:
        """Test transforms registry functionality."""
        try:
            from brainsmith.libraries.transforms.registry import discover_all_transforms
            
            transforms = discover_all_transforms()
            
            return {
                'passed': True,
                'description': f'Transforms registry discovered {len(transforms)} transforms'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import transforms registry: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Transforms registry test failed: {e}'
            }
    
    def _test_transform_steps(self) -> Dict[str, Any]:
        """Test transform steps functionality."""
        try:
            from brainsmith.libraries.transforms.steps import discover_all_steps
            
            steps = discover_all_steps()
            
            return {
                'passed': len(steps) > 0,
                'description': f'Transform steps discovered: {len(steps)} steps'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import transform steps: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Transform steps test failed: {e}'
            }
    
    def _test_transform_operations(self) -> Dict[str, Any]:
        """Test transform operations functionality."""
        try:
            from brainsmith.libraries.transforms.operations.expand_norms import expand_layernorm
            
            return {
                'passed': True,
                'description': 'Transform operations imported successfully'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import transform operations: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Transform operations test failed: {e}'
            }
    
    def _test_transforms_contrib(self) -> Dict[str, Any]:
        """Test transforms contrib directory."""
        try:
            contrib_dir = Path(__file__).parent.parent.parent / 'brainsmith/libraries/transforms/contrib'
            
            if contrib_dir.exists():
                return {
                    'passed': True,
                    'description': 'Transforms contrib directory exists'
                }
            else:
                return {
                    'passed': False,
                    'description': 'Transforms contrib directory not found'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'description': f'Transforms contrib test failed: {e}'
            }
    
    def _test_analysis_library(self) -> Dict[str, Any]:
        """Test analysis library functionality."""
        logger.info("Testing Analysis Library...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test analysis registry
            test_results['test_cases']['analysis_registry'] = self._test_analysis_registry()
            
            # Test profiling tools
            test_results['test_cases']['profiling_tools'] = self._test_profiling_tools()
            
            # Test generation tools
            test_results['test_cases']['generation_tools'] = self._test_generation_tools()
            
            # Test contrib directory
            test_results['test_cases']['analysis_contrib'] = self._test_analysis_contrib()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.6  # Lower threshold for analysis
            
        except Exception as e:
            logger.error(f"Analysis library testing failed: {e}")
            test_results['issues'].append({
                'severity': 'medium',
                'component': 'analysis_library',
                'description': f"Analysis test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_analysis_registry(self) -> Dict[str, Any]:
        """Test analysis registry functionality."""
        try:
            from brainsmith.libraries.analysis.registry import discover_all_analyzers
            
            analyzers = discover_all_analyzers()
            
            return {
                'passed': True,
                'description': f'Analysis registry discovered {len(analyzers)} analyzers'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import analysis registry: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Analysis registry test failed: {e}'
            }
    
    def _test_profiling_tools(self) -> Dict[str, Any]:
        """Test profiling tools functionality."""
        try:
            from brainsmith.libraries.analysis.profiling.roofline import roofline_analysis
            
            return {
                'passed': True,
                'description': 'Profiling tools (roofline) imported successfully'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import profiling tools: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Profiling tools test failed: {e}'
            }
    
    def _test_generation_tools(self) -> Dict[str, Any]:
        """Test generation tools functionality."""
        try:
            # Check if generation tools directory exists
            tools_dir = Path(__file__).parent.parent.parent / 'brainsmith/libraries/analysis/tools'
            
            if tools_dir.exists():
                return {
                    'passed': True,
                    'description': 'Generation tools directory exists'
                }
            else:
                return {
                    'passed': False,
                    'description': 'Generation tools directory not found'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'description': f'Generation tools test failed: {e}'
            }
    
    def _test_analysis_contrib(self) -> Dict[str, Any]:
        """Test analysis contrib directory."""
        try:
            contrib_dir = Path(__file__).parent.parent.parent / 'brainsmith/libraries/analysis/contrib'
            
            if contrib_dir.exists():
                return {
                    'passed': True,
                    'description': 'Analysis contrib directory exists'
                }
            else:
                return {
                    'passed': False,
                    'description': 'Analysis contrib directory not found'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'description': f'Analysis contrib test failed: {e}'
            }
    
    def _test_automation_library(self) -> Dict[str, Any]:
        """Test automation library functionality."""
        logger.info("Testing Automation Library...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test batch processing
            test_results['test_cases']['batch_processing'] = self._test_batch_processing()
            
            # Test parameter sweeps
            test_results['test_cases']['parameter_sweeps'] = self._test_parameter_sweeps()
            
            # Test contrib directory
            test_results['test_cases']['automation_contrib'] = self._test_automation_contrib()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.7
            
        except Exception as e:
            logger.error(f"Automation library testing failed: {e}")
            test_results['issues'].append({
                'severity': 'medium',
                'component': 'automation_library',
                'description': f"Automation test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_batch_processing(self) -> Dict[str, Any]:
        """Test batch processing functionality."""
        try:
            from brainsmith.libraries.automation.batch import batch_process_models
            
            return {
                'passed': True,
                'description': 'Batch processing functions imported successfully'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import batch processing: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Batch processing test failed: {e}'
            }
    
    def _test_parameter_sweeps(self) -> Dict[str, Any]:
        """Test parameter sweeps functionality."""
        try:
            from brainsmith.libraries.automation.sweep import parameter_sweep, find_best
            
            return {
                'passed': True,
                'description': 'Parameter sweep functions imported successfully'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import parameter sweeps: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Parameter sweeps test failed: {e}'
            }
    
    def _test_automation_contrib(self) -> Dict[str, Any]:
        """Test automation contrib directory."""
        try:
            contrib_dir = Path(__file__).parent.parent.parent / 'brainsmith/libraries/automation/contrib'
            
            if contrib_dir.exists():
                return {
                    'passed': True,
                    'description': 'Automation contrib directory exists'
                }
            else:
                return {
                    'passed': False,
                    'description': 'Automation contrib directory not found'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'description': f'Automation contrib test failed: {e}'
            }
    
    def _test_blueprints_library(self) -> Dict[str, Any]:
        """Test blueprints library functionality."""
        logger.info("Testing Blueprints Library...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test blueprints registry
            test_results['test_cases']['blueprints_registry'] = self._test_blueprints_registry()
            
            # Test blueprint templates
            test_results['test_cases']['blueprint_templates'] = self._test_blueprint_templates()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.8  # High threshold for blueprints
            
        except Exception as e:
            logger.error(f"Blueprints library testing failed: {e}")
            test_results['issues'].append({
                'severity': 'medium',
                'component': 'blueprints_library',
                'description': f"Blueprints test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_blueprints_registry(self) -> Dict[str, Any]:
        """Test blueprints registry functionality."""
        try:
            from brainsmith.libraries.blueprints.registry import discover_all_blueprints
            
            blueprints = discover_all_blueprints()
            
            return {
                'passed': len(blueprints) > 0,
                'description': f'Blueprints registry discovered {len(blueprints)} blueprints'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Failed to import blueprints registry: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Blueprints registry test failed: {e}'
            }
    
    def _test_blueprint_templates(self) -> Dict[str, Any]:
        """Test blueprint templates functionality."""
        try:
            # Check for blueprint template files
            blueprints_dir = Path(__file__).parent.parent.parent / 'brainsmith/libraries/blueprints'
            
            template_files = []
            for category_dir in blueprints_dir.iterdir():
                if category_dir.is_dir() and not category_dir.name.startswith('__'):
                    for file in category_dir.iterdir():
                        if file.suffix in ['.yaml', '.yml']:
                            template_files.append(file.name)
            
            return {
                'passed': len(template_files) > 0,
                'description': f'Found {len(template_files)} blueprint template files'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Blueprint templates test failed: {e}'
            }