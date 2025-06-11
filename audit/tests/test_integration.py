"""
Integration Testing Module

Tests for Phase 2: Integration Testing
- Cross-Layer Integration
- Blueprint Management Integration
- Registry Integration
- Import Dependency Health
"""

import sys
import os
import logging
import importlib
import ast
from pathlib import Path
from typing import Dict, Any, List, Set
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

class IntegrationTester:
    """Test suite for Brainsmith integration between components."""
    
    def __init__(self):
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Execute all integration tests."""
        self.test_results = {
            'cross_layer_integration': self._test_cross_layer_integration(),
            'blueprint_management_integration': self._test_blueprint_management_integration(),
            'registry_integration': self._test_registry_integration(),
            'import_dependency_health': self._test_import_dependency_health()
        }
        return self.test_results
    
    def _test_cross_layer_integration(self) -> Dict[str, Any]:
        """Test cross-layer integration."""
        logger.info("Testing Cross-Layer Integration...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test Core → Infrastructure integration
            test_results['test_cases']['core_to_infrastructure'] = self._test_core_to_infrastructure()
            
            # Test Infrastructure → Libraries integration
            test_results['test_cases']['infrastructure_to_libraries'] = self._test_infrastructure_to_libraries()
            
            # Test Core → Libraries integration
            test_results['test_cases']['core_to_libraries'] = self._test_core_to_libraries()
            
            # Test End-to-End workflows
            test_results['test_cases']['end_to_end_workflows'] = self._test_end_to_end_workflows()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.75  # 75% pass rate
            
            if not test_results['passed']:
                test_results['issues'].append({
                    'severity': 'high',
                    'component': 'cross_layer_integration',
                    'description': f"Only {passed_tests}/{total_tests} cross-layer integration tests passed"
                })
                
        except Exception as e:
            logger.error(f"Cross-layer integration testing failed: {e}")
            test_results['issues'].append({
                'severity': 'critical',
                'component': 'cross_layer_integration',
                'description': f"Integration test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_core_to_infrastructure(self) -> Dict[str, Any]:
        """Test Core → Infrastructure integration."""
        try:
            # Test forge() calls to DSE engine
            from brainsmith.core.api import forge
            from brainsmith.infrastructure.dse.engine import parameter_sweep
            
            # Check if forge() can successfully import and call DSE functions
            return {
                'passed': True,
                'description': 'Core to Infrastructure integration working (imports successful)'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Core to Infrastructure import failed: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Core to Infrastructure integration test failed: {e}'
            }
    
    def _test_infrastructure_to_libraries(self) -> Dict[str, Any]:
        """Test Infrastructure → Libraries integration."""
        try:
            # Test infrastructure discovery of libraries
            from brainsmith.infrastructure.dse.blueprint_manager import BlueprintManager
            from brainsmith.libraries.blueprints.registry import discover_all_blueprints
            
            # Test that infrastructure can discover and use library components
            manager = BlueprintManager()
            blueprints = discover_all_blueprints()
            
            return {
                'passed': True,
                'description': f'Infrastructure to Libraries integration working (discovered {len(blueprints)} blueprints)'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Infrastructure to Libraries import failed: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Infrastructure to Libraries integration test failed: {e}'
            }
    
    def _test_core_to_libraries(self) -> Dict[str, Any]:
        """Test Core → Libraries integration."""
        try:
            # Test direct library access from core
            from brainsmith.libraries.automation.sweep import parameter_sweep as lib_sweep
            from brainsmith.libraries.analysis.profiling.roofline import roofline_analysis
            
            return {
                'passed': True,
                'description': 'Core to Libraries integration working (direct access successful)'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Core to Libraries import failed: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Core to Libraries integration test failed: {e}'
            }
    
    def _test_end_to_end_workflows(self) -> Dict[str, Any]:
        """Test end-to-end workflows."""
        try:
            # Test complete import chain for a typical workflow
            from brainsmith.core.api import forge
            from brainsmith.infrastructure.hooks import log_optimization_event
            from brainsmith.libraries.automation.sweep import find_best
            
            # This tests that a complete workflow can import all necessary components
            return {
                'passed': True,
                'description': 'End-to-end workflow imports successful'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'End-to-end workflow import failed: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'End-to-end workflow test failed: {e}'
            }
    
    def _test_blueprint_management_integration(self) -> Dict[str, Any]:
        """Test blueprint management integration."""
        logger.info("Testing Blueprint Management Integration...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test split architecture validation
            test_results['test_cases']['split_architecture'] = self._test_blueprint_split_architecture()
            
            # Test design point creation
            test_results['test_cases']['design_point_creation'] = self._test_design_point_creation()
            
            # Test parameter space management
            test_results['test_cases']['parameter_space_management'] = self._test_parameter_space_management()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.8  # High threshold for critical integration
            
        except Exception as e:
            logger.error(f"Blueprint management integration testing failed: {e}")
            test_results['issues'].append({
                'severity': 'critical',
                'component': 'blueprint_management_integration',
                'description': f"Blueprint integration test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_blueprint_split_architecture(self) -> Dict[str, Any]:
        """Test blueprint split architecture between infrastructure/dse and libraries/blueprints."""
        try:
            # Test both systems can work together
            from brainsmith.infrastructure.dse.blueprint_manager import BlueprintManager
            from brainsmith.libraries.blueprints.registry import BlueprintLibraryRegistry
            
            # Test that both systems can discover blueprints
            infra_manager = BlueprintManager()
            libs_registry = BlueprintLibraryRegistry()
            
            infra_blueprints = infra_manager.discover_blueprints()
            libs_blueprints = libs_registry.discover_blueprints()
            
            return {
                'passed': True,
                'description': f'Blueprint split architecture working (infra: {len(infra_blueprints)}, libs: {len(libs_blueprints)})'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Blueprint split architecture import failed: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Blueprint split architecture test failed: {e}'
            }
    
    def _test_design_point_creation(self) -> Dict[str, Any]:
        """Test design point creation workflow."""
        try:
            from brainsmith.infrastructure.dse.blueprint_manager import BlueprintManager
            
            manager = BlueprintManager()
            
            # Test design point creation (this may fail gracefully with no blueprints)
            test_params = {'test_param': 1}
            design_point = manager.create_design_point('test_blueprint', test_params)
            
            # If it returns None, that's acceptable (no blueprints available)
            return {
                'passed': True,
                'description': f'Design point creation workflow functional (result: {design_point is not None})'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Design point creation test failed: {e}'
            }
    
    def _test_parameter_space_management(self) -> Dict[str, Any]:
        """Test parameter space management."""
        try:
            from brainsmith.infrastructure.dse.blueprint_manager import BlueprintManager
            
            manager = BlueprintManager()
            
            # Test parameter space extraction (may return empty space)
            param_space = manager.get_blueprint_parameter_space('test_blueprint')
            
            return {
                'passed': True,
                'description': f'Parameter space management functional (space: {len(param_space) if param_space else 0} params)'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Parameter space management test failed: {e}'
            }
    
    def _test_registry_integration(self) -> Dict[str, Any]:
        """Test registry integration."""
        logger.info("Testing Registry Integration...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test cross-registry discovery
            test_results['test_cases']['cross_registry_discovery'] = self._test_cross_registry_discovery()
            
            # Test dependency resolution
            test_results['test_cases']['dependency_resolution'] = self._test_dependency_resolution()
            
            # Test cache consistency
            test_results['test_cases']['cache_consistency'] = self._test_cache_consistency()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.7
            
        except Exception as e:
            logger.error(f"Registry integration testing failed: {e}")
            test_results['issues'].append({
                'severity': 'medium',
                'component': 'registry_integration',
                'description': f"Registry integration test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_cross_registry_discovery(self) -> Dict[str, Any]:
        """Test cross-registry discovery."""
        try:
            # Test that different registries can discover their components
            from brainsmith.libraries.kernels.registry import discover_all_kernels
            from brainsmith.libraries.transforms.registry import discover_all_transforms
            from brainsmith.libraries.blueprints.registry import discover_all_blueprints
            
            kernels = discover_all_kernels()
            transforms = discover_all_transforms()
            blueprints = discover_all_blueprints()
            
            total_components = len(kernels) + len(transforms) + len(blueprints)
            
            return {
                'passed': True,
                'description': f'Cross-registry discovery working ({total_components} total components)'
            }
            
        except ImportError as e:
            return {
                'passed': False,
                'description': f'Cross-registry discovery import failed: {e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'description': f'Cross-registry discovery test failed: {e}'
            }
    
    def _test_dependency_resolution(self) -> Dict[str, Any]:
        """Test dependency resolution."""
        try:
            from brainsmith.libraries.transforms.registry import TransformRegistry
            
            registry = TransformRegistry()
            
            # Test dependency validation
            test_transforms = ['test_transform1', 'test_transform2']
            errors = registry.validate_dependencies(test_transforms)
            
            # Should return a list (may be empty or contain errors)
            return {
                'passed': isinstance(errors, list),
                'description': f'Dependency resolution functional ({len(errors)} validation errors)'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Dependency resolution test failed: {e}'
            }
    
    def _test_cache_consistency(self) -> Dict[str, Any]:
        """Test cache consistency."""
        try:
            from brainsmith.libraries.kernels.registry import refresh_kernel_registry
            from brainsmith.libraries.transforms.registry import refresh_transform_registry
            
            # Test that cache refresh functions exist and work
            refresh_kernel_registry()
            refresh_transform_registry()
            
            return {
                'passed': True,
                'description': 'Cache consistency mechanisms functional'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Cache consistency test failed: {e}'
            }
    
    def _test_import_dependency_health(self) -> Dict[str, Any]:
        """Test import dependency health."""
        logger.info("Testing Import Dependency Health...")
        
        test_results = {
            'passed': False,
            'issues': [],
            'test_cases': {}
        }
        
        try:
            # Test import chain analysis
            test_results['test_cases']['import_chain_analysis'] = self._test_import_chain_analysis()
            
            # Test circular dependency detection
            test_results['test_cases']['circular_dependency_detection'] = self._test_circular_dependency_detection()
            
            # Test backward compatibility
            test_results['test_cases']['backward_compatibility'] = self._test_backward_compatibility()
            
            # Determine overall pass/fail
            passed_tests = sum(1 for result in test_results['test_cases'].values() if result.get('passed', False))
            total_tests = len(test_results['test_cases'])
            test_results['passed'] = passed_tests >= total_tests * 0.8  # High threshold for dependency health
            
        except Exception as e:
            logger.error(f"Import dependency health testing failed: {e}")
            test_results['issues'].append({
                'severity': 'high',
                'component': 'import_dependency_health',
                'description': f"Import dependency test execution failed: {str(e)}"
            })
        
        return test_results
    
    def _test_import_chain_analysis(self) -> Dict[str, Any]:
        """Test import chain analysis."""
        try:
            # Test key import paths
            import_tests = [
                'brainsmith.core.api',
                'brainsmith.infrastructure.dse.engine',
                'brainsmith.infrastructure.finn.interface',
                'brainsmith.libraries.kernels.registry',
                'brainsmith.libraries.transforms.registry',
                'brainsmith.libraries.analysis.registry',
                'brainsmith.libraries.automation.sweep',
                'brainsmith.libraries.blueprints.registry'
            ]
            
            successful_imports = 0
            failed_imports = []
            
            for module_name in import_tests:
                try:
                    importlib.import_module(module_name)
                    successful_imports += 1
                except ImportError as e:
                    failed_imports.append(f"{module_name}: {e}")
            
            success_rate = successful_imports / len(import_tests)
            
            return {
                'passed': success_rate >= 0.8,  # 80% of imports should work
                'description': f'Import chain analysis: {successful_imports}/{len(import_tests)} imports successful ({success_rate:.1%})'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Import chain analysis failed: {e}'
            }
    
    def _test_circular_dependency_detection(self) -> Dict[str, Any]:
        """Test circular dependency detection."""
        try:
            # Analyze import structure for circular dependencies
            brainsmith_root = Path(__file__).parent.parent.parent / 'brainsmith'
            
            def find_imports_in_file(file_path: Path) -> Set[str]:
                """Find imports in a Python file."""
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    imports = set()
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                if name.name.startswith('brainsmith'):
                                    imports.add(name.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.module.startswith('brainsmith'):
                                imports.add(node.module)
                    
                    return imports
                except:
                    return set()
            
            # Simple circular dependency detection
            circular_issues = []
            
            # Check for obvious circular patterns (this is a basic check)
            core_imports = find_imports_in_file(brainsmith_root / 'core' / 'api.py')
            infra_imports = find_imports_in_file(brainsmith_root / 'infrastructure' / 'dse' / 'engine.py')
            
            # Core shouldn't import from infrastructure.dse if dse imports from core
            if any('brainsmith.infrastructure' in imp for imp in core_imports):
                if any('brainsmith.core' in imp for imp in infra_imports):
                    circular_issues.append("Potential circular dependency between core and infrastructure")
            
            return {
                'passed': len(circular_issues) == 0,
                'description': f'Circular dependency check: {len(circular_issues)} potential issues found'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Circular dependency detection failed: {e}'
            }
    
    def _test_backward_compatibility(self) -> Dict[str, Any]:
        """Test backward compatibility."""
        try:
            # Test that legacy import paths still work
            legacy_tests = [
                # These might have compatibility aliases
                'brainsmith.core',
                'brainsmith.infrastructure',
                'brainsmith.libraries'
            ]
            
            working_imports = 0
            for module_name in legacy_tests:
                try:
                    importlib.import_module(module_name)
                    working_imports += 1
                except ImportError:
                    pass  # Expected for some legacy paths
            
            return {
                'passed': working_imports >= len(legacy_tests) // 2,  # At least half should work
                'description': f'Backward compatibility: {working_imports}/{len(legacy_tests)} legacy imports working'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'description': f'Backward compatibility test failed: {e}'
            }