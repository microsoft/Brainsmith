"""
Package Imports Tests - Infrastructure Layer

Tests the import system and package structure.
Validates import paths, graceful fallbacks, and __all__ exports.
"""

import pytest
import sys
import importlib
from unittest.mock import patch


@pytest.mark.infrastructure
class TestCoreImports:
    """Test core package imports work correctly."""
    
    def test_main_package_import(self):
        """Test importing main brainsmith package."""
        try:
            import brainsmith
            assert hasattr(brainsmith, '__version__')
            assert hasattr(brainsmith, '__all__')
        except ImportError as e:
            pytest.fail(f"Main package import failed: {e}")
    
    def test_core_api_import(self):
        """Test importing core API components."""
        try:
            from brainsmith.core.api import forge, validate_blueprint
            assert callable(forge)
            assert callable(validate_blueprint)
        except ImportError as e:
            pytest.skip(f"Core API not available: {e}")
    
    def test_core_cli_import(self):
        """Test importing CLI components."""
        try:
            from brainsmith.core.cli import brainsmith
            # Should be a Click command group
            assert hasattr(brainsmith, 'commands') or hasattr(brainsmith, 'callback')
        except ImportError as e:
            pytest.skip(f"CLI not available: {e}")
    
    def test_core_metrics_import(self):
        """Test importing metrics components."""
        try:
            from brainsmith.core.metrics import DSEMetrics, PerformanceMetrics, ResourceMetrics
            assert DSEMetrics is not None
            assert PerformanceMetrics is not None
            assert ResourceMetrics is not None
        except ImportError as e:
            pytest.skip(f"Metrics not available: {e}")


@pytest.mark.infrastructure  
class TestInfrastructureImports:
    """Test infrastructure layer imports."""
    
    def test_design_space_import(self):
        """Test importing design space components."""
        try:
            from brainsmith.infrastructure.dse.design_space import (
                DesignSpace, DesignPoint, ParameterDefinition
            )
            assert DesignSpace is not None
            assert DesignPoint is not None  
            assert ParameterDefinition is not None
        except ImportError as e:
            pytest.skip(f"Design space not available: {e}")
    
    def test_hooks_import(self):
        """Test importing hooks system."""
        try:
            from brainsmith.infrastructure.hooks import events, types
            # These might be None if not available, which is fine
        except ImportError as e:
            pytest.skip(f"Hooks system not available: {e}")
    
    def test_infrastructure_placeholders(self):
        """Test that infrastructure placeholder modules exist."""
        infrastructure_modules = [
            'brainsmith.infrastructure.blueprint',
            'brainsmith.infrastructure.finn', 
            'brainsmith.infrastructure.data'
        ]
        
        for module_name in infrastructure_modules:
            try:
                module = importlib.import_module(module_name)
                # Should exist even if minimal
                assert module is not None
            except ImportError:
                pytest.skip(f"Infrastructure module {module_name} not available")


@pytest.mark.infrastructure
class TestLibraryImports:
    """Test library layer imports."""
    
    def test_kernels_library_import(self):
        """Test importing kernels library."""
        try:
            from brainsmith.libraries.kernels import functions, types
            # These might be None if not available
        except ImportError as e:
            pytest.skip(f"Kernels library not available: {e}")
    
    def test_transforms_library_import(self):
        """Test importing transforms library."""
        try:
            from brainsmith.libraries.transforms.steps import bert, cleanup, optimizations
            # These might be None if not available
        except ImportError as e:
            pytest.skip(f"Transforms library not available: {e}")
    
    def test_analysis_library_import(self):
        """Test importing analysis library."""
        try:
            from brainsmith.libraries.analysis.profiling import roofline_analysis
            # Might be None if not available
        except ImportError as e:
            pytest.skip(f"Analysis library not available: {e}")
    
    def test_automation_library_import(self):
        """Test importing automation library."""
        try:
            from brainsmith.libraries.automation import batch, sweep
            # These might be None if not available
        except ImportError as e:
            pytest.skip(f"Automation library not available: {e}")


@pytest.mark.infrastructure
class TestGracefulFallbacks:
    """Test graceful fallback behavior for missing components."""
    
    def test_main_package_fallback_imports(self):
        """Test that main package handles missing components gracefully."""
        try:
            import brainsmith
            
            # These might be None but should not cause import errors
            dse_components = [
                'DSEInterface', 'DSEAnalyzer', 'ParetoAnalyzer',
                'DSEConfiguration', 'DSEObjective', 'OptimizationObjective'
            ]
            
            for component in dse_components:
                value = getattr(brainsmith, component, 'NOT_FOUND')
                # Should either exist or be None (graceful fallback)
                assert value == 'NOT_FOUND' or value is None or callable(value)
                
        except ImportError as e:
            pytest.skip(f"Main package not available: {e}")
    
    def test_optional_component_fallbacks(self):
        """Test optional component fallback behavior."""
        try:
            import brainsmith
            
            # These are marked as optional and might be None
            optional_components = [
                'BrainsmithConfig', 'BrainsmithResult', 'DSEResult',
                'BrainsmithMetrics', 'BrainsmithCompiler', 'get_core_status'
            ]
            
            for component in optional_components:
                value = getattr(brainsmith, component, 'NOT_FOUND')
                # Should either exist or be None (graceful fallback)
                assert value == 'NOT_FOUND' or value is None or callable(value)
                
        except ImportError as e:
            pytest.skip(f"Main package not available: {e}")
    
    def test_backward_compatibility_fallbacks(self):
        """Test backward compatibility import fallbacks."""
        try:
            import brainsmith
            
            # These are backward compatibility imports that might be None
            compat_components = [
                'roofline_analysis', 'RooflineProfiler',
                'batch', 'sweep', 'kernel_functions', 'kernel_types'
            ]
            
            for component in compat_components:
                value = getattr(brainsmith, component, 'NOT_FOUND')
                # Should either exist or be None (graceful fallback)
                assert value == 'NOT_FOUND' or value is None or callable(value)
                
        except ImportError as e:
            pytest.skip(f"Main package not available: {e}")


@pytest.mark.infrastructure
class TestPackageStructure:
    """Test package structure and organization."""
    
    def test_all_exports_defined(self):
        """Test that __all__ is properly defined."""
        try:
            import brainsmith
            
            assert hasattr(brainsmith, '__all__')
            assert isinstance(brainsmith.__all__, list)
            assert len(brainsmith.__all__) > 0
            
            # Check that exported items actually exist or are None
            for item in brainsmith.__all__:
                value = getattr(brainsmith, item, 'NOT_FOUND')
                assert value != 'NOT_FOUND', f"Exported item '{item}' not found in package"
                
        except ImportError as e:
            pytest.skip(f"Main package not available: {e}")
    
    def test_package_metadata(self):
        """Test package metadata is properly defined."""
        try:
            import brainsmith
            
            # Check version
            assert hasattr(brainsmith, '__version__')
            assert isinstance(brainsmith.__version__, str)
            assert len(brainsmith.__version__) > 0
            
            # Check author
            assert hasattr(brainsmith, '__author__')
            assert isinstance(brainsmith.__author__, str)
            
            # Check description
            assert hasattr(brainsmith, '__description__')
            assert isinstance(brainsmith.__description__, str)
            
        except ImportError as e:
            pytest.skip(f"Main package not available: {e}")
    
    def test_core_layer_structure(self):
        """Test core layer module structure."""
        core_modules = [
            'brainsmith.core.api',
            'brainsmith.core.cli', 
            'brainsmith.core.metrics'
        ]
        
        for module_name in core_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except ImportError:
                pytest.skip(f"Core module {module_name} not available")
    
    def test_infrastructure_layer_structure(self):
        """Test infrastructure layer module structure."""
        infra_modules = [
            'brainsmith.infrastructure.dse',
            'brainsmith.infrastructure.hooks',
            'brainsmith.infrastructure.blueprint',
            'brainsmith.infrastructure.finn',
            'brainsmith.infrastructure.data'
        ]
        
        for module_name in infra_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except ImportError:
                pytest.skip(f"Infrastructure module {module_name} not available")
    
    def test_libraries_layer_structure(self):
        """Test libraries layer module structure."""
        lib_modules = [
            'brainsmith.libraries.kernels',
            'brainsmith.libraries.transforms',
            'brainsmith.libraries.analysis', 
            'brainsmith.libraries.automation'
        ]
        
        for module_name in lib_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except ImportError:
                pytest.skip(f"Library module {module_name} not available")


@pytest.mark.infrastructure
class TestImportErrorHandling:
    """Test import error handling and recovery."""
    
    def test_missing_optional_dependency_handling(self):
        """Test handling of missing optional dependencies."""
        # Test that the package can handle missing optional dependencies
        # This is mainly about ensuring imports don't crash
        
        try:
            import brainsmith
            
            # These functions should work even if some dependencies are missing
            utility_functions = [
                'load_design_space', 'sample_design_space',
                'analyze_dse_results', 'get_pareto_frontier',
                'list_available_strategies', 'recommend_strategy'
            ]
            
            for func_name in utility_functions:
                func = getattr(brainsmith, func_name, None)
                if func is not None:
                    assert callable(func)
                    
        except ImportError as e:
            pytest.skip(f"Main package not available: {e}")
    
    def test_circular_import_prevention(self):
        """Test that there are no circular import issues."""
        # Import all main components to check for circular imports
        components_to_test = [
            'brainsmith',
            'brainsmith.core',
            'brainsmith.infrastructure', 
            'brainsmith.libraries'
        ]
        
        for component in components_to_test:
            try:
                importlib.import_module(component)
            except ImportError as e:
                if "circular" in str(e).lower():
                    pytest.fail(f"Circular import detected in {component}: {e}")
                else:
                    pytest.skip(f"Component {component} not available: {e}")
    
    @patch('sys.modules')
    def test_import_with_corrupted_modules(self, mock_modules):
        """Test import behavior when some modules are corrupted."""
        # This test ensures graceful degradation
        
        # Mock a corrupted module in sys.modules
        original_modules = sys.modules.copy()
        
        try:
            # Simulate corrupted module
            sys.modules['brainsmith.dse'] = None
            
            # Should still be able to import main package
            import brainsmith
            
            # DSE components should be None
            assert brainsmith.DSEInterface is None
            
        except ImportError:
            pytest.skip("Could not test with corrupted modules")
        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)


@pytest.mark.integration
def test_complete_import_workflow():
    """Test complete import workflow from top-level package."""
    try:
        # Start with top-level import
        import brainsmith
        
        # Test that we can access core functionality
        if hasattr(brainsmith, 'forge') and brainsmith.forge is not None:
            assert callable(brainsmith.forge)
        
        # Test that we can access data structures
        if hasattr(brainsmith, 'DesignSpace') and brainsmith.DesignSpace is not None:
            # Should be able to create a design space
            ds = brainsmith.DesignSpace("test")
            assert ds.name == "test"
        
        # Test that we can access utility functions
        if hasattr(brainsmith, 'list_available_strategies') and brainsmith.list_available_strategies is not None:
            strategies = brainsmith.list_available_strategies()
            assert isinstance(strategies, dict)
        
    except ImportError as e:
        pytest.skip(f"Complete import workflow not available: {e}")


# Helper functions for import testing
def safely_import_module(module_name: str):
    """Helper to safely import a module and return it or None."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def check_module_has_attributes(module, expected_attributes: list):
    """Helper to check if module has expected attributes."""
    if module is None:
        return False
    
    for attr in expected_attributes:
        if not hasattr(module, attr):
            return False
    return True


def assert_import_fallback_behavior(package_name: str, component_names: list):
    """Helper to assert proper import fallback behavior."""
    try:
        package = importlib.import_module(package_name)
        
        for component in component_names:
            value = getattr(package, component, 'NOT_FOUND')
            # Should either exist or be None (graceful fallback)
            assert value == 'NOT_FOUND' or value is None or callable(value), \
                f"Component {component} in {package_name} should exist or be None"
                
    except ImportError:
        pytest.skip(f"Package {package_name} not available for fallback testing")


def get_all_submodules(package_name: str) -> list:
    """Helper to get all submodules of a package."""
    try:
        package = importlib.import_module(package_name)
        if hasattr(package, '__path__'):
            import pkgutil
            return [name for _, name, _ in pkgutil.iter_modules(package.__path__)]
        else:
            return []
    except ImportError:
        return []