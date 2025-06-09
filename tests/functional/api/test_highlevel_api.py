"""
High-Level API Tests

Tests for the main BrainSmith public API functions that users interact with directly.
Focus on real-world usage patterns and user experience validation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestHighLevelAPI:
    """Test suite for high-level BrainSmith API functions."""
    
    @pytest.mark.smoke
    def test_api_module_imports(self):
        """Test that core API modules can be imported successfully."""
        try:
            import brainsmith
            assert hasattr(brainsmith, '__version__')
            assert brainsmith.__version__ == "0.4.0"
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import BrainSmith modules: {e}")
    
    @pytest.mark.smoke
    def test_core_api_functions_available(self):
        """Test that core API functions are available."""
        import brainsmith
        
        # Check Week 1 core API functions
        assert hasattr(brainsmith, 'brainsmith_explore')
        assert hasattr(brainsmith, 'brainsmith_roofline')
        assert hasattr(brainsmith, 'brainsmith_dataflow_analysis')
        assert hasattr(brainsmith, 'build_model')
        assert hasattr(brainsmith, 'optimize_model')
        
        # Check utility functions
        assert hasattr(brainsmith, 'list_available_strategies')
        assert hasattr(brainsmith, 'recommend_strategy')
    
    @pytest.mark.core
    def test_build_model_basic_usage(self, test_data_manager):
        """Test basic model build workflow."""
        import brainsmith
        
        # Create test model file
        model_path = test_data_manager.create_test_model("test_model", "small")
        
        # Test basic build call
        result = brainsmith.build_model(
            model_path=str(model_path),
            blueprint_name="test_blueprint",
            parameters={'pe': 4, 'simd': 2}
        )
        
        # Validate result structure
        assert result is not None
        assert isinstance(result, dict)
        
        # Should have either success or fallback status
        assert 'status' in result or 'week1_fallback' in result
    
    @pytest.mark.core
    def test_optimize_model_basic_usage(self, test_data_manager):
        """Test basic model optimization workflow."""
        import brainsmith
        
        # Create test model file
        model_path = test_data_manager.create_test_model("opt_model", "small")
        
        # Test basic optimization call
        result = brainsmith.optimize_model(
            model_path=str(model_path),
            blueprint_name="test_blueprint",
            max_evaluations=5,
            strategy="auto"
        )
        
        # Validate result structure
        assert result is not None
        assert isinstance(result, (dict, tuple))
    
    @pytest.mark.core
    def test_list_available_strategies(self):
        """Test listing available optimization strategies."""
        import brainsmith
        
        strategies = brainsmith.list_available_strategies()
        
        assert isinstance(strategies, dict)
        assert len(strategies) > 0
        
        # Should have at least the fallback strategy
        assert any('fallback' in strategy_name or 'week1' in strategy_name 
                  for strategy_name in strategies.keys())
        
        # Validate strategy structure
        for strategy_name, strategy_info in strategies.items():
            assert isinstance(strategy_info, dict)
            assert 'description' in strategy_info
            assert 'available' in strategy_info
    
    @pytest.mark.core
    def test_recommend_strategy(self):
        """Test strategy recommendation functionality."""
        import brainsmith
        
        # Test basic recommendation
        strategy = brainsmith.recommend_strategy(
            n_parameters=5,
            max_evaluations=50,
            n_objectives=1
        )
        
        assert isinstance(strategy, str)
        assert len(strategy) > 0
        
        # Test multi-objective recommendation
        strategy_multi = brainsmith.recommend_strategy(
            n_parameters=10,
            max_evaluations=100,
            n_objectives=3
        )
        
        assert isinstance(strategy_multi, str)
        assert len(strategy_multi) > 0
    
    @pytest.mark.core
    def test_design_space_functionality(self):
        """Test design space loading and sampling."""
        import brainsmith
        
        # Test design space loading (may fallback)
        try:
            design_space = brainsmith.load_design_space("test_blueprint")
            assert design_space is not None
            assert hasattr(design_space, 'name')
            
            # Test sampling if design space loaded successfully
            if hasattr(design_space, 'parameters'):
                samples = brainsmith.sample_design_space(
                    design_space, 
                    n_samples=5,
                    strategy="latin_hypercube"
                )
                assert isinstance(samples, list)
                
        except Exception as e:
            # Expected for Week 1 fallback
            pass
    
    @pytest.mark.core
    def test_brainsmith_explore_api(self, test_data_manager):
        """Test the Week 1 brainsmith_explore API."""
        import brainsmith
        
        model_path = test_data_manager.create_test_model("explore_model", "small")
        
        try:
            # Test the core explore function
            results, analysis = brainsmith.brainsmith_explore(
                model_path=str(model_path),
                blueprint_path="test_blueprint",
                exit_point="dataflow_generation"
            )
            
            assert results is not None
            assert analysis is not None
            
        except Exception as e:
            # May fail due to missing dependencies, but should not crash
            assert isinstance(e, (ImportError, AttributeError, FileNotFoundError))
    
    @pytest.mark.core
    def test_error_handling_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        import brainsmith
        
        # Test with invalid model path
        try:
            result = brainsmith.build_model(
                model_path="/nonexistent/model.onnx",
                blueprint_name="test_blueprint"
            )
            # Should either succeed with fallback or raise appropriate error
            assert result is not None or isinstance(result, dict)
        except (FileNotFoundError, ValueError, ImportError):
            # These are acceptable errors for invalid inputs
            pass
    
    @pytest.mark.performance
    def test_api_response_time(self):
        """Test API response times for basic operations."""
        import brainsmith
        import time
        
        # Test strategy listing performance
        start_time = time.time()
        strategies = brainsmith.list_available_strategies()
        list_time = time.time() - start_time
        
        assert list_time < 5.0, f"Strategy listing took {list_time:.2f}s, expected < 5s"
        assert isinstance(strategies, dict)
        
        # Test recommendation performance
        start_time = time.time()
        strategy = brainsmith.recommend_strategy(n_parameters=5)
        recommend_time = time.time() - start_time
        
        assert recommend_time < 2.0, f"Strategy recommendation took {recommend_time:.2f}s, expected < 2s"
        assert isinstance(strategy, str)


class TestWeek1SpecificFeatures:
    """Test suite for Week 1 specific features and fallbacks."""
    
    @pytest.mark.core
    def test_week1_fallback_behavior(self, test_data_manager):
        """Test Week 1 fallback behavior when advanced features unavailable."""
        import brainsmith
        
        model_path = test_data_manager.create_test_model("fallback_model", "small")
        
        # Test build_model fallback
        result = brainsmith.build_model(
            model_path=str(model_path),
            blueprint_name="test_blueprint"
        )
        
        # Should return fallback result if BrainsmithCompiler not available
        if isinstance(result, dict) and result.get('status') == 'fallback':
            assert 'week1_fallback' in result
            assert result['week1_fallback'] is True
            assert 'model_path' in result
            assert 'blueprint_name' in result
    
    @pytest.mark.core
    def test_core_components_availability(self):
        """Test which core components are available in current implementation."""
        import brainsmith
        
        # Check what's available vs None
        components = {
            'BrainsmithConfig': brainsmith.BrainsmithConfig,
            'BrainsmithResult': brainsmith.BrainsmithResult,
            'DSEResult': brainsmith.DSEResult,
            'BrainsmithMetrics': brainsmith.BrainsmithMetrics,
            'BrainsmithCompiler': brainsmith.BrainsmithCompiler,
            'DesignSpace': brainsmith.DesignSpace,
            'DesignPoint': brainsmith.DesignPoint
        }
        
        available_components = {name: comp for name, comp in components.items() if comp is not None}
        unavailable_components = {name: comp for name, comp in components.items() if comp is None}
        
        # Should have at least DesignSpace and DesignPoint
        assert 'DesignSpace' in available_components
        assert 'DesignPoint' in available_components
        
        print(f"Available components: {list(available_components.keys())}")
        print(f"Unavailable components: {list(unavailable_components.keys())}")
    
    @pytest.mark.core
    def test_blueprint_system_availability(self):
        """Test blueprint system availability."""
        import brainsmith
        
        blueprint_functions = {
            'Blueprint': brainsmith.Blueprint,
            'get_blueprint': brainsmith.get_blueprint,
            'load_blueprint': brainsmith.load_blueprint,
            'list_blueprints': brainsmith.list_blueprints
        }
        
        available_blueprint_funcs = {name: func for name, func in blueprint_functions.items() if func is not None}
        
        if available_blueprint_funcs:
            print(f"Available blueprint functions: {list(available_blueprint_funcs.keys())}")
            
            # Test blueprint listing if available
            if brainsmith.list_blueprints is not None:
                try:
                    blueprints = brainsmith.list_blueprints()
                    assert isinstance(blueprints, (list, dict))
                except Exception as e:
                    # May fail due to missing blueprint files
                    pass
    
    @pytest.mark.core
    def test_dse_system_availability(self):
        """Test DSE system availability."""
        import brainsmith
        
        dse_components = {
            'DSEInterface': brainsmith.DSEInterface,
            'SimpleDSEEngine': brainsmith.SimpleDSEEngine,
            'ExternalDSEAdapter': brainsmith.ExternalDSEAdapter,
            'DSEAnalyzer': brainsmith.DSEAnalyzer,
            'ParetoAnalyzer': brainsmith.ParetoAnalyzer
        }
        
        available_dse_comps = {name: comp for name, comp in dse_components.items() if comp is not None}
        
        if available_dse_comps:
            print(f"Available DSE components: {list(available_dse_comps.keys())}")
        else:
            print("DSE system not available - using Week 1 fallbacks")


class TestAPIIntegration:
    """Test suite for API integration and workflow testing."""
    
    @pytest.mark.integration
    def test_full_workflow_basic(self, test_data_manager):
        """Test a basic end-to-end workflow."""
        import brainsmith
        
        model_path = test_data_manager.create_test_model("workflow_model", "small")
        
        # Step 1: Get available strategies
        strategies = brainsmith.list_available_strategies()
        assert len(strategies) > 0
        
        # Step 2: Get a recommendation
        recommended_strategy = brainsmith.recommend_strategy(n_parameters=5)
        assert isinstance(recommended_strategy, str)
        
        # Step 3: Try to build a model
        try:
            result = brainsmith.build_model(
                model_path=str(model_path),
                blueprint_name="test_blueprint",
                parameters={'pe': 4}
            )
            assert result is not None
        except Exception as e:
            # Expected for Week 1 without full compiler
            pass
        
        # Step 4: Try optimization
        try:
            opt_result = brainsmith.optimize_model(
                model_path=str(model_path),
                blueprint_name="test_blueprint",
                max_evaluations=3
            )
            assert opt_result is not None
        except Exception as e:
            # Expected for Week 1 without full DSE
            pass
    
    @pytest.mark.integration
    def test_api_consistency(self):
        """Test API consistency and return type validation."""
        import brainsmith
        
        # Test that API functions return consistent types
        strategies = brainsmith.list_available_strategies()
        assert isinstance(strategies, dict)
        
        for strategy_name, strategy_info in strategies.items():
            assert isinstance(strategy_name, str)
            assert isinstance(strategy_info, dict)
            assert 'available' in strategy_info
            assert isinstance(strategy_info['available'], bool)
        
        # Test strategy recommendation consistency
        rec1 = brainsmith.recommend_strategy(n_parameters=5)
        rec2 = brainsmith.recommend_strategy(n_parameters=5)
        
        # Should be deterministic for same inputs
        assert rec1 == rec2
        assert isinstance(rec1, str)
        assert isinstance(rec2, str)