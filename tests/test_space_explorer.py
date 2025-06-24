"""
Comprehensive tests for Design Space Explorer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import time

from brainsmith.core.dse.space_explorer import (
    DesignSpaceExplorer, ExplorationConfig, ExplorationProgress, 
    ExplorationResults, explore_design_space
)
from brainsmith.core.dse.combination_generator import ComponentCombination
from brainsmith.core.blueprint import (
    DesignSpaceDefinition, NodeDesignSpace, TransformDesignSpace,
    ComponentSpace, ExplorationRules, DSEStrategy, DSEStrategies
)


class TestExplorationConfig:
    """Test ExplorationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExplorationConfig()
        
        assert config.max_evaluations == 100
        assert config.strategy_name is None
        assert config.enable_caching == True
        assert config.parallel_evaluations == 1
        assert config.early_termination_patience == 20
        assert config.progress_callback is None
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid max_evaluations
        with pytest.raises(ValueError, match="max_evaluations must be positive"):
            ExplorationConfig(max_evaluations=0)
        
        # Test invalid parallel_evaluations
        with pytest.raises(ValueError, match="parallel_evaluations must be positive"):
            ExplorationConfig(parallel_evaluations=0)
        
        # Test invalid early_termination_patience
        with pytest.raises(ValueError, match="early_termination_patience must be positive"):
            ExplorationConfig(early_termination_patience=0)
    
    def test_custom_config(self):
        """Test custom configuration."""
        callback = Mock()
        
        config = ExplorationConfig(
            max_evaluations=50,
            strategy_name="custom_strategy",
            enable_caching=False,
            parallel_evaluations=4,
            progress_callback=callback
        )
        
        assert config.max_evaluations == 50
        assert config.strategy_name == "custom_strategy"
        assert config.enable_caching == False
        assert config.parallel_evaluations == 4
        assert config.progress_callback == callback


class TestExplorationProgress:
    """Test ExplorationProgress tracking."""
    
    def test_progress_initialization(self):
        """Test progress initialization."""
        progress = ExplorationProgress(total_budget=100)
        
        assert progress.total_budget == 100
        assert progress.evaluations_completed == 0
        assert progress.evaluations_cached == 0
        assert progress.best_score == 0.0
        assert progress.progress_percentage == 0.0
    
    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        progress = ExplorationProgress(total_budget=100)
        
        progress.evaluations_completed = 25
        assert progress.progress_percentage == 25.0
        
        progress.evaluations_completed = 100
        assert progress.progress_percentage == 100.0
    
    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation."""
        progress = ExplorationProgress(total_budget=100)
        
        # Mock start time
        with patch('time.time', return_value=1000.0):
            progress.start_time = 1000.0
        
        with patch('time.time', return_value=1010.0):
            assert progress.elapsed_time == 10.0
    
    def test_completion_time_estimation(self):
        """Test completion time estimation."""
        progress = ExplorationProgress(total_budget=100)
        progress.evaluations_completed = 25
        
        # Mock elapsed time
        with patch.object(progress, 'elapsed_time', 10.0):
            estimated = progress.estimate_completion_time()
            # Should estimate 30 more seconds (75 remaining evals * 0.4 sec/eval)
            assert estimated == 30.0


class TestExplorationResults:
    """Test ExplorationResults container."""
    
    def test_results_creation(self):
        """Test results creation and conversion."""
        combo = ComponentCombination(canonical_ops=["LayerNorm"])
        
        results = ExplorationResults(
            best_combination=combo,
            best_score=0.8,
            all_combinations=[combo],
            performance_data=[{'score': 0.8}],
            pareto_frontier=[combo],
            exploration_summary={'total': 1},
            strategy_metadata={'strategy': 'test'},
            execution_stats={'time': 10.0}
        )
        
        assert results.best_combination == combo
        assert results.best_score == 0.8
        assert len(results.all_combinations) == 1
        assert len(results.pareto_frontier) == 1
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        combo = ComponentCombination(canonical_ops=["LayerNorm"])
        
        results = ExplorationResults(
            best_combination=combo,
            best_score=0.8,
            all_combinations=[combo],
            performance_data=[],
            pareto_frontier=[combo],
            exploration_summary={},
            strategy_metadata={},
            execution_stats={}
        )
        
        result_dict = results.to_dict()
        
        assert result_dict['best_score'] == 0.8
        assert result_dict['total_combinations_evaluated'] == 1
        assert result_dict['pareto_frontier_size'] == 1
        assert 'best_combination' in result_dict
    
    def test_save_to_file(self):
        """Test saving results to file."""
        combo = ComponentCombination(canonical_ops=["LayerNorm"])
        
        results = ExplorationResults(
            best_combination=combo,
            best_score=0.8,
            all_combinations=[combo],
            performance_data=[],
            pareto_frontier=[],
            exploration_summary={},
            strategy_metadata={},
            execution_stats={}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            results.save_to_file(f.name)
            
            # Verify file was created and contains valid JSON
            with open(f.name, 'r') as read_f:
                data = json.load(read_f)
                assert data['best_score'] == 0.8


class TestDesignSpaceExplorer:
    """Test DesignSpaceExplorer main class."""
    
    def create_test_design_space(self) -> DesignSpaceDefinition:
        """Create test design space with strategies."""
        return DesignSpaceDefinition(
            name="explorer_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["LayerNorm", "Softmax"],
                    exploration=ExplorationRules(
                        required=["LayerNorm"],
                        optional=["Softmax"]
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["cleanup", "streamlining"],
                    exploration=ExplorationRules(required=["cleanup"])
                )
            ),
            dse_strategies=DSEStrategies(
                primary_strategy="test_strategy",
                strategies={
                    "test_strategy": DSEStrategy(
                        name="test_strategy",
                        max_evaluations=10,
                        sampling="adaptive"
                    )
                }
            )
        )
    
    def test_explorer_initialization(self):
        """Test explorer initialization."""
        design_space = self.create_test_design_space()
        config = ExplorationConfig(max_evaluations=20)
        
        explorer = DesignSpaceExplorer(design_space, config)
        
        assert explorer.design_space == design_space
        assert explorer.config == config
        assert explorer.progress.total_budget == 20
        assert len(explorer.evaluation_cache) == 0
    
    def test_cache_setup(self):
        """Test cache setup with directory."""
        design_space = self.create_test_design_space()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExplorationConfig(
                enable_caching=True,
                cache_directory=temp_dir
            )
            
            explorer = DesignSpaceExplorer(design_space, config)
            
            # Cache directory should be created
            assert Path(temp_dir).exists()
    
    def test_single_combination_evaluation(self):
        """Test evaluation of single combination."""
        design_space = self.create_test_design_space()
        explorer = DesignSpaceExplorer(design_space)
        
        combo = ComponentCombination(canonical_ops=["LayerNorm"])
        
        # Mock evaluation function
        def mock_eval_func(model_path, combination):
            return {'throughput': 100.0, 'latency': 10.0}
        
        result = explorer._evaluate_single_combination("test_model.onnx", combo, mock_eval_func)
        
        assert result['success'] == True
        assert result['combination'] == combo
        assert result['metrics']['throughput'] == 100.0
        assert result['primary_metric'] == 100.0  # Default to throughput
        assert 'evaluation_time' in result
    
    def test_evaluation_error_handling(self):
        """Test handling of evaluation errors."""
        design_space = self.create_test_design_space()
        explorer = DesignSpaceExplorer(design_space)
        
        combo = ComponentCombination(canonical_ops=["LayerNorm"])
        
        # Mock evaluation function that raises error
        def failing_eval_func(model_path, combination):
            raise RuntimeError("Evaluation failed")
        
        result = explorer._evaluate_single_combination("test_model.onnx", combo, failing_eval_func)
        
        assert result['success'] == False
        assert result['combination'] == combo
        assert result['primary_metric'] == 0.0
        assert 'error' in result
        assert "Evaluation failed" in result['error']
    
    def test_batch_evaluation_sequential(self):
        """Test batch evaluation in sequential mode."""
        design_space = self.create_test_design_space()
        config = ExplorationConfig(parallel_evaluations=1)
        explorer = DesignSpaceExplorer(design_space, config)
        
        combinations = [
            ComponentCombination(canonical_ops=["LayerNorm"]),
            ComponentCombination(canonical_ops=["LayerNorm", "Softmax"])
        ]
        
        def mock_eval_func(model_path, combination):
            return {'throughput': len(combination.canonical_ops) * 50.0}
        
        results = explorer._evaluate_combination_batch("test_model.onnx", combinations, mock_eval_func)
        
        assert len(results) == 2
        assert results[0]['metrics']['throughput'] == 50.0  # 1 op * 50
        assert results[1]['metrics']['throughput'] == 100.0  # 2 ops * 50
    
    def test_batch_evaluation_parallel(self):
        """Test batch evaluation in parallel mode."""
        design_space = self.create_test_design_space()
        config = ExplorationConfig(parallel_evaluations=2)
        explorer = DesignSpaceExplorer(design_space, config)
        
        combinations = [
            ComponentCombination(canonical_ops=["LayerNorm"]),
            ComponentCombination(canonical_ops=["LayerNorm", "Softmax"])
        ]
        
        def mock_eval_func(model_path, combination):
            time.sleep(0.1)  # Simulate evaluation time
            return {'throughput': len(combination.canonical_ops) * 50.0}
        
        start_time = time.time()
        results = explorer._evaluate_combination_batch("test_model.onnx", combinations, mock_eval_func)
        end_time = time.time()
        
        assert len(results) == 2
        # Should take less time than sequential (< 0.2s vs 0.2s)
        assert end_time - start_time < 0.18
    
    def test_caching_functionality(self):
        """Test evaluation result caching."""
        design_space = self.create_test_design_space()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExplorationConfig(
                enable_caching=True,
                cache_directory=temp_dir
            )
            explorer = DesignSpaceExplorer(design_space, config)
            
            combo = ComponentCombination(canonical_ops=["LayerNorm"])
            
            def mock_eval_func(model_path, combination):
                return {'throughput': 100.0}
            
            # First evaluation - should call function
            result1 = explorer._evaluate_single_combination("test_model.onnx", combo, mock_eval_func)
            assert result1['success'] == True
            assert explorer.progress.evaluations_cached == 0
            
            # Second evaluation - should use cache
            result2 = explorer._evaluate_single_combination("test_model.onnx", combo, mock_eval_func)
            assert result2['success'] == True
            assert explorer.progress.evaluations_cached == 1
    
    def test_early_termination_detection(self):
        """Test early termination condition detection."""
        design_space = self.create_test_design_space()
        config = ExplorationConfig(
            early_termination_patience=5,
            early_termination_threshold=0.01
        )
        explorer = DesignSpaceExplorer(design_space, config)
        
        # Simulate stagnant performance
        explorer.performance_history = [
            {'success': True, 'primary_metric': 0.5} for _ in range(10)
        ]
        
        assert explorer._should_terminate_early() == True
        
        # Simulate improving performance
        explorer.performance_history = [
            {'success': True, 'primary_metric': 0.5 + i * 0.1} for i in range(10)
        ]
        
        assert explorer._should_terminate_early() == False
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        design_space = self.create_test_design_space()
        callback_data = []
        
        def progress_callback(data):
            callback_data.append(data)
        
        config = ExplorationConfig(progress_callback=progress_callback)
        explorer = DesignSpaceExplorer(design_space, config)
        
        # Simulate some results
        batch_results = [
            {'success': True, 'primary_metric': 0.8, 'combination': ComponentCombination()}
        ]
        
        context = Mock()
        context.performance_history = []
        
        explorer._update_exploration_state(batch_results, context)
        
        # Callback should have been called
        assert len(callback_data) == 1
        assert 'progress_percentage' in callback_data[0]
        assert 'best_score' in callback_data[0]
    
    def test_exploration_summary_generation(self):
        """Test exploration summary generation."""
        design_space = self.create_test_design_space()
        explorer = DesignSpaceExplorer(design_space)
        
        # Add mock performance history
        explorer.performance_history = [
            {'success': True, 'primary_metric': 0.6},
            {'success': True, 'primary_metric': 0.8},
            {'success': False, 'primary_metric': 0.0},
            {'success': True, 'primary_metric': 0.7}
        ]
        
        summary = explorer._generate_exploration_summary()
        
        assert summary['total_combinations_evaluated'] == 4
        assert summary['successful_evaluations'] == 3
        assert summary['success_rate'] == 0.75
        assert summary['best_score'] == 0.8
        assert summary['worst_score'] == 0.6
        assert 'average_score' in summary
    
    def test_component_frequency_analysis(self):
        """Test component frequency analysis."""
        design_space = self.create_test_design_space()
        explorer = DesignSpaceExplorer(design_space)
        
        # Add mock performance history with combinations
        explorer.performance_history = [
            {
                'success': True,
                'primary_metric': 0.8,
                'combination': ComponentCombination(canonical_ops=["LayerNorm"])
            },
            {
                'success': True,
                'primary_metric': 0.6,
                'combination': ComponentCombination(canonical_ops=["LayerNorm", "Softmax"])
            }
        ]
        
        frequency_analysis = explorer._analyze_component_frequency()
        
        assert 'canonical_op_LayerNorm' in frequency_analysis
        assert frequency_analysis['canonical_op_LayerNorm'] == 2  # In both combinations
        assert 'canonical_op_Softmax' in frequency_analysis
        assert frequency_analysis['canonical_op_Softmax'] == 1  # In one combination


class TestExplorationIntegration:
    """Test integration scenarios for exploration."""
    
    def create_integration_design_space(self) -> DesignSpaceDefinition:
        """Create design space for integration testing."""
        return DesignSpaceDefinition(
            name="integration_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["op1", "op2", "op3"],
                    exploration=ExplorationRules(
                        required=["op1"],
                        optional=["op2", "op3"]
                    )
                ),
                hw_kernels=ComponentSpace(
                    available=[{"kernel1": ["option_a", "option_b"]}],
                    exploration=ExplorationRules(required=["kernel1"])
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["transform1", "transform2"],
                    exploration=ExplorationRules(required=["transform1"])
                )
            ),
            dse_strategies=DSEStrategies(
                primary_strategy="integration_strategy",
                strategies={
                    "integration_strategy": DSEStrategy(
                        name="integration_strategy",
                        max_evaluations=10,
                        sampling="adaptive"
                    )
                }
            )
        )
    
    @patch('brainsmith.core.dse_v2.space_explorer.DesignSpaceExplorer._execute_exploration_loop')
    def test_explore_design_space_main_flow(self, mock_exploration_loop):
        """Test main explore_design_space flow."""
        design_space = self.create_integration_design_space()
        explorer = DesignSpaceExplorer(design_space)
        
        # Mock the exploration loop to return results
        mock_results = ExplorationResults(
            best_combination=ComponentCombination(),
            best_score=0.8,
            all_combinations=[],
            performance_data=[],
            pareto_frontier=[],
            exploration_summary={},
            strategy_metadata={},
            execution_stats={}
        )
        mock_exploration_loop.return_value = mock_results
        
        def mock_eval_func(model_path, combination):
            return {'throughput': 100.0}
        
        results = explorer.explore_design_space("test_model.onnx", mock_eval_func)
        
        assert results == mock_results
        mock_exploration_loop.assert_called_once()
    
    def test_unknown_strategy_error(self):
        """Test error handling for unknown strategy."""
        design_space = self.create_integration_design_space()
        config = ExplorationConfig(strategy_name="unknown_strategy")
        explorer = DesignSpaceExplorer(design_space, config)
        
        def mock_eval_func(model_path, combination):
            return {'throughput': 100.0}
        
        with pytest.raises(ValueError, match="Strategy 'unknown_strategy' not available"):
            explorer.explore_design_space("test_model.onnx", mock_eval_func)
    
    def test_end_to_end_small_exploration(self):
        """Test end-to-end exploration with small design space."""
        design_space = DesignSpaceDefinition(
            name="e2e_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["LayerNorm"],
                    exploration=ExplorationRules(required=["LayerNorm"])
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["cleanup"],
                    exploration=ExplorationRules(required=["cleanup"])
                )
            ),
            dse_strategies=DSEStrategies(
                primary_strategy="simple_strategy",
                strategies={
                    "simple_strategy": DSEStrategy(
                        name="simple_strategy",
                        max_evaluations=3,
                        sampling="random"
                    )
                }
            )
        )
        
        config = ExplorationConfig(max_evaluations=3)
        explorer = DesignSpaceExplorer(design_space, config)
        
        evaluation_count = 0
        
        def mock_eval_func(model_path, combination):
            nonlocal evaluation_count
            evaluation_count += 1
            return {
                'throughput': 100.0 + evaluation_count * 10,  # Improving performance
                'latency': 10.0,
                'resource_efficiency': 0.8
            }
        
        results = explorer.explore_design_space("test_model.onnx", mock_eval_func)
        
        assert isinstance(results, ExplorationResults)
        assert results.best_score > 0
        assert len(results.all_combinations) > 0
        assert evaluation_count > 0


def test_convenience_function():
    """Test convenience function for exploration."""
    design_space = DesignSpaceDefinition(
        name="convenience_test",
        nodes=NodeDesignSpace(
            canonical_ops=ComponentSpace(
                available=["LayerNorm"],
                exploration=ExplorationRules(required=["LayerNorm"])
            )
        ),
        transforms=TransformDesignSpace(
            model_topology=ComponentSpace(
                available=["cleanup"],
                exploration=ExplorationRules(required=["cleanup"])
            )
        ),
        dse_strategies=DSEStrategies(
            primary_strategy="test_strategy",
            strategies={
                "test_strategy": DSEStrategy(
                    name="test_strategy",
                    max_evaluations=2,
                    sampling="random"
                )
            }
        )
    )
    
    def mock_eval_func(model_path, combination):
        return {'throughput': 100.0}
    
    config = ExplorationConfig(max_evaluations=2)
    
    results = explore_design_space(design_space, "test_model.onnx", mock_eval_func, config)
    
    assert isinstance(results, ExplorationResults)
    assert results.best_score >= 0


if __name__ == "__main__":
    pytest.main([__file__])