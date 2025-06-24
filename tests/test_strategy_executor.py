"""
Comprehensive tests for DSE Strategy Executor and strategies.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from brainsmith.core.dse.strategy_executor import (
    StrategyExecutor, ExplorationStrategy, HierarchicalExplorationStrategy,
    AdaptiveExplorationStrategy, ParetoGuidedStrategy, ExplorationContext,
    StrategyResult, StrategyPhase
)
from brainsmith.core.dse.combination_generator import ComponentCombination, CombinationGenerator
from brainsmith.core.blueprint import (
    DesignSpaceDefinition, NodeDesignSpace, TransformDesignSpace,
    ComponentSpace, ExplorationRules, DSEStrategy, DSEStrategies
)


class TestExplorationContext:
    """Test ExplorationContext dataclass."""
    
    def test_context_creation(self):
        """Test exploration context creation."""
        design_space = DesignSpaceDefinition(name="test")
        generator = Mock()
        
        context = ExplorationContext(
            design_space=design_space,
            combination_generator=generator,
            total_budget=100,
            remaining_budget=100
        )
        
        assert context.design_space == design_space
        assert context.total_budget == 100
        assert context.remaining_budget == 100
        assert context.current_phase == StrategyPhase.KERNEL_SELECTION
        assert context.evaluated_combinations == []
        assert context.performance_history == []


class TestStrategyResult:
    """Test StrategyResult dataclass."""
    
    def test_result_creation(self):
        """Test strategy result creation."""
        combinations = [ComponentCombination()]
        
        result = StrategyResult(
            selected_combinations=combinations,
            strategy_metadata={"test": "value"},
            execution_stats={"selected": 1}
        )
        
        assert result.selected_combinations == combinations
        assert result.strategy_metadata["test"] == "value"
        assert result.execution_stats["selected"] == 1
        assert result.recommendations == []


class TestHierarchicalExplorationStrategy:
    """Test HierarchicalExplorationStrategy."""
    
    def create_test_design_space(self) -> DesignSpaceDefinition:
        """Create test design space."""
        return DesignSpaceDefinition(
            name="hierarchical_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["LayerNorm", "Softmax"],
                    exploration=ExplorationRules(
                        required=["LayerNorm"],
                        optional=["Softmax"]
                    )
                ),
                hw_kernels=ComponentSpace(
                    available=[{"MatMul": ["hls", "rtl"]}],
                    exploration=ExplorationRules(required=["MatMul"])
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["cleanup", "streamlining"],
                    exploration=ExplorationRules(required=["cleanup"])
                )
            )
        )
    
    def create_test_context(self) -> ExplorationContext:
        """Create test exploration context."""
        design_space = self.create_test_design_space()
        generator = CombinationGenerator(design_space)
        
        return ExplorationContext(
            design_space=design_space,
            combination_generator=generator,
            total_budget=100,
            remaining_budget=100
        )
    
    def test_strategy_initialization(self):
        """Test hierarchical strategy initialization."""
        config = DSEStrategy(name="hierarchical_test", max_evaluations=100)
        strategy = HierarchicalExplorationStrategy(config)
        
        assert strategy.config == config
        assert StrategyPhase.KERNEL_SELECTION in strategy.phase_budgets
        assert StrategyPhase.TRANSFORM_SELECTION in strategy.phase_budgets
        assert StrategyPhase.FINE_TUNING in strategy.phase_budgets
        
        # Budgets should sum to 1.0
        total_budget = sum(strategy.phase_budgets.values())
        assert abs(total_budget - 1.0) < 0.01
    
    def test_phase1_kernel_exploration(self):
        """Test phase 1 kernel exploration."""
        config = DSEStrategy(name="hierarchical_test", max_evaluations=100)
        strategy = HierarchicalExplorationStrategy(config)
        context = self.create_test_context()
        
        # Should start in kernel selection phase
        assert context.current_phase == StrategyPhase.KERNEL_SELECTION
        
        result = strategy.select_combinations(context)
        
        assert isinstance(result, StrategyResult)
        assert len(result.selected_combinations) > 0
        assert result.strategy_metadata["phase"] == "kernel_selection"
        assert "kernel_groups" in result.strategy_metadata
        
        # All combinations should be valid
        for combo in result.selected_combinations:
            assert isinstance(combo, ComponentCombination)
            assert combo.is_valid
    
    def test_phase2_transform_exploration(self):
        """Test phase 2 transform exploration."""
        config = DSEStrategy(name="hierarchical_test", max_evaluations=100)
        strategy = HierarchicalExplorationStrategy(config)
        context = self.create_test_context()
        
        # Set up context for phase 2
        context.current_phase = StrategyPhase.TRANSFORM_SELECTION
        
        # Add mock performance history
        context.performance_history = [
            {
                "combination": ComponentCombination(
                    canonical_ops=["LayerNorm"],
                    hw_kernels={"MatMul": "hls"}
                ),
                "primary_metric": 0.8
            },
            {
                "combination": ComponentCombination(
                    canonical_ops=["LayerNorm", "Softmax"],
                    hw_kernels={"MatMul": "rtl"}
                ),
                "primary_metric": 0.6
            }
        ]
        
        result = strategy.select_combinations(context)
        
        assert result.strategy_metadata["phase"] == "transform_selection"
        assert len(result.selected_combinations) > 0
    
    def test_phase3_fine_tuning(self):
        """Test phase 3 fine tuning."""
        config = DSEStrategy(name="hierarchical_test", max_evaluations=100)
        strategy = HierarchicalExplorationStrategy(config)
        context = self.create_test_context()
        
        # Set up context for phase 3
        context.current_phase = StrategyPhase.FINE_TUNING
        
        # Add mock performance history
        context.performance_history = [
            {
                "combination": ComponentCombination(
                    canonical_ops=["LayerNorm"],
                    hw_kernels={"MatMul": "hls"},
                    model_topology=["cleanup"]
                ),
                "primary_metric": 0.9
            }
        ]
        
        result = strategy.select_combinations(context)
        
        assert result.strategy_metadata["phase"] == "fine_tuning"
        assert len(result.selected_combinations) >= 0  # May be 0 if no variations possible
    
    def test_adaptation_logic(self):
        """Test strategy adaptation between phases."""
        config = DSEStrategy(name="hierarchical_test", max_evaluations=100)
        strategy = HierarchicalExplorationStrategy(config)
        context = self.create_test_context()
        
        # Start in phase 1
        assert context.current_phase == StrategyPhase.KERNEL_SELECTION
        
        # Simulate completing phase 1 (40% of budget)
        phase1_budget = int(100 * 0.4)
        new_results = [{"primary_metric": 0.5} for _ in range(phase1_budget)]
        
        result = strategy.adapt_selection(context, new_results)
        
        # Should move to phase 2
        assert context.current_phase == StrategyPhase.TRANSFORM_SELECTION
        assert len(context.performance_history) == phase1_budget
    
    def test_is_adaptive(self):
        """Test that hierarchical strategy is adaptive."""
        config = DSEStrategy(name="hierarchical_test")
        strategy = HierarchicalExplorationStrategy(config)
        
        assert strategy.is_adaptive() == True


class TestAdaptiveExplorationStrategy:
    """Test AdaptiveExplorationStrategy."""
    
    def create_test_context(self) -> ExplorationContext:
        """Create test exploration context."""
        design_space = DesignSpaceDefinition(
            name="adaptive_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["op1", "op2", "op3"],
                    exploration=ExplorationRules(
                        required=["op1"],
                        optional=["op2", "op3"]
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["transform1", "transform2"],
                    exploration=ExplorationRules(required=["transform1"])
                )
            )
        )
        generator = CombinationGenerator(design_space)
        
        return ExplorationContext(
            design_space=design_space,
            combination_generator=generator,
            total_budget=50,
            remaining_budget=50
        )
    
    def test_initial_exploration(self):
        """Test initial exploration phase."""
        config = DSEStrategy(name="adaptive_test", max_evaluations=50)
        strategy = AdaptiveExplorationStrategy(config)
        context = self.create_test_context()
        
        result = strategy.select_combinations(context)
        
        assert isinstance(result, StrategyResult)
        assert result.strategy_metadata["phase"] == "initial_exploration"
        assert len(result.selected_combinations) == strategy.adaptation_frequency
    
    def test_adaptive_exploration(self):
        """Test adaptive exploration phase."""
        config = DSEStrategy(name="adaptive_test", max_evaluations=50)
        strategy = AdaptiveExplorationStrategy(config)
        context = self.create_test_context()
        
        # Add enough history to trigger adaptive phase
        context.performance_history = [
            {"primary_metric": 0.5 + i * 0.1} for i in range(15)
        ]
        
        result = strategy.select_combinations(context)
        
        assert result.strategy_metadata["phase"] == "adaptive_exploration"
        assert "trends" in result.strategy_metadata
        assert "exploration_ratio" in result.strategy_metadata
        assert len(result.selected_combinations) == strategy.adaptation_frequency
    
    def test_performance_trend_analysis(self):
        """Test performance trend analysis."""
        config = DSEStrategy(name="adaptive_test")
        strategy = AdaptiveExplorationStrategy(config)
        
        # Test improving trend
        improving_history = [{"primary_metric": 0.1 * i} for i in range(20)]
        trends = strategy._analyze_performance_trends(improving_history)
        assert trends["trend"] == "improving"
        
        # Test declining trend
        declining_history = [{"primary_metric": 1.0 - 0.1 * i} for i in range(20)]
        trends = strategy._analyze_performance_trends(declining_history)
        assert trends["trend"] == "declining"
        
        # Test insufficient data
        short_history = [{"primary_metric": 0.5}]
        trends = strategy._analyze_performance_trends(short_history)
        assert trends["trend"] == "insufficient_data"
    
    def test_promising_region_identification(self):
        """Test identification of promising regions."""
        config = DSEStrategy(name="adaptive_test")
        strategy = AdaptiveExplorationStrategy(config)
        
        history = [
            {"primary_metric": 0.1, "combination": ComponentCombination()},
            {"primary_metric": 0.9, "combination": ComponentCombination()},
            {"primary_metric": 0.5, "combination": ComponentCombination()},
            {"primary_metric": 0.8, "combination": ComponentCombination()},
            {"primary_metric": 0.2, "combination": ComponentCombination()}
        ]
        
        promising = strategy._identify_promising_regions(history)
        
        # Should return top 20% (1 out of 5)
        assert len(promising) == 1
        assert promising[0]["primary_metric"] == 0.9  # Best performer
    
    def test_is_adaptive(self):
        """Test that adaptive strategy is adaptive."""
        config = DSEStrategy(name="adaptive_test")
        strategy = AdaptiveExplorationStrategy(config)
        
        assert strategy.is_adaptive() == True


class TestParetoGuidedStrategy:
    """Test ParetoGuidedStrategy."""
    
    def create_test_context(self) -> ExplorationContext:
        """Create test exploration context."""
        design_space = DesignSpaceDefinition(
            name="pareto_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["op1", "op2"],
                    exploration=ExplorationRules(optional=["op1", "op2"])
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["transform1"],
                    exploration=ExplorationRules(required=["transform1"])
                )
            )
        )
        generator = CombinationGenerator(design_space)
        
        return ExplorationContext(
            design_space=design_space,
            combination_generator=generator,
            total_budget=50,
            remaining_budget=50
        )
    
    def test_strategy_initialization(self):
        """Test Pareto strategy initialization."""
        config = DSEStrategy(
            name="pareto_test",
            objectives=["throughput", "power_efficiency"]
        )
        strategy = ParetoGuidedStrategy(config)
        
        assert strategy.objectives == ["throughput", "power_efficiency"]
        assert strategy.pareto_frontier == []
    
    def test_initial_sampling(self):
        """Test initial diverse sampling."""
        config = DSEStrategy(name="pareto_test")
        strategy = ParetoGuidedStrategy(config)
        context = self.create_test_context()
        
        result = strategy.select_combinations(context)
        
        assert len(result.selected_combinations) == 10  # Initial sample size
        assert result.strategy_metadata["pareto_frontier_size"] == 0
    
    def test_pareto_guided_sampling(self):
        """Test Pareto-guided sampling."""
        config = DSEStrategy(name="pareto_test")
        strategy = ParetoGuidedStrategy(config)
        context = self.create_test_context()
        
        # Add enough history to trigger Pareto guidance
        context.performance_history = [
            {
                "throughput": 0.8,
                "power_efficiency": 0.6,
                "combination": ComponentCombination(canonical_ops=["op1"])
            },
            {
                "throughput": 0.6,
                "power_efficiency": 0.8,
                "combination": ComponentCombination(canonical_ops=["op2"])
            }
        ] * 10  # 20 total results
        
        result = strategy.select_combinations(context)
        
        assert len(result.selected_combinations) == 10
        assert result.strategy_metadata["pareto_frontier_size"] > 0
    
    def test_pareto_frontier_update(self):
        """Test Pareto frontier update logic."""
        config = DSEStrategy(
            name="pareto_test",
            objectives=["throughput", "resource_efficiency"]
        )
        strategy = ParetoGuidedStrategy(config)
        
        history = [
            {
                "throughput": 0.9,
                "resource_efficiency": 0.5,
                "combination": ComponentCombination(canonical_ops=["op1"])
            },
            {
                "throughput": 0.5,
                "resource_efficiency": 0.9,
                "combination": ComponentCombination(canonical_ops=["op2"])
            },
            {
                "throughput": 0.7,
                "resource_efficiency": 0.7,
                "combination": ComponentCombination(canonical_ops=["op1", "op2"])
            }
        ]
        
        strategy._update_pareto_frontier(history)
        
        # Should have solutions for each objective
        assert len(strategy.pareto_frontier) == 2  # One best for each objective
    
    def test_is_adaptive(self):
        """Test that Pareto strategy is adaptive."""
        config = DSEStrategy(name="pareto_test")
        strategy = ParetoGuidedStrategy(config)
        
        assert strategy.is_adaptive() == True


class TestStrategyExecutor:
    """Test StrategyExecutor class."""
    
    def create_test_design_space_with_strategies(self) -> DesignSpaceDefinition:
        """Create design space with DSE strategies."""
        return DesignSpaceDefinition(
            name="executor_test",
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
                primary_strategy="test_adaptive",
                strategies={
                    "test_adaptive": DSEStrategy(
                        name="test_adaptive",
                        sampling="adaptive",
                        max_evaluations=50
                    ),
                    "test_hierarchical": DSEStrategy(
                        name="test_hierarchical_strategy",
                        sampling="hierarchical",
                        max_evaluations=100
                    ),
                    "test_pareto": DSEStrategy(
                        name="test_pareto_strategy",
                        sampling="pareto_guided",
                        max_evaluations=75
                    )
                }
            )
        )
    
    def test_executor_initialization(self):
        """Test strategy executor initialization."""
        design_space = self.create_test_design_space_with_strategies()
        executor = StrategyExecutor(design_space)
        
        assert executor.design_space == design_space
        assert isinstance(executor.combination_generator, CombinationGenerator)
        assert len(executor.strategies) == 3  # Should register all strategies
    
    def test_strategy_registration(self):
        """Test automatic strategy registration."""
        design_space = self.create_test_design_space_with_strategies()
        executor = StrategyExecutor(design_space)
        
        available_strategies = executor.get_available_strategies()
        
        assert "test_adaptive" in available_strategies
        assert "test_hierarchical" in available_strategies
        assert "test_pareto" in available_strategies
    
    def test_strategy_creation(self):
        """Test strategy creation based on configuration."""
        design_space = self.create_test_design_space_with_strategies()
        executor = StrategyExecutor(design_space)
        
        # Check strategy types
        assert isinstance(executor.strategies["test_adaptive"], AdaptiveExplorationStrategy)
        assert isinstance(executor.strategies["test_hierarchical"], HierarchicalExplorationStrategy)
        assert isinstance(executor.strategies["test_pareto"], ParetoGuidedStrategy)
    
    def test_strategy_execution(self):
        """Test strategy execution."""
        design_space = self.create_test_design_space_with_strategies()
        executor = StrategyExecutor(design_space)
        
        result = executor.execute_strategy("test_adaptive", max_evaluations=20)
        
        assert isinstance(result, StrategyResult)
        assert len(result.selected_combinations) > 0
        assert len(result.selected_combinations) <= 20
    
    def test_unknown_strategy_error(self):
        """Test error for unknown strategy."""
        design_space = self.create_test_design_space_with_strategies()
        executor = StrategyExecutor(design_space)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            executor.execute_strategy("non_existent_strategy", max_evaluations=10)
    
    def test_empty_strategies_handling(self):
        """Test handling of design space without strategies."""
        design_space = DesignSpaceDefinition(
            name="no_strategies_test",
            nodes=NodeDesignSpace(),
            transforms=TransformDesignSpace()
        )
        
        executor = StrategyExecutor(design_space)
        
        assert len(executor.strategies) == 0
        assert executor.get_available_strategies() == []


class TestStrategyIntegration:
    """Test integration between strategies and combination generator."""
    
    def test_strategy_with_combination_generator(self):
        """Test strategy integration with combination generator."""
        design_space = DesignSpaceDefinition(
            name="integration_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["op1", "op2", "op3"],
                    exploration=ExplorationRules(
                        required=["op1"],
                        optional=["op2", "op3"]
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["t1", "t2"],
                    exploration=ExplorationRules(required=["t1"])
                )
            )
        )
        
        generator = CombinationGenerator(design_space)
        config = DSEStrategy(name="integration_test", max_evaluations=10)
        strategy = AdaptiveExplorationStrategy(config)
        
        context = ExplorationContext(
            design_space=design_space,
            combination_generator=generator,
            total_budget=10,
            remaining_budget=10
        )
        
        result = strategy.select_combinations(context)
        
        # Verify combinations are valid and from the design space
        assert len(result.selected_combinations) > 0
        
        for combo in result.selected_combinations:
            assert isinstance(combo, ComponentCombination)
            assert "op1" in combo.canonical_ops  # Required component
            assert "t1" in combo.model_topology  # Required transform
    
    def test_strategy_adaptation_cycle(self):
        """Test complete strategy adaptation cycle."""
        design_space = DesignSpaceDefinition(
            name="adaptation_test",
            nodes=NodeDesignSpace(
                canonical_ops=ComponentSpace(
                    available=["base_op", "opt_op1", "opt_op2"],
                    exploration=ExplorationRules(
                        required=["base_op"],
                        optional=["opt_op1", "opt_op2"]
                    )
                )
            ),
            transforms=TransformDesignSpace(
                model_topology=ComponentSpace(
                    available=["base_transform"],
                    exploration=ExplorationRules(required=["base_transform"])
                )
            )
        )
        
        config = DSEStrategy(name="adaptation_test", max_evaluations=20)
        strategy = AdaptiveExplorationStrategy(config)
        generator = CombinationGenerator(design_space)
        
        context = ExplorationContext(
            design_space=design_space,
            combination_generator=generator,
            total_budget=20,
            remaining_budget=20
        )
        
        # Initial selection
        result1 = strategy.select_combinations(context)
        assert len(result1.selected_combinations) > 0
        
        # Simulate evaluation results
        mock_results = [
            {
                "combination": combo,
                "primary_metric": 0.5 + i * 0.1
            }
            for i, combo in enumerate(result1.selected_combinations)
        ]
        
        # Adaptation
        result2 = strategy.adapt_selection(context, mock_results)
        assert len(result2.selected_combinations) > 0
        
        # Context should be updated
        assert len(context.performance_history) == len(mock_results)


if __name__ == "__main__":
    pytest.main([__file__])