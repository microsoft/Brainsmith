"""
DSE Strategy Executor for Blueprint V2

Implements different exploration strategies for design space exploration,
including hierarchical, adaptive, and Pareto-guided strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
import logging
import random
import math
from copy import deepcopy

from .combination_generator import ComponentCombination, CombinationGenerator
from ..blueprint_v2 import DesignSpaceDefinition, DSEStrategy

logger = logging.getLogger(__name__)


class StrategyPhase(Enum):
    """Phases for hierarchical exploration strategy."""
    KERNEL_SELECTION = "kernel_selection"
    TRANSFORM_SELECTION = "transform_selection"
    FINE_TUNING = "fine_tuning"
    COMPLETE = "complete"


@dataclass
class ExplorationContext:
    """Context information for strategy execution."""
    design_space: DesignSpaceDefinition
    combination_generator: CombinationGenerator
    total_budget: int
    remaining_budget: int
    current_phase: StrategyPhase = StrategyPhase.KERNEL_SELECTION
    evaluated_combinations: List[ComponentCombination] = field(default_factory=list)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    best_combinations: List[ComponentCombination] = field(default_factory=list)
    pareto_frontier: List[ComponentCombination] = field(default_factory=list)


@dataclass
class StrategyResult:
    """Result from strategy execution."""
    selected_combinations: List[ComponentCombination]
    strategy_metadata: Dict[str, Any]
    execution_stats: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies."""
    
    def __init__(self, strategy_config: DSEStrategy):
        """Initialize strategy with configuration."""
        self.config = strategy_config
        self.context: Optional[ExplorationContext] = None
        
    @abstractmethod
    def select_combinations(self, context: ExplorationContext) -> StrategyResult:
        """
        Select combinations to evaluate based on strategy.
        
        Args:
            context: Current exploration context
            
        Returns:
            Strategy result with selected combinations
        """
        pass
    
    @abstractmethod
    def adapt_selection(self, context: ExplorationContext, 
                       new_results: List[Dict[str, Any]]) -> StrategyResult:
        """
        Adapt strategy based on new evaluation results.
        
        Args:
            context: Current exploration context
            new_results: New evaluation results
            
        Returns:
            Updated strategy result
        """
        pass
    
    def is_adaptive(self) -> bool:
        """Return True if strategy adapts based on results."""
        return False


class HierarchicalExplorationStrategy(ExplorationStrategy):
    """
    Hierarchical exploration strategy that explores design space in phases:
    1. Kernel selection exploration
    2. Transform selection exploration  
    3. Fine-tuning best combinations
    """
    
    def __init__(self, strategy_config: DSEStrategy):
        """Initialize hierarchical strategy."""
        super().__init__(strategy_config)
        self.phase_budgets = {
            StrategyPhase.KERNEL_SELECTION: 0.4,  # 40% of budget
            StrategyPhase.TRANSFORM_SELECTION: 0.4,  # 40% of budget
            StrategyPhase.FINE_TUNING: 0.2  # 20% of budget
        }
    
    def select_combinations(self, context: ExplorationContext) -> StrategyResult:
        """Select combinations using hierarchical approach."""
        self.context = context
        
        if context.current_phase == StrategyPhase.KERNEL_SELECTION:
            return self._phase1_kernel_exploration(context)
        elif context.current_phase == StrategyPhase.TRANSFORM_SELECTION:
            return self._phase2_transform_exploration(context)
        elif context.current_phase == StrategyPhase.FINE_TUNING:
            return self._phase3_fine_tuning(context)
        else:
            return StrategyResult(
                selected_combinations=[],
                strategy_metadata={"phase": "complete"},
                execution_stats={"message": "All phases complete"}
            )
    
    def _phase1_kernel_exploration(self, context: ExplorationContext) -> StrategyResult:
        """Phase 1: Explore different kernel choices."""
        phase_budget = int(context.total_budget * self.phase_budgets[StrategyPhase.KERNEL_SELECTION])
        
        logger.info(f"Phase 1: Kernel exploration with budget {phase_budget}")
        
        # Generate combinations focusing on kernel diversity
        all_combinations = context.combination_generator.generate_all_combinations()
        
        # Group combinations by kernel choices
        kernel_groups = self._group_by_kernels(all_combinations)
        
        # Sample from each kernel group
        selected_combinations = []
        samples_per_group = max(1, phase_budget // len(kernel_groups))
        
        for kernel_signature, combinations in kernel_groups.items():
            sampled = random.sample(combinations, min(samples_per_group, len(combinations)))
            selected_combinations.extend(sampled)
        
        # Limit to phase budget
        if len(selected_combinations) > phase_budget:
            selected_combinations = random.sample(selected_combinations, phase_budget)
        
        return StrategyResult(
            selected_combinations=selected_combinations,
            strategy_metadata={
                "phase": "kernel_selection",
                "kernel_groups": len(kernel_groups),
                "samples_per_group": samples_per_group
            },
            execution_stats={
                "combinations_selected": len(selected_combinations),
                "budget_used": len(selected_combinations)
            }
        )
    
    def _phase2_transform_exploration(self, context: ExplorationContext) -> StrategyResult:
        """Phase 2: Explore transform choices with best kernels."""
        phase_budget = int(context.total_budget * self.phase_budgets[StrategyPhase.TRANSFORM_SELECTION])
        
        logger.info(f"Phase 2: Transform exploration with budget {phase_budget}")
        
        # Find best performing kernel combinations from phase 1
        best_kernels = self._find_best_kernels(context.performance_history)
        
        # Generate transform variants for best kernels
        selected_combinations = []
        
        for kernel_combo in best_kernels[:5]:  # Top 5 kernel combinations
            transform_variants = self._generate_transform_variants(kernel_combo, context)
            selected_combinations.extend(transform_variants)
        
        # Limit to phase budget
        if len(selected_combinations) > phase_budget:
            selected_combinations = random.sample(selected_combinations, phase_budget)
        
        return StrategyResult(
            selected_combinations=selected_combinations,
            strategy_metadata={
                "phase": "transform_selection",
                "best_kernels_used": len(best_kernels),
                "transform_variants_generated": len(selected_combinations)
            },
            execution_stats={
                "combinations_selected": len(selected_combinations),
                "budget_used": len(selected_combinations)
            }
        )
    
    def _phase3_fine_tuning(self, context: ExplorationContext) -> StrategyResult:
        """Phase 3: Fine-tune best combinations."""
        phase_budget = int(context.total_budget * self.phase_budgets[StrategyPhase.FINE_TUNING])
        
        logger.info(f"Phase 3: Fine-tuning with budget {phase_budget}")
        
        # Find overall best combinations
        best_combinations = self._find_best_combinations(context.performance_history)
        
        # Generate variations of best combinations
        selected_combinations = []
        
        for combo in best_combinations[:phase_budget]:
            # Generate small variations (e.g., enable/disable optional components)
            variations = self._generate_combination_variations(combo, context)
            selected_combinations.extend(variations[:2])  # Up to 2 variations per best combo
        
        # Limit to phase budget
        if len(selected_combinations) > phase_budget:
            selected_combinations = selected_combinations[:phase_budget]
        
        return StrategyResult(
            selected_combinations=selected_combinations,
            strategy_metadata={
                "phase": "fine_tuning",
                "base_combinations": len(best_combinations),
                "variations_generated": len(selected_combinations)
            },
            execution_stats={
                "combinations_selected": len(selected_combinations),
                "budget_used": len(selected_combinations)
            }
        )
    
    def _group_by_kernels(self, combinations: List[ComponentCombination]) -> Dict[str, List[ComponentCombination]]:
        """Group combinations by their kernel signatures."""
        groups = {}
        
        for combo in combinations:
            # Create kernel signature
            kernel_sig = tuple(sorted(f"{k}:{v}" for k, v in combo.hw_kernels.items()))
            kernel_key = "_".join(kernel_sig)
            
            if kernel_key not in groups:
                groups[kernel_key] = []
            groups[kernel_key].append(combo)
        
        return groups
    
    def _find_best_kernels(self, performance_history: List[Dict[str, Any]]) -> List[ComponentCombination]:
        """Find best performing kernel combinations from history."""
        if not performance_history:
            return []
        
        # Sort by performance (assuming higher is better for primary metric)
        sorted_results = sorted(
            performance_history,
            key=lambda x: x.get('primary_metric', 0),
            reverse=True
        )
        
        # Extract combinations from best results
        best_combinations = []
        for result in sorted_results[:10]:  # Top 10
            if 'combination' in result:
                best_combinations.append(result['combination'])
        
        return best_combinations
    
    def _find_best_combinations(self, performance_history: List[Dict[str, Any]]) -> List[ComponentCombination]:
        """Find overall best combinations from history."""
        return self._find_best_kernels(performance_history)  # Same logic for now
    
    def _generate_transform_variants(self, base_combination: ComponentCombination, 
                                   context: ExplorationContext) -> List[ComponentCombination]:
        """Generate transform variants for a base kernel combination."""
        variants = []
        
        # Get available transforms from design space
        model_transforms = context.design_space.transforms.model_topology.get_component_names()
        hw_transforms = context.design_space.transforms.hw_kernel.get_component_names()
        graph_transforms = context.design_space.transforms.hw_graph.get_component_names()
        
        # Generate a few variants with different transform combinations
        for i in range(3):  # Generate 3 variants
            variant = ComponentCombination(
                canonical_ops=base_combination.canonical_ops.copy(),
                hw_kernels=base_combination.hw_kernels.copy(),
                model_topology=random.sample(model_transforms, 
                                           min(len(model_transforms), random.randint(1, 3))),
                hw_kernel_transforms=random.sample(hw_transforms,
                                                 min(len(hw_transforms), random.randint(1, 2))),
                hw_graph_transforms=random.sample(graph_transforms,
                                                min(len(graph_transforms), random.randint(1, 2)))
            )
            variants.append(variant)
        
        return variants
    
    def _generate_combination_variations(self, base_combination: ComponentCombination,
                                       context: ExplorationContext) -> List[ComponentCombination]:
        """Generate small variations of a base combination."""
        variations = []
        
        # Variation 1: Toggle optional canonical ops
        if context.design_space.nodes.canonical_ops.exploration.optional:
            variant1 = deepcopy(base_combination)
            optional_ops = context.design_space.nodes.canonical_ops.exploration.optional
            
            for op in optional_ops:
                if op in variant1.canonical_ops:
                    variant1.canonical_ops.remove(op)
                else:
                    variant1.canonical_ops.append(op)
            
            variations.append(variant1)
        
        # Variation 2: Toggle optional transforms
        if context.design_space.transforms.model_topology.exploration.optional:
            variant2 = deepcopy(base_combination)
            optional_transforms = context.design_space.transforms.model_topology.exploration.optional
            
            for transform in optional_transforms:
                if transform in variant2.model_topology:
                    variant2.model_topology.remove(transform)
                else:
                    variant2.model_topology.append(transform)
            
            variations.append(variant2)
        
        return variations
    
    def adapt_selection(self, context: ExplorationContext, 
                       new_results: List[Dict[str, Any]]) -> StrategyResult:
        """Adapt hierarchical strategy based on results."""
        # Add new results to performance history
        context.performance_history.extend(new_results)
        
        # Determine next phase
        evaluations_done = len(context.performance_history)
        phase1_budget = int(context.total_budget * self.phase_budgets[StrategyPhase.KERNEL_SELECTION])
        phase2_budget = int(context.total_budget * self.phase_budgets[StrategyPhase.TRANSFORM_SELECTION])
        
        if evaluations_done < phase1_budget:
            context.current_phase = StrategyPhase.KERNEL_SELECTION
        elif evaluations_done < phase1_budget + phase2_budget:
            context.current_phase = StrategyPhase.TRANSFORM_SELECTION
        else:
            context.current_phase = StrategyPhase.FINE_TUNING
        
        # Select next combinations
        return self.select_combinations(context)
    
    def is_adaptive(self) -> bool:
        """Hierarchical strategy is adaptive."""
        return True


class AdaptiveExplorationStrategy(ExplorationStrategy):
    """
    Adaptive exploration strategy that analyzes performance trends
    and focuses on promising regions of the design space.
    """
    
    def __init__(self, strategy_config: DSEStrategy):
        """Initialize adaptive strategy."""
        super().__init__(strategy_config)
        self.adaptation_frequency = 10  # Adapt every 10 evaluations
        self.exploration_vs_exploitation = 0.7  # 70% exploration, 30% exploitation
    
    def select_combinations(self, context: ExplorationContext) -> StrategyResult:
        """Select combinations using adaptive approach."""
        if len(context.performance_history) < self.adaptation_frequency:
            # Initial exploration phase
            return self._initial_exploration(context)
        else:
            # Adaptive phase
            return self._adaptive_exploration(context)
    
    def _initial_exploration(self, context: ExplorationContext) -> StrategyResult:
        """Initial broad exploration of design space."""
        combinations = context.combination_generator.generate_sample_combinations(
            self.adaptation_frequency, "diverse"
        )
        
        return StrategyResult(
            selected_combinations=combinations,
            strategy_metadata={"phase": "initial_exploration"},
            execution_stats={
                "combinations_selected": len(combinations),
                "strategy": "diverse_sampling"
            }
        )
    
    def _adaptive_exploration(self, context: ExplorationContext) -> StrategyResult:
        """Adaptive exploration based on performance trends."""
        # Analyze performance trends
        trends = self._analyze_performance_trends(context.performance_history)
        
        # Identify promising regions
        promising_regions = self._identify_promising_regions(context.performance_history)
        
        # Balance exploration vs exploitation
        n_explore = int(self.adaptation_frequency * self.exploration_vs_exploitation)
        n_exploit = self.adaptation_frequency - n_explore
        
        selected_combinations = []
        
        # Exploration: sample from less explored regions
        explore_combinations = self._sample_unexplored_regions(context, n_explore)
        selected_combinations.extend(explore_combinations)
        
        # Exploitation: sample around best performing combinations
        exploit_combinations = self._sample_promising_regions(context, promising_regions, n_exploit)
        selected_combinations.extend(exploit_combinations)
        
        return StrategyResult(
            selected_combinations=selected_combinations,
            strategy_metadata={
                "phase": "adaptive_exploration",
                "trends": trends,
                "promising_regions": len(promising_regions),
                "exploration_ratio": self.exploration_vs_exploitation
            },
            execution_stats={
                "combinations_selected": len(selected_combinations),
                "exploration_samples": n_explore,
                "exploitation_samples": n_exploit
            }
        )
    
    def _analyze_performance_trends(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends in the data."""
        if len(performance_history) < 5:
            return {"trend": "insufficient_data"}
        
        # Simple trend analysis
        recent_performance = [r.get('primary_metric', 0) for r in performance_history[-10:]]
        early_performance = [r.get('primary_metric', 0) for r in performance_history[:10]]
        
        recent_avg = sum(recent_performance) / len(recent_performance)
        early_avg = sum(early_performance) / len(early_performance)
        
        if recent_avg > early_avg * 1.1:
            trend = "improving"
        elif recent_avg < early_avg * 0.9:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_avg": recent_avg,
            "early_avg": early_avg,
            "improvement_ratio": recent_avg / early_avg if early_avg > 0 else 1.0
        }
    
    def _identify_promising_regions(self, performance_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify regions of design space with good performance."""
        if not performance_history:
            return []
        
        # Find top performing combinations
        sorted_results = sorted(
            performance_history,
            key=lambda x: x.get('primary_metric', 0),
            reverse=True
        )
        
        # Take top 20% as promising
        top_count = max(1, len(sorted_results) // 5)
        promising = sorted_results[:top_count]
        
        return promising
    
    def _sample_unexplored_regions(self, context: ExplorationContext, n_samples: int) -> List[ComponentCombination]:
        """Sample from less explored regions of design space."""
        # Generate random combinations (simple exploration for now)
        return context.combination_generator.generate_sample_combinations(n_samples, "random")
    
    def _sample_promising_regions(self, context: ExplorationContext, 
                                promising_regions: List[Dict[str, Any]], 
                                n_samples: int) -> List[ComponentCombination]:
        """Sample around promising regions."""
        if not promising_regions:
            return context.combination_generator.generate_sample_combinations(n_samples, "random")
        
        selected = []
        for i in range(n_samples):
            # Pick a random promising result
            base_result = random.choice(promising_regions)
            base_combination = base_result.get('combination')
            
            if base_combination:
                # Generate variation of promising combination
                variations = self._generate_combination_variations(base_combination, context)
                if variations:
                    selected.append(random.choice(variations))
        
        return selected
    
    def _generate_combination_variations(self, base_combination: ComponentCombination,
                                       context: ExplorationContext) -> List[ComponentCombination]:
        """Generate variations around a base combination."""
        # Simple variation: change one optional component
        variations = []
        
        # Try toggling optional canonical ops
        optional_ops = context.design_space.nodes.canonical_ops.exploration.optional
        for op in optional_ops:
            variant = deepcopy(base_combination)
            if op in variant.canonical_ops:
                variant.canonical_ops.remove(op)
            else:
                variant.canonical_ops.append(op)
            variations.append(variant)
        
        return variations
    
    def adapt_selection(self, context: ExplorationContext, 
                       new_results: List[Dict[str, Any]]) -> StrategyResult:
        """Adapt based on new results."""
        context.performance_history.extend(new_results)
        return self.select_combinations(context)
    
    def is_adaptive(self) -> bool:
        """Adaptive strategy is adaptive."""
        return True


class ParetoGuidedStrategy(ExplorationStrategy):
    """
    Pareto-guided exploration strategy for multi-objective optimization.
    Maintains Pareto frontier and guides sampling toward it.
    """
    
    def __init__(self, strategy_config: DSEStrategy):
        """Initialize Pareto-guided strategy."""
        super().__init__(strategy_config)
        self.objectives = strategy_config.objectives or ["throughput", "resource_efficiency"]
        self.pareto_frontier: List[ComponentCombination] = []
    
    def select_combinations(self, context: ExplorationContext) -> StrategyResult:
        """Select combinations using Pareto guidance."""
        if len(context.performance_history) < 20:
            # Initial diverse sampling
            combinations = context.combination_generator.generate_sample_combinations(10, "diverse")
        else:
            # Pareto-guided sampling
            self._update_pareto_frontier(context.performance_history)
            combinations = self._sample_toward_pareto_frontier(context, 10)
        
        return StrategyResult(
            selected_combinations=combinations,
            strategy_metadata={
                "pareto_frontier_size": len(self.pareto_frontier),
                "objectives": self.objectives
            },
            execution_stats={
                "combinations_selected": len(combinations)
            }
        )
    
    def _update_pareto_frontier(self, performance_history: List[Dict[str, Any]]):
        """Update Pareto frontier based on performance history."""
        # Simple Pareto frontier update (placeholder implementation)
        if not performance_history:
            return
        
        # For now, just keep top performers for each objective
        self.pareto_frontier = []
        
        for objective in self.objectives:
            best_for_objective = max(
                performance_history,
                key=lambda x: x.get(objective, 0),
                default=None
            )
            if best_for_objective and 'combination' in best_for_objective:
                self.pareto_frontier.append(best_for_objective['combination'])
    
    def _sample_toward_pareto_frontier(self, context: ExplorationContext, 
                                     n_samples: int) -> List[ComponentCombination]:
        """Sample combinations guided toward Pareto frontier."""
        if not self.pareto_frontier:
            return context.combination_generator.generate_sample_combinations(n_samples, "random")
        
        # Generate variations of Pareto optimal solutions
        selected = []
        for i in range(n_samples):
            base_combination = random.choice(self.pareto_frontier)
            variations = self._generate_combination_variations(base_combination, context)
            if variations:
                selected.append(random.choice(variations))
            else:
                selected.append(base_combination)
        
        return selected
    
    def _generate_combination_variations(self, base_combination: ComponentCombination,
                                       context: ExplorationContext) -> List[ComponentCombination]:
        """Generate variations for Pareto guidance."""
        # Similar to adaptive strategy
        variations = []
        
        optional_ops = context.design_space.nodes.canonical_ops.exploration.optional
        for op in optional_ops[:2]:  # Limit variations
            variant = deepcopy(base_combination)
            if op in variant.canonical_ops:
                variant.canonical_ops.remove(op)
            else:
                variant.canonical_ops.append(op)
            variations.append(variant)
        
        return variations
    
    def adapt_selection(self, context: ExplorationContext, 
                       new_results: List[Dict[str, Any]]) -> StrategyResult:
        """Adapt Pareto frontier based on new results."""
        context.performance_history.extend(new_results)
        self._update_pareto_frontier(context.performance_history)
        return self.select_combinations(context)
    
    def is_adaptive(self) -> bool:
        """Pareto-guided strategy is adaptive."""
        return True


class StrategyExecutor:
    """Executes exploration strategies and manages strategy lifecycle."""
    
    def __init__(self, design_space: DesignSpaceDefinition):
        """Initialize strategy executor."""
        self.design_space = design_space
        self.combination_generator = CombinationGenerator(design_space)
        self.strategies: Dict[str, ExplorationStrategy] = {}
        
        # Register built-in strategies
        self._register_builtin_strategies()
    
    def _register_builtin_strategies(self):
        """Register built-in exploration strategies."""
        if self.design_space.dse_strategies:
            for name, strategy_config in self.design_space.dse_strategies.strategies.items():
                strategy = self._create_strategy(strategy_config)
                self.strategies[name] = strategy
    
    def _create_strategy(self, strategy_config: DSEStrategy) -> ExplorationStrategy:
        """Create strategy instance based on configuration."""
        sampling = strategy_config.sampling.lower()
        
        if sampling == "adaptive":
            return AdaptiveExplorationStrategy(strategy_config)
        elif "hierarchical" in strategy_config.name.lower():
            return HierarchicalExplorationStrategy(strategy_config)
        elif "pareto" in strategy_config.name.lower():
            return ParetoGuidedStrategy(strategy_config)
        else:
            # Default to adaptive
            return AdaptiveExplorationStrategy(strategy_config)
    
    def execute_strategy(self, strategy_name: str, max_evaluations: int) -> StrategyResult:
        """Execute named strategy with given budget."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        
        # Create exploration context
        context = ExplorationContext(
            design_space=self.design_space,
            combination_generator=self.combination_generator,
            total_budget=max_evaluations,
            remaining_budget=max_evaluations
        )
        
        # Execute strategy
        return strategy.select_combinations(context)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self.strategies.keys())