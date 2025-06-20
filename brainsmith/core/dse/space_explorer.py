"""
Design Space Explorer for Blueprint V2

Main orchestration class that coordinates combination generation, strategy execution,
and result analysis for comprehensive design space exploration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import logging
import time
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, Future
import hashlib

from .combination_generator import ComponentCombination, CombinationGenerator
from .strategy_executor import StrategyExecutor, ExplorationStrategy, ExplorationContext, StrategyResult
from .results_analyzer import DSEResults, ResultsAnalyzer, ParetoFrontierAnalyzer
from ..blueprint import DesignSpaceDefinition

logger = logging.getLogger(__name__)


@dataclass
class ExplorationConfig:
    """Configuration for design space exploration."""
    max_evaluations: int = 100
    strategy_name: Optional[str] = None  # Use primary strategy if None
    enable_caching: bool = True
    cache_directory: Optional[str] = None
    parallel_evaluations: int = 1
    early_termination_patience: int = 20
    early_termination_threshold: float = 0.01
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    checkpoint_frequency: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if self.parallel_evaluations <= 0:
            raise ValueError("parallel_evaluations must be positive")
        if self.early_termination_patience <= 0:
            raise ValueError("early_termination_patience must be positive")


@dataclass
class ExplorationProgress:
    """Tracks progress during exploration."""
    total_budget: int
    evaluations_completed: int = 0
    evaluations_cached: int = 0
    current_strategy: str = ""
    current_phase: str = ""
    best_score: float = 0.0
    pareto_frontier_size: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_completion_time: Optional[float] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        return (self.evaluations_completed / self.total_budget) * 100.0 if self.total_budget > 0 else 0.0
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        return time.time() - self.start_time
    
    def estimate_completion_time(self) -> Optional[float]:
        """Estimate completion time based on current progress."""
        if self.evaluations_completed > 0:
            time_per_evaluation = self.elapsed_time / self.evaluations_completed
            remaining_evaluations = self.total_budget - self.evaluations_completed
            return time_per_evaluation * remaining_evaluations
        return None


@dataclass
class ExplorationResults:
    """Results from design space exploration."""
    best_combination: Optional[ComponentCombination]
    best_score: float
    all_combinations: List[ComponentCombination]
    performance_data: List[Dict[str, Any]]
    pareto_frontier: List[ComponentCombination]
    exploration_summary: Dict[str, Any]
    strategy_metadata: Dict[str, Any]
    execution_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            'best_combination': self.best_combination.to_dict() if self.best_combination else None,
            'best_score': self.best_score,
            'total_combinations_evaluated': len(self.all_combinations),
            'pareto_frontier_size': len(self.pareto_frontier),
            'exploration_summary': self.exploration_summary,
            'strategy_metadata': self.strategy_metadata,
            'execution_stats': self.execution_stats
        }
    
    def save_to_file(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DesignSpaceExplorer:
    """
    Main class for exploring design spaces defined by Blueprint V2.
    
    Coordinates combination generation, strategy execution, evaluation,
    and result analysis to provide comprehensive design space exploration.
    """
    
    def __init__(self, design_space: DesignSpaceDefinition, config: Optional[ExplorationConfig] = None):
        """
        Initialize design space explorer.
        
        Args:
            design_space: Blueprint V2 design space definition
            config: Exploration configuration (uses defaults if None)
        """
        self.design_space = design_space
        self.config = config or ExplorationConfig()
        
        # Initialize components
        self.combination_generator = CombinationGenerator(design_space)
        self.strategy_executor = StrategyExecutor(design_space)
        self.results_analyzer = ResultsAnalyzer()
        self.pareto_analyzer = ParetoFrontierAnalyzer()
        
        # Exploration state
        self.progress = ExplorationProgress(total_budget=self.config.max_evaluations)
        self.evaluation_cache: Dict[str, Dict[str, Any]] = {}
        self.evaluated_combinations: List[ComponentCombination] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Setup caching
        if self.config.enable_caching:
            self._setup_cache()
        
        logger.info(f"Initialized design space explorer for blueprint: {design_space.name}")
        logger.info(f"Exploration config: max_evaluations={self.config.max_evaluations}, "
                   f"strategy={self.config.strategy_name}, parallel={self.config.parallel_evaluations}")
    
    def explore_design_space(self, 
                           model_path: str,
                           evaluation_function: Optional[Callable[[str, ComponentCombination], Dict[str, Any]]] = None) -> ExplorationResults:
        """
        Explore the design space using the configured strategy.
        
        Args:
            model_path: Path to the model to be accelerated
            evaluation_function: Function that evaluates a combination and returns metrics
            
        Returns:
            Complete exploration results
        """
        logger.info(f"Starting design space exploration for model: {model_path}")
        
        # Initialize exploration
        self.progress = ExplorationProgress(total_budget=self.config.max_evaluations)
        self.evaluated_combinations = []
        self.performance_history = []
        
        # Get strategy
        strategy_name = self.config.strategy_name or self.design_space.dse_strategies.primary_strategy
        if strategy_name not in self.strategy_executor.get_available_strategies():
            raise ValueError(f"Strategy '{strategy_name}' not available. "
                           f"Available: {self.strategy_executor.get_available_strategies()}")
        
        self.progress.current_strategy = strategy_name
        
        try:
            # Main exploration loop
            results = self._execute_exploration_loop(model_path, evaluation_function, strategy_name)
            
            logger.info(f"Exploration completed successfully. "
                       f"Evaluated {len(results.all_combinations)} combinations in {self.progress.elapsed_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            raise
    
    def _execute_exploration_loop(self, 
                                model_path: str,
                                evaluation_function: Callable,
                                strategy_name: str) -> ExplorationResults:
        """Execute the main exploration loop."""
        
        # Create exploration context
        context = ExplorationContext(
            design_space=self.design_space,
            combination_generator=self.combination_generator,
            total_budget=self.config.max_evaluations,
            remaining_budget=self.config.max_evaluations
        )
        
        # Get initial combinations from strategy
        strategy_result = self.strategy_executor.execute_strategy(strategy_name, self.config.max_evaluations)
        combinations_to_evaluate = strategy_result.selected_combinations
        
        # Main evaluation loop
        while (self.progress.evaluations_completed < self.config.max_evaluations and 
               combinations_to_evaluate):
            
            # Evaluate batch of combinations
            batch_size = min(
                len(combinations_to_evaluate),
                self.config.max_evaluations - self.progress.evaluations_completed,
                self.config.parallel_evaluations * 5  # Process in reasonable batches
            )
            
            current_batch = combinations_to_evaluate[:batch_size]
            combinations_to_evaluate = combinations_to_evaluate[batch_size:]
            
            # Evaluate batch
            batch_results = self._evaluate_combination_batch(model_path, current_batch, evaluation_function)
            
            # Update exploration state
            self._update_exploration_state(batch_results, context)
            
            # Check early termination
            if self._should_terminate_early():
                logger.info("Early termination triggered")
                break
            
            # Checkpoint progress
            if self.progress.evaluations_completed % self.config.checkpoint_frequency == 0:
                self._checkpoint_progress()
            
            # Adaptive strategy: get more combinations if needed
            if (combinations_to_evaluate == [] and 
                self.progress.evaluations_completed < self.config.max_evaluations):
                
                strategy = self.strategy_executor.strategies[strategy_name]
                if strategy.is_adaptive():
                    adapt_result = strategy.adapt_selection(context, batch_results)
                    combinations_to_evaluate.extend(adapt_result.selected_combinations)
        
        # Generate final results
        return self._generate_final_results(strategy_result.strategy_metadata)
    
    def _evaluate_combination_batch(self, 
                                  model_path: str,
                                  combinations: List[ComponentCombination],
                                  evaluation_function: Callable) -> List[Dict[str, Any]]:
        """Evaluate a batch of combinations."""
        
        logger.debug(f"Evaluating batch of {len(combinations)} combinations")
        
        batch_results = []
        
        if self.config.parallel_evaluations > 1:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=self.config.parallel_evaluations) as executor:
                # Submit evaluations
                future_to_combo = {
                    executor.submit(self._evaluate_single_combination, model_path, combo, evaluation_function): combo
                    for combo in combinations
                }
                
                # Collect results
                for future in future_to_combo:
                    combo = future_to_combo[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Evaluation failed for combination {combo.combination_id}: {e}")
                        # Add failed result
                        batch_results.append({
                            'combination': combo,
                            'success': False,
                            'error': str(e),
                            'primary_metric': 0.0
                        })
        else:
            # Sequential evaluation
            for combo in combinations:
                result = self._evaluate_single_combination(model_path, combo, evaluation_function)
                batch_results.append(result)
        
        return batch_results
    
    def _evaluate_single_combination(self, 
                                   model_path: str,
                                   combination: ComponentCombination,
                                   evaluation_function: Callable) -> Dict[str, Any]:
        """Evaluate a single combination."""
        
        # Check cache first
        if self.config.enable_caching:
            cached_result = self._get_cached_result(combination)
            if cached_result:
                self.progress.evaluations_cached += 1
                logger.debug(f"Using cached result for {combination.combination_id}")
                return cached_result
        
        # Evaluate combination
        start_time = time.time()
        try:
            # Use FINN bridge if no evaluation function provided
            if evaluation_function is None:
                from ..finn import FINNEvaluationBridge
                # Get blueprint config from design space - FIXED: Use raw_blueprint_config
                blueprint_config = getattr(self.design_space, 'raw_blueprint_config', {})
                finn_bridge = FINNEvaluationBridge(blueprint_config)
                metrics = finn_bridge.evaluate_combination(model_path, combination)
            else:
                metrics = evaluation_function(model_path, combination)
            evaluation_time = time.time() - start_time
            
            result = {
                'combination': combination,
                'success': True,
                'metrics': metrics,
                'evaluation_time': evaluation_time,
                'primary_metric': metrics.get('throughput', 0.0),  # Default primary metric
                **metrics  # Flatten metrics for easy access
            }
            
            # Cache result
            if self.config.enable_caching:
                self._cache_result(combination, result)
            
            self.progress.evaluations_completed += 1
            
            return result
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            logger.error(f"Evaluation failed for {combination.combination_id}: {e}")
            
            result = {
                'combination': combination,
                'success': False,
                'error': str(e),
                'evaluation_time': evaluation_time,
                'primary_metric': 0.0
            }
            
            self.progress.evaluations_completed += 1
            return result
    
    def _update_exploration_state(self, batch_results: List[Dict[str, Any]], context: ExplorationContext):
        """Update exploration state with new results."""
        
        # Add to history
        self.performance_history.extend(batch_results)
        context.performance_history.extend(batch_results)
        
        # Update combinations list
        for result in batch_results:
            if result['success']:
                self.evaluated_combinations.append(result['combination'])
        
        # Update best score
        successful_results = [r for r in batch_results if r['success']]
        if successful_results:
            current_best = max(successful_results, key=lambda x: x['primary_metric'])
            if current_best['primary_metric'] > self.progress.best_score:
                self.progress.best_score = current_best['primary_metric']
        
        # Update Pareto frontier
        pareto_combinations = self.pareto_analyzer.update_frontier(
            [r for r in self.performance_history if r['success']],
            objectives=['throughput', 'resource_efficiency']  # Default objectives
        )
        self.progress.pareto_frontier_size = len(pareto_combinations)
        
        # Update estimated completion time
        self.progress.estimated_completion_time = self.progress.estimate_completion_time()
        
        # Call progress callback
        if self.config.progress_callback:
            self.config.progress_callback({
                'progress_percentage': self.progress.progress_percentage,
                'evaluations_completed': self.progress.evaluations_completed,
                'best_score': self.progress.best_score,
                'elapsed_time': self.progress.elapsed_time,
                'estimated_completion_time': self.progress.estimated_completion_time
            })
    
    def _should_terminate_early(self) -> bool:
        """Check if early termination conditions are met."""
        
        if len(self.performance_history) < self.config.early_termination_patience:
            return False
        
        # Check if improvement has stagnated
        recent_scores = [
            r['primary_metric'] for r in self.performance_history[-self.config.early_termination_patience:]
            if r['success']
        ]
        
        if len(recent_scores) < self.config.early_termination_patience:
            return False
        
        # Calculate improvement rate
        early_avg = sum(recent_scores[:len(recent_scores)//2]) / (len(recent_scores)//2)
        recent_avg = sum(recent_scores[len(recent_scores)//2:]) / (len(recent_scores) - len(recent_scores)//2)
        
        if early_avg > 0:
            improvement_rate = (recent_avg - early_avg) / early_avg
            return improvement_rate < self.config.early_termination_threshold
        
        return False
    
    def _checkpoint_progress(self):
        """Save exploration progress checkpoint."""
        if not self.config.cache_directory:
            return
        
        checkpoint_path = Path(self.config.cache_directory) / "exploration_checkpoint.json"
        checkpoint_data = {
            'progress': {
                'evaluations_completed': self.progress.evaluations_completed,
                'best_score': self.progress.best_score,
                'elapsed_time': self.progress.elapsed_time
            },
            'performance_history_size': len(self.performance_history)
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.debug(f"Checkpoint saved: {self.progress.evaluations_completed}/{self.config.max_evaluations} evaluations")
    
    def _generate_final_results(self, strategy_metadata: Dict[str, Any]) -> ExplorationResults:
        """Generate final exploration results."""
        
        # Find best combination
        successful_results = [r for r in self.performance_history if r['success']]
        best_combination = None
        best_score = 0.0
        
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['primary_metric'])
            best_combination = best_result['combination']
            best_score = best_result['primary_metric']
        
        # Generate Pareto frontier
        pareto_frontier = self.pareto_analyzer.update_frontier(
            successful_results,
            objectives=['throughput', 'resource_efficiency']
        )
        
        # Generate exploration summary
        exploration_summary = self._generate_exploration_summary()
        
        # Generate execution statistics
        execution_stats = {
            'total_evaluations': self.progress.evaluations_completed,
            'cached_evaluations': self.progress.evaluations_cached,
            'successful_evaluations': len(successful_results),
            'failed_evaluations': len(self.performance_history) - len(successful_results),
            'total_time': self.progress.elapsed_time,
            'average_evaluation_time': (
                sum(r.get('evaluation_time', 0) for r in self.performance_history) / 
                len(self.performance_history) if self.performance_history else 0
            ),
            'pareto_frontier_size': len(pareto_frontier),
            'design_space_coverage': self._calculate_coverage()
        }
        
        return ExplorationResults(
            best_combination=best_combination,
            best_score=best_score,
            all_combinations=self.evaluated_combinations,
            performance_data=self.performance_history,
            pareto_frontier=pareto_frontier,
            exploration_summary=exploration_summary,
            strategy_metadata=strategy_metadata,
            execution_stats=execution_stats
        )
    
    def _generate_exploration_summary(self) -> Dict[str, Any]:
        """Generate summary of exploration results."""
        successful_results = [r for r in self.performance_history if r['success']]
        
        if not successful_results:
            return {'message': 'No successful evaluations'}
        
        metrics = [r['primary_metric'] for r in successful_results]
        
        return {
            'total_combinations_evaluated': len(self.performance_history),
            'successful_evaluations': len(successful_results),
            'success_rate': len(successful_results) / len(self.performance_history),
            'best_score': max(metrics),
            'worst_score': min(metrics),
            'average_score': sum(metrics) / len(metrics),
            'score_std_dev': (sum((x - sum(metrics)/len(metrics))**2 for x in metrics) / len(metrics))**0.5,
            'improvement_over_baseline': self._calculate_improvement(),
            'most_common_components': self._analyze_component_frequency()
        }
    
    def _calculate_coverage(self) -> float:
        """Calculate design space coverage percentage."""
        # Simplified coverage calculation
        total_possible = self.combination_generator.generate_all_combinations(max_combinations=1000)
        coverage = len(self.evaluated_combinations) / len(total_possible)
        return min(coverage, 1.0) * 100.0
    
    def _calculate_improvement(self) -> float:
        """Calculate improvement over baseline (first evaluation)."""
        if len(self.performance_history) < 2:
            return 0.0
        
        baseline = self.performance_history[0].get('primary_metric', 0)
        current_best = self.progress.best_score
        
        if baseline > 0:
            return (current_best - baseline) / baseline * 100.0
        return 0.0
    
    def _analyze_component_frequency(self) -> Dict[str, int]:
        """Analyze frequency of components in successful evaluations."""
        successful_combinations = [
            r['combination'] for r in self.performance_history 
            if r['success'] and 'combination' in r
        ]
        
        component_counts = {}
        
        for combo in successful_combinations:
            # Count canonical ops
            for op in combo.canonical_ops:
                component_counts[f"canonical_op_{op}"] = component_counts.get(f"canonical_op_{op}", 0) + 1
            
            # Count hw kernels
            for kernel, option in combo.hw_kernels.items():
                component_counts[f"hw_kernel_{kernel}_{option}"] = component_counts.get(f"hw_kernel_{kernel}_{option}", 0) + 1
            
            # Count transforms
            for transform in combo.model_topology:
                component_counts[f"transform_{transform}"] = component_counts.get(f"transform_{transform}", 0) + 1
        
        # Return top 10 most common
        sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_components[:10])
    
    def _setup_cache(self):
        """Setup evaluation cache."""
        if self.config.cache_directory:
            cache_dir = Path(self.config.cache_directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_file = cache_dir / "evaluation_cache.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        self.evaluation_cache = json.load(f)
                    logger.info(f"Loaded {len(self.evaluation_cache)} cached evaluations")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
    
    def _get_cached_result(self, combination: ComponentCombination) -> Optional[Dict[str, Any]]:
        """Get cached evaluation result for combination."""
        cache_key = self._generate_cache_key(combination)
        cached = self.evaluation_cache.get(cache_key)
        if cached:
            # Add combination back to cached result
            result = cached.copy()
            result['combination'] = combination
            return result
        return None
    
    def _cache_result(self, combination: ComponentCombination, result: Dict[str, Any]):
        """Cache evaluation result."""
        cache_key = self._generate_cache_key(combination)
        
        # Create cacheable result (remove non-serializable objects)
        cacheable_result = {
            'success': result['success'],
            'primary_metric': result['primary_metric'],
            'evaluation_time': result['evaluation_time']
        }
        
        if 'metrics' in result:
            cacheable_result['metrics'] = result['metrics']
        if 'error' in result:
            cacheable_result['error'] = result['error']
        
        self.evaluation_cache[cache_key] = cacheable_result
        
        # Periodically save cache
        if len(self.evaluation_cache) % 10 == 0:
            self._save_cache()
    
    def _generate_cache_key(self, combination: ComponentCombination) -> str:
        """Generate cache key for combination."""
        # Use combination ID as cache key
        return combination.combination_id
    
    def _save_cache(self):
        """Save evaluation cache to disk."""
        if not self.config.cache_directory:
            return
        
        cache_file = Path(self.config.cache_directory) / "evaluation_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.evaluation_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


def explore_design_space(design_space: DesignSpaceDefinition,
                        model_path: str,
                        evaluation_function: Callable,
                        config: Optional[ExplorationConfig] = None) -> ExplorationResults:
    """
    Convenience function for design space exploration.
    
    Args:
        design_space: Blueprint V2 design space definition
        model_path: Path to model to accelerate
        evaluation_function: Function to evaluate combinations
        config: Exploration configuration
        
    Returns:
        Exploration results
    """
    explorer = DesignSpaceExplorer(design_space, config)
    return explorer.explore_design_space(model_path, evaluation_function)