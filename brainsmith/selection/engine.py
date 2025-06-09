"""
Core Selection Engine
Main orchestration engine for multi-criteria decision analysis and solution selection.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Type, Union
import numpy as np

from .models import (
    SelectionContext, SelectionCriteria, SelectionConfiguration,
    SelectionReport, SelectionMetrics, RankedSolution,
    CompromiseSolution, DecisionMatrix, ParetoSolution
)

logger = logging.getLogger(__name__)


class SelectionResult:
    """Result container for selection operations."""
    
    def __init__(self, 
                 ranked_solutions: List[RankedSolution],
                 selection_metrics: SelectionMetrics,
                 selection_report: Optional[SelectionReport] = None):
        self.ranked_solutions = ranked_solutions
        self.selection_metrics = selection_metrics
        self.selection_report = selection_report
        self.timestamp = time.time()
    
    @property
    def best_solution(self) -> Optional[RankedSolution]:
        """Get the top-ranked solution."""
        return self.ranked_solutions[0] if self.ranked_solutions else None
    
    @property
    def top_solutions(self, n: int = 5) -> List[RankedSolution]:
        """Get top N solutions."""
        return self.ranked_solutions[:n]
    
    def get_solutions_by_score_threshold(self, threshold: float) -> List[RankedSolution]:
        """Get solutions above score threshold."""
        return [sol for sol in self.ranked_solutions if sol.score >= threshold]


class SelectionEngine:
    """Multi-criteria decision analysis engine for solution selection."""
    
    def __init__(self, configuration: Optional[SelectionConfiguration] = None):
        """Initialize selection engine with configuration."""
        self.config = configuration or SelectionConfiguration()
        self.strategies = self._initialize_strategies()
        self.selection_history = []
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid selection configuration")
        
        logger.info(f"Selection engine initialized with {self.config.algorithm} algorithm")
    
    def _initialize_strategies(self) -> Dict[str, Type['SelectionStrategy']]:
        """Initialize available selection strategies."""
        # Import strategies here to avoid circular imports
        from .strategies import (
            TOPSISSelector, PROMETHEESelector, AHPSelector,
            WeightedSumSelector, WeightedProductSelector, FuzzyTOPSISSelector
        )
        
        return {
            'topsis': TOPSISSelector,
            'promethee': PROMETHEESelector,
            'ahp': AHPSelector,
            'weighted_sum': WeightedSumSelector,
            'weighted_product': WeightedProductSelector,
            'fuzzy_topsis': FuzzyTOPSISSelector
        }
    
    def select_solutions(self, 
                        pareto_solutions: List[ParetoSolution],
                        selection_criteria: SelectionCriteria,
                        max_solutions: Optional[int] = None) -> SelectionResult:
        """
        Select best solutions using specified criteria and algorithm.
        
        Args:
            pareto_solutions: List of Pareto optimal solutions
            selection_criteria: Selection criteria and preferences
            max_solutions: Maximum number of solutions to return
        
        Returns:
            SelectionResult: Ranked solutions with metrics and analysis
        """
        start_time = time.time()
        
        # Validate inputs
        if not pareto_solutions:
            raise ValueError("No Pareto solutions provided")
        
        if not selection_criteria.validate():
            raise ValueError("Invalid selection criteria")
        
        # Create selection context
        context = SelectionContext(
            pareto_solutions=pareto_solutions,
            selection_criteria=selection_criteria
        )
        
        try:
            # Get strategy
            strategy_class = self.strategies.get(self.config.algorithm)
            if strategy_class is None:
                raise ValueError(f"Unknown selection strategy: {self.config.algorithm}")
            
            strategy = strategy_class(self.config)
            
            # Perform selection
            ranked_solutions = strategy.select_solutions(context)
            
            # Limit number of solutions if specified
            max_sols = max_solutions or self.config.max_solutions
            if max_sols > 0:
                ranked_solutions = ranked_solutions[:max_sols]
            
            # Calculate selection metrics
            selection_time = time.time() - start_time
            metrics = self._calculate_metrics(
                ranked_solutions, selection_criteria, selection_time
            )
            
            # Create comprehensive report if enabled
            report = None
            if self.config.include_sensitivity or self.config.include_trade_off:
                report = self._create_selection_report(
                    ranked_solutions, selection_criteria, metrics, context
                )
            
            # Create result
            result = SelectionResult(ranked_solutions, metrics, report)
            
            # Store in history
            self.selection_history.append(result)
            
            logger.info(f"Selection completed: {len(ranked_solutions)} solutions ranked "
                       f"in {selection_time:.3f}s using {self.config.algorithm}")
            
            return result
            
        except Exception as e:
            logger.error(f"Selection failed: {e}")
            raise
    
    def _calculate_metrics(self, 
                          ranked_solutions: List[RankedSolution],
                          criteria: SelectionCriteria,
                          selection_time: float) -> SelectionMetrics:
        """Calculate selection quality metrics."""
        
        # Weight consistency (how well weights sum to 1)
        weight_sum = sum(criteria.weights.values())
        weight_consistency = 1.0 - abs(1.0 - weight_sum)
        
        # Preference satisfaction (based on score distribution)
        scores = [sol.score for sol in ranked_solutions]
        if scores:
            preference_satisfaction = np.mean(scores)
        else:
            preference_satisfaction = 0.0
        
        # Diversity score (based on objective value diversity)
        diversity_score = self._calculate_diversity_score(ranked_solutions)
        
        # Confidence score (based on score differences)
        confidence_score = self._calculate_confidence_score(ranked_solutions)
        
        return SelectionMetrics(
            selection_time=selection_time,
            number_of_solutions=len(ranked_solutions),
            number_of_criteria=len(criteria.objectives),
            weight_consistency=weight_consistency,
            preference_satisfaction=preference_satisfaction,
            diversity_score=diversity_score,
            confidence_score=confidence_score
        )
    
    def _calculate_diversity_score(self, ranked_solutions: List[RankedSolution]) -> float:
        """Calculate diversity score of selected solutions."""
        if len(ranked_solutions) < 2:
            return 1.0
        
        # Calculate pairwise distances in objective space
        distances = []
        for i in range(len(ranked_solutions)):
            for j in range(i + 1, len(ranked_solutions)):
                obj_i = np.array(ranked_solutions[i].objective_values)
                obj_j = np.array(ranked_solutions[j].objective_values)
                
                # Normalize by maximum values to make scale-invariant
                max_vals = np.maximum(np.abs(obj_i), np.abs(obj_j))
                max_vals[max_vals == 0] = 1  # Avoid division by zero
                
                normalized_distance = np.linalg.norm((obj_i - obj_j) / max_vals)
                distances.append(normalized_distance)
        
        # Return average normalized distance
        return np.mean(distances) if distances else 0.0
    
    def _calculate_confidence_score(self, ranked_solutions: List[RankedSolution]) -> float:
        """Calculate confidence score based on score separation."""
        if len(ranked_solutions) < 2:
            return 1.0
        
        scores = [sol.score for sol in ranked_solutions]
        
        # Calculate score gaps between consecutive solutions
        gaps = []
        for i in range(len(scores) - 1):
            gap = scores[i] - scores[i + 1]
            gaps.append(gap)
        
        # Confidence is higher when top solutions are clearly separated
        if gaps:
            # Focus on top gaps (first few solutions)
            top_gaps = gaps[:min(3, len(gaps))]
            confidence = np.mean(top_gaps)
        else:
            confidence = 0.0
        
        # Normalize to [0, 1] range
        return min(1.0, max(0.0, confidence))
    
    def _create_selection_report(self, 
                                ranked_solutions: List[RankedSolution],
                                criteria: SelectionCriteria,
                                metrics: SelectionMetrics,
                                context: SelectionContext) -> SelectionReport:
        """Create comprehensive selection report."""
        
        # Trade-off analysis
        trade_off_analysis = {}
        if self.config.include_trade_off:
            trade_off_analysis = self._analyze_trade_offs(ranked_solutions, criteria)
        
        # Sensitivity analysis
        sensitivity_analysis = {}
        if self.config.include_sensitivity:
            sensitivity_analysis = self._analyze_sensitivity(ranked_solutions, criteria)
        
        # Identify compromise solutions
        compromise_solutions = self._identify_compromise_solutions(
            ranked_solutions, criteria, trade_off_analysis
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            ranked_solutions, criteria, metrics, trade_off_analysis
        )
        
        # Create executive summary
        executive_summary = self._create_executive_summary(
            ranked_solutions, criteria, metrics, recommendations
        )
        
        return SelectionReport(
            ranked_solutions=ranked_solutions,
            selection_criteria=criteria,
            selection_metrics=metrics,
            compromise_solutions=compromise_solutions,
            sensitivity_analysis=sensitivity_analysis,
            recommendations=recommendations,
            trade_off_analysis=trade_off_analysis,
            executive_summary=executive_summary
        )
    
    def _analyze_trade_offs(self, 
                           ranked_solutions: List[RankedSolution],
                           criteria: SelectionCriteria) -> Dict[str, Any]:
        """Analyze trade-offs between objectives."""
        trade_offs = {}
        
        if len(ranked_solutions) < 2:
            return trade_offs
        
        # Calculate objective correlations
        n_objectives = len(criteria.objectives)
        objective_matrix = np.zeros((len(ranked_solutions), n_objectives))
        
        for i, solution in enumerate(ranked_solutions):
            for j in range(n_objectives):
                if j < len(solution.objective_values):
                    objective_matrix[i, j] = solution.objective_values[j]
        
        # Calculate correlation matrix
        correlations = np.corrcoef(objective_matrix.T)
        
        # Identify trade-off relationships
        trade_off_pairs = []
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                corr = correlations[i, j]
                if abs(corr) > 0.3:  # Significant correlation
                    obj_i = criteria.objectives[i]
                    obj_j = criteria.objectives[j]
                    trade_off_pairs.append({
                        'objective_1': obj_i,
                        'objective_2': obj_j,
                        'correlation': corr,
                        'trade_off_strength': abs(corr)
                    })
        
        trade_offs['objective_correlations'] = correlations.tolist()
        trade_offs['trade_off_pairs'] = trade_off_pairs
        
        return trade_offs
    
    def _analyze_sensitivity(self, 
                            ranked_solutions: List[RankedSolution],
                            criteria: SelectionCriteria) -> Dict[str, Any]:
        """Analyze sensitivity to weight changes."""
        sensitivity = {}
        
        if len(ranked_solutions) < 2:
            return sensitivity
        
        # Test weight perturbations
        weight_perturbations = [0.05, 0.1, 0.2]  # 5%, 10%, 20% changes
        
        for obj in criteria.objectives:
            obj_sensitivity = {}
            
            for perturbation in weight_perturbations:
                # Create perturbed weights
                perturbed_weights = criteria.weights.copy()
                original_weight = perturbed_weights[obj]
                
                # Increase weight
                perturbed_weights[obj] = min(1.0, original_weight + perturbation)
                
                # Renormalize other weights
                other_total = sum(w for o, w in perturbed_weights.items() if o != obj)
                if other_total > 0:
                    scale_factor = (1.0 - perturbed_weights[obj]) / other_total
                    for o in perturbed_weights:
                        if o != obj:
                            perturbed_weights[o] *= scale_factor
                
                # Recalculate scores with perturbed weights
                # This is a simplified sensitivity analysis
                # In practice, would re-run the selection algorithm
                ranking_changes = self._count_ranking_changes(
                    ranked_solutions, criteria, perturbed_weights
                )
                
                obj_sensitivity[f'perturbation_{perturbation}'] = ranking_changes
            
            sensitivity[obj] = obj_sensitivity
        
        return sensitivity
    
    def _count_ranking_changes(self, 
                              ranked_solutions: List[RankedSolution],
                              criteria: SelectionCriteria,
                              perturbed_weights: Dict[str, float]) -> float:
        """Count ranking changes due to weight perturbation."""
        # Simplified implementation - would need full re-ranking in practice
        # For now, estimate based on objective importance
        
        changes = 0
        for i, solution in enumerate(ranked_solutions):
            for j, other_solution in enumerate(ranked_solutions):
                if i != j:
                    # Simple scoring with perturbed weights
                    score_i = sum(
                        perturbed_weights.get(obj, 0) * val
                        for obj, val in zip(criteria.objectives, solution.objective_values)
                    )
                    score_j = sum(
                        perturbed_weights.get(obj, 0) * val
                        for obj, val in zip(criteria.objectives, other_solution.objective_values)
                    )
                    
                    # Check if relative ordering changed
                    original_better = solution.rank < other_solution.rank
                    new_better = score_i > score_j
                    
                    if original_better != new_better:
                        changes += 1
        
        # Normalize by total possible changes
        total_comparisons = len(ranked_solutions) * (len(ranked_solutions) - 1)
        return changes / total_comparisons if total_comparisons > 0 else 0.0
    
    def _identify_compromise_solutions(self, 
                                     ranked_solutions: List[RankedSolution],
                                     criteria: SelectionCriteria,
                                     trade_off_analysis: Dict[str, Any]) -> List[CompromiseSolution]:
        """Identify compromise solutions."""
        compromise_solutions = []
        
        if not ranked_solutions:
            return compromise_solutions
        
        # Balanced compromise (highest overall score)
        best_solution = ranked_solutions[0]
        balanced_compromise = CompromiseSolution(
            solution=best_solution,
            compromise_type='balanced',
            trade_off_analysis=trade_off_analysis,
            sensitivity_analysis={},
            robustness_score=best_solution.confidence
        )
        compromise_solutions.append(balanced_compromise)
        
        # If we have trade-off analysis, find other types
        if len(ranked_solutions) > 1:
            # Least regret solution (minimal maximum loss)
            least_regret_idx = self._find_least_regret_solution(ranked_solutions, criteria)
            if least_regret_idx != 0:  # Different from balanced
                least_regret_compromise = CompromiseSolution(
                    solution=ranked_solutions[least_regret_idx],
                    compromise_type='least_regret',
                    trade_off_analysis=trade_off_analysis,
                    sensitivity_analysis={},
                    robustness_score=ranked_solutions[least_regret_idx].confidence
                )
                compromise_solutions.append(least_regret_compromise)
        
        return compromise_solutions
    
    def _find_least_regret_solution(self, 
                                   ranked_solutions: List[RankedSolution],
                                   criteria: SelectionCriteria) -> int:
        """Find solution with least regret (minimal maximum loss)."""
        if len(ranked_solutions) < 2:
            return 0
        
        # Calculate ideal point (best value for each objective)
        n_objectives = len(criteria.objectives)
        ideal_point = np.zeros(n_objectives)
        
        for j in range(n_objectives):
            obj_values = [sol.objective_values[j] for sol in ranked_solutions 
                         if j < len(sol.objective_values)]
            if obj_values:
                if criteria.maximize_objectives.get(criteria.objectives[j], True):
                    ideal_point[j] = max(obj_values)
                else:
                    ideal_point[j] = min(obj_values)
        
        # Calculate regret for each solution
        min_regret = float('inf')
        best_idx = 0
        
        for i, solution in enumerate(ranked_solutions):
            regrets = []
            for j in range(min(n_objectives, len(solution.objective_values))):
                if criteria.maximize_objectives.get(criteria.objectives[j], True):
                    regret = ideal_point[j] - solution.objective_values[j]
                else:
                    regret = solution.objective_values[j] - ideal_point[j]
                regrets.append(max(0, regret))  # No negative regret
            
            max_regret = max(regrets) if regrets else float('inf')
            if max_regret < min_regret:
                min_regret = max_regret
                best_idx = i
        
        return best_idx
    
    def _generate_recommendations(self, 
                                 ranked_solutions: List[RankedSolution],
                                 criteria: SelectionCriteria,
                                 metrics: SelectionMetrics,
                                 trade_off_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not ranked_solutions:
            recommendations.append("No feasible solutions found. Consider relaxing constraints.")
            return recommendations
        
        # Top solution recommendation
        best_solution = ranked_solutions[0]
        recommendations.append(
            f"Recommended solution achieves score {best_solution.score:.3f} "
            f"and ranks #1 out of {len(ranked_solutions)} alternatives."
        )
        
        # Confidence assessment
        if metrics.confidence_score > 0.7:
            recommendations.append("High confidence in top solution selection.")
        elif metrics.confidence_score > 0.4:
            recommendations.append("Moderate confidence - consider multiple top solutions.")
        else:
            recommendations.append("Low confidence - solution selection is highly sensitive to preferences.")
        
        # Diversity assessment
        if metrics.diversity_score > 0.5:
            recommendations.append("Selected solutions show good diversity across objectives.")
        else:
            recommendations.append("Selected solutions are similar - consider broader exploration.")
        
        # Trade-off insights
        if 'trade_off_pairs' in trade_off_analysis:
            strong_trade_offs = [
                pair for pair in trade_off_analysis['trade_off_pairs']
                if pair['trade_off_strength'] > 0.7
            ]
            if strong_trade_offs:
                recommendations.append(
                    f"Strong trade-offs detected between {len(strong_trade_offs)} objective pairs. "
                    "Consider these relationships when making final decision."
                )
        
        return recommendations
    
    def _create_executive_summary(self, 
                                 ranked_solutions: List[RankedSolution],
                                 criteria: SelectionCriteria,
                                 metrics: SelectionMetrics,
                                 recommendations: List[str]) -> str:
        """Create executive summary of selection results."""
        
        if not ranked_solutions:
            return "No feasible solutions were found for the given criteria."
        
        summary_parts = []
        
        # Overview
        summary_parts.append(
            f"Selection analysis completed for {metrics.number_of_solutions} solutions "
            f"using {metrics.number_of_criteria} criteria. "
            f"Analysis took {metrics.selection_time:.2f} seconds."
        )
        
        # Best solution
        best_solution = ranked_solutions[0]
        summary_parts.append(
            f"Top recommended solution achieves overall score of {best_solution.score:.3f}."
        )
        
        # Key metrics
        summary_parts.append(
            f"Selection quality: confidence {metrics.confidence_score:.2f}, "
            f"diversity {metrics.diversity_score:.2f}, "
            f"preference satisfaction {metrics.preference_satisfaction:.2f}."
        )
        
        # Primary recommendation
        if recommendations:
            summary_parts.append(f"Key recommendation: {recommendations[0]}")
        
        return " ".join(summary_parts)
    
    def compare_strategies(self, 
                          pareto_solutions: List[ParetoSolution],
                          selection_criteria: SelectionCriteria,
                          strategies: Optional[List[str]] = None) -> Dict[str, SelectionResult]:
        """Compare multiple selection strategies on the same problem."""
        
        if strategies is None:
            strategies = ['topsis', 'weighted_sum', 'promethee']
        
        results = {}
        original_algorithm = self.config.algorithm
        
        try:
            for strategy in strategies:
                if strategy in self.strategies:
                    # Temporarily change algorithm
                    self.config.algorithm = strategy
                    
                    # Run selection
                    result = self.select_solutions(pareto_solutions, selection_criteria)
                    results[strategy] = result
                    
                    logger.info(f"Strategy {strategy}: {len(result.ranked_solutions)} solutions, "
                               f"best score {result.best_solution.score:.3f}")
        
        finally:
            # Restore original algorithm
            self.config.algorithm = original_algorithm
        
        return results