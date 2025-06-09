"""
Strategy Decision Tracker

Track and analyze optimization strategy decisions and outcomes for learning
and improvement of strategy selection.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
import statistics

from .types import (
    StrategyDecisionRecord,
    StrategyOutcomeRecord,
    EffectivenessReport,
    ProblemContext,
    ProblemCharacteristics,
    PerformanceMetrics
)

logger = logging.getLogger(__name__)

class StrategyDecisionDatabase:
    """Database for storing strategy decisions and outcomes"""
    
    def __init__(self):
        self.decisions = {}  # decision_id -> StrategyDecisionRecord
        self.outcomes = {}   # strategy_id -> List[StrategyOutcomeRecord]
        self.correlations = {}  # problem_type -> strategy effectiveness
    
    def store_decision(self, decision: StrategyDecisionRecord) -> None:
        """Store strategy decision record"""
        if not decision.decision_id:
            decision.decision_id = str(uuid.uuid4())
        
        self.decisions[decision.decision_id] = decision
        logger.debug(f"Stored strategy decision: {decision.decision_id}")
    
    def store_outcome(self, outcome: StrategyOutcomeRecord) -> None:
        """Store strategy outcome record"""
        if outcome.strategy_id not in self.outcomes:
            self.outcomes[outcome.strategy_id] = []
        
        self.outcomes[outcome.strategy_id].append(outcome)
        logger.debug(f"Stored strategy outcome for: {outcome.strategy_id}")
    
    def query_by_problem_type(self, problem_type: str) -> List[StrategyDecisionRecord]:
        """Query decisions by problem type"""
        matching_decisions = []
        
        for decision in self.decisions.values():
            # Simple matching - in practice would be more sophisticated
            if (problem_type.lower() in str(decision.problem_characteristics).lower() or
                problem_type.lower() in decision.selection_rationale.lower()):
                matching_decisions.append(decision)
        
        return matching_decisions
    
    def get_strategy_outcomes(self, strategy: str) -> List[StrategyOutcomeRecord]:
        """Get all outcomes for a specific strategy"""
        all_outcomes = []
        
        for strategy_id, outcomes in self.outcomes.items():
            if strategy.lower() in strategy_id.lower():
                all_outcomes.extend(outcomes)
        
        return all_outcomes

class PerformanceCorrelator:
    """Correlate performance metrics with strategy decisions"""
    
    def __init__(self):
        self.correlation_cache = {}
    
    def correlate_strategy_performance(self, 
                                     decisions: List[StrategyDecisionRecord],
                                     outcomes: List[StrategyOutcomeRecord]) -> Dict[str, float]:
        """Correlate strategy selection with performance outcomes"""
        
        correlations = {}
        
        # Group outcomes by strategy
        strategy_performance = {}
        for outcome in outcomes:
            strategy = outcome.strategy_id.split('_')[0]  # Extract base strategy name
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(outcome.performance_metrics.solution_quality)
        
        # Compute average performance per strategy
        for strategy, performances in strategy_performance.items():
            if len(performances) > 0:
                correlations[strategy] = statistics.mean(performances)
        
        return correlations
    
    def analyze_context_sensitivity(self, 
                                  decisions: List[StrategyDecisionRecord],
                                  outcomes: List[StrategyOutcomeRecord]) -> Dict[str, float]:
        """Analyze how sensitive strategies are to problem context"""
        
        sensitivity_scores = {}
        
        # Group by strategy and analyze variance in performance
        strategy_groups = {}
        for decision in decisions:
            strategy = decision.selected_strategy
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(decision)
        
        for strategy, strategy_decisions in strategy_groups.items():
            if len(strategy_decisions) > 1:
                # Find corresponding outcomes
                strategy_outcomes = [o for o in outcomes if strategy in o.strategy_id]
                if len(strategy_outcomes) > 1:
                    performances = [o.performance_metrics.solution_quality for o in strategy_outcomes]
                    if len(performances) > 1:
                        # Use coefficient of variation as sensitivity measure
                        mean_perf = statistics.mean(performances)
                        std_perf = statistics.stdev(performances)
                        sensitivity_scores[strategy] = std_perf / mean_perf if mean_perf > 0 else 0
        
        return sensitivity_scores

class StrategyAnalyzer:
    """Analyze strategy effectiveness and patterns"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def compute_success_rates(self, 
                            decisions: List[StrategyDecisionRecord],
                            outcomes: List[StrategyOutcomeRecord]) -> Dict[str, float]:
        """Compute success rates for different strategies"""
        
        success_rates = {}
        
        # Group outcomes by strategy
        strategy_outcomes = {}
        for outcome in outcomes:
            strategy = outcome.strategy_id.split('_')[0]
            if strategy not in strategy_outcomes:
                strategy_outcomes[strategy] = []
            strategy_outcomes[strategy].append(outcome)
        
        # Compute success rates
        for strategy, outcomes in strategy_outcomes.items():
            if len(outcomes) > 0:
                successes = sum(1 for o in outcomes if o.optimization_success)
                success_rates[strategy] = successes / len(outcomes)
        
        return success_rates
    
    def compare_strategy_performance(self, 
                                   decisions: List[StrategyDecisionRecord],
                                   outcomes: List[StrategyOutcomeRecord]) -> Dict[str, Dict[str, float]]:
        """Compare performance metrics across strategies"""
        
        comparison = {}
        
        # Group outcomes by strategy
        strategy_outcomes = {}
        for outcome in outcomes:
            strategy = outcome.strategy_id.split('_')[0]
            if strategy not in strategy_outcomes:
                strategy_outcomes[strategy] = []
            strategy_outcomes[strategy].append(outcome)
        
        # Compute performance statistics for each strategy
        for strategy, outcomes in strategy_outcomes.items():
            if len(outcomes) > 0:
                comparison[strategy] = {
                    'avg_solution_quality': statistics.mean([o.performance_metrics.solution_quality for o in outcomes]),
                    'avg_convergence_time': statistics.mean([o.performance_metrics.convergence_time for o in outcomes]),
                    'avg_efficiency': statistics.mean([o.performance_metrics.efficiency for o in outcomes]),
                    'success_rate': sum(1 for o in outcomes if o.optimization_success) / len(outcomes)
                }
        
        return comparison

class StrategyDecisionTracker:
    """Track and analyze optimization strategy decisions and outcomes"""
    
    def __init__(self):
        self.decision_database = StrategyDecisionDatabase()
        self.performance_correlator = PerformanceCorrelator()
        self.strategy_analyzer = StrategyAnalyzer()
    
    def record_strategy_choice(self, 
                             context: ProblemContext, 
                             strategy: str, 
                             rationale: str) -> str:
        """Record strategy selection decision with context"""
        
        decision_record = StrategyDecisionRecord(
            timestamp=datetime.now(),
            problem_context=context,
            selected_strategy=strategy,
            selection_rationale=rationale,
            problem_characteristics=self._extract_problem_characteristics(context),
            available_alternatives=self._get_available_strategies(context),
            confidence_score=self._compute_confidence_score(context, strategy)
        )
        
        self.decision_database.store_decision(decision_record)
        
        # Update strategy effectiveness models
        self._update_strategy_models(decision_record)
        
        logger.info(f"Recorded strategy decision: {strategy} for problem context")
        return decision_record.decision_id
    
    def record_strategy_outcome(self, 
                              strategy: str, 
                              performance: PerformanceMetrics,
                              decision_id: str = None) -> None:
        """Record strategy outcome and performance results"""
        
        strategy_id = self._get_strategy_id(strategy, decision_id)
        
        outcome_record = StrategyOutcomeRecord(
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            performance_metrics=performance,
            optimization_success=self._evaluate_success(performance),
            convergence_metrics=self._extract_convergence_data(performance),
            quality_metrics=self._compute_quality_metrics(performance)
        )
        
        self.decision_database.store_outcome(outcome_record)
        
        # Update strategy effectiveness analysis
        self._update_effectiveness_analysis(outcome_record)
        
        logger.info(f"Recorded strategy outcome for: {strategy}")
    
    def analyze_strategy_effectiveness(self, problem_type: str = None) -> EffectivenessReport:
        """Analyze strategy effectiveness for specific problem types"""
        
        # Retrieve historical data
        if problem_type:
            historical_decisions = self.decision_database.query_by_problem_type(problem_type)
        else:
            historical_decisions = list(self.decision_database.decisions.values())
        
        all_outcomes = []
        for outcomes in self.decision_database.outcomes.values():
            all_outcomes.extend(outcomes)
        
        report = EffectivenessReport()
        
        # Strategy success rates
        report.success_rates = self.strategy_analyzer.compute_success_rates(
            historical_decisions, all_outcomes
        )
        
        # Performance comparisons
        report.performance_comparison = self.strategy_analyzer.compare_strategy_performance(
            historical_decisions, all_outcomes
        )
        
        # Context sensitivity analysis
        report.context_sensitivity = self.performance_correlator.analyze_context_sensitivity(
            historical_decisions, all_outcomes
        )
        
        # Recommendations
        report.recommendations = self._generate_strategy_recommendations(
            problem_type, historical_decisions, all_outcomes
        )
        
        logger.info(f"Generated effectiveness report for problem type: {problem_type}")
        return report
    
    def _extract_problem_characteristics(self, context: ProblemContext) -> ProblemCharacteristics:
        """Extract key characteristics of optimization problem"""
        
        characteristics = ProblemCharacteristics()
        
        # Model characteristics
        model_info = context.model_info
        characteristics.model_size = model_info.get('parameter_count', 0)
        characteristics.model_complexity = self._compute_model_complexity(model_info)
        characteristics.operator_diversity = self._compute_operator_diversity(model_info)
        
        # Target characteristics
        characteristics.performance_targets = context.targets.copy()
        characteristics.constraint_tightness = self._compute_constraint_tightness(context.constraints)
        characteristics.multi_objective_complexity = self._compute_mo_complexity(context.targets)
        
        # Resource characteristics
        platform_info = context.platform
        characteristics.available_resources = platform_info.get('resources', {})
        characteristics.resource_pressure = self._compute_resource_pressure(context)
        
        return characteristics
    
    def _compute_model_complexity(self, model_info: Dict[str, Any]) -> float:
        """Compute model complexity score"""
        complexity = 0.0
        
        # Layer count contribution
        layer_count = model_info.get('layer_count', 0)
        complexity += min(layer_count / 100.0, 1.0) * 0.3
        
        # Parameter count contribution
        param_count = model_info.get('parameter_count', 0)
        complexity += min(param_count / 10_000_000, 1.0) * 0.4
        
        # Operator diversity contribution
        op_types = len(model_info.get('operator_types', []))
        complexity += min(op_types / 20.0, 1.0) * 0.3
        
        return complexity
    
    def _compute_operator_diversity(self, model_info: Dict[str, Any]) -> float:
        """Compute operator diversity score"""
        op_types = model_info.get('operator_types', [])
        total_ops = model_info.get('total_operators', 1)
        
        if total_ops == 0:
            return 0.0
        
        # Shannon entropy-like diversity measure
        unique_types = len(set(op_types))
        return min(unique_types / 15.0, 1.0)  # Normalize to 0-1
    
    def _compute_constraint_tightness(self, constraints: Dict[str, float]) -> float:
        """Compute how tight the constraints are"""
        if not constraints:
            return 0.0
        
        # Assume constraints are normalized to 0-1 where 1 is very tight
        tightness_values = []
        for constraint_name, value in constraints.items():
            if 'max_' in constraint_name or 'limit' in constraint_name:
                # Inverse relationship for max constraints
                tightness_values.append(1.0 - value)
            else:
                tightness_values.append(value)
        
        return statistics.mean(tightness_values) if tightness_values else 0.0
    
    def _compute_mo_complexity(self, targets: Dict[str, float]) -> float:
        """Compute multi-objective complexity"""
        num_objectives = len(targets)
        
        if num_objectives <= 1:
            return 0.0
        elif num_objectives == 2:
            return 0.3
        elif num_objectives == 3:
            return 0.6
        else:
            return min(1.0, 0.6 + (num_objectives - 3) * 0.1)
    
    def _compute_resource_pressure(self, context: ProblemContext) -> float:
        """Compute resource pressure based on targets vs available resources"""
        targets = context.targets
        constraints = context.constraints
        
        if not targets or not constraints:
            return 0.5  # Medium pressure if unknown
        
        # Simple heuristic: higher targets with tighter constraints = higher pressure
        target_intensity = sum(targets.values()) / len(targets) if targets else 0
        constraint_tightness = self._compute_constraint_tightness(constraints)
        
        pressure = (target_intensity * constraint_tightness) / 2.0
        return min(pressure, 1.0)
    
    def _get_available_strategies(self, context: ProblemContext) -> List[str]:
        """Get list of available strategies for the context"""
        # In practice, this would consider context to filter strategies
        return [
            "genetic_algorithm",
            "simulated_annealing", 
            "particle_swarm",
            "bayesian_optimization",
            "random_search",
            "multi_objective"
        ]
    
    def _compute_confidence_score(self, context: ProblemContext, strategy: str) -> float:
        """Compute confidence score for strategy selection"""
        # Simple heuristic - in practice would be more sophisticated
        problem_chars = self._extract_problem_characteristics(context)
        
        base_confidence = 0.5
        
        # Adjust based on problem complexity
        if problem_chars.model_complexity < 0.3:
            base_confidence += 0.2  # Higher confidence for simple problems
        elif problem_chars.model_complexity > 0.7:
            base_confidence -= 0.1  # Lower confidence for complex problems
        
        # Adjust based on constraint tightness
        if problem_chars.constraint_tightness > 0.8:
            base_confidence -= 0.15  # Lower confidence for tight constraints
        
        return max(0.0, min(1.0, base_confidence))
    
    def _get_strategy_id(self, strategy: str, decision_id: str = None) -> str:
        """Generate strategy ID"""
        if decision_id:
            return f"{strategy}_{decision_id[:8]}"
        else:
            return f"{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _evaluate_success(self, performance: PerformanceMetrics) -> bool:
        """Evaluate if optimization was successful"""
        # Simple success criteria - would be more sophisticated in practice
        return (performance.solution_quality > 0.7 and 
                performance.convergence_time < 1000 and
                performance.efficiency > 0.5)
    
    def _extract_convergence_data(self, performance: PerformanceMetrics) -> Dict[str, float]:
        """Extract convergence-related metrics"""
        return {
            'convergence_time': performance.convergence_time,
            'final_quality': performance.solution_quality,
            'efficiency': performance.efficiency
        }
    
    def _compute_quality_metrics(self, performance: PerformanceMetrics) -> Dict[str, float]:
        """Compute quality metrics from performance"""
        return {
            'overall_quality': (performance.solution_quality + performance.efficiency) / 2.0,
            'time_efficiency': 1.0 / (1.0 + performance.convergence_time / 100.0),
            'resource_efficiency': performance.efficiency
        }
    
    def _update_strategy_models(self, decision: StrategyDecisionRecord) -> None:
        """Update strategy effectiveness models with new decision"""
        # Placeholder for model updates
        logger.debug(f"Updated strategy models with decision: {decision.decision_id}")
    
    def _update_effectiveness_analysis(self, outcome: StrategyOutcomeRecord) -> None:
        """Update effectiveness analysis with new outcome"""
        # Placeholder for analysis updates
        logger.debug(f"Updated effectiveness analysis with outcome: {outcome.strategy_id}")
    
    def _generate_strategy_recommendations(self, 
                                         problem_type: str,
                                         decisions: List[StrategyDecisionRecord],
                                         outcomes: List[StrategyOutcomeRecord]) -> List[str]:
        """Generate strategy recommendations based on historical data"""
        
        recommendations = []
        
        # Get performance comparison
        performance_comparison = self.strategy_analyzer.compare_strategy_performance(decisions, outcomes)
        
        # Sort strategies by average solution quality
        if performance_comparison:
            sorted_strategies = sorted(
                performance_comparison.items(),
                key=lambda x: x[1].get('avg_solution_quality', 0),
                reverse=True
            )
            
            top_strategies = sorted_strategies[:3]
            
            for strategy, metrics in top_strategies:
                quality = metrics.get('avg_solution_quality', 0)
                success_rate = metrics.get('success_rate', 0)
                
                if quality > 0.7 and success_rate > 0.8:
                    recommendations.append(
                        f"Recommend {strategy}: high quality ({quality:.2f}) and success rate ({success_rate:.2f})"
                    )
                elif quality > 0.6:
                    recommendations.append(
                        f"Consider {strategy}: good quality ({quality:.2f}) but monitor success rate"
                    )
        
        # Default recommendation if no strong patterns
        if not recommendations:
            recommendations.append("Use genetic_algorithm as a robust general-purpose strategy")
        
        return recommendations
    
    def get_decision_history(self, limit: int = 100) -> List[StrategyDecisionRecord]:
        """Get recent decision history"""
        decisions = list(self.decision_database.decisions.values())
        decisions.sort(key=lambda x: x.timestamp, reverse=True)
        return decisions[:limit]
    
    def get_strategy_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive strategy statistics"""
        all_decisions = list(self.decision_database.decisions.values())
        all_outcomes = []
        for outcomes in self.decision_database.outcomes.values():
            all_outcomes.extend(outcomes)
        
        stats = {}
        
        # Usage statistics
        strategy_usage = {}
        for decision in all_decisions:
            strategy = decision.selected_strategy
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        # Performance statistics
        performance_comparison = self.strategy_analyzer.compare_strategy_performance(
            all_decisions, all_outcomes
        )
        
        # Combine statistics
        for strategy in set(list(strategy_usage.keys()) + list(performance_comparison.keys())):
            stats[strategy] = {
                'usage_count': strategy_usage.get(strategy, 0),
                'performance_metrics': performance_comparison.get(strategy, {}),
                'last_used': self._get_last_usage(strategy, all_decisions)
            }
        
        return stats
    
    def _get_last_usage(self, strategy: str, decisions: List[StrategyDecisionRecord]) -> Optional[datetime]:
        """Get last usage timestamp for strategy"""
        strategy_decisions = [d for d in decisions if d.selected_strategy == strategy]
        if strategy_decisions:
            return max(d.timestamp for d in strategy_decisions)
        return None