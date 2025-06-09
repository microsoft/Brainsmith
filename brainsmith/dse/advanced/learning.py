"""
Learning-Based Search and Adaptive Strategies
Intelligent search that learns from historical analysis and adapts strategies dynamically.
"""

import os
import sys
import time
import logging
import random
import math
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, deque
import pickle

# Import Week 2 components
from ...metrics import HistoricalAnalysisEngine, TrendAnalyzer, RegressionDetector

logger = logging.getLogger(__name__)


@dataclass
class SearchPattern:
    """Represents a learned search pattern."""
    pattern_id: str
    parameter_ranges: Dict[str, Tuple[float, float]]
    objective_correlations: Dict[str, float]
    success_rate: float
    usage_count: int = 0
    avg_improvement: float = 0.0
    context_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Performance metrics for optimization strategies."""
    strategy_name: str
    success_count: int = 0
    total_attempts: int = 0
    avg_improvement: float = 0.0
    avg_convergence_time: float = 0.0
    best_objective_achieved: float = float('-inf')
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=10))
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.total_attempts)
    
    def update(self, improvement: float, convergence_time: float, success: bool):
        """Update performance metrics."""
        self.total_attempts += 1
        if success:
            self.success_count += 1
        
        self.avg_improvement = (self.avg_improvement * (self.total_attempts - 1) + improvement) / self.total_attempts
        self.avg_convergence_time = (self.avg_convergence_time * (self.total_attempts - 1) + convergence_time) / self.total_attempts
        
        if improvement > self.best_objective_achieved:
            self.best_objective_achieved = improvement
        
        self.recent_performance.append({
            'improvement': improvement,
            'convergence_time': convergence_time,
            'success': success,
            'timestamp': time.time()
        })


class SearchMemory:
    """Memory system for storing and retrieving successful search patterns."""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.patterns = {}
        self.pattern_index = {}
        self.access_history = deque(maxlen=memory_size)
        self.pattern_counter = 0
    
    def store_pattern(self, design_parameters: Dict[str, Any], 
                     objective_values: List[float], 
                     context: Dict[str, Any] = None) -> str:
        """Store a successful search pattern."""
        
        pattern_id = f"pattern_{self.pattern_counter}"
        self.pattern_counter += 1
        
        # Extract parameter ranges
        parameter_ranges = {}
        for param, value in design_parameters.items():
            if isinstance(value, (int, float)):
                # Store as range with some tolerance
                tolerance = abs(value) * 0.1 + 0.01
                parameter_ranges[param] = (value - tolerance, value + tolerance)
        
        # Calculate objective correlations (simplified)
        objective_correlations = {}
        if len(objective_values) > 1:
            for i, obj_val in enumerate(objective_values):
                objective_correlations[f"obj_{i}"] = obj_val
        
        pattern = SearchPattern(
            pattern_id=pattern_id,
            parameter_ranges=parameter_ranges,
            objective_correlations=objective_correlations,
            success_rate=1.0,  # Initial success rate
            context_conditions=context or {}
        )
        
        self.patterns[pattern_id] = pattern
        self._update_index(pattern)
        self.access_history.append(pattern_id)
        
        # Maintain memory size limit
        if len(self.patterns) > self.memory_size:
            self._evict_old_patterns()
        
        return pattern_id
    
    def retrieve_similar_patterns(self, design_parameters: Dict[str, Any], 
                                context: Dict[str, Any] = None, 
                                top_k: int = 5) -> List[SearchPattern]:
        """Retrieve patterns similar to current design parameters."""
        
        similarities = []
        
        for pattern_id, pattern in self.patterns.items():
            similarity = self._calculate_similarity(design_parameters, pattern, context)
            similarities.append((similarity, pattern))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [pattern for _, pattern in similarities[:top_k]]
    
    def update_pattern_success(self, pattern_id: str, success: bool, improvement: float):
        """Update pattern success statistics."""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.usage_count += 1
            
            # Update success rate with exponential moving average
            alpha = 0.1
            if success:
                pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * 1.0
                pattern.avg_improvement = (pattern.avg_improvement * (pattern.usage_count - 1) + improvement) / pattern.usage_count
            else:
                pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * 0.0
    
    def _calculate_similarity(self, design_parameters: Dict[str, Any], 
                            pattern: SearchPattern, 
                            context: Dict[str, Any] = None) -> float:
        """Calculate similarity between current parameters and stored pattern."""
        
        total_similarity = 0.0
        param_count = 0
        
        # Parameter similarity
        for param, value in design_parameters.items():
            if param in pattern.parameter_ranges and isinstance(value, (int, float)):
                min_val, max_val = pattern.parameter_ranges[param]
                if min_val <= value <= max_val:
                    # Value is within range
                    range_size = max_val - min_val
                    if range_size > 0:
                        # Calculate relative position within range
                        position = (value - min_val) / range_size
                        # Closer to center is more similar
                        similarity = 1.0 - abs(position - 0.5) * 2.0
                    else:
                        similarity = 1.0  # Exact match
                else:
                    # Value is outside range
                    center = (min_val + max_val) / 2
                    distance = abs(value - center)
                    range_size = max_val - min_val
                    similarity = max(0.0, 1.0 - distance / (range_size + 1.0))
                
                total_similarity += similarity
                param_count += 1
        
        # Context similarity
        context_similarity = 1.0
        if context and pattern.context_conditions:
            context_matches = 0
            context_total = 0
            
            for key, value in context.items():
                if key in pattern.context_conditions:
                    if value == pattern.context_conditions[key]:
                        context_matches += 1
                    context_total += 1
            
            if context_total > 0:
                context_similarity = context_matches / context_total
        
        # Combine similarities
        if param_count > 0:
            param_similarity = total_similarity / param_count
            return 0.7 * param_similarity + 0.3 * context_similarity
        else:
            return 0.0
    
    def _update_index(self, pattern: SearchPattern):
        """Update search index for faster retrieval."""
        # Simple indexing by parameter names
        for param in pattern.parameter_ranges:
            if param not in self.pattern_index:
                self.pattern_index[param] = []
            self.pattern_index[param].append(pattern.pattern_id)
    
    def _evict_old_patterns(self):
        """Evict least recently used patterns."""
        # Find patterns not in recent access history
        recent_patterns = set(self.access_history)
        candidates_for_eviction = [pid for pid in self.patterns.keys() if pid not in recent_patterns]
        
        # Sort by usage count and success rate
        candidates_for_eviction.sort(key=lambda pid: (
            self.patterns[pid].usage_count,
            self.patterns[pid].success_rate
        ))
        
        # Remove lowest scoring patterns
        num_to_evict = len(self.patterns) - self.memory_size + 1
        for i in range(min(num_to_evict, len(candidates_for_eviction))):
            pattern_id = candidates_for_eviction[i]
            del self.patterns[pattern_id]
            
            # Clean up index
            for param_list in self.pattern_index.values():
                if pattern_id in param_list:
                    param_list.remove(pattern_id)
    
    def save_to_file(self, filepath: str):
        """Save memory to file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'patterns': self.patterns,
                    'pattern_index': self.pattern_index,
                    'pattern_counter': self.pattern_counter
                }, f)
        except Exception as e:
            logger.error(f"Failed to save search memory: {e}")
    
    def load_from_file(self, filepath: str):
        """Load memory from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.patterns = data['patterns']
                self.pattern_index = data['pattern_index']
                self.pattern_counter = data['pattern_counter']
        except Exception as e:
            logger.error(f"Failed to load search memory: {e}")


class LearningBasedSearch:
    """Search strategy that learns from historical data and past experiences."""
    
    def __init__(self, historical_analysis_engine: HistoricalAnalysisEngine = None,
                 learning_config: Dict[str, Any] = None):
        self.historical_engine = historical_analysis_engine
        self.learning_config = learning_config or {}
        self.search_memory = SearchMemory(memory_size=self.learning_config.get('memory_size', 1000))
        self.parameter_correlations = {}
        self.learned_patterns = []
        self.learning_history = []
        
        # Learning parameters
        self.exploration_rate = self.learning_config.get('exploration_rate', 0.3)
        self.learning_rate = self.learning_config.get('learning_rate', 0.1)
        self.pattern_confidence_threshold = self.learning_config.get('confidence_threshold', 0.7)
    
    def learn_from_history(self, hours_lookback: int = 168):  # 1 week default
        """Learn patterns from historical optimization data."""
        
        if not self.historical_engine:
            logger.warning("No historical analysis engine available for learning")
            return
        
        try:
            # Get historical trend data
            trend_summary = self.historical_engine.get_trend_summary(hours=hours_lookback)
            
            # Analyze successful optimization patterns
            successful_patterns = self._extract_successful_patterns(trend_summary)
            
            # Learn parameter correlations
            self._learn_parameter_correlations(successful_patterns)
            
            # Update learned patterns
            self.learned_patterns.extend(successful_patterns)
            
            # Store patterns in memory
            for pattern in successful_patterns:
                self.search_memory.store_pattern(
                    pattern['parameters'],
                    pattern['objectives'],
                    pattern.get('context', {})
                )
            
            logger.info(f"Learned {len(successful_patterns)} patterns from {hours_lookback}h of history")
            
        except Exception as e:
            logger.error(f"Failed to learn from history: {e}")
    
    def suggest_next_candidates(self, current_population: List[Dict[str, Any]], 
                               search_state: Dict[str, Any],
                               num_candidates: int = 5) -> List[Dict[str, Any]]:
        """Suggest promising candidates based on learned patterns."""
        
        candidates = []
        
        # Get similar patterns from memory
        if current_population:
            best_current = max(current_population, key=lambda x: x.get('fitness', -float('inf')))
            similar_patterns = self.search_memory.retrieve_similar_patterns(
                best_current, search_state, top_k=3
            )
        else:
            similar_patterns = []
        
        # Generate candidates based on patterns
        for i in range(num_candidates):
            if random.random() < self.exploration_rate or not similar_patterns:
                # Exploration: generate random candidate
                candidate = self._generate_exploration_candidate(search_state)
            else:
                # Exploitation: generate candidate based on learned patterns
                pattern = random.choice(similar_patterns)
                candidate = self._generate_exploitation_candidate(pattern, search_state)
            
            candidates.append(candidate)
        
        return candidates
    
    def update_learning(self, design_parameters: Dict[str, Any], 
                       objective_values: List[float],
                       success: bool, search_context: Dict[str, Any] = None):
        """Update learning based on optimization results."""
        
        # Store successful patterns
        if success and objective_values:
            pattern_id = self.search_memory.store_pattern(
                design_parameters, objective_values, search_context
            )
            
            # Calculate improvement
            improvement = max(objective_values) if objective_values else 0.0
            self.search_memory.update_pattern_success(pattern_id, success, improvement)
        
        # Update parameter correlations
        self._update_parameter_correlations(design_parameters, objective_values, success)
        
        # Record learning history
        self.learning_history.append({
            'timestamp': time.time(),
            'parameters': design_parameters.copy(),
            'objectives': objective_values.copy() if objective_values else [],
            'success': success,
            'context': search_context or {}
        })
    
    def get_parameter_suggestions(self, current_parameters: Dict[str, Any],
                                 parameter_name: str) -> List[Any]:
        """Get suggestions for a specific parameter based on learning."""
        
        suggestions = []
        
        # Get patterns that involve this parameter
        similar_patterns = self.search_memory.retrieve_similar_patterns(current_parameters, top_k=10)
        
        parameter_values = []
        for pattern in similar_patterns:
            if parameter_name in pattern.parameter_ranges:
                min_val, max_val = pattern.parameter_ranges[parameter_name]
                # Suggest center of range and some variations
                center = (min_val + max_val) / 2
                parameter_values.extend([min_val, center, max_val])
        
        # Remove duplicates and sort
        if parameter_values:
            unique_values = list(set(parameter_values))
            unique_values.sort()
            suggestions = unique_values[:5]  # Top 5 suggestions
        
        return suggestions
    
    def _extract_successful_patterns(self, trend_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract successful optimization patterns from trend analysis."""
        
        patterns = []
        
        # Look for metrics with positive trends
        trends = trend_summary.get('trends', {})
        
        for metric_name, trend_data in trends.items():
            if trend_data.get('direction') == 'improving' and trend_data.get('strength', 0) > 0.5:
                # This is a successful pattern
                pattern = {
                    'metric': metric_name,
                    'trend_strength': trend_data.get('strength', 0),
                    'improvement_rate': trend_data.get('slope', 0),
                    'parameters': {},  # Would be extracted from actual data
                    'objectives': [trend_data.get('latest_value', 0)],
                    'context': {'trend_type': 'improving'}
                }
                patterns.append(pattern)
        
        return patterns
    
    def _learn_parameter_correlations(self, patterns: List[Dict[str, Any]]):
        """Learn correlations between parameters and objectives."""
        
        for pattern in patterns:
            parameters = pattern.get('parameters', {})
            objectives = pattern.get('objectives', [])
            
            if not parameters or not objectives:
                continue
            
            # Calculate correlations (simplified)
            for param_name, param_value in parameters.items():
                if isinstance(param_value, (int, float)):
                    if param_name not in self.parameter_correlations:
                        self.parameter_correlations[param_name] = {
                            'positive_correlation': 0.0,
                            'negative_correlation': 0.0,
                            'sample_count': 0
                        }
                    
                    # Update correlation estimates
                    correlation_data = self.parameter_correlations[param_name]
                    
                    # Simplified correlation update
                    avg_objective = sum(objectives) / len(objectives)
                    if avg_objective > 0:
                        correlation_data['positive_correlation'] += 1
                    else:
                        correlation_data['negative_correlation'] += 1
                    
                    correlation_data['sample_count'] += 1
    
    def _generate_exploration_candidate(self, search_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exploration candidate using random search with bias."""
        
        design_space = search_state.get('design_space', {})
        candidate = {}
        
        for param_name, param_range in design_space.items():
            if param_name.startswith('_'):
                continue
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    # Use learned correlations to bias exploration
                    if param_name in self.parameter_correlations:
                        corr_data = self.parameter_correlations[param_name]
                        total_samples = corr_data['sample_count']
                        if total_samples > 0:
                            positive_ratio = corr_data['positive_correlation'] / total_samples
                            if positive_ratio > 0.6:
                                # Bias towards higher values
                                bias_min = min_val + (max_val - min_val) * 0.3
                                bias_max = max_val
                            elif positive_ratio < 0.4:
                                # Bias towards lower values
                                bias_min = min_val
                                bias_max = min_val + (max_val - min_val) * 0.7
                            else:
                                bias_min, bias_max = min_val, max_val
                        else:
                            bias_min, bias_max = min_val, max_val
                    else:
                        bias_min, bias_max = min_val, max_val
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        candidate[param_name] = random.randint(int(bias_min), int(bias_max))
                    else:
                        candidate[param_name] = random.uniform(bias_min, bias_max)
            
            elif isinstance(param_range, (list, tuple)):
                candidate[param_name] = random.choice(param_range)
        
        return candidate
    
    def _generate_exploitation_candidate(self, pattern: SearchPattern, 
                                       search_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exploitation candidate based on learned pattern."""
        
        candidate = {}
        design_space = search_state.get('design_space', {})
        
        for param_name, param_range in design_space.items():
            if param_name.startswith('_'):
                continue
            
            if param_name in pattern.parameter_ranges:
                # Use pattern range
                pattern_min, pattern_max = pattern.parameter_ranges[param_name]
                
                # Add some noise for variation
                noise_factor = 0.1
                range_size = pattern_max - pattern_min
                noise = random.uniform(-range_size * noise_factor, range_size * noise_factor)
                
                # Sample from pattern range with noise
                value = random.uniform(pattern_min, pattern_max) + noise
                
                # Clamp to design space bounds
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    space_min, space_max = param_range
                    if isinstance(space_min, (int, float)) and isinstance(space_max, (int, float)):
                        value = max(space_min, min(space_max, value))
                        
                        if isinstance(space_min, int) and isinstance(space_max, int):
                            candidate[param_name] = int(round(value))
                        else:
                            candidate[param_name] = value
                elif isinstance(param_range, (list, tuple)):
                    candidate[param_name] = random.choice(param_range)
            else:
                # Fall back to design space sampling
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    min_val, max_val = param_range
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            candidate[param_name] = random.randint(min_val, max_val)
                        else:
                            candidate[param_name] = random.uniform(min_val, max_val)
                elif isinstance(param_range, (list, tuple)):
                    candidate[param_name] = random.choice(param_range)
        
        return candidate
    
    def _update_parameter_correlations(self, design_parameters: Dict[str, Any],
                                     objective_values: List[float], success: bool):
        """Update parameter correlation estimates."""
        
        if not objective_values:
            return
        
        avg_objective = sum(objective_values) / len(objective_values)
        
        for param_name, param_value in design_parameters.items():
            if isinstance(param_value, (int, float)):
                if param_name not in self.parameter_correlations:
                    self.parameter_correlations[param_name] = {
                        'positive_correlation': 0.0,
                        'negative_correlation': 0.0,
                        'sample_count': 0,
                        'value_sum': 0.0,
                        'objective_sum': 0.0
                    }
                
                corr_data = self.parameter_correlations[param_name]
                
                # Update running statistics
                corr_data['sample_count'] += 1
                corr_data['value_sum'] += param_value
                corr_data['objective_sum'] += avg_objective
                
                # Update correlation estimates with exponential moving average
                alpha = self.learning_rate
                if avg_objective > 0:
                    corr_data['positive_correlation'] = (1 - alpha) * corr_data['positive_correlation'] + alpha * 1.0
                    corr_data['negative_correlation'] = (1 - alpha) * corr_data['negative_correlation'] + alpha * 0.0
                else:
                    corr_data['positive_correlation'] = (1 - alpha) * corr_data['positive_correlation'] + alpha * 0.0
                    corr_data['negative_correlation'] = (1 - alpha) * corr_data['negative_correlation'] + alpha * 1.0


class AdaptiveStrategySelector:
    """Intelligently select optimization strategy based on problem characteristics and performance."""
    
    def __init__(self):
        self.available_strategies = {
            'genetic_algorithm': {'complexity': 'medium', 'convergence': 'slow', 'exploration': 'high'},
            'simulated_annealing': {'complexity': 'low', 'convergence': 'medium', 'exploration': 'medium'},
            'particle_swarm': {'complexity': 'medium', 'convergence': 'fast', 'exploration': 'low'},
            'multi_objective': {'complexity': 'high', 'convergence': 'slow', 'exploration': 'high'},
            'hybrid': {'complexity': 'high', 'convergence': 'medium', 'exploration': 'high'}
        }
        
        self.strategy_performance = {}
        for strategy in self.available_strategies:
            self.strategy_performance[strategy] = StrategyPerformance(strategy)
        
        self.problem_characteristics_history = []
        self.strategy_selection_history = []
    
    def select_strategy(self, problem_characteristics: Dict[str, Any], 
                       search_state: Dict[str, Any]) -> str:
        """Select best strategy for current problem state."""
        
        # Analyze problem characteristics
        complexity = self._assess_problem_complexity(problem_characteristics)
        dimensionality = problem_characteristics.get('num_parameters', 10)
        num_objectives = problem_characteristics.get('num_objectives', 1)
        time_budget = problem_characteristics.get('time_budget', 3600)  # seconds
        
        # Score each strategy
        strategy_scores = {}
        
        for strategy_name, strategy_props in self.available_strategies.items():
            score = self._calculate_strategy_score(
                strategy_name, strategy_props, complexity, dimensionality, 
                num_objectives, time_budget, search_state
            )
            strategy_scores[strategy_name] = score
        
        # Select best strategy
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        
        # Record selection
        self.strategy_selection_history.append({
            'timestamp': time.time(),
            'selected_strategy': best_strategy,
            'problem_characteristics': problem_characteristics.copy(),
            'strategy_scores': strategy_scores.copy(),
            'search_state': search_state.copy()
        })
        
        logger.info(f"Selected strategy: {best_strategy} (score: {strategy_scores[best_strategy]:.3f})")
        return best_strategy
    
    def update_strategy_performance(self, strategy_name: str, 
                                  improvement: float, 
                                  convergence_time: float,
                                  success: bool):
        """Update performance metrics for a strategy."""
        
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name].update(improvement, convergence_time, success)
    
    def get_strategy_recommendations(self, problem_characteristics: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get ranked list of strategy recommendations."""
        
        complexity = self._assess_problem_complexity(problem_characteristics)
        dimensionality = problem_characteristics.get('num_parameters', 10)
        num_objectives = problem_characteristics.get('num_objectives', 1)
        time_budget = problem_characteristics.get('time_budget', 3600)
        
        recommendations = []
        
        for strategy_name, strategy_props in self.available_strategies.items():
            score = self._calculate_strategy_score(
                strategy_name, strategy_props, complexity, dimensionality,
                num_objectives, time_budget, {}
            )
            recommendations.append((strategy_name, score))
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def _assess_problem_complexity(self, problem_characteristics: Dict[str, Any]) -> str:
        """Assess problem complexity based on characteristics."""
        
        num_parameters = problem_characteristics.get('num_parameters', 10)
        num_objectives = problem_characteristics.get('num_objectives', 1)
        num_constraints = problem_characteristics.get('num_constraints', 0)
        parameter_types = problem_characteristics.get('parameter_types', ['continuous'])
        
        complexity_score = 0
        
        # Parameter count contribution
        if num_parameters <= 5:
            complexity_score += 1
        elif num_parameters <= 15:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # Objective count contribution
        if num_objectives == 1:
            complexity_score += 1
        elif num_objectives <= 3:
            complexity_score += 2
        else:
            complexity_score += 3
        
        # Constraint count contribution
        if num_constraints == 0:
            complexity_score += 0
        elif num_constraints <= 5:
            complexity_score += 1
        else:
            complexity_score += 2
        
        # Parameter type contribution
        if 'discrete' in parameter_types or 'categorical' in parameter_types:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score <= 4:
            return 'low'
        elif complexity_score <= 7:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_strategy_score(self, strategy_name: str, strategy_props: Dict[str, str],
                                complexity: str, dimensionality: int, num_objectives: int,
                                time_budget: float, search_state: Dict[str, Any]) -> float:
        """Calculate score for strategy given problem characteristics."""
        
        score = 0.0
        
        # Base score from strategy properties
        if strategy_props['complexity'] == complexity:
            score += 3.0
        elif (strategy_props['complexity'] == 'medium' and complexity in ['low', 'high']) or \
             (strategy_props['complexity'] == 'high' and complexity == 'medium'):
            score += 2.0
        else:
            score += 1.0
        
        # Dimensionality considerations
        if dimensionality <= 10:
            if strategy_name in ['simulated_annealing', 'particle_swarm']:
                score += 2.0
        elif dimensionality <= 20:
            if strategy_name in ['genetic_algorithm', 'hybrid']:
                score += 2.0
        else:
            if strategy_name in ['genetic_algorithm', 'multi_objective', 'hybrid']:
                score += 2.0
        
        # Objective count considerations
        if num_objectives == 1:
            if strategy_name in ['simulated_annealing', 'particle_swarm', 'genetic_algorithm']:
                score += 2.0
        else:
            if strategy_name in ['multi_objective', 'hybrid']:
                score += 3.0
            elif strategy_name == 'genetic_algorithm':
                score += 1.0
        
        # Time budget considerations
        if time_budget < 300:  # 5 minutes
            if strategy_props['convergence'] == 'fast':
                score += 2.0
        elif time_budget < 1800:  # 30 minutes
            if strategy_props['convergence'] == 'medium':
                score += 2.0
        else:
            if strategy_props['convergence'] == 'slow':
                score += 1.0
        
        # Historical performance consideration
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            if perf.total_attempts > 0:
                # Add performance bonus
                performance_bonus = perf.success_rate * 2.0 + (perf.avg_improvement / 100.0)
                score += performance_bonus
                
                # Recent performance weight
                if perf.recent_performance:
                    recent_success_rate = sum(1 for p in perf.recent_performance if p['success']) / len(perf.recent_performance)
                    score += recent_success_rate * 1.0
        
        # Search state considerations
        generation = search_state.get('generation', 0)
        if generation > 0:
            # Later in search, prefer exploitation
            if strategy_props['exploration'] == 'low':
                score += 1.0
        
        return score


class SearchSpacePruner:
    """Intelligently prune search space using predictive models and constraints."""
    
    def __init__(self):
        self.pruning_rules = []
        self.constraint_violations = defaultdict(int)
        self.parameter_ranges = {}
        self.pruning_history = []
    
    def add_pruning_rule(self, rule_func: Callable, rule_name: str, priority: int = 1):
        """Add a pruning rule."""
        self.pruning_rules.append({
            'function': rule_func,
            'name': rule_name,
            'priority': priority,
            'usage_count': 0
        })
        
        # Sort by priority
        self.pruning_rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def prune_design_space(self, design_space: Dict[str, Any], 
                          constraints: List[Any] = None,
                          historical_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prune design space based on constraints and historical data."""
        
        pruned_space = design_space.copy()
        pruning_applied = []
        
        # Apply constraint-based pruning
        if constraints:
            pruned_space, constraint_pruning = self._apply_constraint_pruning(pruned_space, constraints)
            pruning_applied.extend(constraint_pruning)
        
        # Apply historical data-based pruning
        if historical_data:
            pruned_space, historical_pruning = self._apply_historical_pruning(pruned_space, historical_data)
            pruning_applied.extend(historical_pruning)
        
        # Apply custom pruning rules
        for rule in self.pruning_rules:
            try:
                pruned_space, rule_pruning = rule['function'](pruned_space)
                rule['usage_count'] += 1
                pruning_applied.extend(rule_pruning)
            except Exception as e:
                logger.error(f"Pruning rule {rule['name']} failed: {e}")
        
        # Record pruning history
        self.pruning_history.append({
            'timestamp': time.time(),
            'original_space_size': self._calculate_space_size(design_space),
            'pruned_space_size': self._calculate_space_size(pruned_space),
            'pruning_applied': pruning_applied
        })
        
        pruning_ratio = 1.0 - (self._calculate_space_size(pruned_space) / max(1, self._calculate_space_size(design_space)))
        logger.info(f"Pruned design space by {pruning_ratio:.2%}")
        
        return pruned_space
    
    def suggest_promising_regions(self, design_space: Dict[str, Any],
                                historical_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Suggest promising regions of design space."""
        
        promising_regions = []
        
        if not historical_data:
            return promising_regions
        
        # Analyze successful designs
        successful_designs = [d for d in historical_data if d.get('success', False)]
        
        if not successful_designs:
            return promising_regions
        
        # Cluster successful designs
        parameter_clusters = self._cluster_successful_designs(successful_designs)
        
        for cluster in parameter_clusters:
            region = self._define_region_from_cluster(cluster, design_space)
            if region:
                promising_regions.append(region)
        
        return promising_regions
    
    def _apply_constraint_pruning(self, design_space: Dict[str, Any], 
                                 constraints: List[Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply constraint-based pruning."""
        
        pruned_space = design_space.copy()
        pruning_applied = []
        
        for constraint in constraints:
            # Simple constraint pruning - remove parameter values that violate constraints
            constraint_param = getattr(constraint, 'parameter', None)
            constraint_threshold = getattr(constraint, 'threshold', None)
            constraint_operator = getattr(constraint, 'operator', None)
            
            if constraint_param and constraint_param in pruned_space:
                param_range = pruned_space[constraint_param]
                
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    min_val, max_val = param_range
                    
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        if constraint_operator == '<=':
                            new_max = min(max_val, constraint_threshold)
                            if new_max != max_val:
                                pruned_space[constraint_param] = (min_val, new_max)
                                pruning_applied.append(f"Constraint pruning: {constraint_param} <= {constraint_threshold}")
                        
                        elif constraint_operator == '>=':
                            new_min = max(min_val, constraint_threshold)
                            if new_min != min_val:
                                pruned_space[constraint_param] = (new_min, max_val)
                                pruning_applied.append(f"Constraint pruning: {constraint_param} >= {constraint_threshold}")
        
        return pruned_space, pruning_applied
    
    def _apply_historical_pruning(self, design_space: Dict[str, Any],
                                 historical_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply historical data-based pruning."""
        
        pruned_space = design_space.copy()
        pruning_applied = []
        
        # Analyze parameter ranges of successful designs
        successful_designs = [d for d in historical_data if d.get('success', False)]
        
        if not successful_designs:
            return pruned_space, pruning_applied
        
        for param_name in design_space:
            if param_name.startswith('_'):
                continue
            
            # Extract parameter values from successful designs
            param_values = []
            for design in successful_designs:
                if 'parameters' in design and param_name in design['parameters']:
                    value = design['parameters'][param_name]
                    if isinstance(value, (int, float)):
                        param_values.append(value)
            
            if len(param_values) >= 3:  # Need sufficient data
                # Calculate parameter range from successful designs
                min_successful = min(param_values)
                max_successful = max(param_values)
                
                # Add some margin
                margin = (max_successful - min_successful) * 0.2
                historical_min = min_successful - margin
                historical_max = max_successful + margin
                
                # Apply to design space if it's a range
                current_range = design_space[param_name]
                if isinstance(current_range, tuple) and len(current_range) == 2:
                    space_min, space_max = current_range
                    if isinstance(space_min, (int, float)) and isinstance(space_max, (int, float)):
                        # Intersect with current range
                        new_min = max(space_min, historical_min)
                        new_max = min(space_max, historical_max)
                        
                        if new_min > space_min or new_max < space_max:
                            if isinstance(space_min, int) and isinstance(space_max, int):
                                pruned_space[param_name] = (int(new_min), int(new_max))
                            else:
                                pruned_space[param_name] = (new_min, new_max)
                            
                            pruning_applied.append(f"Historical pruning: {param_name} [{new_min:.2f}, {new_max:.2f}]")
        
        return pruned_space, pruning_applied
    
    def _cluster_successful_designs(self, successful_designs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Cluster successful designs to identify promising regions."""
        
        # Simple clustering - group designs with similar parameter values
        clusters = []
        
        for design in successful_designs:
            parameters = design.get('parameters', {})
            
            # Find closest cluster
            best_cluster = None
            best_distance = float('inf')
            
            for cluster in clusters:
                avg_distance = self._calculate_average_distance(parameters, cluster)
                if avg_distance < best_distance:
                    best_distance = avg_distance
                    best_cluster = cluster
            
            # Add to cluster if close enough, otherwise create new cluster
            if best_cluster is not None and best_distance < 0.5:  # Threshold for similarity
                best_cluster.append(design)
            else:
                clusters.append([design])
        
        return clusters
    
    def _calculate_average_distance(self, parameters: Dict[str, Any], 
                                   cluster: List[Dict[str, Any]]) -> float:
        """Calculate average distance to cluster."""
        
        if not cluster:
            return float('inf')
        
        total_distance = 0.0
        valid_comparisons = 0
        
        for design in cluster:
            cluster_params = design.get('parameters', {})
            distance = self._calculate_parameter_distance(parameters, cluster_params)
            if distance < float('inf'):
                total_distance += distance
                valid_comparisons += 1
        
        return total_distance / max(1, valid_comparisons)
    
    def _calculate_parameter_distance(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate normalized distance between parameter sets."""
        
        common_params = set(params1.keys()) & set(params2.keys())
        
        if not common_params:
            return float('inf')
        
        total_distance = 0.0
        param_count = 0
        
        for param in common_params:
            val1, val2 = params1[param], params2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize distance by parameter range
                distance = abs(val1 - val2)
                # Simple normalization - in practice would use actual parameter ranges
                normalized_distance = distance / max(abs(val1), abs(val2), 1.0)
                total_distance += normalized_distance
                param_count += 1
        
        return total_distance / max(1, param_count)
    
    def _define_region_from_cluster(self, cluster: List[Dict[str, Any]], 
                                   design_space: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Define a promising region from a cluster of successful designs."""
        
        if not cluster:
            return None
        
        region = {}
        
        # Get all parameter names
        all_params = set()
        for design in cluster:
            all_params.update(design.get('parameters', {}).keys())
        
        for param in all_params:
            # Extract parameter values from cluster
            values = []
            for design in cluster:
                if param in design.get('parameters', {}):
                    value = design['parameters'][param]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if len(values) >= 2:
                min_val = min(values)
                max_val = max(values)
                
                # Add margin
                margin = (max_val - min_val) * 0.1
                region_min = min_val - margin
                region_max = max_val + margin
                
                # Clamp to design space
                if param in design_space:
                    space_range = design_space[param]
                    if isinstance(space_range, tuple) and len(space_range) == 2:
                        space_min, space_max = space_range
                        if isinstance(space_min, (int, float)) and isinstance(space_max, (int, float)):
                            region_min = max(space_min, region_min)
                            region_max = min(space_max, region_max)
                
                region[param] = (region_min, region_max)
        
        return region if region else None
    
    def _calculate_space_size(self, design_space: Dict[str, Any]) -> float:
        """Calculate approximate size of design space."""
        
        total_size = 1.0
        
        for param_name, param_range in design_space.items():
            if param_name.startswith('_'):
                continue
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    total_size *= (max_val - min_val + 1)
            elif isinstance(param_range, (list, tuple)):
                total_size *= len(param_range)
        
        return total_size