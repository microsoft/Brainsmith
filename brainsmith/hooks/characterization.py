"""
Problem Characterization System

Characterize and classify optimization problems for strategy selection.
Provides automated problem analysis and strategy recommendations.
"""

import logging
from typing import Dict, Any, List, Optional
import math

from .types import (
    ProblemCharacteristics,
    ProblemType,
    DesignSpaceCharacteristics,
    ObjectiveCharacteristics,
    ProblemComplexity
)

logger = logging.getLogger(__name__)

class ProblemFeatureExtractor:
    """Extract features from optimization problems for classification"""
    
    def __init__(self):
        self.feature_weights = self._load_feature_weights()
    
    def extract_features(self, characteristics: ProblemCharacteristics) -> Dict[str, float]:
        """Extract normalized features for problem classification"""
        
        features = {}
        
        # Model complexity features
        features['model_size_norm'] = min(characteristics.model_size / 10_000_000, 1.0)
        features['model_complexity'] = characteristics.model_complexity
        features['operator_diversity'] = characteristics.operator_diversity
        
        # Objective features
        features['num_objectives'] = min(len(characteristics.performance_targets), 5) / 5.0
        features['constraint_tightness'] = characteristics.constraint_tightness
        features['multi_objective_complexity'] = characteristics.multi_objective_complexity
        
        # Resource features
        features['resource_pressure'] = characteristics.resource_pressure
        features['design_space_size_norm'] = min(math.log10(max(characteristics.design_space_size, 1)) / 10, 1.0)
        
        # Variable type features
        total_vars = sum(characteristics.variable_types.values()) if characteristics.variable_types else 1
        features['continuous_ratio'] = characteristics.variable_types.get('continuous', 0) / total_vars
        features['discrete_ratio'] = characteristics.variable_types.get('discrete', 0) / total_vars
        features['integer_ratio'] = characteristics.variable_types.get('integer', 0) / total_vars
        
        return features
    
    def _load_feature_weights(self) -> Dict[str, float]:
        """Load feature importance weights"""
        return {
            'model_size_norm': 0.15,
            'model_complexity': 0.20,
            'operator_diversity': 0.10,
            'num_objectives': 0.15,
            'constraint_tightness': 0.15,
            'multi_objective_complexity': 0.10,
            'resource_pressure': 0.10,
            'design_space_size_norm': 0.05
        }

class ProblemClassifier:
    """Classify optimization problems based on characteristics"""
    
    def __init__(self):
        self.problem_templates = self._load_problem_templates()
        self.classification_rules = self._load_classification_rules()
    
    def classify(self, features: Dict[str, float]) -> ProblemType:
        """Classify problem type based on extracted features"""
        
        # Calculate similarity to each problem template
        similarities = {}
        for template_name, template_features in self.problem_templates.items():
            similarity = self._calculate_similarity(features, template_features)
            similarities[template_name] = similarity
        
        # Find best match
        best_match = max(similarities.items(), key=lambda x: x[1])
        problem_type_name = best_match[0]
        confidence = best_match[1]
        
        # Determine complexity
        complexity = self._determine_complexity(features)
        
        # Generate explanation
        explanation = self._generate_explanation(problem_type_name, features)
        
        # Get recommended strategies
        recommended_strategies = self._get_recommended_strategies(problem_type_name, complexity)
        
        return ProblemType(
            type_name=problem_type_name,
            complexity=complexity,
            confidence=confidence,
            explanation=explanation,
            recommended_strategies=recommended_strategies
        )
    
    def get_confidence(self, features: Dict[str, float]) -> float:
        """Get classification confidence"""
        # Calculate confidence based on feature clarity
        confidence_factors = []
        
        # Clear model complexity indication
        if features.get('model_complexity', 0) > 0.7 or features.get('model_complexity', 0) < 0.3:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Clear resource pressure indication
        if features.get('resource_pressure', 0) > 0.7 or features.get('resource_pressure', 0) < 0.3:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Clear multi-objective indication
        if features.get('num_objectives', 0) > 0.4 or features.get('num_objectives', 0) < 0.2:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def explain_classification(self, features: Dict[str, float]) -> str:
        """Explain classification reasoning"""
        explanations = []
        
        # Model complexity explanation
        complexity = features.get('model_complexity', 0)
        if complexity > 0.7:
            explanations.append("High model complexity detected")
        elif complexity < 0.3:
            explanations.append("Low model complexity detected")
        
        # Multi-objective explanation
        num_obj = features.get('num_objectives', 0)
        if num_obj > 0.4:
            explanations.append("Multi-objective optimization problem")
        
        # Resource pressure explanation
        pressure = features.get('resource_pressure', 0)
        if pressure > 0.7:
            explanations.append("High resource pressure constraints")
        elif pressure < 0.3:
            explanations.append("Relaxed resource constraints")
        
        # Design space explanation
        space_size = features.get('design_space_size_norm', 0)
        if space_size > 0.7:
            explanations.append("Large design space")
        elif space_size < 0.3:
            explanations.append("Small design space")
        
        return "; ".join(explanations) if explanations else "Standard optimization problem"
    
    def _load_problem_templates(self) -> Dict[str, Dict[str, float]]:
        """Load problem type templates"""
        return {
            'simple_single_objective': {
                'model_complexity': 0.2,
                'num_objectives': 0.2,
                'constraint_tightness': 0.3,
                'resource_pressure': 0.2,
                'design_space_size_norm': 0.3
            },
            'complex_single_objective': {
                'model_complexity': 0.8,
                'num_objectives': 0.2,
                'constraint_tightness': 0.6,
                'resource_pressure': 0.7,
                'design_space_size_norm': 0.8
            },
            'multi_objective_moderate': {
                'model_complexity': 0.5,
                'num_objectives': 0.6,
                'constraint_tightness': 0.5,
                'resource_pressure': 0.5,
                'design_space_size_norm': 0.6
            },
            'multi_objective_complex': {
                'model_complexity': 0.8,
                'num_objectives': 0.8,
                'constraint_tightness': 0.7,
                'resource_pressure': 0.8,
                'design_space_size_norm': 0.9
            },
            'resource_constrained': {
                'model_complexity': 0.6,
                'num_objectives': 0.4,
                'constraint_tightness': 0.9,
                'resource_pressure': 0.9,
                'design_space_size_norm': 0.5
            },
            'large_scale': {
                'model_complexity': 0.7,
                'num_objectives': 0.3,
                'constraint_tightness': 0.4,
                'resource_pressure': 0.6,
                'design_space_size_norm': 1.0
            }
        }
    
    def _load_classification_rules(self) -> Dict[str, Any]:
        """Load classification rules"""
        return {
            'complexity_thresholds': {
                'simple': 0.3,
                'moderate': 0.6,
                'complex': 0.8
            },
            'multi_objective_threshold': 0.4,
            'resource_pressure_threshold': 0.7,
            'large_scale_threshold': 0.8
        }
    
    def _calculate_similarity(self, 
                            features: Dict[str, float],
                            template: Dict[str, float]) -> float:
        """Calculate similarity between features and template"""
        
        total_similarity = 0.0
        feature_count = 0
        
        for feature_name, template_value in template.items():
            if feature_name in features:
                feature_value = features[feature_name]
                # Use inverse squared difference as similarity measure
                diff = abs(feature_value - template_value)
                similarity = 1.0 / (1.0 + diff**2)
                total_similarity += similarity
                feature_count += 1
        
        return total_similarity / feature_count if feature_count > 0 else 0.0
    
    def _determine_complexity(self, features: Dict[str, float]) -> ProblemComplexity:
        """Determine problem complexity level"""
        
        complexity_indicators = [
            features.get('model_complexity', 0),
            features.get('multi_objective_complexity', 0),
            features.get('constraint_tightness', 0),
            features.get('resource_pressure', 0),
            features.get('design_space_size_norm', 0)
        ]
        
        avg_complexity = sum(complexity_indicators) / len(complexity_indicators)
        
        if avg_complexity < 0.3:
            return ProblemComplexity.SIMPLE
        elif avg_complexity < 0.6:
            return ProblemComplexity.MODERATE
        elif avg_complexity < 0.8:
            return ProblemComplexity.COMPLEX
        else:
            return ProblemComplexity.VERY_COMPLEX
    
    def _generate_explanation(self, problem_type: str, features: Dict[str, float]) -> str:
        """Generate explanation for classification"""
        
        explanations = {
            'simple_single_objective': "Simple single-objective problem with low complexity",
            'complex_single_objective': "Complex single-objective problem requiring sophisticated optimization",
            'multi_objective_moderate': "Multi-objective problem with moderate complexity",
            'multi_objective_complex': "Complex multi-objective problem with conflicting objectives",
            'resource_constrained': "Resource-constrained problem requiring efficient solutions",
            'large_scale': "Large-scale problem with extensive design space"
        }
        
        base_explanation = explanations.get(problem_type, "Standard optimization problem")
        
        # Add specific feature insights
        insights = []
        if features.get('model_complexity', 0) > 0.7:
            insights.append("high model complexity")
        if features.get('num_objectives', 0) > 0.4:
            insights.append("multiple objectives")
        if features.get('resource_pressure', 0) > 0.7:
            insights.append("tight resource constraints")
        
        if insights:
            return f"{base_explanation} characterized by {', '.join(insights)}"
        else:
            return base_explanation
    
    def _get_recommended_strategies(self, 
                                  problem_type: str,
                                  complexity: ProblemComplexity) -> List[str]:
        """Get recommended optimization strategies for problem type"""
        
        strategy_mapping = {
            'simple_single_objective': ['gradient_descent', 'simulated_annealing', 'random_search'],
            'complex_single_objective': ['genetic_algorithm', 'bayesian_optimization', 'hybrid'],
            'multi_objective_moderate': ['multi_objective', 'genetic_algorithm', 'particle_swarm'],
            'multi_objective_complex': ['multi_objective', 'hybrid', 'genetic_algorithm'],
            'resource_constrained': ['simulated_annealing', 'genetic_algorithm', 'gradient_descent'],
            'large_scale': ['genetic_algorithm', 'particle_swarm', 'hybrid']
        }
        
        base_strategies = strategy_mapping.get(problem_type, ['genetic_algorithm', 'simulated_annealing'])
        
        # Adjust based on complexity
        if complexity == ProblemComplexity.VERY_COMPLEX:
            # Prefer more sophisticated strategies for very complex problems
            if 'hybrid' not in base_strategies:
                base_strategies.insert(0, 'hybrid')
            if 'genetic_algorithm' not in base_strategies:
                base_strategies.insert(1, 'genetic_algorithm')
        elif complexity == ProblemComplexity.SIMPLE:
            # Prefer simpler strategies for simple problems
            simple_strategies = ['gradient_descent', 'simulated_annealing', 'random_search']
            base_strategies = [s for s in base_strategies if s in simple_strategies] + \
                            [s for s in simple_strategies if s not in base_strategies]
        
        return base_strategies[:5]  # Return top 5 recommendations

class StrategyRecommender:
    """Recommend optimization strategies based on problem classification"""
    
    def __init__(self):
        self.strategy_database = self._load_strategy_database()
        self.performance_history = self._load_performance_history()
    
    def recommend(self, 
                 problem_type: ProblemType,
                 top_k: int = 5,
                 include_reasoning: bool = True) -> List[Dict[str, Any]]:
        """Recommend optimization strategies for problem type"""
        
        recommendations = []
        
        # Get base recommendations from problem type
        base_strategies = problem_type.recommended_strategies
        
        # Score each strategy
        for strategy in base_strategies[:top_k]:
            score = self._score_strategy(strategy, problem_type)
            
            recommendation = {
                'strategy': strategy,
                'score': score,
                'confidence': problem_type.confidence
            }
            
            if include_reasoning:
                recommendation['reasoning'] = self._generate_reasoning(strategy, problem_type)
            
            recommendations.append(recommendation)
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def _load_strategy_database(self) -> Dict[str, Dict[str, Any]]:
        """Load strategy information database"""
        return {
            'genetic_algorithm': {
                'complexity_preference': 'high',
                'multi_objective_capable': True,
                'scalability': 'excellent',
                'convergence_speed': 'moderate'
            },
            'simulated_annealing': {
                'complexity_preference': 'moderate',
                'multi_objective_capable': False,
                'scalability': 'good',
                'convergence_speed': 'moderate'
            },
            'particle_swarm': {
                'complexity_preference': 'moderate',
                'multi_objective_capable': True,
                'scalability': 'good',
                'convergence_speed': 'fast'
            },
            'gradient_descent': {
                'complexity_preference': 'low',
                'multi_objective_capable': False,
                'scalability': 'excellent',
                'convergence_speed': 'very_fast'
            },
            'bayesian_optimization': {
                'complexity_preference': 'high',
                'multi_objective_capable': True,
                'scalability': 'moderate',
                'convergence_speed': 'slow'
            },
            'multi_objective': {
                'complexity_preference': 'high',
                'multi_objective_capable': True,
                'scalability': 'good',
                'convergence_speed': 'moderate'
            },
            'hybrid': {
                'complexity_preference': 'very_high',
                'multi_objective_capable': True,
                'scalability': 'excellent',
                'convergence_speed': 'variable'
            },
            'random_search': {
                'complexity_preference': 'low',
                'multi_objective_capable': False,
                'scalability': 'excellent',
                'convergence_speed': 'slow'
            }
        }
    
    def _load_performance_history(self) -> Dict[str, Dict[str, float]]:
        """Load historical performance data"""
        # Placeholder for actual performance history
        return {
            'genetic_algorithm': {'avg_quality': 0.85, 'success_rate': 0.88},
            'simulated_annealing': {'avg_quality': 0.78, 'success_rate': 0.82},
            'particle_swarm': {'avg_quality': 0.75, 'success_rate': 0.80},
            'gradient_descent': {'avg_quality': 0.70, 'success_rate': 0.90},
            'bayesian_optimization': {'avg_quality': 0.88, 'success_rate': 0.75},
            'multi_objective': {'avg_quality': 0.80, 'success_rate': 0.78},
            'hybrid': {'avg_quality': 0.90, 'success_rate': 0.85},
            'random_search': {'avg_quality': 0.60, 'success_rate': 0.95}
        }
    
    def _score_strategy(self, strategy: str, problem_type: ProblemType) -> float:
        """Score strategy suitability for problem type"""
        
        strategy_info = self.strategy_database.get(strategy, {})
        performance_info = self.performance_history.get(strategy, {})
        
        score = 0.0
        
        # Complexity match score
        complexity_pref = strategy_info.get('complexity_preference', 'moderate')
        complexity_score = self._compute_complexity_match(complexity_pref, problem_type.complexity)
        score += 0.3 * complexity_score
        
        # Multi-objective capability score
        if len(problem_type.recommended_strategies) > 1:  # Proxy for multi-objective
            if strategy_info.get('multi_objective_capable', False):
                score += 0.2
        else:
            score += 0.1  # Small bonus for single-objective
        
        # Historical performance score
        avg_quality = performance_info.get('avg_quality', 0.5)
        success_rate = performance_info.get('success_rate', 0.5)
        score += 0.3 * avg_quality + 0.2 * success_rate
        
        return min(score, 1.0)
    
    def _compute_complexity_match(self, 
                                complexity_pref: str,
                                problem_complexity: ProblemComplexity) -> float:
        """Compute how well strategy complexity preference matches problem"""
        
        complexity_mapping = {
            'low': 1, 'moderate': 2, 'high': 3, 'very_high': 4
        }
        
        problem_mapping = {
            ProblemComplexity.SIMPLE: 1,
            ProblemComplexity.MODERATE: 2,
            ProblemComplexity.COMPLEX: 3,
            ProblemComplexity.VERY_COMPLEX: 4
        }
        
        strategy_level = complexity_mapping.get(complexity_pref, 2)
        problem_level = problem_mapping.get(problem_complexity, 2)
        
        # Perfect match gets score 1.0, larger differences get lower scores
        diff = abs(strategy_level - problem_level)
        return max(0.0, 1.0 - diff * 0.25)
    
    def _generate_reasoning(self, strategy: str, problem_type: ProblemType) -> str:
        """Generate reasoning for strategy recommendation"""
        
        strategy_info = self.strategy_database.get(strategy, {})
        
        reasoning_parts = []
        
        # Complexity reasoning
        complexity_pref = strategy_info.get('complexity_preference', 'moderate')
        reasoning_parts.append(f"Suitable for {complexity_pref} complexity problems")
        
        # Multi-objective reasoning
        if strategy_info.get('multi_objective_capable', False):
            reasoning_parts.append("Capable of handling multiple objectives")
        
        # Scalability reasoning
        scalability = strategy_info.get('scalability', 'moderate')
        reasoning_parts.append(f"{scalability} scalability")
        
        # Performance reasoning
        performance_info = self.performance_history.get(strategy, {})
        avg_quality = performance_info.get('avg_quality', 0.5)
        if avg_quality > 0.8:
            reasoning_parts.append("Strong historical performance")
        elif avg_quality > 0.6:
            reasoning_parts.append("Good historical performance")
        
        return "; ".join(reasoning_parts)

class ProblemCharacterizer:
    """Characterize and classify optimization problems for strategy selection"""
    
    def __init__(self):
        self.feature_extractor = ProblemFeatureExtractor()
        self.classifier = ProblemClassifier()
        self.strategy_recommender = StrategyRecommender()
    
    def capture_problem_characteristics(self, problem: Dict[str, Any]) -> ProblemCharacteristics:
        """Capture comprehensive problem characteristics"""
        
        characteristics = ProblemCharacteristics()
        
        # Design space characteristics
        characteristics.design_space_size = problem.get('design_space_size', 0)
        characteristics.variable_types = problem.get('variable_types', {})
        
        # Model characteristics
        model_info = problem.get('model', {})
        characteristics.model_size = model_info.get('parameter_count', 0)
        characteristics.model_complexity = self._compute_model_complexity(model_info)
        characteristics.operator_diversity = self._compute_operator_diversity(model_info)
        
        # Objective and constraint characteristics
        characteristics.performance_targets = problem.get('targets', {})
        characteristics.constraint_tightness = self._compute_constraint_tightness(problem.get('constraints', {}))
        characteristics.multi_objective_complexity = self._compute_mo_complexity(characteristics.performance_targets)
        
        # Resource characteristics
        characteristics.available_resources = problem.get('resources', {})
        characteristics.resource_pressure = self._compute_resource_pressure(problem)
        
        logger.debug("Captured problem characteristics")
        return characteristics
    
    def classify_problem_type(self, characteristics: ProblemCharacteristics) -> ProblemType:
        """Classify problem type based on characteristics"""
        
        # Extract classification features
        features = self.feature_extractor.extract_features(characteristics)
        
        # Classify using trained classifier
        problem_type = self.classifier.classify(features)
        
        logger.info(f"Classified problem as: {problem_type.type_name} (confidence: {problem_type.confidence:.2f})")
        return problem_type
    
    def recommend_strategies(self, problem_type: ProblemType) -> List[str]:
        """Recommend optimization strategies for problem type"""
        
        recommendations = self.strategy_recommender.recommend(
            problem_type,
            top_k=5,
            include_reasoning=True
        )
        
        strategy_names = [rec['strategy'] for rec in recommendations]
        logger.info(f"Recommended strategies: {strategy_names}")
        return strategy_names
    
    def _compute_model_complexity(self, model_info: Dict[str, Any]) -> float:
        """Compute normalized model complexity score"""
        complexity = 0.0
        
        # Parameter count contribution
        param_count = model_info.get('parameter_count', 0)
        complexity += min(param_count / 10_000_000, 1.0) * 0.4
        
        # Layer count contribution
        layer_count = model_info.get('layer_count', 0)
        complexity += min(layer_count / 100, 1.0) * 0.3
        
        # Operator diversity contribution
        op_types = len(model_info.get('operator_types', []))
        complexity += min(op_types / 20, 1.0) * 0.3
        
        return complexity
    
    def _compute_operator_diversity(self, model_info: Dict[str, Any]) -> float:
        """Compute operator diversity score"""
        op_types = model_info.get('operator_types', [])
        if not op_types:
            return 0.0
        
        unique_types = len(set(op_types))
        return min(unique_types / 15.0, 1.0)
    
    def _compute_constraint_tightness(self, constraints: Dict[str, Any]) -> float:
        """Compute constraint tightness score"""
        if not constraints:
            return 0.0
        
        # Simple heuristic based on constraint values
        tightness_scores = []
        for name, value in constraints.items():
            if isinstance(value, (int, float)):
                # Assume normalized constraints where smaller values = tighter
                if 'max' in name.lower() or 'limit' in name.lower():
                    tightness_scores.append(1.0 - min(value, 1.0))
                else:
                    tightness_scores.append(min(value, 1.0))
        
        return sum(tightness_scores) / len(tightness_scores) if tightness_scores else 0.5
    
    def _compute_mo_complexity(self, targets: Dict[str, Any]) -> float:
        """Compute multi-objective complexity"""
        num_objectives = len(targets)
        
        if num_objectives <= 1:
            return 0.0
        elif num_objectives == 2:
            return 0.4
        elif num_objectives == 3:
            return 0.7
        else:
            return min(1.0, 0.7 + (num_objectives - 3) * 0.1)
    
    def _compute_resource_pressure(self, problem: Dict[str, Any]) -> float:
        """Compute resource pressure score"""
        targets = problem.get('targets', {})
        constraints = problem.get('constraints', {})
        resources = problem.get('resources', {})
        
        if not targets or not constraints:
            return 0.5
        
        # Heuristic: high targets + tight constraints + limited resources = high pressure
        target_intensity = len(targets) / 5.0  # Normalize assuming max 5 targets
        constraint_tightness = self._compute_constraint_tightness(constraints)
        resource_limitation = 1.0 - (len(resources) / 10.0)  # Assume max 10 resource types
        
        pressure = (target_intensity + constraint_tightness + resource_limitation) / 3.0
        return min(pressure, 1.0)