"""
Core data types for Automation Hooks Framework.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class StrategyType(Enum):
    """Optimization strategy types"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_DESCENT = "gradient_descent"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    MULTI_OBJECTIVE = "multi_objective"
    HYBRID = "hybrid"

class ProblemComplexity(Enum):
    """Problem complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class ParameterChangeType(Enum):
    """Types of parameter changes"""
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    REFINEMENT = "refinement"
    CORRECTION = "correction"

@dataclass
class ProblemContext:
    """Context information for optimization problem"""
    model_info: Dict[str, Any] = field(default_factory=dict)
    targets: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)
    platform: Dict[str, Any] = field(default_factory=dict)
    problem_size: int = 0
    complexity_indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_info': self.model_info,
            'targets': self.targets,
            'constraints': self.constraints,
            'platform': self.platform,
            'problem_size': self.problem_size,
            'complexity_indicators': self.complexity_indicators
        }

@dataclass
class ProblemCharacteristics:
    """Comprehensive problem characteristics"""
    model_size: int = 0
    model_complexity: float = 0.0
    operator_diversity: float = 0.0
    performance_targets: Dict[str, float] = field(default_factory=dict)
    constraint_tightness: float = 0.0
    multi_objective_complexity: float = 0.0
    available_resources: Dict[str, float] = field(default_factory=dict)
    resource_pressure: float = 0.0
    design_space_size: int = 0
    variable_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_size': self.model_size,
            'model_complexity': self.model_complexity,
            'operator_diversity': self.operator_diversity,
            'performance_targets': self.performance_targets,
            'constraint_tightness': self.constraint_tightness,
            'multi_objective_complexity': self.multi_objective_complexity,
            'available_resources': self.available_resources,
            'resource_pressure': self.resource_pressure,
            'design_space_size': self.design_space_size,
            'variable_types': self.variable_types
        }

@dataclass
class StrategyDecisionRecord:
    """Record of strategy selection decision"""
    timestamp: datetime
    problem_context: ProblemContext
    selected_strategy: str
    selection_rationale: str
    problem_characteristics: ProblemCharacteristics
    available_alternatives: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    decision_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'problem_context': self.problem_context.to_dict(),
            'selected_strategy': self.selected_strategy,
            'selection_rationale': self.selection_rationale,
            'problem_characteristics': self.problem_characteristics.to_dict(),
            'available_alternatives': self.available_alternatives,
            'confidence_score': self.confidence_score,
            'decision_id': self.decision_id
        }

@dataclass
class PerformanceMetrics:
    """Performance metrics for strategy outcomes"""
    throughput: float = 0.0
    latency: float = 0.0
    power: float = 0.0
    area: float = 0.0
    efficiency: float = 0.0
    convergence_time: float = 0.0
    solution_quality: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'throughput': self.throughput,
            'latency': self.latency,
            'power': self.power,
            'area': self.area,
            'efficiency': self.efficiency,
            'convergence_time': self.convergence_time,
            'solution_quality': self.solution_quality
        }

@dataclass
class StrategyOutcomeRecord:
    """Record of strategy execution outcome"""
    timestamp: datetime
    strategy_id: str
    performance_metrics: PerformanceMetrics
    optimization_success: bool = False
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'performance_metrics': self.performance_metrics.to_dict(),
            'optimization_success': self.optimization_success,
            'convergence_metrics': self.convergence_metrics,
            'quality_metrics': self.quality_metrics,
            'execution_time': self.execution_time
        }

@dataclass
class EffectivenessReport:
    """Strategy effectiveness analysis report"""
    success_rates: Dict[str, float] = field(default_factory=dict)
    performance_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    context_sensitivity: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence_intervals: Dict[str, tuple] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success_rates': self.success_rates,
            'performance_comparison': self.performance_comparison,
            'context_sensitivity': self.context_sensitivity,
            'recommendations': self.recommendations,
            'confidence_intervals': {k: list(v) for k, v in self.confidence_intervals.items()}
        }

@dataclass
class ParameterChangeRecord:
    """Record of parameter changes during optimization"""
    timestamp: datetime
    parameter_changes: Dict[str, Any]
    change_context: Dict[str, Any] = field(default_factory=dict)
    change_magnitude: float = 0.0
    change_type: ParameterChangeType = ParameterChangeType.EXPLORATION
    iteration_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'parameter_changes': self.parameter_changes,
            'change_context': self.change_context,
            'change_magnitude': self.change_magnitude,
            'change_type': self.change_type.value,
            'iteration_number': self.iteration_number
        }

@dataclass
class ImpactMeasurement:
    """Measurement of parameter impact"""
    magnitude: float = 0.0
    direction: str = "neutral"  # positive, negative, neutral
    confidence: float = 0.0
    statistical_significance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'magnitude': self.magnitude,
            'direction': self.direction,
            'confidence': self.confidence,
            'statistical_significance': self.statistical_significance
        }

@dataclass
class InteractionEffect:
    """Parameter interaction effect"""
    parameters: List[str] = field(default_factory=list)
    strength: float = 0.0
    type: str = "synergistic"  # synergistic, antagonistic, neutral
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameters': self.parameters,
            'strength': self.strength,
            'type': self.type,
            'confidence': self.confidence
        }

@dataclass
class ImpactAnalysis:
    """Comprehensive impact analysis"""
    direct_impact: Dict[str, ImpactMeasurement] = field(default_factory=dict)
    interaction_effects: List[InteractionEffect] = field(default_factory=list)
    sensitivity_coefficients: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direct_impact': {k: v.to_dict() for k, v in self.direct_impact.items()},
            'interaction_effects': [e.to_dict() for e in self.interaction_effects],
            'sensitivity_coefficients': self.sensitivity_coefficients,
            'statistical_significance': self.statistical_significance
        }

@dataclass
class SensitivityInsight:
    """Insight from sensitivity analysis"""
    type: str = ""
    parameter: str = ""
    parameters: List[str] = field(default_factory=list)
    impact_magnitude: float = 0.0
    interaction_strength: float = 0.0
    recommendation: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'parameter': self.parameter,
            'parameters': self.parameters,
            'impact_magnitude': self.impact_magnitude,
            'interaction_strength': self.interaction_strength,
            'recommendation': self.recommendation,
            'confidence': self.confidence
        }

@dataclass
class SensitivityData:
    """Sensitivity analysis data collection"""
    parameters: List[str] = field(default_factory=list)
    measurements: List[ParameterChangeRecord] = field(default_factory=list)
    impacts: List[ImpactAnalysis] = field(default_factory=list)
    insights: List[SensitivityInsight] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameters': self.parameters,
            'measurements': [m.to_dict() for m in self.measurements],
            'impacts': [i.to_dict() for i in self.impacts],
            'insights': [i.to_dict() for i in self.insights]
        }

@dataclass
class DesignSpaceCharacteristics:
    """Design space characteristics"""
    dimensionality: int = 0
    variable_types: Dict[str, int] = field(default_factory=dict)
    space_size: int = 0
    constraint_density: float = 0.0
    linearity_measure: float = 0.0
    separability_measure: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dimensionality': self.dimensionality,
            'variable_types': self.variable_types,
            'space_size': self.space_size,
            'constraint_density': self.constraint_density,
            'linearity_measure': self.linearity_measure,
            'separability_measure': self.separability_measure
        }

@dataclass
class ObjectiveCharacteristics:
    """Objective function characteristics"""
    num_objectives: int = 1
    conflicting_objectives: bool = False
    objective_scales: Dict[str, float] = field(default_factory=dict)
    noise_level: float = 0.0
    evaluation_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_objectives': self.num_objectives,
            'conflicting_objectives': self.conflicting_objectives,
            'objective_scales': self.objective_scales,
            'noise_level': self.noise_level,
            'evaluation_cost': self.evaluation_cost
        }

@dataclass
class ProblemType:
    """Classified problem type"""
    type_name: str = ""
    complexity: ProblemComplexity = ProblemComplexity.MODERATE
    confidence: float = 0.0
    explanation: str = ""
    recommended_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type_name': self.type_name,
            'complexity': self.complexity.value,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'recommended_strategies': self.recommended_strategies
        }