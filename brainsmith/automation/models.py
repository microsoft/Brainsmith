"""
Data models for the Automation Framework
Defines core data structures used throughout the automation system.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import uuid

# Import from other frameworks
try:
    from ..selection.models import RankedSolution, SelectionCriteria
    from ..analysis.models import PerformanceAnalysis, BenchmarkResult
    from ..dse.models import OptimizationResult, ParetoFront
except ImportError:
    # Fallback definitions if other frameworks not available
    @dataclass
    class RankedSolution:
        solution: Any
        rank: int
        score: float
    
    @dataclass
    class SelectionCriteria:
        objectives: List[str]
        weights: Dict[str, float]
    
    @dataclass
    class PerformanceAnalysis:
        analysis_id: str
        timestamp: datetime
    
    @dataclass
    class BenchmarkResult:
        design_id: str
        benchmark_category: str
    
    @dataclass
    class OptimizationResult:
        pareto_solutions: List[Any]
        convergence_history: List[float]
    
    @dataclass
    class ParetoFront:
        solutions: List[Any]


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStep(Enum):
    """Steps in automated workflow."""
    INITIALIZATION = "initialization"
    DSE_OPTIMIZATION = "dse_optimization"
    SOLUTION_SELECTION = "solution_selection"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    BENCHMARKING = "benchmarking"
    RECOMMENDATION = "recommendation"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


class RecommendationCategory(Enum):
    """Categories of design recommendations."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    POWER_OPTIMIZATION = "power_optimization"
    TIMING_IMPROVEMENT = "timing_improvement"
    AREA_REDUCTION = "area_reduction"
    ALGORITHM_SELECTION = "algorithm_selection"
    PARAMETER_TUNING = "parameter_tuning"
    ARCHITECTURE_CHANGE = "architecture_change"


class RecommendationConfidence(Enum):
    """Confidence levels for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXPERIMENTAL = "experimental"


class ComponentStatus(Enum):
    """Status of automation components."""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class DesignTarget:
    """Target specifications for design optimization."""
    application_type: str
    performance_targets: Dict[str, float]
    constraints: Dict[str, Union[float, bool, str]]
    optimization_objectives: List[str]
    priority_weights: Dict[str, float] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate design target specification."""
        if not self.application_type:
            return False
        if not self.performance_targets:
            return False
        if not self.optimization_objectives:
            return False
        return True


@dataclass
class WorkflowConfiguration:
    """Configuration for automated workflow."""
    optimization_budget: int = 1800  # seconds
    quality_threshold: float = 0.75
    enable_learning: bool = True
    max_iterations: int = 50
    convergence_tolerance: float = 0.01
    parallel_execution: bool = True
    max_workers: int = 4
    save_intermediate_results: bool = True
    validation_enabled: bool = True
    
    def validate(self) -> bool:
        """Validate workflow configuration."""
        if self.optimization_budget <= 0:
            return False
        if not (0 < self.quality_threshold <= 1):
            return False
        if self.max_iterations <= 0:
            return False
        if self.convergence_tolerance <= 0:
            return False
        return True


@dataclass
class OptimizationJob:
    """Job specification for optimization."""
    job_id: str
    design_target: DesignTarget
    workflow_config: WorkflowConfiguration
    timestamp: datetime = field(default_factory=datetime.now)
    status: WorkflowStatus = WorkflowStatus.PENDING
    priority: int = 1
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate job ID if not provided."""
        if not self.job_id:
            self.job_id = f"job_{uuid.uuid4().hex[:8]}"


@dataclass
class DesignRecommendation:
    """Design recommendation with rationale."""
    category: RecommendationCategory
    confidence: RecommendationConfidence
    title: str
    description: str
    rationale: str
    impact_estimate: Dict[str, float]
    implementation_effort: str
    priority: int = 1
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_impact_summary(self) -> str:
        """Get summary of expected impact."""
        if not self.impact_estimate:
            return "Impact not quantified"
        
        impacts = []
        for metric, change in self.impact_estimate.items():
            direction = "increase" if change > 0 else "decrease"
            impacts.append(f"{abs(change):.1f}% {direction} in {metric}")
        
        return "; ".join(impacts)


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    overall_score: float
    completeness: float
    accuracy: float
    consistency: float
    reliability: float
    confidence: float
    detailed_scores: Dict[str, float] = field(default_factory=dict)
    
    def get_quality_level(self) -> QualityLevel:
        """Get overall quality level."""
        if self.overall_score >= 0.9:
            return QualityLevel.EXCELLENT
        elif self.overall_score >= 0.8:
            return QualityLevel.GOOD
        elif self.overall_score >= 0.6:
            return QualityLevel.ACCEPTABLE
        elif self.overall_score >= 0.4:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILED


@dataclass
class ValidationResult:
    """Result of automated validation."""
    validation_id: str
    timestamp: datetime
    passed: bool
    quality_metrics: QualityMetrics
    validation_tests: List[Dict[str, Any]]
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def get_pass_rate(self) -> float:
        """Get percentage of tests passed."""
        if not self.validation_tests:
            return 0.0
        
        passed_tests = sum(1 for test in self.validation_tests if test.get('passed', False))
        return (passed_tests / len(self.validation_tests)) * 100.0


@dataclass
class AutomationMetrics:
    """Metrics for automation performance."""
    total_runtime: float
    step_times: Dict[WorkflowStep, float]
    resource_utilization: Dict[str, float]
    quality_scores: Dict[str, float]
    success_rate: float
    efficiency_score: float
    user_satisfaction: Optional[float] = None
    
    def get_bottleneck_step(self) -> Optional[WorkflowStep]:
        """Identify the slowest workflow step."""
        if not self.step_times:
            return None
        
        return max(self.step_times.keys(), key=lambda step: self.step_times[step])


@dataclass
class HistoricalPatterns:
    """Historical patterns for learning."""
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    success_rate: float
    context: Dict[str, Any]
    last_observed: datetime
    
    def is_reliable(self, min_frequency: int = 5, min_confidence: float = 0.7) -> bool:
        """Check if pattern is reliable for use."""
        return self.frequency >= min_frequency and self.confidence >= min_confidence


@dataclass
class AdaptiveParameters:
    """Parameters that adapt based on learning."""
    parameter_name: str
    current_value: Union[float, int, str]
    learned_value: Union[float, int, str]
    confidence: float
    adaptation_history: List[Tuple[datetime, Any]] = field(default_factory=list)
    
    def should_adapt(self, confidence_threshold: float = 0.8) -> bool:
        """Check if parameter should be adapted."""
        return self.confidence >= confidence_threshold


@dataclass
class AutomationContext:
    """Context for automation execution."""
    job: OptimizationJob
    current_step: WorkflowStep
    step_results: Dict[WorkflowStep, Any] = field(default_factory=dict)
    component_status: Dict[str, ComponentStatus] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step_result(self, step: WorkflowStep, result: Any) -> None:
        """Add result for a workflow step."""
        self.step_results[step] = result
        self.execution_history.append({
            'step': step.value,
            'timestamp': datetime.now(),
            'status': 'completed'
        })
    
    def get_step_result(self, step: WorkflowStep) -> Optional[Any]:
        """Get result for a specific step."""
        return self.step_results.get(step)


@dataclass
class WorkflowDefinition:
    """Definition of an automated workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    dependencies: Dict[WorkflowStep, List[WorkflowStep]] = field(default_factory=dict)
    step_configs: Dict[WorkflowStep, Dict[str, Any]] = field(default_factory=dict)
    timeout_seconds: int = 3600
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    
    def validate_dependencies(self) -> bool:
        """Validate that dependencies form a valid DAG."""
        # Simple cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(step):
            visited.add(step)
            rec_stack.add(step)
            
            for dependent in self.dependencies.get(step, []):
                if dependent not in visited:
                    if has_cycle(dependent):
                        return True
                elif dependent in rec_stack:
                    return True
            
            rec_stack.remove(step)
            return False
        
        for step in self.steps:
            if step not in visited:
                if has_cycle(step):
                    return False
        
        return True


@dataclass
class AutomationResult:
    """Result of automated workflow execution."""
    job_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Core results
    optimization_result: Optional[OptimizationResult] = None
    selected_solutions: List[RankedSolution] = field(default_factory=list)
    performance_analysis: Optional[PerformanceAnalysis] = None
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    recommendations: List[DesignRecommendation] = field(default_factory=list)
    
    # Quality and metrics
    quality_metrics: Optional[QualityMetrics] = None
    automation_metrics: Optional[AutomationMetrics] = None
    validation_result: Optional[ValidationResult] = None
    
    # Additional data
    learned_patterns: List[HistoricalPatterns] = field(default_factory=list)
    adapted_parameters: List[AdaptiveParameters] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)
    error_log: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def best_solution(self) -> Optional[RankedSolution]:
        """Get best selected solution."""
        if self.selected_solutions:
            return min(self.selected_solutions, key=lambda s: s.rank)
        return None
    
    def get_success_summary(self) -> Dict[str, Any]:
        """Get summary of workflow success."""
        return {
            'status': self.status.value,
            'duration': self.duration,
            'solutions_found': len(self.selected_solutions),
            'recommendations_generated': len(self.recommendations),
            'quality_score': self.quality_metrics.overall_score if self.quality_metrics else 0.0,
            'success': self.status == WorkflowStatus.COMPLETED
        }


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    report_id: str
    timestamp: datetime
    job_id: str
    
    # Quality assessments
    optimization_quality: QualityMetrics
    selection_quality: QualityMetrics
    analysis_quality: QualityMetrics
    overall_quality: QualityMetrics
    
    # Validation results
    validation_results: List[ValidationResult]
    
    # Issues and recommendations
    quality_issues: List[str] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)
    
    def get_overall_grade(self) -> str:
        """Get overall quality grade."""
        score = self.overall_quality.overall_score
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"


# Type aliases for better code readability
WorkflowResults = Dict[WorkflowStep, Any]
ComponentStatusMap = Dict[str, ComponentStatus]
QualityScores = Dict[str, float]
RecommendationList = List[DesignRecommendation]
PatternDatabase = Dict[str, HistoricalPatterns]