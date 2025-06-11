"""
DSE Type Definitions

Data structures and type definitions for design space exploration functionality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum

# Import key classes to re-export them
try:
    from .design_space import DesignPoint, DesignSpace, ParameterDefinition as DesignSpaceParameterDefinition
except ImportError:
    # Graceful fallback if design_space module not available
    DesignPoint = None
    DesignSpace = None
    DesignSpaceParameterDefinition = None

# Type aliases for clarity
ParameterSpace = Dict[str, List[Any]]
ParameterSet = Dict[str, Any]


class OptimizationObjective(Enum):
    """Optimization objective directions."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SamplingStrategy(Enum):
    """Available sampling strategies for design space exploration."""
    RANDOM = "random"
    GRID = "grid"
    LATIN_HYPERCUBE = "lhs"
    SOBOL = "sobol"


@dataclass
class DSEObjective:
    """Definition of a design space exploration objective."""
    name: str
    direction: OptimizationObjective
    weight: float = 1.0
    target_value: Optional[float] = None
    
    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("Objective weight must be positive")


@dataclass
class DSEConfiguration:
    """Configuration for design space exploration."""
    
    # Execution settings
    max_parallel: int = 1
    timeout_seconds: Optional[int] = 3600
    continue_on_failure: bool = True
    
    # Sampling settings
    sampling_strategy: SamplingStrategy = SamplingStrategy.RANDOM
    max_evaluations: int = 50
    random_seed: Optional[int] = None
    
    # Optimization settings
    objectives: List[DSEObjective] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Blueprint and design space
    blueprint_path: Optional[str] = None
    parameter_space: ParameterSpace = field(default_factory=dict)
    
    def __post_init__(self):
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1")


@dataclass
class DSEResult:
    """Result from a single design space exploration evaluation."""
    
    # Input parameters
    parameters: ParameterSet
    
    # Execution results
    build_success: bool
    build_time: float
    
    # Performance metrics (from core.metrics.DSEMetrics)
    metrics: Optional[Any] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_objective_value(self, objective_name: str) -> Optional[float]:
        """Extract objective value from metrics."""
        if not self.metrics:
            return None
        
        try:
            # Navigate nested metric structure
            value = self.metrics
            for part in objective_name.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            
            return float(value) if value is not None else None
        except (AttributeError, ValueError, TypeError):
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'parameters': self.parameters,
            'build_success': self.build_success,
            'build_time': self.build_time,
            'metrics': self.metrics.to_dict() if hasattr(self.metrics, 'to_dict') else self.metrics,
            'metadata': self.metadata
        }


@dataclass
class DSEResults:
    """Collection of DSE results from a complete exploration."""
    
    # All results
    results: List[DSEResult]
    
    # Configuration used
    configuration: DSEConfiguration
    
    # Summary information
    total_evaluations: int = 0
    successful_evaluations: int = 0
    total_time: float = 0.0
    
    # Best results
    best_result: Optional[DSEResult] = None
    pareto_points: List[DSEResult] = field(default_factory=list)
    
    # Analysis data
    convergence: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'total_evaluations') or self.total_evaluations == 0:
            self.total_evaluations = len(self.results)
        
        if not hasattr(self, 'successful_evaluations') or self.successful_evaluations == 0:
            self.successful_evaluations = sum(1 for r in self.results if r.build_success)
    
    def get_successful_results(self) -> List[DSEResult]:
        """Get only successful results."""
        return [r for r in self.results if r.build_success]
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_evaluations / self.total_evaluations if self.total_evaluations > 0 else 0.0
    
    def get_best_by_objective(self, objective_name: str, direction: OptimizationObjective) -> Optional[DSEResult]:
        """Find best result for a specific objective."""
        successful = self.get_successful_results()
        if not successful:
            return None
        
        def get_value(result: DSEResult) -> float:
            value = result.get_objective_value(objective_name)
            if value is None:
                return float('-inf') if direction == OptimizationObjective.MAXIMIZE else float('inf')
            return value
        
        if direction == OptimizationObjective.MAXIMIZE:
            return max(successful, key=get_value)
        else:
            return min(successful, key=get_value)


@dataclass
class ComparisonResult:
    """Result of comparing multiple DSE results."""
    
    best_result: Optional[DSEResult]
    ranking: List[DSEResult]
    comparison_metric: str
    summary_stats: Dict[str, Any]
    
    def get_improvement_ratio(self, baseline_result: DSEResult, objective: str) -> Optional[float]:
        """Calculate improvement ratio compared to baseline."""
        if not self.best_result:
            return None
        
        baseline_value = baseline_result.get_objective_value(objective)
        best_value = self.best_result.get_objective_value(objective)
        
        if baseline_value is None or best_value is None or baseline_value == 0:
            return None
        
        return best_value / baseline_value


@dataclass
class ParameterDefinition:
    """Definition of a design space parameter."""
    
    name: str
    param_type: str  # 'integer', 'float', 'categorical', 'boolean'
    default_value: Any = None
    
    # For numeric parameters
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    
    # For categorical parameters
    choices: Optional[List[Any]] = None
    
    # Metadata
    description: str = ""
    units: str = ""
    
    def __post_init__(self):
        if self.param_type not in ['integer', 'float', 'categorical', 'boolean']:
            raise ValueError(f"Invalid parameter type: {self.param_type}")
        
        if self.param_type == 'categorical' and not self.choices:
            raise ValueError("Categorical parameters must have choices")
        
        if self.param_type in ['integer', 'float']:
            if self.min_value is not None and self.max_value is not None:
                if self.min_value >= self.max_value:
                    raise ValueError("min_value must be less than max_value")
    
    def generate_values(self, n_samples: Optional[int] = None) -> List[Any]:
        """Generate parameter values based on definition."""
        if self.param_type == 'boolean':
            return [True, False]
        
        elif self.param_type == 'categorical':
            return self.choices or []
        
        elif self.param_type in ['integer', 'float'] and self.min_value is not None and self.max_value is not None:
            if n_samples is None:
                # Generate reasonable number of samples
                if self.param_type == 'integer':
                    step = self.step or 1
                    return list(range(int(self.min_value), int(self.max_value) + 1, int(step)))
                else:
                    step = self.step or (self.max_value - self.min_value) / 10
                    values = []
                    current = self.min_value
                    while current <= self.max_value:
                        values.append(current)
                        current += step
                    return values
            else:
                # Generate n_samples between min and max
                if self.param_type == 'integer':
                    import random
                    return [random.randint(int(self.min_value), int(self.max_value)) for _ in range(n_samples)]
                else:
                    import random
                    return [random.uniform(self.min_value, self.max_value) for _ in range(n_samples)]
        
        else:
            return [self.default_value] if self.default_value is not None else []
    
    def validate_value(self, value: Any) -> bool:
        """Validate if a value is valid for this parameter."""
        if self.param_type == 'boolean':
            return isinstance(value, bool)
        
        elif self.param_type == 'categorical':
            return value in (self.choices or [])
        
        elif self.param_type == 'integer':
            if not isinstance(value, int):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            return True
        
        elif self.param_type == 'float':
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            return True
        
        return False


@dataclass
class ExplorationStatistics:
    """Statistics about design space exploration progress."""
    
    # Evaluation counts
    total_evaluations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    
    # Timing
    total_time: float = 0.0
    average_evaluation_time: float = 0.0
    
    # Convergence
    best_objective_value: Optional[float] = None
    objective_history: List[float] = field(default_factory=list)
    convergence_rate: Optional[float] = None
    
    # Space coverage
    space_coverage: float = 0.0
    unique_configurations: int = 0
    
    def update(self, result: DSEResult, objective_name: str = None):
        """Update statistics with new result."""
        self.total_evaluations += 1
        
        if result.build_success:
            self.successful_evaluations += 1
        else:
            self.failed_evaluations += 1
        
        self.total_time += result.build_time
        self.average_evaluation_time = self.total_time / self.total_evaluations
        
        if objective_name and result.build_success:
            obj_value = result.get_objective_value(objective_name)
            if obj_value is not None:
                self.objective_history.append(obj_value)
                if self.best_objective_value is None or obj_value > self.best_objective_value:
                    self.best_objective_value = obj_value
    
    def get_success_rate(self) -> float:
        """Get evaluation success rate."""
        return self.successful_evaluations / self.total_evaluations if self.total_evaluations > 0 else 0.0


# Utility type for parameter space validation
@dataclass
class ParameterSpaceValidation:
    """Result of parameter space validation."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_combinations: int = 0
    estimated_runtime: float = 0.0
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


# Export list for module
__all__ = [
    # Core types from this module
    'ParameterSpace', 'ParameterSet',
    'OptimizationObjective', 'SamplingStrategy',
    'DSEObjective', 'DSEConfiguration', 'DSEResult', 'DSEResults',
    'ComparisonResult', 'ParameterDefinition', 'ExplorationStatistics',
    'ParameterSpaceValidation',
    
    # Re-exported from design_space module
    'DesignPoint', 'DesignSpace', 'DesignSpaceParameterDefinition'
]