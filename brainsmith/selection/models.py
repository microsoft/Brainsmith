"""
Data models for the Selection Framework
Defines core data structures used throughout the selection system.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

# Import from Week 2 Advanced DSE
try:
    from ..dse.advanced.multi_objective import ParetoSolution
except ImportError:
    # Fallback definition if advanced DSE not available
    @dataclass
    class ParetoSolution:
        design_parameters: Dict[str, Any]
        objective_values: List[float]
        constraint_violations: List[float] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
        
        @property
        def is_feasible(self) -> bool:
            return all(violation <= 0 for violation in self.constraint_violations)


class PreferenceType(Enum):
    """Types of preference functions for MCDA algorithms."""
    USUAL = "usual"              # No preference
    U_SHAPE = "u_shape"          # U-shape preference
    V_SHAPE = "v_shape"          # V-shape preference
    LEVEL = "level"              # Level preference
    LINEAR = "linear"            # Linear preference
    GAUSSIAN = "gaussian"        # Gaussian preference


class RankingMethod(Enum):
    """Methods for solution ranking."""
    SCORE_BASED = "score_based"
    DOMINANCE_BASED = "dominance_based"
    REFERENCE_BASED = "reference_based"
    LEXICOGRAPHIC = "lexicographic"


@dataclass
class PreferenceFunction:
    """Preference function for PROMETHEE algorithm."""
    preference_type: PreferenceType
    threshold: Optional[float] = None
    indifference_threshold: Optional[float] = None
    sigma: Optional[float] = None  # For Gaussian preference
    
    def __call__(self, difference: float) -> float:
        """Calculate preference value for given difference."""
        if self.preference_type == PreferenceType.USUAL:
            return 1.0 if difference > 0 else 0.0
        
        elif self.preference_type == PreferenceType.U_SHAPE:
            if self.indifference_threshold is None:
                raise ValueError("U-shape preference requires indifference_threshold")
            return 0.0 if abs(difference) <= self.indifference_threshold else 1.0
        
        elif self.preference_type == PreferenceType.V_SHAPE:
            if self.threshold is None:
                raise ValueError("V-shape preference requires threshold")
            return min(1.0, abs(difference) / self.threshold) if difference != 0 else 0.0
        
        elif self.preference_type == PreferenceType.LEVEL:
            if self.indifference_threshold is None or self.threshold is None:
                raise ValueError("Level preference requires both thresholds")
            abs_diff = abs(difference)
            if abs_diff <= self.indifference_threshold:
                return 0.0
            elif abs_diff >= self.threshold:
                return 1.0
            else:
                return 0.5
        
        elif self.preference_type == PreferenceType.LINEAR:
            if self.indifference_threshold is None or self.threshold is None:
                raise ValueError("Linear preference requires both thresholds")
            abs_diff = abs(difference)
            if abs_diff <= self.indifference_threshold:
                return 0.0
            elif abs_diff >= self.threshold:
                return 1.0
            else:
                return (abs_diff - self.indifference_threshold) / (self.threshold - self.indifference_threshold)
        
        elif self.preference_type == PreferenceType.GAUSSIAN:
            if self.sigma is None:
                raise ValueError("Gaussian preference requires sigma")
            return 1.0 - np.exp(-(difference ** 2) / (2 * self.sigma ** 2)) if difference != 0 else 0.0
        
        else:
            raise ValueError(f"Unknown preference type: {self.preference_type}")


@dataclass
class SelectionCriteria:
    """Multi-criteria selection specification."""
    objectives: List[str]
    weights: Dict[str, float]
    constraints: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    preference_functions: Dict[str, PreferenceFunction] = field(default_factory=dict)
    thresholds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    maximize_objectives: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize criteria after initialization."""
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {obj: w/total_weight for obj, w in self.weights.items()}
        
        # Set default maximize directions
        for obj in self.objectives:
            if obj not in self.maximize_objectives:
                # Default assumption based on objective name
                self.maximize_objectives[obj] = (
                    'maximize' in obj.lower() or 
                    'throughput' in obj.lower() or
                    'efficiency' in obj.lower() or
                    'accuracy' in obj.lower()
                )
    
    def validate(self) -> bool:
        """Validate selection criteria consistency."""
        # Check weight coverage
        for obj in self.objectives:
            if obj not in self.weights:
                return False
        
        # Check weight sum
        if not np.isclose(sum(self.weights.values()), 1.0, rtol=1e-6):
            return False
        
        return True


@dataclass
class UserPreferences:
    """User preference specification."""
    importance_weights: Dict[str, float]
    threshold_preferences: Dict[str, Tuple[float, float]]
    constraint_preferences: List[str]
    trade_off_preferences: Dict[str, float] = field(default_factory=dict)
    aspiration_levels: Dict[str, float] = field(default_factory=dict)
    reservation_levels: Dict[str, float] = field(default_factory=dict)


@dataclass
class RankedSolution:
    """Solution with ranking information."""
    solution: ParetoSolution
    rank: int
    score: float
    selection_criteria: SelectionCriteria
    ranking_method: RankingMethod
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    
    @property
    def design_parameters(self) -> Dict[str, Any]:
        """Get design parameters from underlying solution."""
        return self.solution.design_parameters
    
    @property
    def objective_values(self) -> List[float]:
        """Get objective values from underlying solution."""
        return self.solution.objective_values
    
    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible."""
        return self.solution.is_feasible


@dataclass
class CompromiseSolution:
    """Compromise solution with trade-off information."""
    solution: RankedSolution
    compromise_type: str  # 'balanced', 'least_regret', 'ideal_distance'
    trade_off_analysis: Dict[str, float]
    sensitivity_analysis: Dict[str, float]
    robustness_score: float = 0.0


@dataclass
class DecisionMatrix:
    """Decision matrix for MCDA algorithms."""
    alternatives: List[str]  # Solution identifiers
    criteria: List[str]      # Objective names
    matrix: np.ndarray       # Performance matrix
    weights: np.ndarray      # Criteria weights
    maximize: np.ndarray     # Maximize flags for each criterion
    
    def __post_init__(self):
        """Validate matrix dimensions after initialization."""
        n_alternatives, n_criteria = self.matrix.shape
        
        if len(self.alternatives) != n_alternatives:
            raise ValueError(f"Number of alternatives ({len(self.alternatives)}) "
                           f"doesn't match matrix rows ({n_alternatives})")
        
        if len(self.criteria) != n_criteria:
            raise ValueError(f"Number of criteria ({len(self.criteria)}) "
                           f"doesn't match matrix columns ({n_criteria})")
        
        if len(self.weights) != n_criteria:
            raise ValueError(f"Number of weights ({len(self.weights)}) "
                           f"doesn't match number of criteria ({n_criteria})")
        
        if len(self.maximize) != n_criteria:
            raise ValueError(f"Number of maximize flags ({len(self.maximize)}) "
                           f"doesn't match number of criteria ({n_criteria})")
    
    def normalize(self, method: str = 'vector') -> 'DecisionMatrix':
        """Normalize decision matrix."""
        if method == 'vector':
            # Vector normalization
            norms = np.linalg.norm(self.matrix, axis=0)
            normalized_matrix = self.matrix / norms
        elif method == 'minmax':
            # Min-max normalization
            min_vals = np.min(self.matrix, axis=0)
            max_vals = np.max(self.matrix, axis=0)
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1  # Avoid division by zero
            normalized_matrix = (self.matrix - min_vals) / ranges
        elif method == 'max':
            # Max normalization
            max_vals = np.max(self.matrix, axis=0)
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            normalized_matrix = self.matrix / max_vals
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return DecisionMatrix(
            alternatives=self.alternatives.copy(),
            criteria=self.criteria.copy(),
            matrix=normalized_matrix,
            weights=self.weights.copy(),
            maximize=self.maximize.copy()
        )


@dataclass
class SelectionContext:
    """Context for solution selection process."""
    pareto_solutions: List[ParetoSolution]
    selection_criteria: SelectionCriteria
    user_preferences: Optional[UserPreferences] = None
    historical_selections: List[RankedSolution] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    time_constraints: Optional[float] = None
    
    def create_decision_matrix(self) -> DecisionMatrix:
        """Create decision matrix from Pareto solutions."""
        if not self.pareto_solutions:
            raise ValueError("No Pareto solutions provided")
        
        n_solutions = len(self.pareto_solutions)
        n_objectives = len(self.selection_criteria.objectives)
        
        # Create performance matrix
        matrix = np.zeros((n_solutions, n_objectives))
        
        for i, solution in enumerate(self.pareto_solutions):
            for j, objective in enumerate(self.selection_criteria.objectives):
                if j < len(solution.objective_values):
                    matrix[i, j] = solution.objective_values[j]
                else:
                    raise ValueError(f"Solution {i} missing objective {j} ({objective})")
        
        # Create alternative identifiers
        alternatives = [f"Solution_{i}" for i in range(n_solutions)]
        
        # Extract weights and maximize flags
        weights = np.array([self.selection_criteria.weights.get(obj, 0.0) 
                           for obj in self.selection_criteria.objectives])
        
        maximize = np.array([self.selection_criteria.maximize_objectives.get(obj, True)
                            for obj in self.selection_criteria.objectives])
        
        return DecisionMatrix(
            alternatives=alternatives,
            criteria=self.selection_criteria.objectives,
            matrix=matrix,
            weights=weights,
            maximize=maximize
        )


@dataclass
class SelectionMetrics:
    """Metrics for evaluating selection quality."""
    selection_time: float
    number_of_solutions: int
    number_of_criteria: int
    weight_consistency: float
    preference_satisfaction: float
    diversity_score: float
    confidence_score: float
    robustness_score: float = 0.0
    sensitivity_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionReport:
    """Comprehensive selection report."""
    ranked_solutions: List[RankedSolution]
    selection_criteria: SelectionCriteria
    selection_metrics: SelectionMetrics
    compromise_solutions: List[CompromiseSolution]
    sensitivity_analysis: Dict[str, Any]
    recommendations: List[str]
    trade_off_analysis: Dict[str, Any]
    executive_summary: str = ""
    
    def get_top_solutions(self, n: int = 5) -> List[RankedSolution]:
        """Get top N solutions."""
        return self.ranked_solutions[:n]
    
    def get_compromise_solution(self, compromise_type: str = 'balanced') -> Optional[CompromiseSolution]:
        """Get compromise solution of specified type."""
        for comp_sol in self.compromise_solutions:
            if comp_sol.compromise_type == compromise_type:
                return comp_sol
        return None


@dataclass
class SelectionConfiguration:
    """Configuration for selection algorithms."""
    algorithm: str = 'topsis'
    normalization_method: str = 'vector'
    aggregation_method: str = 'weighted_sum'
    preference_threshold: float = 0.01
    indifference_threshold: float = 0.005
    similarity_threshold: float = 0.1
    max_solutions: int = 10
    include_sensitivity: bool = True
    include_trade_off: bool = True
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        valid_algorithms = ['topsis', 'promethee', 'ahp', 'weighted_sum', 'weighted_product', 'fuzzy_topsis']
        if self.algorithm not in valid_algorithms:
            return False
        
        valid_normalizations = ['vector', 'minmax', 'max']
        if self.normalization_method not in valid_normalizations:
            return False
        
        if not (0 <= self.confidence_level <= 1):
            return False
        
        if self.max_solutions <= 0:
            return False
        
        return True


# Type aliases for better code readability
SolutionScore = Tuple[int, float]  # (solution_index, score)
PreferenceWeights = Dict[str, float]
ObjectiveValues = List[float]
PerformanceMatrix = np.ndarray