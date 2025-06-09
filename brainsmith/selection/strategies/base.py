"""
Base Selection Strategy Interface
Defines the abstract interface that all selection strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import numpy as np

from ..models import (
    SelectionContext, RankedSolution, SelectionConfiguration,
    DecisionMatrix, RankingMethod
)

logger = logging.getLogger(__name__)


class SelectionStrategy(ABC):
    """Abstract base class for all selection strategies."""
    
    def __init__(self, configuration: SelectionConfiguration):
        """Initialize strategy with configuration."""
        self.config = configuration
        self.ranking_method = RankingMethod.SCORE_BASED
        
    @abstractmethod
    def select_solutions(self, context: SelectionContext) -> List[RankedSolution]:
        """
        Select and rank solutions based on the strategy algorithm.
        
        Args:
            context: Selection context with solutions and criteria
            
        Returns:
            List of ranked solutions in descending order of preference
        """
        pass
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the name of the algorithm."""
        pass
    
    def prepare_decision_matrix(self, context: SelectionContext) -> DecisionMatrix:
        """Prepare and normalize decision matrix from context."""
        # Create decision matrix from context
        decision_matrix = context.create_decision_matrix()
        
        # Normalize the matrix
        normalized_matrix = decision_matrix.normalize(self.config.normalization_method)
        
        return normalized_matrix
    
    def create_ranked_solutions(self, 
                               context: SelectionContext,
                               scores: np.ndarray,
                               sort_descending: bool = True) -> List[RankedSolution]:
        """Create ranked solutions from scores."""
        # Sort solutions by score
        if sort_descending:
            sorted_indices = np.argsort(scores)[::-1]  # Descending order
        else:
            sorted_indices = np.argsort(scores)  # Ascending order
            
        ranked_solutions = []
        
        for rank, idx in enumerate(sorted_indices):
            solution = context.pareto_solutions[idx]
            score = scores[idx]
            
            ranked_solution = RankedSolution(
                solution=solution,
                rank=rank + 1,  # 1-based ranking
                score=float(score),
                selection_criteria=context.selection_criteria,
                ranking_method=self.ranking_method,
                confidence=self._calculate_solution_confidence(score, scores)
            )
            
            ranked_solutions.append(ranked_solution)
        
        return ranked_solutions
    
    def _calculate_solution_confidence(self, score: float, all_scores: np.ndarray) -> float:
        """Calculate confidence level for a solution based on its score."""
        if len(all_scores) <= 1:
            return 1.0
        
        # Calculate percentile rank
        percentile = np.sum(all_scores <= score) / len(all_scores)
        
        # Calculate separation from other scores
        other_scores = all_scores[all_scores != score]
        if len(other_scores) > 0:
            min_distance = np.min(np.abs(other_scores - score))
            score_range = np.max(all_scores) - np.min(all_scores)
            separation = min_distance / max(score_range, 1e-10)
        else:
            separation = 1.0
        
        # Combine percentile and separation
        confidence = 0.7 * percentile + 0.3 * separation
        
        return min(1.0, max(0.0, confidence))
    
    def validate_context(self, context: SelectionContext) -> None:
        """Validate selection context before processing."""
        if not context.pareto_solutions:
            raise ValueError("No Pareto solutions provided")
        
        if not context.selection_criteria.validate():
            raise ValueError("Invalid selection criteria")
        
        # Check that all solutions have the same number of objectives
        n_objectives = len(context.selection_criteria.objectives)
        for i, solution in enumerate(context.pareto_solutions):
            if len(solution.objective_values) != n_objectives:
                raise ValueError(
                    f"Solution {i} has {len(solution.objective_values)} objectives, "
                    f"expected {n_objectives}"
                )
    
    def log_selection_info(self, context: SelectionContext, scores: np.ndarray) -> None:
        """Log information about the selection process."""
        logger.info(
            f"{self.algorithm_name} selection: {len(context.pareto_solutions)} solutions, "
            f"{len(context.selection_criteria.objectives)} objectives"
        )
        
        if len(scores) > 0:
            logger.info(
                f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}], "
                f"mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}"
            )


class WeightedStrategy(SelectionStrategy):
    """Base class for weighted aggregation strategies."""
    
    def __init__(self, configuration: SelectionConfiguration):
        super().__init__(configuration)
        
    def apply_weights(self, 
                     normalized_matrix: np.ndarray,
                     weights: np.ndarray,
                     maximize: np.ndarray) -> np.ndarray:
        """Apply weights to normalized decision matrix."""
        # Convert minimization objectives to maximization
        adjusted_matrix = normalized_matrix.copy()
        for j, is_maximize in enumerate(maximize):
            if not is_maximize:
                # For minimization objectives, subtract from 1 (assuming normalized to [0,1])
                adjusted_matrix[:, j] = 1.0 - adjusted_matrix[:, j]
        
        # Apply weights
        weighted_matrix = adjusted_matrix * weights
        
        return weighted_matrix


class DistanceBasedStrategy(SelectionStrategy):
    """Base class for distance-based strategies like TOPSIS."""
    
    def __init__(self, configuration: SelectionConfiguration):
        super().__init__(configuration)
    
    def calculate_ideal_points(self, 
                              weighted_matrix: np.ndarray,
                              maximize: np.ndarray) -> tuple:
        """Calculate ideal and negative-ideal points."""
        ideal_point = np.zeros(weighted_matrix.shape[1])
        negative_ideal_point = np.zeros(weighted_matrix.shape[1])
        
        for j in range(weighted_matrix.shape[1]):
            if maximize[j]:
                ideal_point[j] = np.max(weighted_matrix[:, j])
                negative_ideal_point[j] = np.min(weighted_matrix[:, j])
            else:
                ideal_point[j] = np.min(weighted_matrix[:, j])
                negative_ideal_point[j] = np.max(weighted_matrix[:, j])
        
        return ideal_point, negative_ideal_point
    
    def calculate_distances(self, 
                           weighted_matrix: np.ndarray,
                           ideal_point: np.ndarray,
                           negative_ideal_point: np.ndarray) -> tuple:
        """Calculate distances to ideal and negative-ideal points."""
        distances_to_ideal = np.sqrt(
            np.sum((weighted_matrix - ideal_point) ** 2, axis=1)
        )
        
        distances_to_negative_ideal = np.sqrt(
            np.sum((weighted_matrix - negative_ideal_point) ** 2, axis=1)
        )
        
        return distances_to_ideal, distances_to_negative_ideal


class OutrankingStrategy(SelectionStrategy):
    """Base class for outranking strategies like PROMETHEE."""
    
    def __init__(self, configuration: SelectionConfiguration):
        super().__init__(configuration)
        self.ranking_method = RankingMethod.DOMINANCE_BASED
    
    def calculate_preference_matrix(self, 
                                   decision_matrix: DecisionMatrix,
                                   preference_functions: Dict[str, Any]) -> np.ndarray:
        """Calculate preference matrix for outranking methods."""
        n_alternatives = decision_matrix.matrix.shape[0]
        n_criteria = decision_matrix.matrix.shape[1]
        
        # Initialize preference matrix
        preference_matrix = np.zeros((n_alternatives, n_alternatives, n_criteria))
        
        for k in range(n_criteria):
            criterion_name = decision_matrix.criteria[k]
            pref_func = preference_functions.get(criterion_name)
            
            if pref_func is None:
                # Default preference function (usual type)
                for i in range(n_alternatives):
                    for j in range(n_alternatives):
                        diff = decision_matrix.matrix[i, k] - decision_matrix.matrix[j, k]
                        if decision_matrix.maximize[k]:
                            preference_matrix[i, j, k] = 1.0 if diff > 0 else 0.0
                        else:
                            preference_matrix[i, j, k] = 1.0 if diff < 0 else 0.0
            else:
                # Apply custom preference function
                for i in range(n_alternatives):
                    for j in range(n_alternatives):
                        diff = decision_matrix.matrix[i, k] - decision_matrix.matrix[j, k]
                        if not decision_matrix.maximize[k]:
                            diff = -diff  # Reverse for minimization
                        preference_matrix[i, j, k] = pref_func(diff)
        
        return preference_matrix
    
    def calculate_outranking_flows(self, 
                                  preference_matrix: np.ndarray,
                                  weights: np.ndarray) -> tuple:
        """Calculate positive and negative outranking flows."""
        n_alternatives = preference_matrix.shape[0]
        
        # Calculate aggregated preference indices
        pi_matrix = np.zeros((n_alternatives, n_alternatives))
        
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:
                    pi_matrix[i, j] = np.sum(weights * preference_matrix[i, j, :])
        
        # Calculate outranking flows
        positive_flows = np.zeros(n_alternatives)
        negative_flows = np.zeros(n_alternatives)
        
        for i in range(n_alternatives):
            positive_flows[i] = np.sum(pi_matrix[i, :]) / (n_alternatives - 1)
            negative_flows[i] = np.sum(pi_matrix[:, i]) / (n_alternatives - 1)
        
        return positive_flows, negative_flows