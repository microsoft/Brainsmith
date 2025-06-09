"""
TOPSIS (Technique for Order Preference by Similarity) Implementation
A classic MCDA method that ranks alternatives based on their distance to ideal solutions.
"""

import numpy as np
from typing import List
import logging

from .base import DistanceBasedStrategy
from ..models import SelectionContext, RankedSolution

logger = logging.getLogger(__name__)


class TOPSISSelector(DistanceBasedStrategy):
    """
    TOPSIS algorithm implementation for solution selection.
    
    TOPSIS ranks alternatives based on their relative closeness to the ideal solution.
    The ideal solution maximizes benefit criteria and minimizes cost criteria.
    """
    
    @property
    def algorithm_name(self) -> str:
        return "TOPSIS"
    
    def select_solutions(self, context: SelectionContext) -> List[RankedSolution]:
        """
        Select solutions using TOPSIS algorithm.
        
        Steps:
        1. Normalize decision matrix
        2. Calculate weighted normalized matrix
        3. Determine ideal and negative-ideal solutions
        4. Calculate separation measures
        5. Calculate relative closeness to ideal solution
        6. Rank alternatives
        """
        # Validate input
        self.validate_context(context)
        
        # Prepare decision matrix
        decision_matrix = self.prepare_decision_matrix(context)
        
        # Step 1-2: Get weighted normalized matrix
        weighted_matrix = self.apply_weights(
            decision_matrix.matrix,
            decision_matrix.weights,
            decision_matrix.maximize
        )
        
        # Step 3: Determine ideal and negative-ideal solutions
        ideal_point, negative_ideal_point = self.calculate_ideal_points(
            weighted_matrix, decision_matrix.maximize
        )
        
        # Step 4: Calculate separation measures
        distances_to_ideal, distances_to_negative_ideal = self.calculate_distances(
            weighted_matrix, ideal_point, negative_ideal_point
        )
        
        # Step 5: Calculate relative closeness
        relative_closeness = self._calculate_relative_closeness(
            distances_to_ideal, distances_to_negative_ideal
        )
        
        # Step 6: Create ranked solutions
        ranked_solutions = self.create_ranked_solutions(
            context, relative_closeness, sort_descending=True
        )
        
        # Log results
        self.log_selection_info(context, relative_closeness)
        
        return ranked_solutions
    
    def _calculate_relative_closeness(self, 
                                     distances_to_ideal: np.ndarray,
                                     distances_to_negative_ideal: np.ndarray) -> np.ndarray:
        """Calculate relative closeness to ideal solution."""
        
        # Handle edge cases
        total_distances = distances_to_ideal + distances_to_negative_ideal
        
        # Avoid division by zero
        relative_closeness = np.zeros_like(distances_to_ideal)
        valid_mask = total_distances > 1e-10
        
        relative_closeness[valid_mask] = (
            distances_to_negative_ideal[valid_mask] / total_distances[valid_mask]
        )
        
        # For cases where both distances are zero (identical to both ideal points)
        # assign maximum closeness
        zero_distance_mask = total_distances <= 1e-10
        relative_closeness[zero_distance_mask] = 1.0
        
        return relative_closeness


class ModifiedTOPSISSelector(TOPSISSelector):
    """
    Modified TOPSIS with additional features:
    - Entropy-based weight adjustment
    - Reference point modification
    - Uncertainty handling
    """
    
    @property
    def algorithm_name(self) -> str:
        return "Modified TOPSIS"
    
    def select_solutions(self, context: SelectionContext) -> List[RankedSolution]:
        """Select solutions using modified TOPSIS with entropy weights."""
        
        # Validate input
        self.validate_context(context)
        
        # Prepare decision matrix
        decision_matrix = self.prepare_decision_matrix(context)
        
        # Calculate entropy-adjusted weights
        entropy_weights = self._calculate_entropy_weights(decision_matrix.matrix)
        
        # Combine with user weights
        combined_weights = self._combine_weights(
            decision_matrix.weights, entropy_weights
        )
        
        # Apply combined weights
        weighted_matrix = self.apply_weights(
            decision_matrix.matrix, combined_weights, decision_matrix.maximize
        )
        
        # Use reference points if available in preferences
        ideal_point, negative_ideal_point = self._calculate_reference_points(
            weighted_matrix, decision_matrix.maximize, context
        )
        
        # Calculate distances with uncertainty consideration
        distances_to_ideal, distances_to_negative_ideal = self._calculate_fuzzy_distances(
            weighted_matrix, ideal_point, negative_ideal_point
        )
        
        # Calculate relative closeness
        relative_closeness = self._calculate_relative_closeness(
            distances_to_ideal, distances_to_negative_ideal
        )
        
        # Create ranked solutions
        ranked_solutions = self.create_ranked_solutions(
            context, relative_closeness, sort_descending=True
        )
        
        # Add entropy information to additional metrics
        for i, solution in enumerate(ranked_solutions):
            solution.additional_metrics['entropy_weight_contribution'] = (
                entropy_weights[i % len(entropy_weights)]
            )
        
        self.log_selection_info(context, relative_closeness)
        
        return ranked_solutions
    
    def _calculate_entropy_weights(self, decision_matrix: np.ndarray) -> np.ndarray:
        """Calculate entropy-based weights for criteria."""
        n_alternatives, n_criteria = decision_matrix.shape
        
        # Normalize matrix to [0, 1]
        normalized_matrix = np.zeros_like(decision_matrix)
        for j in range(n_criteria):
            col = decision_matrix[:, j]
            min_val, max_val = np.min(col), np.max(col)
            if max_val > min_val:
                normalized_matrix[:, j] = (col - min_val) / (max_val - min_val)
            else:
                normalized_matrix[:, j] = 0.5  # Equal preference if no variation
        
        # Calculate entropy for each criterion
        entropy_weights = np.zeros(n_criteria)
        k = 1.0 / np.log(n_alternatives) if n_alternatives > 1 else 1.0
        
        for j in range(n_criteria):
            # Calculate proportions
            proportions = normalized_matrix[:, j]
            proportions = proportions / np.sum(proportions) if np.sum(proportions) > 0 else proportions
            
            # Calculate entropy
            entropy = 0.0
            for p in proportions:
                if p > 1e-10:  # Avoid log(0)
                    entropy -= p * np.log(p)
            
            entropy *= k
            
            # Diversity (1 - entropy)
            diversity = 1.0 - entropy
            entropy_weights[j] = diversity
        
        # Normalize weights
        total_weight = np.sum(entropy_weights)
        if total_weight > 0:
            entropy_weights /= total_weight
        else:
            entropy_weights.fill(1.0 / n_criteria)
        
        return entropy_weights
    
    def _combine_weights(self, 
                        user_weights: np.ndarray,
                        entropy_weights: np.ndarray,
                        alpha: float = 0.7) -> np.ndarray:
        """Combine user weights with entropy weights."""
        # Weighted combination: alpha * user_weights + (1-alpha) * entropy_weights
        combined = alpha * user_weights + (1 - alpha) * entropy_weights
        
        # Normalize
        total = np.sum(combined)
        if total > 0:
            combined /= total
        
        return combined
    
    def _calculate_reference_points(self, 
                                   weighted_matrix: np.ndarray,
                                   maximize: np.ndarray,
                                   context: SelectionContext) -> tuple:
        """Calculate reference points considering user preferences."""
        
        # Check if aspiration/reservation levels are specified
        if (context.user_preferences and 
            hasattr(context.user_preferences, 'aspiration_levels') and
            context.user_preferences.aspiration_levels):
            
            # Use user-specified aspiration levels as ideal point
            ideal_point = np.zeros(weighted_matrix.shape[1])
            for j, criterion in enumerate(context.selection_criteria.objectives):
                if criterion in context.user_preferences.aspiration_levels:
                    ideal_point[j] = context.user_preferences.aspiration_levels[criterion]
                else:
                    # Fall back to matrix-based calculation
                    if maximize[j]:
                        ideal_point[j] = np.max(weighted_matrix[:, j])
                    else:
                        ideal_point[j] = np.min(weighted_matrix[:, j])
        else:
            # Standard TOPSIS ideal point calculation
            ideal_point, _ = self.calculate_ideal_points(weighted_matrix, maximize)
        
        # Similar for negative ideal point with reservation levels
        if (context.user_preferences and 
            hasattr(context.user_preferences, 'reservation_levels') and
            context.user_preferences.reservation_levels):
            
            negative_ideal_point = np.zeros(weighted_matrix.shape[1])
            for j, criterion in enumerate(context.selection_criteria.objectives):
                if criterion in context.user_preferences.reservation_levels:
                    negative_ideal_point[j] = context.user_preferences.reservation_levels[criterion]
                else:
                    # Fall back to matrix-based calculation
                    if maximize[j]:
                        negative_ideal_point[j] = np.min(weighted_matrix[:, j])
                    else:
                        negative_ideal_point[j] = np.max(weighted_matrix[:, j])
        else:
            # Standard TOPSIS negative ideal point calculation
            _, negative_ideal_point = self.calculate_ideal_points(weighted_matrix, maximize)
        
        return ideal_point, negative_ideal_point
    
    def _calculate_fuzzy_distances(self, 
                                  weighted_matrix: np.ndarray,
                                  ideal_point: np.ndarray,
                                  negative_ideal_point: np.ndarray) -> tuple:
        """Calculate distances with uncertainty consideration."""
        
        # Standard distance calculation
        distances_to_ideal, distances_to_negative_ideal = self.calculate_distances(
            weighted_matrix, ideal_point, negative_ideal_point
        )
        
        # Add small uncertainty factor to avoid zero distances
        uncertainty_factor = 1e-6
        distances_to_ideal += uncertainty_factor
        distances_to_negative_ideal += uncertainty_factor
        
        return distances_to_ideal, distances_to_negative_ideal