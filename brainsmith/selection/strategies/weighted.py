"""
Weighted Sum and Weighted Product Method Implementations
Simple but effective MCDA methods based on weighted aggregation.
"""

import numpy as np
from typing import List
import logging

from .base import WeightedStrategy
from ..models import SelectionContext, RankedSolution

logger = logging.getLogger(__name__)


class WeightedSumSelector(WeightedStrategy):
    """
    Weighted Sum Method (WSM) implementation.
    
    The simplest MCDA method that calculates a weighted sum of normalized criteria values.
    """
    
    @property
    def algorithm_name(self) -> str:
        return "Weighted Sum"
    
    def select_solutions(self, context: SelectionContext) -> List[RankedSolution]:
        """
        Select solutions using Weighted Sum Method.
        
        Steps:
        1. Normalize decision matrix
        2. Apply weights to normalized values
        3. Sum weighted values for each alternative
        4. Rank alternatives by total score
        """
        # Validate input
        self.validate_context(context)
        
        # Prepare decision matrix
        decision_matrix = self.prepare_decision_matrix(context)
        
        # Apply weights
        weighted_matrix = self.apply_weights(
            decision_matrix.matrix,
            decision_matrix.weights,
            decision_matrix.maximize
        )
        
        # Calculate weighted sum for each alternative
        scores = np.sum(weighted_matrix, axis=1)
        
        # Create ranked solutions
        ranked_solutions = self.create_ranked_solutions(
            context, scores, sort_descending=True
        )
        
        # Log results
        self.log_selection_info(context, scores)
        
        return ranked_solutions


class WeightedProductSelector(WeightedStrategy):
    """
    Weighted Product Method (WPM) implementation.
    
    Uses weighted geometric mean instead of arithmetic mean.
    More suitable when criteria are not commensurate.
    """
    
    @property
    def algorithm_name(self) -> str:
        return "Weighted Product"
    
    def select_solutions(self, context: SelectionContext) -> List[RankedSolution]:
        """
        Select solutions using Weighted Product Method.
        
        Steps:
        1. Normalize decision matrix
        2. Raise normalized values to the power of their weights
        3. Calculate product for each alternative
        4. Rank alternatives by product score
        """
        # Validate input
        self.validate_context(context)
        
        # Prepare decision matrix
        decision_matrix = self.prepare_decision_matrix(context)
        
        # Prepare matrix for product calculation
        adjusted_matrix = self._prepare_product_matrix(
            decision_matrix.matrix,
            decision_matrix.maximize
        )
        
        # Calculate weighted product
        scores = self._calculate_weighted_product(
            adjusted_matrix, decision_matrix.weights
        )
        
        # Create ranked solutions
        ranked_solutions = self.create_ranked_solutions(
            context, scores, sort_descending=True
        )
        
        # Log results
        self.log_selection_info(context, scores)
        
        return ranked_solutions
    
    def _prepare_product_matrix(self, 
                               matrix: np.ndarray,
                               maximize: np.ndarray) -> np.ndarray:
        """Prepare matrix for weighted product calculation."""
        
        # Ensure all values are positive for product calculation
        adjusted_matrix = matrix.copy()
        
        # Add small constant to avoid zero values
        epsilon = 1e-10
        adjusted_matrix += epsilon
        
        # For minimization criteria, take reciprocal
        for j, is_maximize in enumerate(maximize):
            if not is_maximize:
                # Take reciprocal for minimization (smaller is better becomes larger is better)
                adjusted_matrix[:, j] = 1.0 / adjusted_matrix[:, j]
        
        return adjusted_matrix
    
    def _calculate_weighted_product(self, 
                                   matrix: np.ndarray,
                                   weights: np.ndarray) -> np.ndarray:
        """Calculate weighted geometric mean (product) for each alternative."""
        
        # Take logarithm to convert product to sum
        log_matrix = np.log(matrix)
        
        # Apply weights
        weighted_log_matrix = log_matrix * weights
        
        # Sum weighted logarithms
        log_scores = np.sum(weighted_log_matrix, axis=1)
        
        # Convert back to product by taking exponential
        scores = np.exp(log_scores)
        
        return scores


class HybridWeightedSelector(WeightedStrategy):
    """
    Hybrid approach combining Weighted Sum and Weighted Product methods.
    
    Useful when some criteria are commensurate (use sum) and others are not (use product).
    """
    
    def __init__(self, configuration, sum_weight: float = 0.5):
        """
        Initialize hybrid selector.
        
        Args:
            configuration: Selection configuration
            sum_weight: Weight for weighted sum component (0-1)
        """
        super().__init__(configuration)
        self.sum_weight = max(0.0, min(1.0, sum_weight))
        self.product_weight = 1.0 - self.sum_weight
    
    @property
    def algorithm_name(self) -> str:
        return f"Hybrid Weighted (Sum: {self.sum_weight:.1f}, Product: {self.product_weight:.1f})"
    
    def select_solutions(self, context: SelectionContext) -> List[RankedSolution]:
        """
        Select solutions using hybrid weighted approach.
        
        Combines normalized scores from both WSM and WPM.
        """
        # Validate input
        self.validate_context(context)
        
        # Get scores from both methods
        wsm_selector = WeightedSumSelector(self.config)
        wpm_selector = WeightedProductSelector(self.config)
        
        wsm_solutions = wsm_selector.select_solutions(context)
        wpm_solutions = wpm_selector.select_solutions(context)
        
        # Extract and normalize scores
        wsm_scores = np.array([sol.score for sol in wsm_solutions])
        wpm_scores = np.array([sol.score for sol in wpm_solutions])
        
        # Normalize scores to [0, 1]
        wsm_normalized = self._normalize_scores(wsm_scores)
        wpm_normalized = self._normalize_scores(wpm_scores)
        
        # Combine scores
        combined_scores = (
            self.sum_weight * wsm_normalized + 
            self.product_weight * wpm_normalized
        )
        
        # Create ranked solutions
        ranked_solutions = self.create_ranked_solutions(
            context, combined_scores, sort_descending=True
        )
        
        # Add component scores to additional metrics
        for i, solution in enumerate(ranked_solutions):
            original_idx = next(
                j for j, sol in enumerate(context.pareto_solutions)
                if sol == solution.solution
            )
            solution.additional_metrics.update({
                'wsm_score': wsm_scores[original_idx],
                'wpm_score': wpm_scores[original_idx],
                'wsm_normalized': wsm_normalized[original_idx],
                'wpm_normalized': wpm_normalized[original_idx]
            })
        
        # Log results
        self.log_selection_info(context, combined_scores)
        
        return ranked_solutions
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            # All scores are equal
            return np.ones_like(scores) * 0.5


class AdaptiveWeightedSelector(WeightedStrategy):
    """
    Adaptive weighted method that adjusts aggregation based on data characteristics.
    
    Automatically chooses between sum and product based on:
    - Scale differences between criteria
    - Correlation between criteria
    - Variability of criteria values
    """
    
    @property
    def algorithm_name(self) -> str:
        return "Adaptive Weighted"
    
    def select_solutions(self, context: SelectionContext) -> List[RankedSolution]:
        """
        Select solutions using adaptive weighted approach.
        
        Analyzes data characteristics to choose optimal aggregation method.
        """
        # Validate input
        self.validate_context(context)
        
        # Prepare decision matrix
        decision_matrix = self.prepare_decision_matrix(context)
        
        # Analyze data characteristics
        method_weights = self._analyze_data_characteristics(decision_matrix)
        
        # Create selectors
        wsm_selector = WeightedSumSelector(self.config)
        wpm_selector = WeightedProductSelector(self.config)
        
        # Get solutions from both methods
        wsm_solutions = wsm_selector.select_solutions(context)
        wpm_solutions = wpm_selector.select_solutions(context)
        
        # Combine based on adaptive weights
        wsm_scores = np.array([sol.score for sol in wsm_solutions])
        wpm_scores = np.array([sol.score for sol in wpm_solutions])
        
        # Normalize and combine
        wsm_normalized = self._normalize_scores(wsm_scores)
        wpm_normalized = self._normalize_scores(wpm_scores)
        
        combined_scores = (
            method_weights['sum'] * wsm_normalized +
            method_weights['product'] * wpm_normalized
        )
        
        # Create ranked solutions
        ranked_solutions = self.create_ranked_solutions(
            context, combined_scores, sort_descending=True
        )
        
        # Add analysis results to metadata
        for solution in ranked_solutions:
            solution.additional_metrics.update({
                'adaptive_sum_weight': method_weights['sum'],
                'adaptive_product_weight': method_weights['product'],
                'scale_heterogeneity': method_weights['scale_analysis'],
                'correlation_strength': method_weights['correlation_analysis']
            })
        
        logger.info(f"Adaptive weights: Sum={method_weights['sum']:.3f}, "
                   f"Product={method_weights['product']:.3f}")
        
        return ranked_solutions
    
    def _analyze_data_characteristics(self, decision_matrix) -> dict:
        """Analyze data characteristics to determine optimal aggregation method."""
        
        matrix = decision_matrix.matrix
        
        # 1. Scale heterogeneity analysis
        scale_heterogeneity = self._calculate_scale_heterogeneity(matrix)
        
        # 2. Correlation analysis
        correlation_strength = self._calculate_correlation_strength(matrix)
        
        # 3. Variability analysis
        variability_score = self._calculate_variability_score(matrix)
        
        # Determine method weights based on characteristics
        # High scale heterogeneity favors product method
        # High correlation favors sum method
        # High variability favors product method
        
        product_bias = 0.0
        sum_bias = 0.0
        
        # Scale heterogeneity (0-1, higher means more heterogeneous)
        if scale_heterogeneity > 0.5:
            product_bias += (scale_heterogeneity - 0.5) * 2.0
        else:
            sum_bias += (0.5 - scale_heterogeneity) * 2.0
        
        # Correlation strength (0-1, higher means more correlated)
        if correlation_strength > 0.3:
            sum_bias += correlation_strength
        else:
            product_bias += (0.3 - correlation_strength) * 1.5
        
        # Variability score (0-1, higher means more variable)
        if variability_score > 0.4:
            product_bias += (variability_score - 0.4) * 1.5
        
        # Normalize to get weights
        total_bias = sum_bias + product_bias
        if total_bias > 0:
            sum_weight = sum_bias / total_bias
            product_weight = product_bias / total_bias
        else:
            # Default to equal weights
            sum_weight = product_weight = 0.5
        
        return {
            'sum': sum_weight,
            'product': product_weight,
            'scale_analysis': scale_heterogeneity,
            'correlation_analysis': correlation_strength,
            'variability_analysis': variability_score
        }
    
    def _calculate_scale_heterogeneity(self, matrix: np.ndarray) -> float:
        """Calculate scale heterogeneity across criteria."""
        if matrix.shape[1] < 2:
            return 0.0
        
        # Calculate coefficient of variation for each criterion
        cvs = []
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            mean_val = np.mean(col)
            std_val = np.std(col)
            if mean_val > 1e-10:
                cvs.append(std_val / mean_val)
            else:
                cvs.append(0.0)
        
        # Heterogeneity is the coefficient of variation of the CVs
        if len(cvs) > 1:
            cv_mean = np.mean(cvs)
            cv_std = np.std(cvs)
            if cv_mean > 1e-10:
                heterogeneity = cv_std / cv_mean
            else:
                heterogeneity = 0.0
        else:
            heterogeneity = 0.0
        
        # Normalize to [0, 1]
        return min(1.0, heterogeneity)
    
    def _calculate_correlation_strength(self, matrix: np.ndarray) -> float:
        """Calculate average correlation strength between criteria."""
        if matrix.shape[1] < 2:
            return 0.0
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(matrix.T)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(corr_matrix, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        if len(correlations) > 0:
            # Return average absolute correlation
            return np.mean(np.abs(correlations))
        else:
            return 0.0
    
    def _calculate_variability_score(self, matrix: np.ndarray) -> float:
        """Calculate overall variability score."""
        if matrix.size == 0:
            return 0.0
        
        # Calculate coefficient of variation for the entire matrix
        overall_mean = np.mean(matrix)
        overall_std = np.std(matrix)
        
        if overall_mean > 1e-10:
            variability = overall_std / overall_mean
        else:
            variability = 0.0
        
        # Normalize to [0, 1]
        return min(1.0, variability)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return np.ones_like(scores) * 0.5