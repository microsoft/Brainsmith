"""
Selection Strategies for Multi-Criteria Decision Analysis

This module implements various MCDA algorithms for solution selection:
- TOPSIS: Technique for Order Preference by Similarity
- PROMETHEE: Preference Ranking Organization Method
- AHP: Analytic Hierarchy Process
- Weighted Sum/Product methods
- Fuzzy TOPSIS for uncertainty handling

Each strategy implements the SelectionStrategy interface and provides
specific algorithmic approaches to ranking Pareto solutions.
"""

from .base import SelectionStrategy
from .topsis import TOPSISSelector
from .promethee import PROMETHEESelector
from .ahp import AHPSelector
from .weighted import WeightedSumSelector, WeightedProductSelector
from .fuzzy import FuzzyTOPSISSelector

__all__ = [
    'SelectionStrategy',
    'TOPSISSelector',
    'PROMETHEESelector',
    'AHPSelector',
    'WeightedSumSelector',
    'WeightedProductSelector',
    'FuzzyTOPSISSelector'
]