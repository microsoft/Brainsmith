"""
Intelligent Solution Selection Framework

This module provides multi-criteria decision analysis (MCDA) algorithms and tools
for automated selection of optimal designs from Pareto solution sets.

Key Features:
- Multi-criteria decision analysis algorithms (TOPSIS, PROMETHEE, AHP)
- User preference specification and elicitation
- Automated solution ranking and filtering
- Trade-off analysis and compromise solutions

Main Components:
1. Selection Engine: Core MCDA algorithms and selection logic
2. Selection Strategies: Specific MCDA algorithm implementations
3. Preference Management: User preference specification and elicitation
4. Solution Ranking: Automated ranking and filtering capabilities

Example Usage:
    from brainsmith.selection import SelectionEngine, SelectionCriteria
    
    # Create selection criteria
    criteria = SelectionCriteria(
        objectives=['maximize_throughput', 'minimize_power'],
        weights={'maximize_throughput': 0.6, 'minimize_power': 0.4},
        constraints=['lut_budget', 'timing_closure']
    )
    
    # Select best solutions
    engine = SelectionEngine(strategy='topsis')
    ranked_solutions = engine.select_solutions(pareto_solutions, criteria)
    
    # Get top recommendations
    best_solutions = ranked_solutions[:5]
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

# Core selection components
from .engine import (
    SelectionEngine,
    SelectionResult,
    SelectionConfiguration
)

# Selection strategies
from .strategies import (
    SelectionStrategy,
    TOPSISSelector,
    PROMETHEESelector,
    AHPSelector,
    WeightedSumSelector,
    WeightedProductSelector,
    FuzzyTOPSISSelector
)

# Preference management
from .preferences import (
    PreferenceManager,
    SelectionCriteria,
    UserPreferences,
    PreferenceElicitation,
    InteractiveElicitation,
    WeightElicitation
)

# Solution ranking
from .ranking import (
    SolutionRanker,
    RankedSolution,
    RankingMethod,
    TradeOffAnalyzer,
    CompromiseSolution
)

# Data models
from .models import (
    SelectionContext,
    SelectionMetrics,
    SelectionReport,
    PreferenceFunction,
    DecisionMatrix
)

# Utilities
from .utils import (
    normalize_matrix,
    calculate_weights,
    validate_preferences,
    create_selection_criteria
)

# Setup logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BrainSmith Development Team"

# Export all public components
__all__ = [
    # Core engine
    'SelectionEngine',
    'SelectionResult',
    'SelectionConfiguration',
    
    # Selection strategies
    'SelectionStrategy',
    'TOPSISSelector',
    'PROMETHEESelector',
    'AHPSelector',
    'WeightedSumSelector',
    'WeightedProductSelector',
    'FuzzyTOPSISSelector',
    
    # Preference management
    'PreferenceManager',
    'SelectionCriteria',
    'UserPreferences',
    'PreferenceElicitation',
    'InteractiveElicitation',
    'WeightElicitation',
    
    # Solution ranking
    'SolutionRanker',
    'RankedSolution',
    'RankingMethod',
    'TradeOffAnalyzer',
    'CompromiseSolution',
    
    # Data models
    'SelectionContext',
    'SelectionMetrics',
    'SelectionReport',
    'PreferenceFunction',
    'DecisionMatrix',
    
    # Utilities
    'normalize_matrix',
    'calculate_weights',
    'validate_preferences',
    'create_selection_criteria'
]

# Initialize logging
logger.info(f"Selection Framework v{__version__} initialized")
logger.info("Available MCDA algorithms: TOPSIS, PROMETHEE, AHP, Weighted Sum/Product, Fuzzy TOPSIS")