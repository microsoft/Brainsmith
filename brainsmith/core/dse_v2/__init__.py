"""
DSE Engine V2 - Design Space Exploration for Blueprint V2

This module provides the V2 design space exploration engine that works with
Blueprint V2 design space definitions to generate and evaluate component
combinations for the 6-entrypoint FINN architecture.

Main Components:
- ComponentCombination: Represents a specific combination of components
- CombinationGenerator: Generates valid combinations from design spaces
- StrategyExecutor: Executes different exploration strategies
- DesignSpaceExplorer: Main orchestration class

Usage:
    from brainsmith.core.dse_v2 import DesignSpaceExplorer
    from brainsmith.core.blueprint_v2 import load_blueprint_v2
    
    blueprint = load_blueprint_v2("bert_accelerator_v2.yaml")
    explorer = DesignSpaceExplorer(blueprint)
    results = explorer.explore_design_space("model.onnx")
"""

from .combination_generator import (
    ComponentCombination,
    CombinationGenerator,
    generate_component_combinations
)

from .strategy_executor import (
    StrategyExecutor,
    ExplorationStrategy,
    HierarchicalExplorationStrategy,
    AdaptiveExplorationStrategy,
    ParetoGuidedStrategy
)

from .space_explorer import (
    DesignSpaceExplorer,
    ExplorationResults
)

from .results_analyzer import (
    DSEResultsV2,
    ResultsAnalyzer,
    ParetoFrontierAnalyzer
)

__all__ = [
    # Combination generation
    'ComponentCombination',
    'CombinationGenerator', 
    'generate_component_combinations',
    
    # Strategy execution
    'StrategyExecutor',
    'ExplorationStrategy',
    'HierarchicalExplorationStrategy',
    'AdaptiveExplorationStrategy',
    'ParetoGuidedStrategy',
    
    # Main exploration
    'DesignSpaceExplorer',
    'ExplorationResults',
    
    # Results analysis
    'DSEResultsV2',
    'ResultsAnalyzer',
    'ParetoFrontierAnalyzer'
]

__version__ = "2.0.0"
__author__ = "BrainSmith Development Team"