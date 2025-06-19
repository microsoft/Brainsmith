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
    from brainsmith.core.dse import DesignSpaceExplorer
    from brainsmith.core.blueprint import load_blueprint

    blueprint = load_blueprint("bert_accelerator.yaml")
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
    DSEResults,
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
    'DSEResults',
    'ResultsAnalyzer',
    'ParetoFrontierAnalyzer'
]

__version__ = "2.0.0"
__author__ = "BrainSmith Development Team"