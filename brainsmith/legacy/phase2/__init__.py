"""
Phase 2: Design Space Explorer

This module handles systematic exploration of design spaces created in Phase 1.
"""

from .data_structures import (
    BuildConfig,
    BuildResult,
    BuildStatus,
    ExplorationResults,
)
from .interfaces import BuildRunnerInterface, MockBuildRunner
from .explorer import ExplorerEngine
# from .combination_generator import CombinationGenerator  # Removed - obsolete with execution tree
from .results_aggregator import ResultsAggregator
from .hooks import ExplorationHook, LoggingHook, CachingHook, HookRegistry
from .progress import ProgressTracker

# Convenience function
def explore(design_space, build_runner_factory, hooks=None, resume_from=None):
    """
    Explore a design space with the given build runner.
    
    Args:
        design_space: DesignSpace object from Phase 1
        build_runner_factory: Callable that returns a BuildRunnerInterface
        hooks: Optional list of ExplorationHook instances
        resume_from: Optional checkpoint ID to resume from
        
    Returns:
        ExplorationResults with all build outcomes and analysis
    """
    explorer = ExplorerEngine(build_runner_factory, hooks)
    return explorer.explore(design_space, resume_from)


__all__ = [
    # Data structures
    "BuildConfig",
    "BuildResult", 
    "BuildStatus",
    "ExplorationResults",
    # Core components
    "BuildRunnerInterface",
    "MockBuildRunner",
    "ExplorerEngine",
    # "CombinationGenerator",  # Removed - obsolete with execution tree
    "ResultsAggregator",
    # Hooks
    "ExplorationHook",
    "LoggingHook",
    "CachingHook",
    "HookRegistry",
    # Progress
    "ProgressTracker",
    # API
    "explore",
]