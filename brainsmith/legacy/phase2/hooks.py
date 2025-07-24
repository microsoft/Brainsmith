"""
Hook system for extending exploration behavior.

This module provides a hook-based extension mechanism that allows users to
inject custom behavior at various points during the exploration process.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .data_structures import BuildConfig, BuildResult, BuildStatus, ExplorationResults
from ..phase1.data_structures import DesignSpace


logger = logging.getLogger(__name__)


class ExplorationHook(ABC):
    """
    Abstract base class for exploration hooks.
    
    Hooks allow injection of custom behavior at key points during exploration.
    """
    
    @abstractmethod
    def on_exploration_start(
        self,
        design_space: DesignSpace,
        exploration_results: ExplorationResults
    ):
        """
        Called when exploration begins.
        
        Args:
            design_space: The design space being explored
            exploration_results: The exploration results container
        """
        pass
    
    @abstractmethod
    def on_combinations_generated(self, configs: List[BuildConfig]):
        """
        Called after all combinations have been generated.
        
        Args:
            configs: List of all configurations to be evaluated
        """
        pass
    
    @abstractmethod
    def on_build_complete(self, config: BuildConfig, result: BuildResult):
        """
        Called after each build completes.
        
        Args:
            config: The configuration that was built
            result: The result of the build
        """
        pass
    
    @abstractmethod
    def on_exploration_complete(self, exploration_results: ExplorationResults):
        """
        Called when exploration finishes.
        
        Args:
            exploration_results: Final exploration results
        """
        pass


class LoggingHook(ExplorationHook):
    """
    Hook that provides detailed logging of the exploration process.
    """
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize the logging hook.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file path to write logs to
        """
        self.log_level = log_level
        self.log_file = log_file
        self.start_time: Optional[datetime] = None
        
        # Set up file handler if requested
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setLevel(getattr(logging, log_level))
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def on_exploration_start(
        self,
        design_space: DesignSpace,
        exploration_results: ExplorationResults
    ):
        """Log exploration start."""
        self.start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("DESIGN SPACE EXPLORATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Model: {design_space.model_path}")
        logger.info(f"Design Space ID: {exploration_results.design_space_id}")
        logger.info(f"Search Strategy: {design_space.search_config.strategy.value}")
        logger.info(f"Total Combinations: {exploration_results.total_combinations}")
        logger.info(f"Max Evaluations: {design_space.search_config.max_evaluations or 'None'}")
        logger.info(f"Timeout: {design_space.search_config.timeout_minutes or 'None'} minutes")
        logger.info("=" * 80)
    
    def on_combinations_generated(self, configs: List[BuildConfig]):
        """Log combination generation summary."""
        logger.info(f"Generated {len(configs)} configurations to evaluate")
        
        if logger.isEnabledFor(logging.DEBUG):
            # Log first few configs as examples
            for i, config in enumerate(configs[:3]):
                logger.debug(f"Example config {i+1}: {config}")
    
    def on_build_complete(self, config: BuildConfig, result: BuildResult):
        """Log build completion."""
        status_emoji = {
            BuildStatus.SUCCESS: "✅",
            BuildStatus.FAILED: "❌",
            BuildStatus.TIMEOUT: "⏰",
            BuildStatus.SKIPPED: "⏩",
        }.get(result.status, "❓")
        
        msg = f"{status_emoji} Build {config.id} ({config.combination_index + 1}/{config.total_combinations}): {result.status.value}"
        
        if result.status == BuildStatus.SUCCESS and result.metrics:
            msg += f" | Throughput: {result.metrics.throughput:.2f} | Latency: {result.metrics.latency:.2f}μs"
        elif result.status == BuildStatus.FAILED:
            msg += f" | Error: {result.error_message or 'Unknown'}"
        
        logger.info(msg)
    
    def on_exploration_complete(self, exploration_results: ExplorationResults):
        """Log exploration completion summary."""
        duration = (exploration_results.end_time - exploration_results.start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("DESIGN SPACE EXPLORATION COMPLETED")
        logger.info("=" * 80)
        logger.info(exploration_results.get_summary_string())
        logger.info(f"\nTotal Duration: {duration:.1f} seconds")
        
        # Log failure summary if any
        failed_results = exploration_results.get_failed_results()
        if failed_results:
            logger.info(f"\nFailure Summary ({len(failed_results)} failures):")
            error_counts: Dict[str, int] = {}
            for result in failed_results:
                error = result.error_message or "Unknown error"
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {error}: {count} occurrences")
        
        logger.info("=" * 80)


class CachingHook(ExplorationHook):
    """
    Hook that implements result caching to support resume functionality.
    """
    
    def __init__(self, cache_dir: str = ".brainsmith_cache"):
        """
        Initialize the caching hook.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file: Optional[Path] = None
        self.design_space_id: Optional[str] = None
    
    def on_exploration_start(
        self,
        design_space: DesignSpace,
        exploration_results: ExplorationResults
    ):
        """Initialize cache for this exploration."""
        self.design_space_id = exploration_results.design_space_id
        
        # Use exploration directory instead of generic cache dir
        exploration_dir = Path(design_space.global_config.working_directory) / self.design_space_id
        exploration_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache file in exploration directory
        self.cache_file = exploration_dir / "exploration_cache.jsonl"
        
        # Load existing results if resuming
        if self.cache_file.exists():
            logger.info(f"Loading cached results from {self.cache_file}")
            self._load_cached_results(exploration_results)
    
    def on_combinations_generated(self, configs: List[BuildConfig]):
        """No action needed."""
        pass
    
    def on_build_complete(self, config: BuildConfig, result: BuildResult):
        """Cache each result as it completes."""
        if not self.cache_file:
            return
        
        # Append result to cache file
        cache_entry = {
            "config_id": result.config_id,
            "status": result.status.value,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "duration_seconds": result.duration_seconds,
            "error_message": result.error_message,
            "artifacts": result.artifacts,
            "metrics": result.metrics.__dict__ if result.metrics else None,
        }
        
        with open(self.cache_file, "a") as f:
            f.write(json.dumps(cache_entry) + "\n")
        
        logger.debug(f"Cached result for {config.id}")
    
    def on_exploration_complete(self, exploration_results: ExplorationResults):
        """Save final exploration summary."""
        if not self.cache_file:
            return
        
        # Save summary file in exploration directory
        exploration_dir = self.cache_file.parent
        summary_file = exploration_dir / "exploration_summary.json"
        summary = {
            "design_space_id": exploration_results.design_space_id,
            "start_time": exploration_results.start_time.isoformat(),
            "end_time": exploration_results.end_time.isoformat(),
            "total_combinations": exploration_results.total_combinations,
            "evaluated_count": exploration_results.evaluated_count,
            "success_count": exploration_results.success_count,
            "failure_count": exploration_results.failure_count,
            "skipped_count": exploration_results.skipped_count,
            "best_config_id": exploration_results.best_config.id if exploration_results.best_config else None,
            "pareto_optimal_ids": [c.id for c in exploration_results.pareto_optimal],
            "metrics_summary": exploration_results.metrics_summary,
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved exploration summary to {summary_file}")
    
    def _load_cached_results(self, exploration_results: ExplorationResults):
        """Load previously cached results."""
        loaded_count = 0
        
        with open(self.cache_file, "r") as f:
            for line in f:
                try:
                    cache_entry = json.loads(line.strip())
                    
                    # Reconstruct BuildResult
                    result = BuildResult(
                        config_id=cache_entry["config_id"],
                        status=BuildStatus(cache_entry["status"]),
                        start_time=datetime.fromisoformat(cache_entry["start_time"]),
                        error_message=cache_entry.get("error_message"),
                        artifacts=cache_entry.get("artifacts", {}),
                    )
                    
                    if cache_entry.get("end_time"):
                        result.end_time = datetime.fromisoformat(cache_entry["end_time"])
                        result.duration_seconds = cache_entry.get("duration_seconds", 0)
                    
                    # Reconstruct metrics if present
                    if cache_entry.get("metrics"):
                        from ..phase1.data_structures import BuildMetrics
                        result.metrics = BuildMetrics(**cache_entry["metrics"])
                    
                    exploration_results.evaluations.append(result)
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load cached result: {str(e)}")
        
        exploration_results.update_counts()
        logger.info(f"Loaded {loaded_count} cached results")


class HookRegistry:
    """
    Registry for managing exploration hooks.
    """
    
    def __init__(self):
        """Initialize the hook registry."""
        self.hooks: List[ExplorationHook] = []
    
    def register(self, hook: ExplorationHook):
        """
        Register a new hook.
        
        Args:
            hook: The hook to register
        """
        self.hooks.append(hook)
        logger.debug(f"Registered hook: {hook.__class__.__name__}")
    
    def unregister(self, hook: ExplorationHook):
        """
        Unregister a hook.
        
        Args:
            hook: The hook to unregister
        """
        if hook in self.hooks:
            self.hooks.remove(hook)
            logger.debug(f"Unregistered hook: {hook.__class__.__name__}")
    
    def get_all(self) -> List[ExplorationHook]:
        """Get all registered hooks."""
        return self.hooks.copy()