"""
Explorer engine for systematic design space exploration.

This module provides the main exploration engine that coordinates the entire
exploration process, including combination generation, build execution,
result aggregation, and hook management.
"""

import logging
import os
import time
from datetime import datetime
from typing import Callable, List, Optional

from .data_structures import BuildConfig, BuildResult, BuildStatus, ExplorationResults
from .combination_generator import CombinationGenerator
from .results_aggregator import ResultsAggregator
from .interfaces import BuildRunnerInterface
from .progress import ProgressTracker
from .hooks import ExplorationHook
from ..phase1.data_structures import DesignSpace, SearchStrategy


logger = logging.getLogger(__name__)


class ExplorerEngine:
    """
    Main engine for design space exploration.
    
    This class orchestrates the entire exploration process, managing the
    generation of configurations, execution of builds, and collection of results.
    """
    
    def __init__(
        self,
        build_runner_factory: Callable[[], BuildRunnerInterface],
        hooks: Optional[List[ExplorationHook]] = None
    ):
        """
        Initialize the explorer engine.
        
        Args:
            build_runner_factory: Factory function that creates BuildRunnerInterface instances
            hooks: Optional list of exploration hooks
        """
        self.build_runner_factory = build_runner_factory
        self.hooks = hooks or []
        self.progress_tracker: Optional[ProgressTracker] = None
        self.exploration_results: Optional[ExplorationResults] = None
    
    def explore(
        self,
        design_space: DesignSpace,
        resume_from: Optional[str] = None
    ) -> ExplorationResults:
        """
        Explore the design space systematically.
        
        Args:
            design_space: The design space to explore
            resume_from: Optional configuration ID to resume from
            
        Returns:
            ExplorationResults with all build outcomes and analysis
        """
        logger.info(f"Starting exploration of design space from {design_space.model_path}")
        start_time = datetime.now()
        
        # Initialize exploration results
        combination_gen = CombinationGenerator()
        design_space_id = combination_gen._generate_design_space_id(design_space)
        
        self.exploration_results = ExplorationResults(
            design_space_id=design_space_id,
            start_time=start_time,
            end_time=start_time  # Will be updated at the end
        )
        
        # Create directory structure for exploration
        exploration_dir = os.path.join(
            design_space.global_config.working_directory,
            design_space_id
        )
        builds_dir = os.path.join(exploration_dir, "builds")
        
        # Create directories
        os.makedirs(exploration_dir, exist_ok=True)
        os.makedirs(builds_dir, exist_ok=True)
        logger.info(f"Created exploration directory: {exploration_dir}")
        
        # Fire exploration start hook
        self._fire_hook("on_exploration_start", design_space, self.exploration_results)
        
        # Generate all combinations
        logger.info("Generating combinations from design space")
        all_configs = combination_gen.generate_all(design_space)
        self.exploration_results.total_combinations = len(all_configs)
        
        # Store all configs for later reference
        for config in all_configs:
            self.exploration_results.add_config(config)
        
        # Apply resume if requested
        if resume_from:
            logger.info(f"Resuming from configuration {resume_from}")
            configs_to_evaluate = combination_gen.filter_by_resume(all_configs, resume_from)
        else:
            configs_to_evaluate = all_configs
        
        # Fire combination generated hook
        self._fire_hook("on_combinations_generated", configs_to_evaluate)
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            total_configs=len(configs_to_evaluate),
            start_time=datetime.now()
        )
        
        # Create build runner
        build_runner = self.build_runner_factory()
        
        # Initialize results aggregator
        aggregator = ResultsAggregator(self.exploration_results)
        
        # Main exploration loop
        logger.info(f"Evaluating {len(configs_to_evaluate)} configurations")
        
        for i, config in enumerate(configs_to_evaluate):
            # Check early stopping conditions
            if self._should_stop_early(design_space, i):
                logger.info("Early stopping conditions met")
                break
            
            # Log progress
            if i % 10 == 0:
                progress_summary = self.progress_tracker.get_summary()
                logger.info(progress_summary)
            
            # Evaluate configuration
            logger.debug(f"Evaluating configuration {config.id}")
            result = self._evaluate_config(config, build_runner)
            
            # Add to results
            aggregator.add_result(result)
            
            # Update progress
            self.progress_tracker.update(result)
            
            # Fire build complete hook
            self._fire_hook("on_build_complete", config, result)
        
        # Finalize results
        logger.info("Finalizing exploration results")
        self.exploration_results.end_time = datetime.now()
        aggregator.finalize()
        
        # Fire exploration complete hook
        self._fire_hook("on_exploration_complete", self.exploration_results)
        
        # Log final summary
        logger.info("\n" + self.exploration_results.get_summary_string())
        
        return self.exploration_results
    
    def _evaluate_config(
        self,
        config: BuildConfig,
        build_runner: BuildRunnerInterface
    ) -> BuildResult:
        """
        Evaluate a single configuration.
        
        Args:
            config: The configuration to evaluate
            build_runner: The build runner to use
            
        Returns:
            BuildResult from the evaluation
        """
        try:
            # Run the build (model path is now in config)
            result = build_runner.run(config)
            
            # Log outcome
            if result.status == BuildStatus.SUCCESS:
                if result.metrics:
                    logger.debug(
                        f"Build {config.id} succeeded: "
                        f"throughput={result.metrics.throughput:.2f}, "
                        f"latency={result.metrics.latency:.2f}μs"
                    )
                else:
                    logger.debug(f"Build {config.id} succeeded with no metrics")
            else:
                logger.debug(
                    f"Build {config.id} failed: {result.error_message or 'Unknown error'}"
                )
            
            return result
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error evaluating {config.id}: {str(e)}")
            
            # Create failed result
            result = BuildResult(
                config_id=config.id,
                status=BuildStatus.FAILED,
                error_message=f"Unexpected error: {str(e)}"
            )
            result.complete(BuildStatus.FAILED, error_message=str(e))
            
            return result
    
    def _should_stop_early(self, design_space: DesignSpace, current_index: int) -> bool:
        """
        Check if exploration should stop early.
        
        Args:
            design_space: The design space being explored
            current_index: Current evaluation index
            
        Returns:
            True if exploration should stop
        """
        # Check max evaluations
        if design_space.search_config.max_evaluations:
            if current_index >= design_space.search_config.max_evaluations:
                logger.info(
                    f"Reached max evaluations limit: "
                    f"{design_space.search_config.max_evaluations}"
                )
                return True
        
        # Check timeout
        if design_space.search_config.timeout_minutes:
            elapsed = (datetime.now() - self.exploration_results.start_time).total_seconds() / 60
            if elapsed >= design_space.search_config.timeout_minutes:
                logger.info(
                    f"Reached timeout limit: "
                    f"{design_space.search_config.timeout_minutes} minutes"
                )
                return True
        
        # Check if all constraints are satisfied
        # (In advanced strategies, we might stop if we found good enough solutions)
        
        return False
    
    def _fire_hook(self, method_name: str, *args, **kwargs):
        """
        Fire a hook method on all registered hooks.
        
        Args:
            method_name: Name of the hook method to call
            *args: Positional arguments to pass to the hook
            **kwargs: Keyword arguments to pass to the hook
        """
        for hook in self.hooks:
            try:
                method = getattr(hook, method_name, None)
                if method and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook {hook.__class__.__name__}.{method_name}: {str(e)}")
                # Continue with other hooks even if one fails