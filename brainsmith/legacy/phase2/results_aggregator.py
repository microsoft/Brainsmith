"""
Results aggregator for analyzing and summarizing exploration outcomes.

This module provides functionality to aggregate build results, find optimal
configurations, and calculate summary statistics across the exploration.
"""

import logging
import statistics
from typing import Dict, List, Optional, Tuple

from .data_structures import BuildResult, BuildStatus, ExplorationResults
from ..phase1.data_structures import BuildMetrics


logger = logging.getLogger(__name__)


class ResultsAggregator:
    """
    Aggregate and analyze results from design space exploration.
    
    This class handles finding best configurations, computing Pareto frontiers,
    and calculating summary statistics across all build results.
    """
    
    def __init__(self, exploration_results: ExplorationResults):
        """
        Initialize the aggregator with exploration results.
        
        Args:
            exploration_results: The exploration results to aggregate
        """
        self.results = exploration_results
        self._successful_results: Optional[List[BuildResult]] = None
    
    def add_result(self, result: BuildResult):
        """
        Add a new result to the exploration results.
        
        Args:
            result: The build result to add
        """
        self.results.evaluations.append(result)
        self.results.update_counts()
        
        # Clear cached successful results
        self._successful_results = None
        
        logger.debug(f"Added result for {result.config_id}: {result.status.value}")
    
    def finalize(self):
        """
        Finalize the exploration results with analysis.
        
        This method finds the best configuration, calculates the Pareto frontier,
        and computes summary statistics for all metrics.
        """
        logger.info("Finalizing exploration results")
        
        # Update counts
        self.results.update_counts()
        
        # Find best configuration
        self.results.best_config = self._find_best_config()
        
        # Find Pareto optimal configurations
        self.results.pareto_optimal = self._find_pareto_optimal()
        
        # Calculate metrics summary
        self.results.metrics_summary = self._calculate_metrics_summary()
        
        logger.info(
            f"Finalized results: {self.results.success_count} successful, "
            f"{self.results.failure_count} failed, "
            f"{len(self.results.pareto_optimal)} Pareto optimal"
        )
    
    def _get_successful_results(self) -> List[BuildResult]:
        """Get cached list of successful results."""
        if self._successful_results is None:
            self._successful_results = self.results.get_successful_results()
        return self._successful_results
    
    def _find_best_config(self):
        """
        Find the best configuration based on primary metric (throughput).
        
        Returns:
            The BuildConfig with highest throughput, or None if no successful builds
        """
        successful = self._get_successful_results()
        if not successful:
            logger.warning("No successful builds to find best configuration")
            return None
        
        # Find result with highest throughput
        best_result = max(
            successful,
            key=lambda r: r.metrics.throughput if r.metrics else 0
        )
        
        # Get the corresponding config
        best_config = self.results.get_config(best_result.config_id)
        
        if best_config and best_result.metrics:
            logger.info(
                f"Best configuration: {best_config.id} with "
                f"throughput={best_result.metrics.throughput:.2f}"
            )
        
        return best_config
    
    def _find_pareto_optimal(self) -> List:
        """
        Find Pareto optimal configurations (2D: throughput vs resources).
        
        A configuration is Pareto optimal if no other configuration is better
        in all objectives. We optimize for high throughput and low resource usage.
        
        Returns:
            List of BuildConfig objects on the Pareto frontier
        """
        successful = self._get_successful_results()
        if not successful:
            return []
        
        # Extract points: (throughput, -resource_usage, result)
        # Using negative resource usage so we can maximize both
        points = []
        for result in successful:
            if result.metrics:
                # Combine resource metrics (lower is better, so negate)
                resource_usage = (
                    result.metrics.lut_utilization +
                    result.metrics.dsp_utilization +
                    result.metrics.bram_utilization
                ) / 3.0
                
                points.append((
                    result.metrics.throughput,
                    -resource_usage,  # Negative so higher is better
                    result
                ))
        
        if not points:
            return []
        
        # Find Pareto frontier using simple algorithm
        pareto_results = []
        
        for i, (tp1, res1, result1) in enumerate(points):
            is_dominated = False
            
            for j, (tp2, res2, result2) in enumerate(points):
                if i == j:
                    continue
                
                # Check if point j dominates point i
                # (higher throughput AND lower resource usage)
                if tp2 >= tp1 and res2 >= res1 and (tp2 > tp1 or res2 > res1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_results.append(result1)
        
        # Get configs for Pareto optimal results
        pareto_configs = []
        for result in pareto_results:
            config = self.results.get_config(result.config_id)
            if config:
                pareto_configs.append(config)
        
        logger.info(f"Found {len(pareto_configs)} Pareto optimal configurations")
        return pareto_configs
    
    def _calculate_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for each metric.
        
        Returns:
            Dictionary mapping metric names to statistics (min, max, mean, std)
        """
        successful = self._get_successful_results()
        if not successful:
            return {}
        
        # Collect all metrics
        metric_values = {
            "throughput": [],
            "latency": [],
            "clock_frequency": [],
            "lut_utilization": [],
            "dsp_utilization": [],
            "bram_utilization": [],
            "total_power": [],
            "accuracy": [],
        }
        
        for result in successful:
            if result.metrics:
                for metric_name in metric_values:
                    value = getattr(result.metrics, metric_name, None)
                    if value is not None:
                        metric_values[metric_name].append(value)
        
        # Calculate statistics
        summary = {}
        for metric_name, values in metric_values.items():
            if values:
                summary[metric_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
        
        logger.debug(f"Calculated summary for {len(summary)} metrics")
        return summary
    
    def get_top_n_configs(self, n: int = 10, metric: str = "throughput") -> List[Tuple]:
        """
        Get the top N configurations by a specific metric.
        
        Args:
            n: Number of configurations to return
            metric: Metric to sort by (default: throughput)
            
        Returns:
            List of (BuildConfig, BuildResult) tuples sorted by metric
        """
        successful = self._get_successful_results()
        if not successful:
            return []
        
        # Sort by metric
        sorted_results = sorted(
            successful,
            key=lambda r: getattr(r.metrics, metric, 0) if r.metrics else 0,
            reverse=True  # Higher is better for most metrics
        )
        
        # Get top N with their configs
        top_configs = []
        for result in sorted_results[:n]:
            config = self.results.get_config(result.config_id)
            if config:
                top_configs.append((config, result))
        
        return top_configs
    
    def get_failed_summary(self) -> Dict[str, int]:
        """
        Get summary of failure reasons.
        
        Returns:
            Dictionary mapping error messages to counts
        """
        failed = self.results.get_failed_results()
        if not failed:
            return {}
        
        error_counts = {}
        for result in failed:
            error = result.error_message or "Unknown error"
            error_counts[error] = error_counts.get(error, 0) + 1
        
        return error_counts