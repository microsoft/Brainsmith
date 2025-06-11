"""
Data Management Functions

Higher-level data management functionality including data lifecycle,
caching, and automated data processing workflows.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import hashlib

from .collection import collect_build_metrics, collect_dse_metrics, summarize_data
from .export import export_metrics, export_summary, export_dse_analysis
from .types import BuildMetrics, MetricsList, DataSummary

logger = logging.getLogger(__name__)


class DataManager:
    """
    Centralized data manager for build metrics and analysis results.
    
    Provides high-level interface for data collection, caching, and export.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data manager.
        
        Args:
            cache_dir: Directory for caching data (default: ./cache)
        """
        self.cache_dir = Path(cache_dir or "./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._metrics_cache: Dict[str, BuildMetrics] = {}
        self._summary_cache: Dict[str, DataSummary] = {}
        
        logger.info(f"Data manager initialized with cache: {self.cache_dir}")
    
    def collect_and_cache_metrics(
        self,
        build_result: Any,
        model_path: Optional[str] = None,
        blueprint_path: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None
    ) -> BuildMetrics:
        """
        Collect metrics and store in cache.
        
        Args:
            build_result: Build result to collect metrics from
            model_path: Path to model file
            blueprint_path: Path to blueprint file
            parameters: Build parameters
            cache_key: Optional cache key (auto-generated if not provided)
            
        Returns:
            Collected BuildMetrics
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(model_path, blueprint_path, parameters)
        
        # Check cache first
        if cache_key in self._metrics_cache:
            logger.debug(f"Using cached metrics for key: {cache_key}")
            return self._metrics_cache[cache_key]
        
        # Collect metrics
        metrics = collect_build_metrics(build_result, model_path, blueprint_path, parameters)
        
        # Store in cache
        self._metrics_cache[cache_key] = metrics
        
        # Optionally persist to disk
        self._persist_metrics(metrics, cache_key)
        
        logger.info(f"Collected and cached metrics: {cache_key}")
        return metrics
    
    def collect_and_export_dse_results(
        self,
        dse_results: List[Any],
        output_dir: str,
        analysis_name: str = "dse_analysis"
    ) -> MetricsList:
        """
        Collect DSE metrics and export complete analysis.
        
        Args:
            dse_results: List of DSE results
            output_dir: Directory to export analysis
            analysis_name: Name for this analysis
            
        Returns:
            List of collected BuildMetrics
        """
        logger.info(f"Processing {len(dse_results)} DSE results for analysis: {analysis_name}")
        
        # Collect metrics from all DSE results
        metrics_list = collect_dse_metrics(dse_results)
        
        # Export complete analysis
        export_dse_analysis(metrics_list, output_dir, include_pareto=True)
        
        # Cache summary
        summary = summarize_data(metrics_list)
        summary_key = f"{analysis_name}_summary"
        self._summary_cache[summary_key] = summary
        
        logger.info(f"DSE analysis complete: {len(metrics_list)} metrics processed")
        return metrics_list
    
    def batch_process_results(
        self,
        results: List[Dict[str, Any]],
        output_dir: str,
        batch_name: str = "batch_analysis"
    ) -> DataSummary:
        """
        Batch process multiple build results.
        
        Args:
            results: List of result dictionaries with 'build_result', 'model_path', etc.
            output_dir: Directory to export results
            batch_name: Name for this batch
            
        Returns:
            Summary of all processed results
        """
        logger.info(f"Batch processing {len(results)} results: {batch_name}")
        
        metrics_list = []
        
        for i, result_info in enumerate(results):
            try:
                metrics = collect_build_metrics(
                    build_result=result_info.get('build_result'),
                    model_path=result_info.get('model_path'),
                    blueprint_path=result_info.get('blueprint_path'),
                    parameters=result_info.get('parameters', {})
                )
                
                # Add batch metadata
                metrics.metadata.update({
                    'batch_name': batch_name,
                    'batch_index': i,
                    'batch_total': len(results)
                })
                
                metrics_list.append(metrics)
                
            except Exception as e:
                logger.error(f"Failed to process result {i}: {e}")
                continue
        
        # Export batch results
        export_metrics(metrics_list, f"{output_dir}/{batch_name}_results.json", 'json')
        export_metrics(metrics_list, f"{output_dir}/{batch_name}_results.csv", 'csv')
        
        # Create and export summary
        summary = summarize_data(metrics_list)
        export_summary(summary, f"{output_dir}/{batch_name}_summary.json")
        
        logger.info(f"Batch processing complete: {len(metrics_list)} successful, {summary.success_rate:.1%} success rate")
        return summary
    
    def get_cached_metrics(self, cache_key: str) -> Optional[BuildMetrics]:
        """Get metrics from cache."""
        return self._metrics_cache.get(cache_key)
    
    def get_cached_summary(self, summary_key: str) -> Optional[DataSummary]:
        """Get summary from cache."""
        return self._summary_cache.get(summary_key)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._metrics_cache.clear()
        self._summary_cache.clear()
        logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'metrics_cached': len(self._metrics_cache),
            'summaries_cached': len(self._summary_cache),
            'cache_dir': str(self.cache_dir),
            'cache_size_mb': self._get_cache_size_mb()
        }
    
    def _generate_cache_key(
        self,
        model_path: Optional[str],
        blueprint_path: Optional[str], 
        parameters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key from inputs."""
        key_components = [
            model_path or "no_model",
            blueprint_path or "no_blueprint",
            str(parameters) if parameters else "no_params",
            str(int(time.time() // 3600))  # Hour-based cache expiry
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _persist_metrics(self, metrics: BuildMetrics, cache_key: str):
        """Persist metrics to disk cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                f.write(metrics.to_json())
        except Exception as e:
            logger.warning(f"Failed to persist metrics: {e}")
    
    def _get_cache_size_mb(self) -> float:
        """Calculate cache directory size in MB."""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0


# Global data manager instance
_global_data_manager: Optional[DataManager] = None


def get_data_manager(cache_dir: Optional[str] = None) -> DataManager:
    """
    Get global data manager instance.
    
    Args:
        cache_dir: Cache directory (only used for first initialization)
        
    Returns:
        Global DataManager instance
    """
    global _global_data_manager
    
    if _global_data_manager is None:
        _global_data_manager = DataManager(cache_dir)
    
    return _global_data_manager


def set_data_manager(data_manager: DataManager):
    """Set global data manager instance."""
    global _global_data_manager
    _global_data_manager = data_manager


# Convenience functions using global data manager

def collect_and_cache(
    build_result: Any,
    model_path: Optional[str] = None,
    blueprint_path: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> BuildMetrics:
    """Collect and cache metrics using global data manager."""
    return get_data_manager().collect_and_cache_metrics(
        build_result, model_path, blueprint_path, parameters
    )


def export_complete_analysis(
    dse_results: List[Any],
    output_dir: str,
    analysis_name: str = "analysis"
) -> MetricsList:
    """Export complete DSE analysis using global data manager."""
    return get_data_manager().collect_and_export_dse_results(
        dse_results, output_dir, analysis_name
    )


def process_batch_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    batch_name: str = "batch"
) -> DataSummary:
    """Process batch results using global data manager."""
    return get_data_manager().batch_process_results(
        results, output_dir, batch_name
    )


# Data validation and quality functions

def validate_metrics_quality(metrics_list: MetricsList) -> Dict[str, Any]:
    """
    Validate quality of collected metrics.
    
    Args:
        metrics_list: List of metrics to validate
        
    Returns:
        Validation report
    """
    report = {
        'total_metrics': len(metrics_list),
        'successful_builds': 0,
        'complete_metrics': 0,
        'issues': [],
        'quality_score': 0.0
    }
    
    if not metrics_list:
        report['issues'].append("No metrics to validate")
        return report
    
    complete_count = 0
    
    for i, metrics in enumerate(metrics_list):
        # Check build success
        if metrics.build.build_success:
            report['successful_builds'] += 1
        
        # Check completeness
        has_performance = metrics.performance.throughput_ops_sec is not None
        has_resources = metrics.resources.lut_utilization_percent is not None
        
        if has_performance and has_resources:
            complete_count += 1
        elif metrics.build.build_success:
            report['issues'].append(f"Metric {i}: successful build but incomplete data")
    
    report['complete_metrics'] = complete_count
    
    # Calculate quality score
    success_rate = report['successful_builds'] / len(metrics_list)
    completeness_rate = complete_count / len(metrics_list)
    report['quality_score'] = (success_rate + completeness_rate) / 2.0
    
    if report['quality_score'] < 0.5:
        report['issues'].append("Low overall data quality (< 50%)")
    
    return report


def cleanup_old_cache(cache_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean up old cache files.
    
    Args:
        cache_dir: Cache directory path
        max_age_hours: Maximum age of cache files in hours
        
    Returns:
        Number of files cleaned up
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return 0
    
    cutoff_time = time.time() - (max_age_hours * 3600)
    cleaned_count = 0
    
    try:
        for cache_file in cache_path.glob("*.json"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old cache files")
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        return 0