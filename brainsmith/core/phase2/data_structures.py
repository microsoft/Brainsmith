"""
Data structures for Phase 2: Design Space Explorer.

This module defines the core data structures used during design space exploration,
including build configurations, results, and aggregated exploration outcomes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from ..phase1.data_structures import ProcessingStep, GlobalConfig, BuildMetrics


class BuildStatus(Enum):
    """Status of a build execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class BuildConfig:
    """
    Configuration for a single build run.
    
    This represents a specific point in the design space with all parameters
    fixed to specific values.
    """
    id: str                              # Unique identifier (e.g., "config_00001")
    design_space_id: str                 # Links to parent design space
    model_path: str                      # Path to ONNX model file for this build
    
    # Specific selections from the design space
    kernels: List[Tuple[str, List[str]]] # Selected kernel configurations
    transforms: Dict[str, List[str]]      # Selected transforms by stage/phase
    preprocessing: List[ProcessingStep]   # Selected preprocessing steps
    postprocessing: List[ProcessingStep]  # Selected postprocessing steps
    
    # Fixed configuration from design space
    build_steps: List[str]               # From hw_compiler_space
    config_flags: Dict[str, Any]         # From hw_compiler_space
    global_config: GlobalConfig          # From design space
    
    # Output directory for this specific build
    output_dir: str = ""                 # Path where Phase 3 will store artifacts
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    combination_index: int = 0           # Which combination number this is
    total_combinations: int = 0          # Total in the design space
    
    def __str__(self) -> str:
        kernel_str = ", ".join(f"{k[0]}[{','.join(k[1])}]" for k in self.kernels)
        # Format transforms by stage
        transform_parts = []
        for stage, transforms in self.transforms.items():
            if transforms:
                transform_parts.append(f"{stage}: {', '.join(transforms)}")
        transform_str = " | ".join(transform_parts) if transform_parts else "none"
        
        return (
            f"BuildConfig {self.id} ({self.combination_index + 1}/{self.total_combinations})\n"
            f"  Kernels: {kernel_str}\n"
            f"  Transforms: {transform_str}\n"
            f"  Output: {self.output_dir}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "design_space_id": self.design_space_id,
            "model_path": self.model_path,
            "kernels": self.kernels,
            "transforms": self.transforms,
            "preprocessing": [
                {"name": s.name, "type": s.type, "parameters": s.parameters, "enabled": s.enabled}
                for s in self.preprocessing
            ],
            "postprocessing": [
                {"name": s.name, "type": s.type, "parameters": s.parameters, "enabled": s.enabled}
                for s in self.postprocessing
            ],
            "build_steps": self.build_steps,
            "config_flags": self.config_flags,
            "global_config": {
                "output_stage": self.global_config.output_stage.value,
                "working_directory": self.global_config.working_directory,
                "cache_results": self.global_config.cache_results,
                "save_artifacts": self.global_config.save_artifacts,
                "log_level": self.global_config.log_level,
                "start_step": self.global_config.start_step,
                "stop_step": self.global_config.stop_step,
                "input_type": self.global_config.input_type,
                "output_type": self.global_config.output_type,
            },
            "timestamp": self.timestamp.isoformat(),
            "combination_index": self.combination_index,
            "total_combinations": self.total_combinations,
            "output_dir": self.output_dir,
        }


@dataclass
class BuildResult:
    """
    Result from a single build run.
    
    Captures the outcome of executing a BuildConfig, including success/failure,
    metrics, timing information, and any artifacts produced.
    """
    config_id: str                       # Links back to BuildConfig
    status: BuildStatus                  # Success, failure, timeout, skipped
    
    # Metrics (if successful)
    metrics: Optional[BuildMetrics] = None
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Artifacts and logs
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> file_path
    logs: Dict[str, str] = field(default_factory=dict)       # log_type -> content
    error_message: Optional[str] = None  # If failed
    
    def complete(self, status: BuildStatus, error_message: Optional[str] = None):
        """Mark the build as complete with the given status."""
        self.status = status
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        if error_message:
            self.error_message = error_message
    
    def __str__(self) -> str:
        status_str = self.status.value
        if self.metrics:
            metrics_str = (
                f"throughput={self.metrics.throughput:.2f}, "
                f"latency={self.metrics.latency:.2f}μs"
            )
        else:
            metrics_str = "no metrics"
        
        return f"BuildResult({self.config_id}: {status_str}, {metrics_str})"


@dataclass
class ExplorationResults:
    """
    Aggregated results from design space exploration.
    
    Contains all individual build results plus summary statistics and analysis.
    """
    design_space_id: str
    start_time: datetime
    end_time: datetime
    
    # All build results
    evaluations: List[BuildResult] = field(default_factory=list)
    
    # Summary statistics
    total_combinations: int = 0
    evaluated_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    
    # Best configurations
    best_config: Optional[BuildConfig] = None      # Based on primary metric
    pareto_optimal: List[BuildConfig] = field(default_factory=list)  # Pareto frontier
    
    # Aggregated metrics
    metrics_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)  # metric -> {min, max, mean, std}
    
    # Store configs for reference
    _config_map: Dict[str, BuildConfig] = field(default_factory=dict, init=False)
    
    def add_config(self, config: BuildConfig):
        """Store a build configuration for later reference."""
        self._config_map[config.id] = config
    
    def get_config(self, config_id: str) -> Optional[BuildConfig]:
        """Retrieve a build configuration by ID."""
        return self._config_map.get(config_id)
    
    def get_successful_results(self) -> List[BuildResult]:
        """Get only successful build results."""
        return [r for r in self.evaluations if r.status == BuildStatus.SUCCESS]
    
    def get_failed_results(self) -> List[BuildResult]:
        """Get only failed build results."""
        return [r for r in self.evaluations if r.status == BuildStatus.FAILED]
    
    def update_counts(self):
        """Update summary counts based on current evaluations."""
        self.evaluated_count = len(self.evaluations)
        self.success_count = len([r for r in self.evaluations if r.status == BuildStatus.SUCCESS])
        self.failure_count = len([r for r in self.evaluations if r.status == BuildStatus.FAILED])
        self.skipped_count = len([r for r in self.evaluations if r.status == BuildStatus.SKIPPED])
    
    def get_summary_string(self) -> str:
        """Get a human-readable summary of the exploration."""
        duration = (self.end_time - self.start_time).total_seconds()
        success_rate = (self.success_count / self.evaluated_count * 100) if self.evaluated_count > 0 else 0
        
        summary = [
            f"Exploration Results Summary",
            f"==========================",
            f"Design Space ID: {self.design_space_id}",
            f"Duration: {duration:.1f} seconds",
            f"Total Combinations: {self.total_combinations}",
            f"Evaluated: {self.evaluated_count}",
            f"Successful: {self.success_count} ({success_rate:.1f}%)",
            f"Failed: {self.failure_count}",
            f"Skipped: {self.skipped_count}",
        ]
        
        if self.best_config:
            summary.append(f"\nBest Configuration: {self.best_config.id}")
            if self.best_config.id in self._config_map:
                best_result = next(
                    (r for r in self.evaluations if r.config_id == self.best_config.id),
                    None
                )
                if best_result and best_result.metrics:
                    summary.append(
                        f"  Throughput: {best_result.metrics.throughput:.2f} inferences/sec"
                    )
                    summary.append(
                        f"  Latency: {best_result.metrics.latency:.2f} μs"
                    )
        
        if self.pareto_optimal:
            summary.append(f"\nPareto Optimal Set: {len(self.pareto_optimal)} configurations")
        
        return "\n".join(summary)