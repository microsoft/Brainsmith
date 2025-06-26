"""
Data structures for Phase 3: Build Runner.

This module defines the core data structures used by the build runner system,
including build status tracking, metrics collection, and result aggregation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class BuildStatus(Enum):
    """Status of a build execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class BuildMetrics:
    """Standardized build metrics across all backends."""
    
    # Performance metrics
    throughput: Optional[float] = None          # inferences/second
    latency: Optional[float] = None             # microseconds
    clock_frequency: Optional[float] = None     # MHz
    
    # Resource metrics  
    lut_utilization: Optional[float] = None     # 0.0 to 1.0
    dsp_utilization: Optional[float] = None     # 0.0 to 1.0
    bram_utilization: Optional[float] = None    # 0.0 to 1.0
    uram_utilization: Optional[float] = None    # 0.0 to 1.0
    total_power: Optional[float] = None         # watts
    
    # Quality metrics
    accuracy: Optional[float] = None            # 0.0 to 1.0
    
    # Raw metrics for debugging/analysis
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildResult:
    """Result from a single build execution."""
    
    config_id: str
    status: BuildStatus = BuildStatus.SKIPPED
    metrics: Optional[BuildMetrics] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> file_path
    logs: Dict[str, str] = field(default_factory=dict)      # log_name -> content_or_path
    error_message: Optional[str] = None
    
    def complete(self, status: BuildStatus, error_message: Optional[str] = None):
        """Mark build as complete with given status."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = status
        self.error_message = error_message
        
    def is_successful(self) -> bool:
        """Check if build completed successfully."""
        return self.status == BuildStatus.SUCCESS
        
    def has_metrics(self) -> bool:
        """Check if build has valid metrics."""
        return self.metrics is not None and self.status == BuildStatus.SUCCESS