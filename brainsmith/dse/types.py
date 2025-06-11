"""
Simple DSE Data Types - North Star Aligned

Essential data structures for design space exploration.
Integrates seamlessly with streamlined BrainSmith modules.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json


@dataclass
class DSEResult:
    """
    Simple DSE result container.
    
    Integrates with:
    - brainsmith.core.metrics.DSEMetrics
    - brainsmith.hooks for event logging
    - brainsmith.analysis for data export
    """
    parameters: Dict[str, Any]
    metrics: Any  # DSEMetrics from brainsmith.core.metrics
    build_success: bool = True
    build_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis hooks and external tools."""
        return {
            'parameters': self.parameters,
            'metrics': self.metrics.to_dict() if hasattr(self.metrics, 'to_dict') else str(self.metrics),
            'build_success': self.build_success,
            'build_time': self.build_time,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string for export."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DSEResult':
        """Create DSEResult from dictionary."""
        # Import here to avoid circular imports
        from ..core.metrics import DSEMetrics
        
        metrics_data = data.get('metrics', {})
        if isinstance(metrics_data, dict):
            metrics = DSEMetrics.from_dict(metrics_data)
        else:
            metrics = metrics_data
        
        return cls(
            parameters=data.get('parameters', {}),
            metrics=metrics,
            build_success=data.get('build_success', True),
            build_time=data.get('build_time', 0.0),
            metadata=data.get('metadata', {})
        )


@dataclass 
class ParameterSet:
    """Parameter combination definition for organized DSE."""
    name: str
    parameters: Dict[str, Any]
    description: str = ""
    priority: int = 0  # Higher priority evaluated first
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'description': self.description,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSet':
        """Create ParameterSet from dictionary."""
        return cls(
            name=data.get('name', ''),
            parameters=data.get('parameters', {}),
            description=data.get('description', ''),
            priority=data.get('priority', 0)
        )


@dataclass
class ComparisonResult:
    """Result of comparing multiple DSE results."""
    best_result: DSEResult
    ranking: List[DSEResult]
    comparison_metric: str
    summary_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'best_result': self.best_result.to_dict(),
            'ranking': [result.to_dict() for result in self.ranking],
            'comparison_metric': self.comparison_metric,
            'summary_stats': self.summary_stats
        }
    
    def get_top_n(self, n: int = 5) -> List[DSEResult]:
        """Get top N results from ranking."""
        return self.ranking[:n]
    
    def get_success_rate(self) -> float:
        """Calculate success rate of all results."""
        if not self.ranking:
            return 0.0
        return sum(1 for r in self.ranking if r.build_success) / len(self.ranking)


@dataclass
class DSEConfiguration:
    """Simple DSE configuration for parameter sweeps."""
    max_parallel: int = 1
    timeout_seconds: int = 3600
    continue_on_failure: bool = True
    export_format: str = 'pandas'  # 'pandas', 'csv', 'json'
    output_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_parallel': self.max_parallel,
            'timeout_seconds': self.timeout_seconds,
            'continue_on_failure': self.continue_on_failure,
            'export_format': self.export_format,
            'output_dir': self.output_dir
        }


# Type aliases for better readability
ParameterSpace = Dict[str, List[Any]]
ParameterCombination = Dict[str, Any]
MetricName = str