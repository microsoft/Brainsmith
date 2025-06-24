"""
Base classes for FINN evaluation backends.

Provides abstract interface and data structures for different FINN workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class EvaluationRequest:
    """Request for hardware evaluation."""
    model_path: str
    combination: Dict[str, Any]  # Either components or build steps
    work_dir: str
    timeout: Optional[int] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.model_path:
            raise ValueError("model_path is required")
        if not self.work_dir:
            raise ValueError("work_dir is required")
        
        # Ensure paths are strings
        self.model_path = str(self.model_path)
        self.work_dir = str(self.work_dir)


@dataclass
class EvaluationResult:
    """Result of hardware evaluation."""
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    reports: Dict[str, str] = field(default_factory=dict)  # report_type -> path
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'metrics': self.metrics,
            'reports': self.reports,
            'error': self.error,
            'warnings': self.warnings
        }
    
    @classmethod
    def from_error(cls, error: str) -> 'EvaluationResult':
        """Create error result."""
        return cls(
            success=False,
            error=error
        )


class EvaluationBackend(ABC):
    """Abstract backend for hardware evaluation."""
    
    def __init__(self, blueprint_config: Dict[str, Any]):
        """
        Initialize backend with blueprint configuration.
        
        Args:
            blueprint_config: Blueprint configuration dict
        """
        self.blueprint_config = blueprint_config
    
    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evaluate hardware configuration.
        
        Args:
            request: Evaluation request with model and configuration
            
        Returns:
            EvaluationResult with metrics and reports
        """
        pass
    
    def validate_request(self, request: EvaluationRequest) -> Optional[str]:
        """
        Validate evaluation request.
        
        Args:
            request: Request to validate
            
        Returns:
            Error message if invalid, None if valid
        """
        # Check model exists
        if not Path(request.model_path).exists():
            return f"Model file not found: {request.model_path}"
        
        # Check work directory
        work_path = Path(request.work_dir)
        if not work_path.exists():
            return f"Work directory does not exist: {request.work_dir}"
        if not work_path.is_dir():
            return f"Work path is not a directory: {request.work_dir}"
            
        # Check timeout
        if request.timeout is not None and request.timeout <= 0:
            return f"Invalid timeout: {request.timeout} (must be positive)"
            
        return None