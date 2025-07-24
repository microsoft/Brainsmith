"""Minimal data structures for segment execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import time


@dataclass
class SegmentResult:
    """Result of executing a single segment."""
    success: bool
    segment_id: str
    output_model: Optional[Path] = None
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    cached: bool = False


@dataclass 
class TreeExecutionResult:
    """Result of executing the entire tree."""
    segment_results: Dict[str, SegmentResult]
    total_time: float
    
    @property
    def stats(self) -> Dict[str, int]:
        """Calculate statistics from results."""
        return {
            "total": len(self.segment_results),
            "successful": sum(1 for r in self.segment_results.values() 
                            if r.success and not r.cached),
            "failed": sum(1 for r in self.segment_results.values() 
                         if not r.success and r.error != "Skipped"),
            "skipped": sum(1 for r in self.segment_results.values() 
                          if r.error == "Skipped"),
            "cached": sum(1 for r in self.segment_results.values() if r.cached)
        }


class ExecutionError(Exception):
    """Raised when fail-fast mode is enabled and execution fails."""
    pass