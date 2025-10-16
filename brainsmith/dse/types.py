# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict


class SegmentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputType(Enum):
    ESTIMATES = "estimates"
    RTL = "rtl"
    BITFILE = "bitfile"


@dataclass
class SegmentResult:
    success: bool
    segment_id: str
    output_model: Optional[Path] = None
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    execution_time: float = 0
    cached: bool = False


@dataclass
class TreeExecutionResult:
    segment_results: Dict[str, SegmentResult]
    total_time: float

    @property
    def stats(self) -> Dict[str, int]:
        return {
            'total': len(self.segment_results),
            'successful': sum(1 for r in self.segment_results.values() if r.success and not r.cached),
            'failed': sum(1 for r in self.segment_results.values() if not r.success),
            'cached': sum(1 for r in self.segment_results.values() if r.cached),
            'skipped': sum(1 for r in self.segment_results.values() if r.error == "Skipped")
        }


class ExecutionError(Exception):
    pass

