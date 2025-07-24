"""Explorer module for executing segment-based execution trees with FINN."""

from .types import SegmentResult, TreeExecutionResult, ExecutionError
from .explorer import explore_execution_tree
from .executor import Executor
from .finn_adapter import FINNAdapter

__all__ = [
    "SegmentResult",
    "TreeExecutionResult", 
    "ExecutionError",
    "explore_execution_tree",
    "Executor",
    "FINNAdapter",
]