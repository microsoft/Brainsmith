"""
Backend infrastructure for FINN evaluation.

This module provides separate backends for different FINN workflows:
- 6-entrypoint workflow (modern DSE-based)
- Legacy workflow (direct build steps)
"""

from .base import EvaluationRequest, EvaluationResult, EvaluationBackend
from .factory import create_backend
from .workflow_detector import WorkflowType, detect_workflow

__all__ = [
    'EvaluationRequest',
    'EvaluationResult', 
    'EvaluationBackend',
    'create_backend',
    'WorkflowType',
    'detect_workflow'
]