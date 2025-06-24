"""
Factory for creating appropriate FINN evaluation backends.

Automatically selects the correct backend based on blueprint structure.
"""

from typing import Dict, Any
import logging

from .workflow_detector import detect_workflow, validate_workflow_config, WorkflowType
from .base import EvaluationBackend

logger = logging.getLogger(__name__)


def create_backend(blueprint: Dict[str, Any]) -> EvaluationBackend:
    """
    Create appropriate backend based on blueprint structure.
    
    Args:
        blueprint: Blueprint configuration dictionary
        
    Returns:
        EvaluationBackend instance configured for the workflow type
        
    Raises:
        ValueError: If workflow type cannot be determined or is unsupported
        ImportError: If required backend implementation is not available
    """
    # Detect workflow type
    workflow_type = detect_workflow(blueprint)
    logger.info(f"Creating backend for {workflow_type.value} workflow")
    
    # Validate configuration for detected workflow
    validate_workflow_config(blueprint, workflow_type)
    
    # Create appropriate backend
    if workflow_type == WorkflowType.SIX_ENTRYPOINT:
        try:
            from .six_entrypoint import SixEntrypointBackend
            logger.debug("Initializing SixEntrypointBackend")
            return SixEntrypointBackend(blueprint)
        except ImportError as e:
            raise ImportError(
                f"Failed to import SixEntrypointBackend: {e}\n"
                "Ensure six_entrypoint.py is implemented"
            )
            
    elif workflow_type == WorkflowType.LEGACY:
        try:
            from .legacy_finn import LegacyFINNBackend
            logger.debug("Initializing LegacyFINNBackend")
            return LegacyFINNBackend(blueprint)
        except ImportError as e:
            raise ImportError(
                f"Failed to import LegacyFINNBackend: {e}\n"
                "Ensure legacy_finn.py is implemented"
            )
    else:
        # This should never happen due to enum constraints, but defensive programming
        raise ValueError(f"Unknown workflow type: {workflow_type}")


def get_backend_info(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about which backend would be used for a blueprint.
    
    Args:
        blueprint: Blueprint configuration
        
    Returns:
        Dictionary with backend information
    """
    try:
        workflow_type = detect_workflow(blueprint)
        validate_workflow_config(blueprint, workflow_type)
        
        return {
            'workflow_type': workflow_type.value,
            'backend_class': (
                'SixEntrypointBackend' if workflow_type == WorkflowType.SIX_ENTRYPOINT
                else 'LegacyFINNBackend'
            ),
            'valid': True,
            'error': None
        }
    except Exception as e:
        return {
            'workflow_type': None,
            'backend_class': None,
            'valid': False,
            'error': str(e)
        }