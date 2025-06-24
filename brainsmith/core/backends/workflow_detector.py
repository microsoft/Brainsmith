"""
Workflow detection for FINN backends.

Determines whether a blueprint uses the modern 6-entrypoint workflow
or the legacy direct build steps workflow.
"""

from enum import Enum
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """FINN workflow types."""
    SIX_ENTRYPOINT = "six_entrypoint"
    LEGACY = "legacy"


def detect_workflow(blueprint: Dict[str, Any]) -> WorkflowType:
    """
    Detect workflow type from blueprint structure.
    
    Args:
        blueprint: Blueprint configuration dictionary
        
    Returns:
        WorkflowType enum value
        
    Raises:
        ValueError: If workflow type cannot be determined
    """
    logger.debug(f"Detecting workflow type for blueprint: {blueprint.get('name', 'unnamed')}")
    
    # Check for legacy workflow indicators
    finn_config = blueprint.get('finn_config', {})
    
    # Legacy blueprints specify build steps directly
    if 'build_steps' in finn_config:
        logger.info("Detected LEGACY workflow - found 'build_steps' in finn_config")
        return WorkflowType.LEGACY
    
    # Check for 6-entrypoint workflow indicators
    # 6-entrypoint uses component spaces for nodes and transforms
    has_nodes = 'nodes' in blueprint
    has_transforms = 'transforms' in blueprint
    
    if has_nodes and has_transforms:
        logger.info("Detected SIX_ENTRYPOINT workflow - found 'nodes' and 'transforms'")
        return WorkflowType.SIX_ENTRYPOINT
    
    # If we have either nodes or transforms but not both, it's likely a malformed blueprint
    if has_nodes or has_transforms:
        raise ValueError(
            f"Invalid blueprint structure: found {'nodes' if has_nodes else 'transforms'} "
            f"but not {'transforms' if has_nodes else 'nodes'}. "
            "6-entrypoint blueprints require both 'nodes' and 'transforms' sections."
        )
    
    # No clear indicators found
    raise ValueError(
        "Unable to detect workflow type. Blueprint must either:\n"
        "1. Define 'build_steps' in 'finn_config' section for legacy workflow\n"
        "2. Define both 'nodes' and 'transforms' sections for 6-entrypoint workflow"
    )


def validate_workflow_config(blueprint: Dict[str, Any], workflow_type: WorkflowType) -> None:
    """
    Validate that blueprint has required configuration for detected workflow.
    
    Args:
        blueprint: Blueprint configuration
        workflow_type: Detected workflow type
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    if workflow_type == WorkflowType.LEGACY:
        # Validate legacy configuration
        finn_config = blueprint.get('finn_config', {})
        
        if 'build_steps' not in finn_config:
            raise ValueError(
                "Legacy workflow requires 'build_steps' list in 'finn_config' section"
            )
            
        build_steps = finn_config['build_steps']
        
        if not isinstance(build_steps, list):
            raise ValueError(
                f"'build_steps' must be a list, got {type(build_steps).__name__}"
            )
            
        if len(build_steps) == 0:
            raise ValueError("'build_steps' list cannot be empty")
            
        logger.debug(f"Legacy workflow validated with {len(build_steps)} build steps")
        
    elif workflow_type == WorkflowType.SIX_ENTRYPOINT:
        # Validate 6-entrypoint configuration
        nodes = blueprint.get('nodes', {})
        transforms = blueprint.get('transforms', {})
        
        if not nodes:
            raise ValueError(
                "6-entrypoint workflow requires 'nodes' section with component definitions"
            )
            
        if not transforms:
            raise ValueError(
                "6-entrypoint workflow requires 'transforms' section with component definitions"
            )
            
        # Check for at least one component space in each
        if not isinstance(nodes, dict) or len(nodes) == 0:
            raise ValueError("'nodes' section must contain at least one component space")
            
        if not isinstance(transforms, dict) or len(transforms) == 0:
            raise ValueError("'transforms' section must contain at least one component space")
            
        logger.debug(
            f"6-entrypoint workflow validated with {len(nodes)} node spaces "
            f"and {len(transforms)} transform spaces"
        )