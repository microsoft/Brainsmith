"""
Forge API - Main entry point for the Design Space Constructor.

This module provides the high-level API for constructing design spaces
from ONNX models and Blueprint YAML files.

Note: This module will eventually grow to become the end-to-end orchestrator
that coordinates all three phases (Design Space Construction, Exploration, 
and Build Execution) to provide a unified interface for the entire DSE flow.
"""

import os
from pathlib import Path

import os
import logging
from .data_structures import DesignSpace
from .parser import BlueprintParser, load_blueprint
from .exceptions import (
    BrainsmithError,
    BlueprintParseError,
    ValidationError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


class ForgeAPI:
    """
    Main API for constructing design spaces.
    
    The ForgeAPI orchestrates the blueprint parsing
    and design space construction process.
    """
    
    def __init__(self):
        """Initialize the Forge API."""
        self.parser = BlueprintParser()
    
    def forge(self, model_path: str, blueprint_path: str) -> DesignSpace:
        """
        Construct a DesignSpace from model and blueprint.
        
        Args:
            model_path: Path to ONNX model file
            blueprint_path: Path to Blueprint YAML file
            
        Returns:
            DesignSpace object ready for exploration
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            BlueprintParseError: If blueprint is invalid
        """
        # Verify model exists (fail fast)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load and parse blueprint
        blueprint_data = load_blueprint(blueprint_path)
        design_space = self.parser.parse(blueprint_data, os.path.abspath(model_path))
        
        # The DesignSpace.__post_init__ already validates size
        
        # Log summary
        size = design_space._estimate_size()
        logger.info(f"✅ Created design space with ~{size:,} configurations")
        
        return design_space


# Convenience function for simple usage
def forge(model_path: str, blueprint_path: str) -> DesignSpace:
    """
    Create design space from model and blueprint.
    
    This is the main entry point for Phase 1. In future versions, this function
    will expand to orchestrate the entire DSE pipeline (exploration + builds).
    
    Args:
        model_path: Path to ONNX model file
        blueprint_path: Path to Blueprint YAML file
        
    Returns:
        DesignSpace object ready for exploration
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        BlueprintParseError: If blueprint is invalid
    
    Example:
        >>> from brainsmith.core.phase1 import forge
        >>> design_space = forge("model.onnx", "blueprint.yaml")
        >>> print(f"Total combinations: {design_space._estimate_size():,}")
        
    Future API (planned):
        >>> results = forge("model.onnx", "blueprint.yaml", explore=True, build=True)
        >>> print(f"Best config: {results.pareto_front[0]}")
    """
    logger.info(f"Forging design space from {model_path} and {blueprint_path}")
    
    # Verify model exists (fail fast)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Parse blueprint (now handles v4.0)
    parser = BlueprintParser()
    design_space = parser.parse(load_blueprint(blueprint_path), model_path)
    
    # The DesignSpace.__post_init__ already validates size
    
    # Log summary
    size = design_space._estimate_size()
    logger.info(f"✅ Created design space with ~{size:,} configurations")
    
    return design_space