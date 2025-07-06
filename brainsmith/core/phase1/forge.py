"""
Forge API - Main entry point for the Design Space Constructor.

This module provides the high-level API for constructing design spaces
from ONNX models and Blueprint YAML files.
"""

import os
from pathlib import Path

from .data_structures import DesignSpace
from .parser import BlueprintParser, load_blueprint
from .validator import DesignSpaceValidator
from .exceptions import (
    BrainsmithError,
    BlueprintParseError,
    ValidationError,
    ConfigurationError,
)


class ForgeAPI:
    """
    Main API for constructing design spaces.
    
    The ForgeAPI orchestrates the blueprint parsing, validation,
    and design space construction process.
    """
    
    def __init__(self):
        """Initialize the Forge API."""
        self.parser = BlueprintParser()
        self.validator = DesignSpaceValidator()
    
    def forge(self, model_path: str, blueprint_path: str) -> DesignSpace:
        """
        Construct a validated DesignSpace from model and blueprint.
        
        Args:
            model_path: Path to ONNX model file
            blueprint_path: Path to Blueprint YAML file
            
        Returns:
            Validated DesignSpace object ready for exploration
            
        Raises:
            ConfigurationError: If model file doesn't exist
            BlueprintParseError: If blueprint is invalid
            ValidationError: If design space validation fails
        """
        # Validate model file exists
        if not os.path.exists(model_path):
            raise ConfigurationError(f"Model file not found: {model_path}")
        
        # Load and parse blueprint
        blueprint_data = load_blueprint(blueprint_path)
        design_space = self.parser.parse(blueprint_data, os.path.abspath(model_path))
        
        # Validate design space
        validation_result = self.validator.validate(design_space)
        
        if not validation_result.is_valid:
            raise ValidationError(
                "Design space validation failed",
                errors=validation_result.errors
            )
        
        return design_space


# Convenience function for simple usage
def forge(model_path: str, blueprint_path: str) -> DesignSpace:
    """
    Convenience function to construct a design space.
    
    Args:
        model_path: Path to ONNX model file
        blueprint_path: Path to Blueprint YAML file
        
    Returns:
        Validated DesignSpace object
        
    Raises:
        ConfigurationError: If model file doesn't exist
        BlueprintParseError: If blueprint is invalid
        ValidationError: If design space validation fails
    
    Example:
        >>> from brainsmith.core.phase1 import forge
        >>> design_space = forge("model.onnx", "blueprint.yaml")
        >>> print(f"Total combinations: {design_space.get_total_combinations()}")
    """
    api = ForgeAPI()
    return api.forge(model_path, blueprint_path)