"""
Forge API - Main entry point for the Design Space Constructor.

This module provides the high-level API for constructing design spaces
from ONNX models and Blueprint YAML files.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from .data_structures import DesignSpace
from .parser import BlueprintParser, load_blueprint
from .validator import DesignSpaceValidator
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
    
    The ForgeAPI orchestrates the blueprint parsing, validation,
    and design space construction process.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the Forge API.
        
        Args:
            verbose: Enable verbose logging output
        """
        self.parser = BlueprintParser()
        self.validator = DesignSpaceValidator()
        
        # Configure logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def forge(self, model_path: str, blueprint_path: str) -> DesignSpace:
        """
        Construct a validated DesignSpace from model and blueprint.
        
        This is the main entry point for creating design spaces. It handles:
        1. Loading and validating the model path
        2. Loading and parsing the blueprint
        3. Constructing the design space
        4. Validating the complete design space
        5. Logging summary information
        
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
        logger.info(f"Forging design space from:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Blueprint: {blueprint_path}")
        
        # Validate model file exists
        model_path = self._validate_model_path(model_path)
        
        # Load and parse blueprint
        logger.debug("Loading blueprint file...")
        blueprint_data = load_blueprint(blueprint_path)
        
        logger.debug("Parsing blueprint...")
        design_space = self.parser.parse(blueprint_data, model_path)
        
        # Validate design space
        logger.debug("Validating design space...")
        validation_result = self.validator.validate(design_space)
        
        if not validation_result.is_valid:
            raise ValidationError(
                "Design space validation failed",
                errors=validation_result.errors,
                warnings=validation_result.warnings
            )
        
        # Log warnings if any
        if validation_result.warnings:
            logger.warning("Validation warnings:")
            for warning in validation_result.warnings:
                logger.warning(f"  - {warning}")
        
        # Log summary
        self._log_summary(design_space, blueprint_data)
        
        logger.info("Design space successfully constructed!")
        return design_space
    
    def _validate_model_path(self, model_path: str) -> str:
        """
        Validate that the model file exists and return absolute path.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Absolute path to model file
            
        Raises:
            ConfigurationError: If model file doesn't exist
        """
        if not model_path:
            raise ConfigurationError("Model path cannot be empty")
        
        # Convert to absolute path
        abs_path = os.path.abspath(model_path)
        
        if not os.path.exists(abs_path):
            raise ConfigurationError(f"Model file not found: {model_path}")
        
        if not os.path.isfile(abs_path):
            raise ConfigurationError(f"Model path is not a file: {model_path}")
        
        return abs_path
    
    def _log_summary(self, design_space: DesignSpace, blueprint_data: dict):
        """Log a summary of the constructed design space."""
        name = blueprint_data.get("name", "Unnamed")
        description = blueprint_data.get("description", "No description")
        
        logger.info("\n" + "="*60)
        logger.info("DESIGN SPACE SUMMARY")
        logger.info("="*60)
        logger.info(f"Name: {name}")
        logger.info(f"Description: {description}")
        logger.info(f"Model: {Path(design_space.model_path).name}")
        logger.info(f"Total Combinations: {design_space.get_total_combinations():,}")
        
        # Log key configuration
        logger.info("\nConfiguration:")
        logger.info(f"  Search Strategy: {design_space.search_config.strategy.value}")
        logger.info(f"  Constraints: {len(design_space.search_config.constraints)}")
        logger.info(f"  Output Stage: {design_space.global_config.output_stage.value}")
        logger.info(f"  Working Dir: {design_space.global_config.working_directory}")
        
        # Log space details
        kernel_count = len(design_space.hw_compiler_space.kernels)
        if isinstance(design_space.hw_compiler_space.transforms, dict):
            transform_count = sum(
                len(t) for t in design_space.hw_compiler_space.transforms.values()
            )
        else:
            transform_count = len(design_space.hw_compiler_space.transforms)
        
        logger.info("\nDesign Space:")
        logger.info(f"  Kernels: {kernel_count}")
        logger.info(f"  Transforms: {transform_count}")
        logger.info(f"  Build Steps: {len(design_space.hw_compiler_space.build_steps)}")
        
        # Show constraints
        if design_space.search_config.constraints:
            logger.info("\nConstraints:")
            for constraint in design_space.search_config.constraints:
                logger.info(f"  - {constraint}")
        
        logger.info("="*60 + "\n")


# Convenience function for simple usage
def forge(model_path: str, blueprint_path: str, verbose: bool = False) -> DesignSpace:
    """
    Convenience function to construct a design space.
    
    This provides a simple interface for the common case of constructing
    a design space with default settings.
    
    Args:
        model_path: Path to ONNX model file
        blueprint_path: Path to Blueprint YAML file
        verbose: Enable verbose logging
        
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
    api = ForgeAPI(verbose=verbose)
    return api.forge(model_path, blueprint_path)