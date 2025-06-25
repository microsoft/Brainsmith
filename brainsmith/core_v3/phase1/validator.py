"""
Design space validator for DSE V3.

This module validates design spaces for correctness, feasibility,
and provides helpful warnings about potential issues.
"""

from dataclasses import dataclass
from typing import List
import os
import logging

from .data_structures import (
    DesignSpace,
    HWCompilerSpace,
    ProcessingSpace,
    SearchConfig,
    GlobalConfig,
)
from .exceptions import ValidationError
from ..config import get_config


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of design space validation.
    
    Attributes:
        is_valid: Whether the design space passed validation
        errors: List of error messages (must be fixed)
        warnings: List of warning messages (informational)
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __str__(self) -> str:
        if self.is_valid:
            status = "VALID"
        else:
            status = f"INVALID ({len(self.errors)} errors)"
        
        msg = f"Validation Result: {status}"
        if self.warnings:
            msg += f" with {len(self.warnings)} warnings"
        return msg


class DesignSpaceValidator:
    """
    Validate design spaces for correctness and feasibility.
    
    This validator checks:
    - Model file existence
    - Configuration validity
    - Constraint feasibility
    - Design space size warnings
    """
    
    # Thresholds for warnings
    MAX_COMBINATIONS_WARNING = 1000
    MAX_COMBINATIONS_ERROR = 10000
    HIGH_PARALLEL_BUILDS = 32
    
    # Valid values for certain fields
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
    VALID_CONSTRAINT_OPERATORS = ["<=", ">=", "==", "<", ">"]
    
    def validate(self, design_space: DesignSpace) -> ValidationResult:
        """
        Perform comprehensive validation of a design space.
        
        Args:
            design_space: The design space to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Validate model path
        self._validate_model_path(design_space.model_path, errors, warnings)
        
        # Validate HW compiler space
        self._validate_hw_compiler(design_space.hw_compiler_space, errors, warnings)
        
        # Validate processing space
        self._validate_processing(design_space.processing_space, errors, warnings)
        
        # Validate search configuration
        self._validate_search(design_space.search_config, errors, warnings)
        
        # Validate global configuration
        self._validate_global(design_space.global_config, errors, warnings)
        
        # Check total combinations
        self._check_combinations(design_space, errors, warnings)
        
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
        
        logger.info(f"Validation completed: {result}")
        return result
    
    def _validate_model_path(self, model_path: str, errors: List[str], warnings: List[str]):
        """Validate model path exists and is an ONNX file."""
        if not model_path:
            errors.append("Model path is required")
            return
        
        if not os.path.exists(model_path):
            errors.append(f"Model file not found: {model_path}")
            return
        
        if not model_path.endswith('.onnx'):
            warnings.append(
                f"Model file does not have .onnx extension: {model_path}"
            )
    
    def _validate_hw_compiler(self, hw_space: HWCompilerSpace, errors: List[str], warnings: List[str]):
        """Validate hardware compiler configuration."""
        # Check kernels
        if not hw_space.kernels:
            warnings.append("No kernels defined in hw_compiler space")
        else:
            # Validate kernel formats
            for i, kernel in enumerate(hw_space.kernels):
                self._validate_kernel_format(kernel, i, errors, warnings)
        
        # Check transforms
        if not hw_space.transforms:
            warnings.append("No transforms defined in hw_compiler space")
        else:
            # Validate transform formats
            if isinstance(hw_space.transforms, dict):
                # Phase-based transforms
                for phase, transforms in hw_space.transforms.items():
                    if not isinstance(transforms, list):
                        errors.append(
                            f"Transform phase '{phase}' must contain a list"
                        )
                    else:
                        for i, transform in enumerate(transforms):
                            self._validate_transform_format(
                                transform, f"{phase}[{i}]", errors, warnings
                            )
            elif isinstance(hw_space.transforms, list):
                # Flat transforms
                for i, transform in enumerate(hw_space.transforms):
                    self._validate_transform_format(
                        transform, str(i), errors, warnings
                    )
        
        # Check build steps
        if not hw_space.build_steps:
            errors.append("Build steps are required")
        else:
            # Check for commonly required steps
            common_steps = ["ConvertToHW", "PrepareIP"]
            for step in common_steps:
                if step not in hw_space.build_steps:
                    warnings.append(
                        f"Build step '{step}' is commonly required but not present"
                    )
    
    def _validate_kernel_format(self, kernel: any, index: int, errors: List[str], warnings: List[str]):
        """Validate a single kernel configuration format."""
        if isinstance(kernel, str):
            # Simple string format - valid
            pass
        elif isinstance(kernel, tuple):
            if len(kernel) != 2:
                errors.append(
                    f"Kernel tuple at index {index} must have exactly 2 elements (name, backends)"
                )
            else:
                name, backends = kernel
                if not isinstance(name, str):
                    errors.append(
                        f"Kernel name at index {index} must be a string"
                    )
                if not isinstance(backends, list):
                    errors.append(
                        f"Kernel backends at index {index} must be a list"
                    )
                elif not backends:
                    errors.append(
                        f"Kernel '{name}' at index {index} has empty backends list"
                    )
        elif isinstance(kernel, list):
            # Mutually exclusive group
            if not kernel:
                errors.append(
                    f"Kernel group at index {index} is empty"
                )
            # Recursively validate items in the group
            for i, item in enumerate(kernel):
                if item is not None and item != "~":
                    self._validate_kernel_format(
                        item, f"{index}.{i}", errors, warnings
                    )
        else:
            errors.append(
                f"Invalid kernel format at index {index}: {type(kernel).__name__}"
            )
    
    def _validate_transform_format(self, transform: any, location: str, errors: List[str], warnings: List[str]):
        """Validate a single transform configuration format."""
        if isinstance(transform, str):
            # Simple string format - valid
            if transform.startswith("~") and len(transform) == 1:
                errors.append(
                    f"Invalid optional transform at {location}: '~' must be followed by a name"
                )
        elif isinstance(transform, list):
            # Mutually exclusive group
            if not transform:
                errors.append(
                    f"Transform group at {location} is empty"
                )
            # Recursively validate items in the group
            for i, item in enumerate(transform):
                if item is not None and item != "~":
                    self._validate_transform_format(
                        item, f"{location}.{i}", errors, warnings
                    )
        else:
            errors.append(
                f"Invalid transform format at {location}: {type(transform).__name__}"
            )
    
    def _validate_processing(self, proc_space: ProcessingSpace, errors: List[str], warnings: List[str]):
        """Validate processing configuration."""
        # Check if any processing is defined
        has_preprocessing = bool(proc_space.preprocessing)
        has_postprocessing = bool(proc_space.postprocessing)
        
        if not has_preprocessing and not has_postprocessing:
            warnings.append("No processing steps defined")
        
        # Validate preprocessing steps
        for i, step_alternatives in enumerate(proc_space.preprocessing):
            if not step_alternatives:
                errors.append(f"Preprocessing step {i} has no alternatives")
            for step in step_alternatives:
                if not step.name:
                    errors.append(f"Preprocessing step at index {i} has no name")
        
        # Validate postprocessing steps
        for i, step_alternatives in enumerate(proc_space.postprocessing):
            if not step_alternatives:
                errors.append(f"Postprocessing step {i} has no alternatives")
            for step in step_alternatives:
                if not step.name:
                    errors.append(f"Postprocessing step at index {i} has no name")
    
    def _validate_search(self, search_config: SearchConfig, errors: List[str], warnings: List[str]):
        """Validate search configuration."""
        # Validate constraints
        for i, constraint in enumerate(search_config.constraints):
            if constraint.operator not in self.VALID_CONSTRAINT_OPERATORS:
                errors.append(
                    f"Constraint {i} has invalid operator '{constraint.operator}'. "
                    f"Valid operators: {self.VALID_CONSTRAINT_OPERATORS}"
                )
            
            if not constraint.metric:
                errors.append(f"Constraint {i} has no metric specified")
        
        # Check parallel builds
        if search_config.parallel_builds < 1:
            errors.append("Parallel builds must be at least 1")
        elif search_config.parallel_builds > self.HIGH_PARALLEL_BUILDS:
            warnings.append(
                f"High parallel builds ({search_config.parallel_builds}) "
                f"may cause resource contention"
            )
        
        # Check evaluation limits
        if search_config.max_evaluations is not None and search_config.max_evaluations < 1:
            errors.append("max_evaluations must be at least 1")
        
        if search_config.timeout_minutes is not None and search_config.timeout_minutes < 1:
            errors.append("timeout_minutes must be at least 1")
    
    def _validate_global(self, global_config: GlobalConfig, errors: List[str], warnings: List[str]):
        """Validate global configuration."""
        # Check working directory
        if not global_config.working_directory:
            errors.append("Working directory is required")
        
        # Check log level
        if global_config.log_level not in self.VALID_LOG_LEVELS:
            errors.append(
                f"Invalid log level '{global_config.log_level}'. "
                f"Valid levels: {self.VALID_LOG_LEVELS}"
            )
        
        # Check if working directory parent exists
        if global_config.working_directory:
            parent_dir = os.path.dirname(global_config.working_directory)
            if parent_dir and not os.path.exists(parent_dir):
                warnings.append(
                    f"Parent directory of working directory does not exist: {parent_dir}"
                )
    
    def _check_combinations(self, design_space: DesignSpace, errors: List[str], warnings: List[str]):
        """Check total number of combinations and issue appropriate warnings."""
        try:
            total = design_space.get_total_combinations()
            
            # Get effective max_combinations limit
            # Priority: blueprint > global config > legacy constant
            max_combinations_limit = self.MAX_COMBINATIONS_ERROR  # Legacy default
            
            # Check global config
            global_config = get_config()
            if global_config.max_combinations is not None:
                max_combinations_limit = global_config.max_combinations
            
            # Blueprint override takes precedence
            if design_space.global_config.max_combinations is not None:
                max_combinations_limit = design_space.global_config.max_combinations
            
            if total == 0:
                errors.append("Design space has no valid combinations")
            elif total > max_combinations_limit:
                errors.append(
                    f"Design space has {total:,} combinations, "
                    f"exceeding maximum of {max_combinations_limit:,}. "
                    f"You can increase this limit by setting max_combinations in the blueprint's "
                    f"global section, or in ~/.brainsmith/config.yaml, or via "
                    f"BRAINSMITH_MAX_COMBINATIONS environment variable."
                )
            elif total > self.MAX_COMBINATIONS_WARNING:
                warnings.append(
                    f"Design space has {total:,} combinations, "
                    f"which may take significant time to explore"
                )
            
            # Estimate time if parallel builds are specified
            if total > 100 and design_space.search_config.parallel_builds > 1:
                estimated_batches = total // design_space.search_config.parallel_builds
                warnings.append(
                    f"With {design_space.search_config.parallel_builds} parallel builds, "
                    f"exploration will require approximately {estimated_batches} batches"
                )
                
        except Exception as e:
            errors.append(f"Failed to calculate total combinations: {str(e)}")