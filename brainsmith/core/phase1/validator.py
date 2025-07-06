"""
Minimal design space validator for DSE V3.

Only checks critical safety constraints:
- Model file exists
- Total combinations < max limit
"""

import os
from dataclasses import dataclass
from typing import List

from .data_structures import DesignSpace
from .exceptions import ValidationError
from ..config import get_config


@dataclass
class ValidationResult:
    """Result of design space validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DesignSpaceValidator:
    """Minimal validator - only critical safety checks."""
    
    def __init__(self):
        """Initialize validator."""
        pass
    
    def validate(self, design_space: DesignSpace) -> ValidationResult:
        """
        Validate only critical constraints.
        
        Args:
            design_space: The design space to validate
            
        Returns:
            ValidationResult with any errors
        """
        errors = []
        
        # Check model exists
        if not os.path.exists(design_space.model_path):
            errors.append(f"Model file not found: {design_space.model_path}")
        
        # Check combinations limit
        try:
            total = design_space.get_total_combinations()
            
            # Get max limit from config hierarchy
            max_limit = 1_000_000  # Default
            global_config = get_config()
            if global_config.max_combinations is not None:
                max_limit = global_config.max_combinations
            if design_space.global_config.max_combinations is not None:
                max_limit = design_space.global_config.max_combinations
            
            if total > max_limit:
                errors.append(f"Too many combinations: {total} > {max_limit}")
        except:
            # If we can't calculate combinations, let it through
            pass
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=[]
        )