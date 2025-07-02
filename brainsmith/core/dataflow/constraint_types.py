############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Shared constraint types for dataflow modeling.

This module contains constraint types that are shared between the kernel
modeling system and the hardware kernel generator, preventing circular
dependencies.
"""

from dataclasses import dataclass
from typing import List
from qonnx.core.datatype import BaseDataType


@dataclass
class DatatypeConstraintGroup:
    """
    Simple constraint group: [DTYPE, MIN_WIDTH, MAX_WIDTH]
    
    Examples:
        DatatypeConstraintGroup("INT", 4, 8)    # INT4, INT5, INT6, INT7, INT8
        DatatypeConstraintGroup("UINT", 8, 16)  # UINT8, UINT16
        DatatypeConstraintGroup("FIXED", 8, 16) # FIXED<8,N>, FIXED<16,N>
        DatatypeConstraintGroup("ANY", 8, 32)   # Any datatype from 8 to 32 bits
    """
    base_type: str      # "INT", "UINT", "FIXED", "FLOAT", "BIPOLAR", "TERNARY"
    min_width: int      # Minimum bit width (inclusive)
    max_width: int      # Maximum bit width (inclusive)
    
    def __post_init__(self):
        """Validate constraint group parameters."""
        if self.min_width <= 0:
            raise ValueError(f"min_width must be positive, got {self.min_width}")
        if self.max_width < self.min_width:
            raise ValueError(f"max_width ({self.max_width}) must be >= min_width ({self.min_width})")
        
        valid_base_types = ["INT", "UINT", "FIXED", "FLOAT", "BIPOLAR", "TERNARY", "BINARY", "ANY"]
        if self.base_type not in valid_base_types:
            raise ValueError(f"Invalid base_type '{self.base_type}'. Must be one of {valid_base_types}")


def validate_datatype_against_constraints(
    datatype: BaseDataType, 
    constraint_groups: List[DatatypeConstraintGroup]
) -> bool:
    """
    Check if a QONNX datatype satisfies any constraint group.
    
    Args:
        datatype: QONNX BaseDataType instance to validate
        constraint_groups: List of constraint groups to check against
        
    Returns:
        True if datatype satisfies at least one constraint group
    """
    if not constraint_groups:
        return True  # No constraints = allow anything
        
    for group in constraint_groups:
        if _matches_constraint_group(datatype, group):
            return True
    return False


def _matches_constraint_group(datatype: BaseDataType, group: DatatypeConstraintGroup) -> bool:
    """Check if datatype matches a single constraint group."""
    # Check bitwidth range first (applies to all types including ANY)
    bitwidth = datatype.bitwidth()
    if not (group.min_width <= bitwidth <= group.max_width):
        return False
    
    # Special case: ANY matches any type (only bitwidth matters)
    if group.base_type == "ANY":
        return True
    
    # Extract base type from QONNX canonical name
    canonical_name = datatype.get_canonical_name()
    
    # Check base type
    if group.base_type == "INT" and not (canonical_name.startswith("INT") and datatype.signed()):
        return False
    elif group.base_type == "UINT" and not (canonical_name.startswith("UINT") or canonical_name == "BINARY"):
        return False
    elif group.base_type == "FIXED" and not canonical_name.startswith("FIXED<"):
        return False
    elif group.base_type == "FLOAT" and not canonical_name.startswith("FLOAT"):
        return False
    elif group.base_type == "BIPOLAR" and canonical_name != "BIPOLAR":
        return False
    elif group.base_type == "TERNARY" and canonical_name != "TERNARY":
        return False
    elif group.base_type == "BINARY" and canonical_name != "BINARY":
        return False
    
    return True


__all__ = [
    "DatatypeConstraintGroup",
    "validate_datatype_against_constraints",
]