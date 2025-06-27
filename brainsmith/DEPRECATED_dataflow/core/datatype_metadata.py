############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Datatype metadata for binding RTL parameters to datatype properties.

This module provides DatatypeMetadata, which creates explicit mappings between
RTL parameters and datatype properties for code generation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatatypeMetadata:
    """
    Explicit binding between RTL parameters and datatype properties.
    
    This class provides a structured way to map RTL parameter names to
    specific datatype properties used in code generation. All properties
    are optional RTL parameter names, allowing flexible datatype definitions.
    
    Attributes:
        name: Identifier for this datatype (e.g., "in", "accumulator", "threshold")
        width: RTL parameter name for bit width
        signed: RTL parameter name for signedness (0/1)
        format: RTL parameter name for format selection (INT/UINT/FIXED/FLOAT)
        bias: RTL parameter name for bias/offset value
        fractional_width: RTL parameter name for fractional bits (FIXED types)
        exponent_width: RTL parameter name for exponent bits (FLOAT types)
        mantissa_width: RTL parameter name for mantissa bits (FLOAT types)
        description: Optional human-readable description
    """
    name: str  # Required - identifier for this datatype
    width: Optional[str] = None
    signed: Optional[str] = None
    format: Optional[str] = None
    bias: Optional[str] = None
    fractional_width: Optional[str] = None
    exponent_width: Optional[str] = None
    mantissa_width: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata parameters."""
        if not self.name:
            raise ValueError("DatatypeMetadata name cannot be empty")
    
    def get_all_parameters(self) -> list[str]:
        """
        Get list of all RTL parameter names referenced by this metadata.
        
        Returns:
            List of parameter names (non-None values only)
        """
        params = []
        
        # Add all non-None parameters
        if self.width is not None:
            params.append(self.width)
        if self.signed is not None:
            params.append(self.signed)
        if self.format is not None:
            params.append(self.format)
        if self.bias is not None:
            params.append(self.bias)
        if self.fractional_width is not None:
            params.append(self.fractional_width)
        if self.exponent_width is not None:
            params.append(self.exponent_width)
        if self.mantissa_width is not None:
            params.append(self.mantissa_width)
            
        return params
    
    def update(self, **kwargs) -> 'DatatypeMetadata':
        """
        Create a new DatatypeMetadata with updated fields.
        
        Args:
            **kwargs: Field values to update
            
        Returns:
            New DatatypeMetadata instance with updated fields
        """
        # Get current values
        current = {
            'name': self.name,
            'width': self.width,
            'signed': self.signed,
            'format': self.format,
            'bias': self.bias,
            'fractional_width': self.fractional_width,
            'exponent_width': self.exponent_width,
            'mantissa_width': self.mantissa_width,
            'description': self.description
        }
        
        # Update with provided values
        current.update(kwargs)
        
        return DatatypeMetadata(**current)