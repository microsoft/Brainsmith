############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Input interface schema"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any
from .base import BaseSchema
from .qonnx_types import BaseDataType, DatatypeConstraintGroup, validate_datatype_against_constraints

@dataclass
class InputSchema(BaseSchema):
    """Schema for an input interface
    
    Defines the schema and constraints for an input that can be
    instantiated with different tensor dimensions.
    """
    
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    block_tiling: Optional[List[Union[int, str]]] = None
    stream_tiling: Optional[List[Union[int, str]]] = None
    optional: bool = False
    is_weight: bool = False  # Explicitly mark weight inputs for FINN integration
    
    # Node attribute for datatype
    datatype_attr: Optional[str] = None
    
    def validates_datatype(self, datatype: BaseDataType) -> bool:
        """Check if datatype satisfies constraints"""
        if not self.datatype_constraints:
            return True  # No constraints = allow any
        return validate_datatype_against_constraints(datatype, self.datatype_constraints)
    
    
    def validate(self) -> List[str]:
        """Validate schema consistency"""
        errors = []
        
        if not self.name:
            errors.append("Input name cannot be empty")
        
        return errors
    
    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]
        
        if self.datatype_constraints:
            parts.append(f"datatype_constraints={len(self.datatype_constraints)}")
        
        if self.datatype_attr:
            parts.append(f"datatype_attr='{self.datatype_attr}'")
        
        if self.block_tiling:
            parts.append(f"block_tiling={self.block_tiling}")
        
        if self.stream_tiling:
            parts.append(f"stream_tiling={self.stream_tiling}")
        
        if self.is_weight:
            parts.append("is_weight=True")
        
        if self.optional:
            parts.append("optional=True")
        
        return f"InputSchema({', '.join(parts)})"