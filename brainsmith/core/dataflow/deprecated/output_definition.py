############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Output interface schema"""

from dataclasses import dataclass, field
from typing import Optional, Union, List
from .base import BaseSchema
from .qonnx_types import BaseDataType, DatatypeConstraintGroup, validate_datatype_against_constraints

@dataclass
class OutputSchema(BaseSchema):
    """Schema for an output interface
    
    Defines the schema for an output. Outputs don't have
    configurable SDIM - their streaming rate is determined
    by the kernel computation.
    """
    
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    block_tiling: Optional[List[Union[int, str]]] = None
    optional: bool = False
    
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
            errors.append("Output name cannot be empty")
        
        return errors
    
    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]
        
        if self.datatype_constraints:
            parts.append(f"datatype_constraints={len(self.datatype_constraints)}")
        
        if self.datatype_attr:
            parts.append(f"datatype_attr='{self.datatype_attr}'")
        
        if self.block_tiling:
            parts.append(f"block_tiling={self.block_tiling}")
        
        if self.optional:
            parts.append("optional=True")
        
        return f"OutputSchema({', '.join(parts)})"