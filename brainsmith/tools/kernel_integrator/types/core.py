"""
Core types for kernel integrator with no dependencies.

This module contains fundamental types that other modules can depend on
without creating circular dependencies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import re

# Import fundamental types from dataflow
from brainsmith.core.dataflow.types import InterfaceType, ShapeSpec


class PortDirection(Enum):
    """Direction of RTL ports."""
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


@dataclass(frozen=True)
class DatatypeSpec:
    """Immutable datatype specification.
    
    Represents a hardware datatype with template parameters and bit width.
    Examples: ap_fixed<16,6>, ap_uint<32>
    """
    type_name: str
    template_params: Dict[str, int]
    bit_width: int
    signed: bool = True
    
    @classmethod
    def from_string(cls, type_str: str) -> "DatatypeSpec":
        """Parse from string like 'ap_fixed<16,6>' or 'ap_uint<32>'.
        
        Args:
            type_str: String representation of the datatype
            
        Returns:
            DatatypeSpec instance
            
        Raises:
            ValueError: If the string cannot be parsed
        """
        # Match patterns like ap_fixed<16,6> or ap_uint<32>
        match = re.match(r'(\w+)<([^>]+)>', type_str.strip())
        if not match:
            raise ValueError(f"Invalid datatype string: {type_str}")
        
        type_name = match.group(1)
        params_str = match.group(2)
        
        # Parse template parameters
        params = params_str.split(',')
        
        if type_name in ['ap_fixed', 'ap_ufixed']:
            if len(params) != 2:
                raise ValueError(f"{type_name} requires 2 parameters, got {len(params)}")
            bit_width = int(params[0].strip())
            int_width = int(params[1].strip())
            template_params = {'width': bit_width, 'int_width': int_width}
            signed = type_name == 'ap_fixed'
        elif type_name in ['ap_int', 'ap_uint']:
            if len(params) != 1:
                raise ValueError(f"{type_name} requires 1 parameter, got {len(params)}")
            bit_width = int(params[0].strip())
            template_params = {'width': bit_width}
            signed = type_name == 'ap_int'
        else:
            # Generic handling for unknown types
            bit_width = int(params[0].strip()) if params else 32
            template_params = {f'param{i}': int(p.strip()) for i, p in enumerate(params)}
            signed = True
        
        return cls(
            type_name=type_name,
            template_params=template_params,
            bit_width=bit_width,
            signed=signed
        )
    
    def to_string(self) -> str:
        """Convert back to string representation."""
        if self.type_name in ['ap_fixed', 'ap_ufixed']:
            return f"{self.type_name}<{self.template_params['width']},{self.template_params['int_width']}>"
        elif self.type_name in ['ap_int', 'ap_uint']:
            return f"{self.type_name}<{self.template_params['width']}>"
        else:
            # Generic handling
            params = ','.join(str(v) for v in self.template_params.values())
            return f"{self.type_name}<{params}>"


@dataclass(frozen=True)
class DimensionSpec:
    """Dimension specification using dataflow shape types.
    
    Represents block (BDIM) and stream (SDIM) dimensions for an interface.
    Dimensions can be concrete integers or symbolic parameter names.
    """
    bdim: ShapeSpec  # Block dimensions using unified type
    sdim: ShapeSpec  # Stream dimensions using unified type
    
    def to_concrete_shape(self, params: Dict[str, int]) -> Tuple[List[int], List[int]]:
        """Resolve symbolic dimensions to concrete values.
        
        Args:
            params: Dictionary mapping parameter names to values
            
        Returns:
            Tuple of (bdim_concrete, sdim_concrete) as lists of integers
        """
        bdim_concrete = []
        for d in self.bdim:
            if isinstance(d, str):
                if d in params:
                    bdim_concrete.append(params[d])
                else:
                    raise ValueError(f"Unknown parameter: {d}")
            else:
                bdim_concrete.append(d)
                
        sdim_concrete = []
        for d in self.sdim:
            if isinstance(d, str):
                if d in params:
                    sdim_concrete.append(params[d])
                else:
                    raise ValueError(f"Unknown parameter: {d}")
            else:
                sdim_concrete.append(d)
        
        return bdim_concrete, sdim_concrete
    
    @property
    def total_elements(self) -> Optional[int]:
        """Calculate total number of elements if all dimensions are concrete.
        
        Returns:
            Total element count or None if dimensions contain parameters
        """
        # Check if all dimensions are concrete
        if any(isinstance(d, str) for d in self.bdim + self.sdim):
            return None
            
        # Calculate product of all dimensions
        total = 1
        for d in self.bdim + self.sdim:
            total *= d
        return total
    
    def get_parameters(self) -> List[str]:
        """Get list of parameter names used in dimensions.
        
        Returns:
            List of unique parameter names
        """
        params = set()
        for d in self.bdim + self.sdim:
            if isinstance(d, str) and d not in ['1', '*']:
                params.add(d)
        return sorted(list(params))