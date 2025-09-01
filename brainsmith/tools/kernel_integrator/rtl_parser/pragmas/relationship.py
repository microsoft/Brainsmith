############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Relationship pragma implementation for interface relationships.

This module contains pragmas for defining relationships between interfaces
in the Kernel Modeling system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import logging

from .base import Pragma, PragmaError
from brainsmith.tools.kernel_integrator.metadata import KernelMetadata
from ..types import PragmaType
from brainsmith.core.dataflow.relationships import DimensionRelationship, RelationType

logger = logging.getLogger(__name__)


@dataclass
class RelationshipPragma(Pragma):
    """Defines relationships between interfaces.
    
    Format: @brainsmith RELATIONSHIP <source> <target> <type> [args...]
    
    Types:
    - EQUAL: All dimensions must match
    - DEPENDENT <src_dim> <tgt_dim> <dep_type>: Dimension dependency
    - MULTIPLE <src_dim> <tgt_dim> [factor=N]: Multiple relationship
    
    Examples:
    - @brainsmith RELATIONSHIP input0 output0 EQUAL
    - @brainsmith RELATIONSHIP input0 output0 DEPENDENT 0 0 copy
    - @brainsmith RELATIONSHIP input0 output0 DEPENDENT 1 1 scaled SCALE_FACTOR
    - @brainsmith RELATIONSHIP input0 output0 MULTIPLE 0 0 factor=4
    """
    
    def __post_init__(self):
        """Perform post-initialization processing."""
        # Call parent __post_init__ first
        super().__post_init__()
    
    def _parse_inputs(self) -> Dict[str, Any]:
        """Parse relationship pragma inputs.
        
        Returns:
            Dict with parsed relationship data
        """
        pos = self.inputs['positional']
        named = self.inputs['named']
        
        if len(pos) < 3:
            raise PragmaError("RELATIONSHIP pragma requires source, target, and type")
        
        result = {
            "source_interface": pos[0],
            "target_interface": pos[1], 
            "relationship_type": pos[2].upper()
        }
        
        # Validate interface names
        for iface in [result["source_interface"], result["target_interface"]]:
            if not iface.replace('_', '').replace('V', '').isalnum():
                raise PragmaError(f"Invalid interface name '{iface}' in RELATIONSHIP pragma")
        
        # Parse type-specific arguments
        rel_type = result["relationship_type"]
        
        if rel_type == "EQUAL":
            # EQUAL takes no additional arguments
            if len(pos) > 3:
                logger.warning(f"EQUAL relationship ignoring extra arguments: {pos[3:]}")
        
        elif rel_type == "DEPENDENT":
            # DEPENDENT requires: src_dim tgt_dim dep_type [scale_factor]
            if len(pos) < 6:
                raise PragmaError("DEPENDENT relationship requires source_dim, target_dim, and dependency_type")
            
            try:
                result["source_dim"] = int(pos[3])
                result["target_dim"] = int(pos[4])
            except ValueError:
                raise PragmaError("DEPENDENT dimension indices must be integers")
            
            result["dependency_type"] = pos[5]
            
            # Validate dependency type
            valid_dep_types = ["copy", "scaled", "min"]
            if result["dependency_type"] not in valid_dep_types:
                raise PragmaError(f"Invalid dependency type '{result['dependency_type']}'. "
                                f"Must be one of: {valid_dep_types}")
            
            # For scaled dependency, get scale factor
            if result["dependency_type"] == "scaled":
                if len(pos) < 7:
                    raise PragmaError("DEPENDENT relationship with 'scaled' type requires scale factor")
                result["scale_factor"] = pos[6]
                
        elif rel_type == "MULTIPLE":
            # MULTIPLE requires: src_dim tgt_dim [factor=N]
            if len(pos) < 5:
                raise PragmaError(f"{rel_type} relationship requires source_dim and target_dim")
            
            try:
                result["source_dim"] = int(pos[3])
                result["target_dim"] = int(pos[4])
            except ValueError:
                raise PragmaError(f"{rel_type} dimension indices must be integers")
            
            # Check for optional factor parameter in named arguments
            if "factor" in named:
                # Try to parse as int, otherwise keep as string (parameter name)
                try:
                    result["scale_factor"] = int(named["factor"])
                except ValueError:
                    result["scale_factor"] = named["factor"]
        
        else:
            raise PragmaError(f"Unknown relationship type '{rel_type}'. "
                            "Valid types: EQUAL, DEPENDENT, MULTIPLE")
        
        return result
    
    def apply_to_kernel(self, kernel: KernelMetadata) -> None:
        """Apply relationship pragma to kernel metadata.
        
        Args:
            kernel: KernelMetadata to update with relationship
        """
        logger.debug(f"Applying RELATIONSHIP pragma to kernel '{kernel.name}'")
        
        # Validate that both interfaces exist
        interface_names = [iface.name for iface in kernel.interfaces]
        source = self.parsed_data["source_interface"]
        target = self.parsed_data["target_interface"]
        
        if source not in interface_names:
            raise PragmaError(f"Source interface '{source}' not found in kernel. "
                            f"Available interfaces: {interface_names}")
        
        if target not in interface_names:
            raise PragmaError(f"Target interface '{target}' not found in kernel. "
                            f"Available interfaces: {interface_names}")
        
        # Add relationships list if it doesn't exist
        if not hasattr(kernel, 'relationships'):
            kernel.relationships = []
        
        # Map string relationship type to enum
        rel_type_str = self.parsed_data["relationship_type"]
        rel_type_map = {
            "EQUAL": RelationType.EQUAL,
            "DEPENDENT": RelationType.DEPENDENT,
            "MULTIPLE": RelationType.MULTIPLE,
        }
        
        if rel_type_str not in rel_type_map:
            raise PragmaError(f"Cannot map relationship type '{rel_type_str}' to RelationType enum")
        
        rel_type = rel_type_map[rel_type_str]
        
        # Create DimensionRelationship
        relationship = DimensionRelationship(
            source_interface=source,
            target_interface=target,
            relation=rel_type,
            source_dim=self.parsed_data.get("source_dim"),
            target_dim=self.parsed_data.get("target_dim"),
            factor=self.parsed_data.get("scale_factor"),
            dependency_type=self.parsed_data.get("dependency_type"),
            description=f"From pragma: {rel_type_str} relationship"
        )
        
        kernel.relationships.append(relationship)
        logger.debug(f"Added {relationship.relation.value} relationship between "
                    f"'{source}' and '{target}' to kernel metadata")