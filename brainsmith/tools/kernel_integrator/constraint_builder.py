"""
Constraint builders for kernel metadata.

This module provides functions to build dataflow constraints from
kernel integrator metadata structures.
"""

from typing import Dict, List, Optional, Set, Union, Any, Tuple
from dataclasses import dataclass

from brainsmith.core.dataflow.constraint_types import (
    DatatypeConstraintGroup,
    validate_datatype_against_constraints
)
from brainsmith.core.dataflow.types import ShapeSpec

from .types.metadata import DatatypeMetadata
from .types.rtl import Parameter


# Simple constraint types for integration layer
@dataclass
class DimensionConstraint:
    """Constraint on a dimension."""
    dimension_index: int
    constraint_type: str  # "parameter", "fixed", "range"
    parameter_name: Optional[str] = None
    fixed_value: Optional[int] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    description: Optional[str] = None


@dataclass
class ParameterConstraint:
    """Constraint on a parameter."""
    parameter_name: str
    constraint_type: str  # "value", "range", "alias", "derived", "runtime"
    default_value: Optional[Any] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    target_parameter: Optional[str] = None  # For alias
    expression: Optional[str] = None  # For derived
    interface: Optional[str] = None  # For runtime (e.g., "axilite")
    is_exposed: bool = False
    description: Optional[str] = None


@dataclass 
class ShapeConstraint:
    """Constraint on shape dimensions."""
    dimension_type: str  # "block" or "stream"
    shape_spec: List[Union[int, str]]
    parameter_names: List[str]
    description: Optional[str] = None


@dataclass
class RelationshipConstraint:
    """Constraint on interface relationships."""
    source_interface: str
    target_interface: str
    relationship_type: str  # "equal", "dependent", "multiple", "divisible"
    source_dimension: Optional[int] = None
    target_dimension: Optional[int] = None
    factor: Optional[Union[int, str]] = None
    metadata: Optional[Dict[str, Any]] = None


def build_datatype_constraints(
    existing_constraints: List[DatatypeConstraintGroup],
    datatype_metadata: Optional[DatatypeMetadata]
) -> List[DatatypeConstraintGroup]:
    """
    Build datatype constraints from metadata.
    
    Combines existing constraints with parameter-based constraints
    from datatype metadata.
    
    Args:
        existing_constraints: Pre-existing datatype constraints
        datatype_metadata: Optional metadata with parameter mappings
        
    Returns:
        List of datatype constraint groups
    """
    # Start with existing constraints
    constraints = list(existing_constraints) if existing_constraints else []
    
    # Add constraints from datatype metadata
    if datatype_metadata:
        # Create parameter-based constraints
        param_constraints = []
        
        if datatype_metadata.width:
            param_constraints.append({
                "property": "bit_width",
                "parameter": datatype_metadata.width,
                "description": f"Width from parameter {datatype_metadata.width}"
            })
        
        if datatype_metadata.signed:
            param_constraints.append({
                "property": "signed",
                "parameter": datatype_metadata.signed,
                "description": f"Signedness from parameter {datatype_metadata.signed}"
            })
        
        if datatype_metadata.format:
            param_constraints.append({
                "property": "format",
                "parameter": datatype_metadata.format,
                "description": f"Format from parameter {datatype_metadata.format}"
            })
        
        # Add as a new constraint group if we have parameter constraints
        if param_constraints:
            param_group = DatatypeConstraintGroup(
                description=f"Parameter-based constraints for {datatype_metadata.name}",
                constraints=[],  # Would need proper constraint objects
                metadata={"parameter_constraints": param_constraints}
            )
            constraints.append(param_group)
    
    return constraints


def build_dimension_constraints(
    bdim_shape: Optional[List[Union[int, str]]],
    sdim_shape: Optional[List[Union[int, str]]],
    bdim_params: Optional[List[str]],
    sdim_params: Optional[List[str]]
) -> List[DimensionConstraint]:
    """
    Build dimension constraints from shape specifications.
    
    Args:
        bdim_shape: Block dimension shape specification
        sdim_shape: Stream dimension shape specification
        bdim_params: RTL parameters for block dimensions
        sdim_params: RTL parameters for stream dimensions
        
    Returns:
        List of dimension constraints
    """
    constraints = []
    
    # Build block dimension constraints
    if bdim_shape:
        block_constraint = ShapeConstraint(
            dimension_type="block",
            shape_spec=bdim_shape,
            parameter_names=bdim_params or [],
            description="Block dimensions from RTL"
        )
        constraints.append(block_constraint)
    
    # Build stream dimension constraints
    if sdim_shape:
        stream_constraint = ShapeConstraint(
            dimension_type="stream",
            shape_spec=sdim_shape,
            parameter_names=sdim_params or [],
            description="Stream dimensions from RTL"
        )
        constraints.append(stream_constraint)
    
    # Add parameter-based constraints
    if bdim_params:
        for i, param in enumerate(bdim_params):
            if param != "1":  # Skip singleton dimensions
                constraints.append(
                    DimensionConstraint(
                        dimension_index=i,
                        constraint_type="parameter",
                        parameter_name=param,
                        description=f"Block dimension {i} from parameter {param}"
                    )
                )
    
    if sdim_params:
        for i, param in enumerate(sdim_params):
            if param != "1":  # Skip singleton dimensions
                constraints.append(
                    DimensionConstraint(
                        dimension_index=i,
                        constraint_type="parameter",
                        parameter_name=param,
                        description=f"Stream dimension {i} from parameter {param}"
                    )
                )
    
    return constraints


def build_parameter_constraints(
    parameters: List[Parameter],
    exposed_parameters: List[str],
    linked_parameters: Dict[str, Dict[str, str]]
) -> List[ParameterConstraint]:
    """
    Build parameter constraints from metadata.
    
    Args:
        parameters: All kernel parameters
        exposed_parameters: Parameters exposed to user
        linked_parameters: Parameter linkage information
        
    Returns:
        List of parameter constraints
    """
    constraints = []
    param_dict = {p.name: p for p in parameters}
    
    # Create constraints for exposed parameters
    for param_name in exposed_parameters:
        if param_name in param_dict:
            param = param_dict[param_name]
            constraint = _create_parameter_constraint(param, is_exposed=True)
            constraints.append(constraint)
    
    # Create constraints for linked parameters
    if "aliases" in linked_parameters:
        for alias, target in linked_parameters["aliases"].items():
            constraints.append(
                ParameterConstraint(
                    parameter_name=alias,
                    constraint_type="alias",
                    target_parameter=target,
                    description=f"Alias for {target}"
                )
            )
    
    if "derived" in linked_parameters:
        for derived, expression in linked_parameters["derived"].items():
            constraints.append(
                ParameterConstraint(
                    parameter_name=derived,
                    constraint_type="derived",
                    expression=expression,
                    description=f"Derived from {expression}"
                )
            )
    
    if "axilite" in linked_parameters:
        for param_name in linked_parameters["axilite"]:
            constraints.append(
                ParameterConstraint(
                    parameter_name=param_name,
                    constraint_type="runtime",
                    interface="axilite",
                    description="Runtime configurable via AXI-Lite"
                )
            )
    
    return constraints


def _create_parameter_constraint(
    param: Parameter,
    is_exposed: bool = False
) -> ParameterConstraint:
    """Create a parameter constraint from RTL parameter."""
    # Determine constraint type based on parameter type
    constraint_type = "value"
    min_value = None
    max_value = None
    allowed_values = None
    
    if param.param_type == "integer":
        constraint_type = "range"
        # Could parse default value for range hints
        if param.default_value:
            try:
                val = int(param.default_value)
                if val > 0:
                    min_value = 1
                    max_value = 2**31 - 1  # Reasonable max
            except ValueError:
                pass
    
    return ParameterConstraint(
        parameter_name=param.name,
        constraint_type=constraint_type,
        min_value=min_value,
        max_value=max_value,
        allowed_values=allowed_values,
        default_value=param.default_value,
        is_exposed=is_exposed,
        description=param.description
    )


def build_relationship_constraints(
    relationships: List[Any],
    interfaces: Dict[str, Any]
) -> List[RelationshipConstraint]:
    """
    Build relationship constraints between interfaces.
    
    Args:
        relationships: Dimension relationships from metadata
        interfaces: Interface definitions for validation
        
    Returns:
        List of relationship constraints
    """
    constraints = []
    
    for rel in relationships:
        # Validate interfaces exist
        if (rel.source_interface not in interfaces or 
            rel.target_interface not in interfaces):
            continue
        
        constraint = RelationshipConstraint(
            source_interface=rel.source_interface,
            target_interface=rel.target_interface,
            relationship_type=rel.relation.value,
            source_dimension=rel.source_dim,
            target_dimension=rel.target_dim,
            factor=rel.factor,
            metadata={
                "dependency_type": rel.dependency_type,
                "description": rel.description
            }
        )
        constraints.append(constraint)
    
    return constraints


def merge_constraints(
    existing: List[Any],
    new: List[Any],
    merge_strategy: str = "append"
) -> List[Any]:
    """
    Merge two lists of constraints.
    
    Args:
        existing: Existing constraints
        new: New constraints to merge
        merge_strategy: How to merge ("append", "replace", "unique")
        
    Returns:
        Merged constraint list
    """
    if merge_strategy == "replace":
        return new
    elif merge_strategy == "append":
        return existing + new
    elif merge_strategy == "unique":
        # Remove duplicates based on constraint identity
        seen = set()
        result = []
        for constraint in existing + new:
            key = _constraint_key(constraint)
            if key not in seen:
                seen.add(key)
                result.append(constraint)
        return result
    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")


def _constraint_key(constraint: Any) -> str:
    """Generate a unique key for constraint deduplication."""
    if hasattr(constraint, 'parameter_name'):
        return f"param:{constraint.parameter_name}"
    elif hasattr(constraint, 'dimension_index'):
        return f"dim:{constraint.dimension_index}:{constraint.constraint_type}"
    elif hasattr(constraint, 'source_interface') and hasattr(constraint, 'target_interface'):
        return f"rel:{constraint.source_interface}:{constraint.target_interface}"
    else:
        # Fallback to string representation
        return str(constraint)


def validate_constraints(
    constraints: List[Any],
    context: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate a set of constraints for consistency.
    
    Args:
        constraints: Constraints to validate
        context: Context with parameters, interfaces, etc.
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check parameter constraints reference existing parameters
    param_constraints = [c for c in constraints if hasattr(c, 'parameter_name')]
    available_params = set(context.get('parameters', {}).keys())
    
    for constraint in param_constraints:
        if constraint.parameter_name not in available_params:
            errors.append(
                f"Parameter constraint references unknown parameter: {constraint.parameter_name}"
            )
    
    # Check dimension constraints are within bounds
    dim_constraints = [c for c in constraints if hasattr(c, 'dimension_index')]
    for constraint in dim_constraints:
        # Would need interface dimension info to validate properly
        if constraint.dimension_index < 0:
            errors.append(
                f"Invalid dimension index: {constraint.dimension_index}"
            )
    
    # Check relationship constraints reference valid interfaces
    rel_constraints = [c for c in constraints if hasattr(c, 'source_interface')]
    available_interfaces = set(context.get('interfaces', {}).keys())
    
    for constraint in rel_constraints:
        if constraint.source_interface not in available_interfaces:
            errors.append(
                f"Relationship references unknown source interface: {constraint.source_interface}"
            )
        if constraint.target_interface not in available_interfaces:
            errors.append(
                f"Relationship references unknown target interface: {constraint.target_interface}"
            )
    
    return len(errors) == 0, errors


# Type aliases for clarity
from typing import Tuple
ConstraintValidation = Tuple[bool, List[str]]