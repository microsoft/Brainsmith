############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Validation helper functions for RTL Parser testing.

This module provides comprehensive validation functions to ensure RTL Parser
output correctness across all components. These helpers support systematic
validation of KernelMetadata structure and pragma application effects.

Functions:
- validate_interface_metadata_complete(): Validates InterfaceMetadata structure
- validate_pragma_application(): Validates pragma effects on metadata
- validate_parameter_exposure(): Validates parameter exposure control
- validate_internal_datatypes(): Validates internal datatype creation
- validate_kernel_metadata_structure(): Validates complete KernelMetadata
"""

from typing import List, Set, Dict, Optional, Any
from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata, DatatypeMetadata
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup


def validate_interface_metadata_complete(interface: InterfaceMetadata) -> List[str]:
    """
    Validate that InterfaceMetadata has complete and consistent structure.
    
    Args:
        interface: InterfaceMetadata object to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields validation
    if not interface.name:
        errors.append("Interface name is required")
    
    if not isinstance(interface.interface_type, InterfaceType):
        errors.append(f"Interface type must be InterfaceType enum, got {type(interface.interface_type)}")
    
    if not isinstance(interface.datatype_constraints, list):
        errors.append(f"datatype_constraints must be list, got {type(interface.datatype_constraints)}")
    
    # Validate datatype constraints
    for i, constraint in enumerate(interface.datatype_constraints):
        if not isinstance(constraint, DatatypeConstraintGroup):
            errors.append(f"datatype_constraints[{i}] must be DatatypeConstraintGroup")
        else:
            if constraint.min_width <= 0:
                errors.append(f"datatype_constraints[{i}] min_width must be positive")
            if constraint.max_width < constraint.min_width:
                errors.append(f"datatype_constraints[{i}] max_width must be >= min_width")
    
    # Validate optional fields when present
    if interface.datatype_metadata is not None:
        if not isinstance(interface.datatype_metadata, DatatypeMetadata):
            errors.append("datatype_metadata must be DatatypeMetadata or None")
        else:
            # Validate datatype metadata structure
            dt_errors = validate_datatype_metadata(interface.datatype_metadata)
            errors.extend([f"datatype_metadata.{err}" for err in dt_errors])
    
    # Validate dimension parameters
    if interface.bdim_params is not None:
        if not isinstance(interface.bdim_params, list):
            errors.append("bdim_params must be list or None")
        elif not all(isinstance(p, str) for p in interface.bdim_params):
            errors.append("bdim_params must contain only strings")
    
    if interface.sdim_params is not None:
        if not isinstance(interface.sdim_params, list):
            errors.append("sdim_params must be list or None")
        elif not all(isinstance(p, str) for p in interface.sdim_params):
            errors.append("sdim_params must contain only strings")
    
    # Interface type specific validations
    if interface.interface_type in [InterfaceType.INPUT, InterfaceType.WEIGHT]:
        # Input and weight interfaces should have dimension parameters
        if interface.bdim_params is None:
            errors.append(f"{interface.interface_type.value} interface should have bdim_params")
        if interface.sdim_params is None:
            errors.append(f"{interface.interface_type.value} interface should have sdim_params")
    
    return errors


def validate_datatype_metadata(datatype: DatatypeMetadata) -> List[str]:
    """
    Validate DatatypeMetadata structure and consistency.
    
    Args:
        datatype: DatatypeMetadata object to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if not datatype.name:
        errors.append("name is required")
    
    # All parameter fields should be strings or None
    param_fields = ['width', 'signed', 'format', 'bias', 'fractional_width', 
                   'exponent_width', 'mantissa_width']
    
    for field in param_fields:
        value = getattr(datatype, field)
        if value is not None and not isinstance(value, str):
            errors.append(f"{field} must be string or None, got {type(value)}")
    
    return errors


def validate_pragma_application(kernel_metadata: KernelMetadata, 
                               expected_pragma_effects: Dict[str, Any]) -> List[str]:
    """
    Validate that pragmas have been correctly applied to kernel metadata.
    
    Args:
        kernel_metadata: KernelMetadata to validate
        expected_pragma_effects: Dictionary of expected effects
            Format: {
                'datatype_constraints': {interface_name: [(base_type, min_width, max_width), ...]},
                'bdim_params': {interface_name: [param_names]},
                'sdim_params': {interface_name: [param_names]},
                'weight_interfaces': [interface_names],
                'hidden_parameters': [param_names],
                'exposed_parameters': [param_names],
                'internal_datatypes': [datatype_names],
                'relationships': int  # expected count
            }
            
    Returns:
        List of validation errors
    """
    errors = []
    
    # Build interface lookup
    interfaces_by_name = {i.name: i for i in kernel_metadata.interfaces}
    
    # Validate datatype constraints
    if 'datatype_constraints' in expected_pragma_effects:
        for interface_name, expected_constraints in expected_pragma_effects['datatype_constraints'].items():
            if interface_name not in interfaces_by_name:
                errors.append(f"Expected interface '{interface_name}' not found")
                continue
                
            interface = interfaces_by_name[interface_name]
            if len(interface.datatype_constraints) != len(expected_constraints):
                errors.append(
                    f"Interface '{interface_name}' expected {len(expected_constraints)} "
                    f"constraints, got {len(interface.datatype_constraints)}"
                )
                continue
            
            for i, (expected_base, expected_min, expected_max) in enumerate(expected_constraints):
                if i >= len(interface.datatype_constraints):
                    break
                constraint = interface.datatype_constraints[i]
                if constraint.base_type != expected_base:
                    errors.append(
                        f"Interface '{interface_name}' constraint {i} expected base_type "
                        f"'{expected_base}', got '{constraint.base_type}'"
                    )
                if constraint.min_width != expected_min:
                    errors.append(
                        f"Interface '{interface_name}' constraint {i} expected min_width "
                        f"{expected_min}, got {constraint.min_width}"
                    )
                if constraint.max_width != expected_max:
                    errors.append(
                        f"Interface '{interface_name}' constraint {i} expected max_width "
                        f"{expected_max}, got {constraint.max_width}"
                    )
    
    # Validate BDIM parameter assignments
    if 'bdim_params' in expected_pragma_effects:
        for interface_name, expected_params in expected_pragma_effects['bdim_params'].items():
            if interface_name not in interfaces_by_name:
                errors.append(f"Expected interface '{interface_name}' not found")
                continue
            
            interface = interfaces_by_name[interface_name]
            if interface.bdim_params != expected_params:
                errors.append(
                    f"Interface '{interface_name}' expected bdim_params {expected_params}, "
                    f"got {interface.bdim_params}"
                )
    
    # Validate SDIM parameter assignments
    if 'sdim_params' in expected_pragma_effects:
        for interface_name, expected_params in expected_pragma_effects['sdim_params'].items():
            if interface_name not in interfaces_by_name:
                errors.append(f"Expected interface '{interface_name}' not found")
                continue
            
            interface = interfaces_by_name[interface_name]
            if interface.sdim_params != expected_params:
                errors.append(
                    f"Interface '{interface_name}' expected sdim_params {expected_params}, "
                    f"got {interface.sdim_params}"
                )
    
    # Validate weight interface detection
    if 'weight_interfaces' in expected_pragma_effects:
        actual_weight_names = {
            i.name for i in kernel_metadata.interfaces 
            if i.interface_type == InterfaceType.WEIGHT
        }
        expected_weight_names = set(expected_pragma_effects['weight_interfaces'])
        
        if actual_weight_names != expected_weight_names:
            errors.append(
                f"Expected weight interfaces {expected_weight_names}, "
                f"got {actual_weight_names}"
            )
    
    # Validate parameter exposure
    if 'hidden_parameters' in expected_pragma_effects:
        for param_name in expected_pragma_effects['hidden_parameters']:
            if param_name in kernel_metadata.exposed_parameters:
                errors.append(f"Parameter '{param_name}' should be hidden but is exposed")
    
    if 'exposed_parameters' in expected_pragma_effects:
        for param_name in expected_pragma_effects['exposed_parameters']:
            if param_name not in kernel_metadata.exposed_parameters:
                errors.append(f"Parameter '{param_name}' should be exposed but is hidden")
    
    # Validate internal datatypes
    if 'internal_datatypes' in expected_pragma_effects:
        actual_dt_names = {dt.name for dt in kernel_metadata.internal_datatypes}
        expected_dt_names = set(expected_pragma_effects['internal_datatypes'])
        
        if actual_dt_names != expected_dt_names:
            errors.append(
                f"Expected internal datatypes {expected_dt_names}, "
                f"got {actual_dt_names}"
            )
    
    # Validate relationships
    if 'relationships' in expected_pragma_effects:
        expected_count = expected_pragma_effects['relationships']
        actual_count = len(kernel_metadata.relationships)
        if actual_count != expected_count:
            errors.append(
                f"Expected {expected_count} relationships, got {actual_count}"
            )
    
    return errors


def validate_parameter_exposure(kernel_metadata: KernelMetadata,
                               should_be_hidden: Set[str],
                               should_be_exposed: Set[str]) -> List[str]:
    """
    Validate parameter exposure control.
    
    Args:
        kernel_metadata: KernelMetadata to validate
        should_be_hidden: Set of parameter names that should be hidden
        should_be_exposed: Set of parameter names that should be exposed
        
    Returns:
        List of validation errors
    """
    errors = []
    
    exposed_set = set(kernel_metadata.exposed_parameters)
    
    # Check hidden parameters
    for param_name in should_be_hidden:
        if param_name in exposed_set:
            errors.append(f"Parameter '{param_name}' should be hidden but is exposed")
    
    # Check exposed parameters
    for param_name in should_be_exposed:
        if param_name not in exposed_set:
            errors.append(f"Parameter '{param_name}' should be exposed but is hidden")
    
    return errors


def validate_internal_datatypes(kernel_metadata: KernelMetadata,
                               expected_datatypes: Dict[str, Dict[str, str]]) -> List[str]:
    """
    Validate internal datatype creation and structure.
    
    Args:
        kernel_metadata: KernelMetadata to validate
        expected_datatypes: Dict mapping datatype name to property dict
            Format: {
                'datatype_name': {
                    'width': 'PARAM_NAME',
                    'signed': 'PARAM_NAME', 
                    # ... other properties
                }
            }
            
    Returns:
        List of validation errors
    """
    errors = []
    
    # Build datatype lookup
    datatypes_by_name = {dt.name: dt for dt in kernel_metadata.internal_datatypes}
    
    # Check expected datatypes exist
    for dt_name, expected_props in expected_datatypes.items():
        if dt_name not in datatypes_by_name:
            errors.append(f"Expected internal datatype '{dt_name}' not found")
            continue
        
        datatype = datatypes_by_name[dt_name]
        
        # Validate properties
        for prop_name, expected_param in expected_props.items():
            actual_param = getattr(datatype, prop_name, None)
            if actual_param != expected_param:
                errors.append(
                    f"Internal datatype '{dt_name}' expected {prop_name}='{expected_param}', "
                    f"got '{actual_param}'"
                )
    
    return errors


def validate_kernel_metadata_structure(kernel_metadata: KernelMetadata) -> List[str]:
    """
    Validate complete KernelMetadata structure and consistency.
    
    Args:
        kernel_metadata: KernelMetadata to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Basic required fields
    if not kernel_metadata.name:
        errors.append("Kernel name is required")
    
    if not kernel_metadata.source_file:
        errors.append("Source file is required")
    
    # Validate collections
    if not isinstance(kernel_metadata.parameters, list):
        errors.append("Parameters must be a list")
    
    if not isinstance(kernel_metadata.interfaces, list):
        errors.append("Interfaces must be a list")
    
    if not isinstance(kernel_metadata.exposed_parameters, list):
        errors.append("Exposed parameters must be a list")
    
    if not isinstance(kernel_metadata.pragmas, list):
        errors.append("Pragmas must be a list")
    
    if not isinstance(kernel_metadata.internal_datatypes, list):
        errors.append("Internal datatypes must be a list")
    
    if not isinstance(kernel_metadata.relationships, list):
        errors.append("Relationships must be a list")
    
    # Validate interfaces
    for i, interface in enumerate(kernel_metadata.interfaces):
        interface_errors = validate_interface_metadata_complete(interface)
        errors.extend([f"interfaces[{i}].{err}" for err in interface_errors])
    
    # Validate internal datatypes
    for i, datatype in enumerate(kernel_metadata.internal_datatypes):
        dt_errors = validate_datatype_metadata(datatype)
        errors.extend([f"internal_datatypes[{i}].{err}" for err in dt_errors])
    
    # Validate parameter consistency 
    # Note: exposed_parameters may contain ALIAS targets that aren't in original parameters
    all_param_names = {p.name for p in kernel_metadata.parameters}
    alias_targets = set()
    
    # Extract alias targets from linked_parameters if available
    if hasattr(kernel_metadata, 'linked_parameters') and kernel_metadata.linked_parameters:
        alias_targets = set(kernel_metadata.linked_parameters.get("aliases", {}).values())
    
    for exposed_param in kernel_metadata.exposed_parameters:
        if exposed_param not in all_param_names and exposed_param not in alias_targets:
            errors.append(f"Exposed parameter '{exposed_param}' not found in parameters list or alias targets")
    
    # Validate interface name uniqueness
    interface_names = [i.name for i in kernel_metadata.interfaces]
    if len(interface_names) != len(set(interface_names)):
        errors.append("Interface names must be unique")
    
    # Validate internal datatype name uniqueness
    dt_names = [dt.name for dt in kernel_metadata.internal_datatypes]
    if len(dt_names) != len(set(dt_names)):
        errors.append("Internal datatype names must be unique")
    
    return errors


def assert_valid_kernel_metadata(kernel_metadata: KernelMetadata) -> None:
    """
    Assert that KernelMetadata is valid, raising AssertionError with details if not.
    
    Args:
        kernel_metadata: KernelMetadata to validate
        
    Raises:
        AssertionError: If validation fails
    """
    errors = validate_kernel_metadata_structure(kernel_metadata)
    if errors:
        error_msg = "KernelMetadata validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise AssertionError(error_msg)


def assert_pragma_effects(kernel_metadata: KernelMetadata, 
                         expected_effects: Dict[str, Any]) -> None:
    """
    Assert that pragma effects are correctly applied, raising AssertionError if not.
    
    Args:
        kernel_metadata: KernelMetadata to validate
        expected_effects: Expected pragma effects (see validate_pragma_application)
        
    Raises:
        AssertionError: If validation fails
    """
    errors = validate_pragma_application(kernel_metadata, expected_effects)
    if errors:
        error_msg = "Pragma application validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise AssertionError(error_msg)