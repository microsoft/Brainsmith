############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Validation utilities for RTL Parser test results.

This module provides utilities for validating and comparing parser outputs,
including deep comparison of KernelMetadata objects with helpful diff output.
"""

from typing import Any, List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, fields
import difflib

from brainsmith.tools.kernel_integrator.types.metadata import KernelMetadata, InterfaceMetadata
from brainsmith.core.dataflow.types import InterfaceType
from brainsmith.core.dataflow.constraint_types import DatatypeConstraintGroup
from brainsmith.tools.kernel_integrator.types.metadata import DatatypeMetadata


@dataclass
class ValidationResult:
    """Result of validating a KernelMetadata object."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def assert_valid(self):
        """Assert that validation passed, with helpful error message."""
        if not self.is_valid:
            error_msg = "KernelMetadata validation failed:\n"
            if self.errors:
                error_msg += "\nErrors:\n" + "\n".join(f"  - {e}" for e in self.errors)
            if self.warnings:
                error_msg += "\nWarnings:\n" + "\n".join(f"  - {w}" for w in self.warnings)
            assert False, error_msg


def validate_kernel_metadata(km: KernelMetadata) -> ValidationResult:
    """Comprehensive validation of generated KernelMetadata."""
    errors = []
    warnings = []
    
    # Check required fields
    if not km.module_name:
        errors.append("Missing module_name")
    
    if km.interfaces is None:
        errors.append("interfaces is None (should be empty list if no interfaces)")
    elif not isinstance(km.interfaces, list):
        errors.append(f"interfaces should be list, got {type(km.interfaces)}")
    
    if km.exposed_parameters is None:
        errors.append("exposed_parameters is None (should be empty list if no parameters)")
    elif not isinstance(km.exposed_parameters, list):
        errors.append(f"exposed_parameters should be list, got {type(km.exposed_parameters)}")
    
    # Validate interfaces
    if isinstance(km.interfaces, list):
        interface_names = set()
        compiler_names = set()
        
        for i, interface in enumerate(km.interfaces):
            # Check interface fields
            if not interface.name:
                errors.append(f"Interface {i} missing name")
            elif interface.name in interface_names:
                errors.append(f"Duplicate interface name: {interface.name}")
            else:
                interface_names.add(interface.name)
            
            if not interface.compiler_name:
                errors.append(f"Interface {interface.name} missing compiler_name")
            elif interface.compiler_name in compiler_names:
                errors.append(f"Duplicate compiler_name: {interface.compiler_name}")
            else:
                compiler_names.add(interface.compiler_name)
            
            if interface.interface_type is None:
                errors.append(f"Interface {interface.name} missing interface_type")
            
            if interface.direction is None:
                errors.append(f"Interface {interface.name} missing direction")
            
            # Validate compiler name format
            if interface.compiler_name and interface.interface_type:
                expected_prefix = {
                    InterfaceType.INPUT: "input",
                    InterfaceType.OUTPUT: "output",
                    InterfaceType.WEIGHT: "weight",
                    InterfaceType.CONTROL: "global"
                }.get(interface.interface_type)
                
                if expected_prefix and not interface.compiler_name.startswith(expected_prefix):
                    warnings.append(
                        f"Interface {interface.name} has compiler_name '{interface.compiler_name}' "
                        f"but type {interface.interface_type.name} typically uses prefix '{expected_prefix}'"
                    )
            
            # Validate datatype constraints
            if interface.datatype_constraints:
                for j, constraint in enumerate(interface.datatype_constraints):
                    if not isinstance(constraint, DatatypeConstraintGroup):
                        errors.append(
                            f"Interface {interface.name} constraint {j} is not DatatypeConstraintGroup"
                        )
                    elif constraint.min_width > constraint.max_width:
                        errors.append(
                            f"Interface {interface.name} constraint {j} has min_width > max_width"
                        )
    
    # Validate exposed parameters
    if isinstance(km.exposed_parameters, list):
        param_names = set()
        for param in km.exposed_parameters:
            if not isinstance(param, str):
                errors.append(f"Exposed parameter is not string: {param}")
            elif param in param_names:
                errors.append(f"Duplicate exposed parameter: {param}")
            else:
                param_names.add(param)
    
    # Validate relationships
    if km.relationships:
        for i, rel in enumerate(km.relationships):
            if not isinstance(rel, RelationshipMetadata):
                errors.append(f"Relationship {i} is not RelationshipMetadata")
            else:
                # Check that referenced interfaces exist
                for iface in [rel.source_interface, rel.target_interface]:
                    if iface and iface not in interface_names:
                        warnings.append(
                            f"Relationship {i} references non-existent interface: {iface}"
                        )
    
    # Validate internal datatypes
    if km.internal_datatypes:
        dt_names = set()
        for dt in km.internal_datatypes:
            if not isinstance(dt, DatatypeMetadata):
                errors.append(f"Internal datatype is not DatatypeMetadata: {dt}")
            elif not dt.name:
                errors.append("Internal datatype missing name")
            elif dt.name in dt_names:
                errors.append(f"Duplicate internal datatype: {dt.name}")
            else:
                dt_names.add(dt.name)
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def assert_kernel_metadata_equal(expected: KernelMetadata, actual: KernelMetadata,
                               ignore_fields: Optional[Set[str]] = None) -> None:
    """Deep comparison with helpful diff output."""
    differences = []
    ignore_fields = ignore_fields or set()
    
    # Compare basic fields
    for field in ['module_name', 'finn_module']:
        if field in ignore_fields:
            continue
        expected_val = getattr(expected, field)
        actual_val = getattr(actual, field)
        if expected_val != actual_val:
            differences.append(f"{field}: expected {expected_val!r}, got {actual_val!r}")
    
    # Compare exposed parameters
    if 'exposed_parameters' not in ignore_fields:
        expected_params = set(expected.exposed_parameters or [])
        actual_params = set(actual.exposed_parameters or [])
        
        missing = expected_params - actual_params
        extra = actual_params - expected_params
        
        if missing:
            differences.append(f"Missing exposed parameters: {sorted(missing)}")
        if extra:
            differences.append(f"Extra exposed parameters: {sorted(extra)}")
    
    # Compare interfaces
    if 'interfaces' not in ignore_fields:
        expected_ifaces = {i.name: i for i in (expected.interfaces or [])}
        actual_ifaces = {i.name: i for i in (actual.interfaces or [])}
        
        missing_ifaces = set(expected_ifaces.keys()) - set(actual_ifaces.keys())
        extra_ifaces = set(actual_ifaces.keys()) - set(expected_ifaces.keys())
        
        if missing_ifaces:
            differences.append(f"Missing interfaces: {sorted(missing_ifaces)}")
        if extra_ifaces:
            differences.append(f"Extra interfaces: {sorted(extra_ifaces)}")
        
        # Compare common interfaces
        for name in expected_ifaces.keys() & actual_ifaces.keys():
            iface_diffs = _compare_interfaces(expected_ifaces[name], actual_ifaces[name])
            if iface_diffs:
                differences.append(f"Interface '{name}' differences:")
                differences.extend(f"  {d}" for d in iface_diffs)
    
    # Compare relationships
    if 'relationships' not in ignore_fields and (expected.relationships or actual.relationships):
        expected_rels = expected.relationships or []
        actual_rels = actual.relationships or []
        
        if len(expected_rels) != len(actual_rels):
            differences.append(
                f"Relationship count mismatch: expected {len(expected_rels)}, got {len(actual_rels)}"
            )
        else:
            for i, (exp_rel, act_rel) in enumerate(zip(expected_rels, actual_rels)):
                rel_diffs = _compare_relationships(exp_rel, act_rel)
                if rel_diffs:
                    differences.append(f"Relationship {i} differences:")
                    differences.extend(f"  {d}" for d in rel_diffs)
    
    # Assert with helpful message
    if differences:
        diff_msg = "KernelMetadata comparison failed:\n\n"
        diff_msg += "\n".join(differences)
        
        # Add visual diff of string representations
        diff_msg += "\n\nFull diff:\n"
        expected_str = _format_kernel_metadata(expected).splitlines()
        actual_str = _format_kernel_metadata(actual).splitlines()
        
        diff = difflib.unified_diff(
            expected_str, actual_str,
            fromfile="expected", tofile="actual",
            lineterm=""
        )
        diff_msg += "\n".join(diff)
        
        assert False, diff_msg


def assert_interface_metadata_equal(expected: InterfaceMetadata, actual: InterfaceMetadata,
                                  ignore_fields: Optional[Set[str]] = None) -> None:
    """Compare InterfaceMetadata objects with helpful diff output."""
    differences = _compare_interfaces(expected, actual, ignore_fields)
    
    if differences:
        diff_msg = "InterfaceMetadata comparison failed:\n\n"
        diff_msg += "\n".join(differences)
        assert False, diff_msg


def _compare_interfaces(expected: InterfaceMetadata, actual: InterfaceMetadata,
                       ignore_fields: Optional[Set[str]] = None) -> List[str]:
    """Compare two interfaces and return differences."""
    differences = []
    ignore_fields = ignore_fields or set()
    
    # Compare basic fields
    for field in ['name', 'interface_type', 'direction', 'compiler_name']:
        if field in ignore_fields:
            continue
        expected_val = getattr(expected, field)
        actual_val = getattr(actual, field)
        if expected_val != actual_val:
            differences.append(f"{field}: expected {expected_val!r}, got {actual_val!r}")
    
    # Compare ports
    if 'ports' not in ignore_fields and (expected.ports or actual.ports):
        expected_ports = {p.name: p for p in (expected.ports or [])}
        actual_ports = {p.name: p for p in (actual.ports or [])}
        
        missing = set(expected_ports.keys()) - set(actual_ports.keys())
        extra = set(actual_ports.keys()) - set(expected_ports.keys())
        
        if missing:
            differences.append(f"Missing ports: {sorted(missing)}")
        if extra:
            differences.append(f"Extra ports: {sorted(extra)}")
    
    # Compare datatype constraints
    if 'datatype_constraints' not in ignore_fields:
        exp_constraints = expected.datatype_constraints or []
        act_constraints = actual.datatype_constraints or []
        
        if len(exp_constraints) != len(act_constraints):
            differences.append(
                f"Constraint count mismatch: expected {len(exp_constraints)}, "
                f"got {len(act_constraints)}"
            )
        else:
            for i, (exp, act) in enumerate(zip(exp_constraints, act_constraints)):
                if (exp.base_type != act.base_type or 
                    exp.min_width != act.min_width or 
                    exp.max_width != act.max_width):
                    differences.append(
                        f"Constraint {i}: expected {exp}, got {act}"
                    )
    
    # Compare datatype metadata
    if 'datatype_metadata' not in ignore_fields:
        if (expected.datatype_metadata is None) != (actual.datatype_metadata is None):
            differences.append(
                f"datatype_metadata presence mismatch: "
                f"expected {'present' if expected.datatype_metadata else 'None'}, "
                f"got {'present' if actual.datatype_metadata else 'None'}"
            )
        elif expected.datatype_metadata and actual.datatype_metadata:
            dt_diffs = _compare_datatype_metadata(
                expected.datatype_metadata, 
                actual.datatype_metadata
            )
            if dt_diffs:
                differences.extend(f"datatype_metadata.{d}" for d in dt_diffs)
    
    return differences


def _compare_relationships(expected: RelationshipMetadata, actual: RelationshipMetadata) -> List[str]:
    """Compare two relationships and return differences."""
    differences = []
    
    for field in fields(RelationshipMetadata):
        expected_val = getattr(expected, field.name)
        actual_val = getattr(actual, field.name)
        if expected_val != actual_val:
            differences.append(f"{field.name}: expected {expected_val!r}, got {actual_val!r}")
    
    return differences


def _compare_datatype_metadata(expected: DatatypeMetadata, actual: DatatypeMetadata) -> List[str]:
    """Compare two DatatypeMetadata objects and return differences."""
    differences = []
    
    # Compare all fields
    for field in ['name', 'width', 'signed', 'format', 'bias', 'fractional_width']:
        expected_val = getattr(expected, field, None)
        actual_val = getattr(actual, field, None)
        if expected_val != actual_val:
            differences.append(f"{field}: expected {expected_val!r}, got {actual_val!r}")
    
    return differences


def _format_kernel_metadata(km: KernelMetadata) -> str:
    """Format KernelMetadata for readable diff output."""
    lines = []
    lines.append(f"KernelMetadata(")
    lines.append(f"  module_name={km.module_name!r},")
    lines.append(f"  finn_module={km.finn_module!r},")
    
    lines.append(f"  exposed_parameters=[")
    for param in sorted(km.exposed_parameters or []):
        lines.append(f"    {param!r},")
    lines.append(f"  ],")
    
    lines.append(f"  interfaces=[")
    for iface in sorted(km.interfaces or [], key=lambda i: i.name):
        lines.append(f"    InterfaceMetadata(")
        lines.append(f"      name={iface.name!r},")
        lines.append(f"      type={iface.interface_type},")
        lines.append(f"      direction={iface.direction},")
        lines.append(f"      compiler_name={iface.compiler_name!r},")
        if iface.datatype_constraints:
            lines.append(f"      constraints={iface.datatype_constraints},")
        if iface.datatype_metadata:
            lines.append(f"      datatype={iface.datatype_metadata},")
        lines.append(f"    ),")
    lines.append(f"  ],")
    
    if km.relationships:
        lines.append(f"  relationships=[")
        for rel in km.relationships:
            lines.append(f"    {rel},")
        lines.append(f"  ],")
    
    if km.internal_datatypes:
        lines.append(f"  internal_datatypes=[")
        for dt in km.internal_datatypes:
            lines.append(f"    {dt},")
        lines.append(f"  ],")
    
    lines.append(f")")
    
    return "\n".join(lines)


# Assertion helpers for common patterns
def assert_has_interface(km: KernelMetadata, name: str, 
                        interface_type: Optional[InterfaceType] = None) -> InterfaceMetadata:
    """Assert that kernel has interface with given name and optionally type."""
    interface = None
    for iface in km.interfaces:
        if iface.name == name:
            interface = iface
            break
    
    assert interface is not None, f"No interface named '{name}' found in kernel"
    
    if interface_type is not None:
        assert interface.interface_type == interface_type, \
            f"Interface '{name}' has type {interface.interface_type}, expected {interface_type}"
    
    return interface


def assert_parameter_exposed(km: KernelMetadata, param_name: str) -> None:
    """Assert that parameter is in exposed_parameters list."""
    assert param_name in km.exposed_parameters, \
        f"Parameter '{param_name}' not in exposed_parameters: {km.exposed_parameters}"


def assert_parameter_not_exposed(km: KernelMetadata, param_name: str) -> None:
    """Assert that parameter is NOT in exposed_parameters list."""
    assert param_name not in km.exposed_parameters, \
        f"Parameter '{param_name}' should not be exposed but found in: {km.exposed_parameters}"


def assert_interface_count(km: KernelMetadata, expected_count: int) -> None:
    """Assert the number of interfaces."""
    actual_count = len(km.interfaces)
    assert actual_count == expected_count, \
        f"Expected {expected_count} interfaces, got {actual_count}: " + \
        f"{[i.name for i in km.interfaces]}"


def assert_no_errors(result: Any) -> None:
    """Assert that no errors occurred during parsing."""
    if hasattr(result, 'errors') and result.errors:
        assert False, f"Unexpected errors: {result.errors}"


def assert_has_errors(result: Any, expected_count: Optional[int] = None) -> None:
    """Assert that errors occurred during parsing."""
    assert hasattr(result, 'errors'), "Result should have errors attribute"
    if expected_count is not None:
        assert len(result.errors) == expected_count, \
            f"Expected {expected_count} errors, got {len(result.errors)}: {result.errors}"
    else:
        assert len(result.errors) > 0, "Expected errors but none occurred"