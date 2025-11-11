############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Modular parameter linking service for automatic parameter assignment.

This module provides automatic linking of RTL parameters to interface properties
based on naming conventions. Parameters are moved from kernel.parameters to 
appropriate interface fields based on pattern matching.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from re import Pattern

from brainsmith.dataflow.types import InterfaceType
from brainsmith.tools.kernel_integrator.metadata import (
    AXILiteMetadata,
    AXIStreamMetadata,
    DatatypeParameters,
    KernelMetadata,
)

from .types import Parameter

logger = logging.getLogger(__name__)


# Pattern definitions for each parameter type
BDIM_PATTERNS = [(r"^(.+)_BDIM$", "single"), (r"^(.+)_BDIM(\d+)$", "indexed")]

SDIM_PATTERNS = [(r"^(.+)_SDIM$", "single"), (r"^(.+)_SDIM(\d+)$", "indexed")]

DTYPE_PATTERNS = {
    "width": [r"_WIDTH$", r"_W$", r"_BITS$"],
    "signed": [r"_SIGNED$", r"_SIGN$"],
    "bias": [r"_BIAS$"],
    "format": [r"_FORMAT$", r"_FMT$"],
    "fractional_width": [r"_FRACTIONAL_WIDTH$", r"_FRAC_WIDTH$"],
    "exponent_width": [r"_EXPONENT_WIDTH$", r"_EXP_WIDTH$"],
    "mantissa_width": [r"_MANTISSA_WIDTH$", r"_MANT_WIDTH$"],
}

AXILITE_PATTERNS = {
    "enable": [r"^USE_(.+)$", r"^(.+)_EN$", r"^ENABLE_(.+)$"],
    "data_width": [r"^(.+)_DATA_WIDTH$", r"^(.+)_DW$"],
    "addr_width": [r"^(.+)_ADDR_WIDTH$", r"^(.+)_AW$"],
}


@dataclass
class LinkingRule:
    """A rule for linking parameters based on patterns."""

    name: str
    pattern: Pattern[str]
    handler: Callable[[Parameter, KernelMetadata, re.Match, dict | None], bool]
    priority: int = 100  # Lower = higher priority
    metadata: dict | None = None  # Additional data for handler


class ParameterLinker:
    """Modular parameter linker with extensible rule system."""

    def __init__(self):
        self.rules: list[LinkingRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up the default linking rules."""
        # BDIM rules (priority 50) - using consolidated handler
        self.rules.append(
            LinkingRule(
                name="bdim",
                pattern=self._build_pattern(BDIM_PATTERNS),
                handler=link_dimension_parameter,
                priority=50,
                metadata={"dimension": "bdim", "patterns": BDIM_PATTERNS},
            )
        )

        # SDIM rules (priority 60) - using consolidated handler
        self.rules.append(
            LinkingRule(
                name="sdim",
                pattern=self._build_pattern(SDIM_PATTERNS),
                handler=link_dimension_parameter,
                priority=60,
                metadata={"dimension": "sdim", "patterns": SDIM_PATTERNS},
            )
        )

        # AXI-Lite rules (priority 65 - before datatype to avoid _WIDTH collision)
        self.rules.append(
            LinkingRule(
                name="axilite",
                pattern=self._build_pattern(AXILITE_PATTERNS),
                handler=link_axilite_parameter,
                priority=65,
            )
        )

        # Datatype rules (priority 70)
        self.rules.append(
            LinkingRule(
                name="dtype",
                pattern=self._build_pattern(DTYPE_PATTERNS, pattern_type="suffix"),
                handler=link_dtype_parameter,
                priority=70,
            )
        )

        # Sort by priority
        self.rules.sort(key=lambda x: x.priority)

    def _build_pattern(self, pattern_source, pattern_type="standard"):
        """Build regex pattern from various source formats.

        Args:
            pattern_source: List of patterns or dict of property->patterns
            pattern_type: 'standard', 'suffix', or 'dimension'
        """
        patterns = []

        # Extract patterns from source
        if isinstance(pattern_source, dict):
            # Flatten dict values (AXILITE_PATTERNS, DTYPE_PATTERNS)
            for patterns_list in pattern_source.values():
                patterns.extend(patterns_list)
        else:
            # List of patterns (BDIM_PATTERNS, SDIM_PATTERNS)
            if pattern_source and isinstance(pattern_source[0], tuple):
                patterns = [p for p, _ in pattern_source]
            else:
                patterns = pattern_source

        # Build regex based on type
        if pattern_type == "suffix":
            # For dtype patterns - strip anchors, escape, re-anchor
            suffixes = [p.replace("$", "") for p in patterns]
            return re.compile(".+(" + "|".join(re.escape(s) for s in suffixes) + ")$")
        else:
            # Standard - join patterns with | (patterns already have their own groups)
            return re.compile("|".join(patterns))

    def link_parameters(self, kernel: KernelMetadata) -> None:
        """Link parameters using registered rules.

        Processes each parameter in kernel.parameters and attempts to link it
        to an appropriate interface or internal group based on naming patterns.
        Parameters that don't match any pattern remain in kernel.parameters.

        Args:
            kernel: KernelMetadata to process (modified in place)
        """
        remaining_params = []

        logger.info(
            f"Starting parameter linking for kernel '{kernel.name}' with {len(kernel.parameters)} parameters"
        )

        for param in kernel.parameters:
            linked = False

            # Try each rule in priority order
            for rule in self.rules:
                match = rule.pattern.match(param.name)
                if match:
                    logger.debug(f"Parameter '{param.name}' matches rule '{rule.name}'")
                    # Try to link using the handler
                    if rule.handler(param, kernel, match, rule.metadata):
                        logger.debug(
                            f"Parameter '{param.name}' successfully linked by rule '{rule.name}'"
                        )
                        linked = True
                        break

            if not linked:
                logger.debug(
                    f"Parameter '{param.name}' not linked by any rule, keeping in kernel.parameters"
                )
                remaining_params.append(param)

        kernel.parameters = remaining_params
        logger.info(
            f"Parameter linking complete. {len(remaining_params)} parameters remain unlinked"
        )


# Handler functions for each parameter type


def link_dimension_parameter(
    param: Parameter, kernel: KernelMetadata, match: re.Match, metadata: dict | None
) -> bool:
    """Try to link parameter as dimension (BDIM or SDIM).

    This is a consolidated handler that uses the rule metadata to determine
    which dimension type and patterns to use.
    """
    if not metadata:
        return False

    dimension_type = metadata.get("dimension")  # 'bdim' or 'sdim'

    # Find the first non-None capture group (since we have multiple patterns combined)
    interface_name = None
    for i in range(1, match.lastindex + 1 if match.lastindex else 1):
        group_val = match.group(i)
        if group_val is not None and not group_val.isdigit():  # Skip index groups
            interface_name = group_val
            break

    if not interface_name:
        logger.debug(f"Could not extract interface name from match for '{param.name}'")
        return False

    logger.debug(
        f"{dimension_type.upper()} pattern matched for '{param.name}', interface_name='{interface_name}'"
    )
    interface = _find_stream_interface(interface_name, kernel)

    if not interface:
        logger.debug(f"No interface found with name '{interface_name}'")
        return False

    # For SDIM, check interface type
    if dimension_type == "sdim" and interface.interface_type not in [
        InterfaceType.INPUT,
        InterfaceType.WEIGHT,
    ]:
        logger.debug(f"SDIM not applicable to {interface.interface_type} interface")
        return False

    attr_name = f"{dimension_type}_params"

    # Check if this is an indexed pattern by looking for a digit group
    index_group = None
    if match.lastindex:
        for i in range(1, match.lastindex + 1):
            group_val = match.group(i)
            if group_val is not None and group_val.isdigit():
                index_group = int(group_val)
                break

    if index_group is not None:
        # Indexed parameter
        if not getattr(interface, attr_name):
            _add_indexed_dimension_param(interface, attr_name, param, index_group)
            logger.debug(
                f"Successfully linked indexed '{param.name}' to interface '{interface.name}'"
            )
            return True
        else:
            logger.debug(f"Interface '{interface.name}' already has {attr_name}, skipping")
            return False
    else:
        # Single parameter case
        context = f"'{param.name}' to interface '{interface.name}' as {attr_name}"
        return _assign_if_empty(interface, attr_name, [param], context)


def link_dtype_parameter(
    param: Parameter, kernel: KernelMetadata, match: re.Match, metadata: dict | None
) -> bool:
    """Try to link parameter as datatype property."""
    # Find the best (longest) matching pattern to handle compound suffixes correctly
    best_match = None
    best_property = None
    best_prefix = None

    for property_name, patterns in DTYPE_PATTERNS.items():
        for pattern_str in patterns:
            suffix = pattern_str.replace("$", "")
            if param.name.endswith(suffix):
                prefix = param.name[: -len(suffix)]
                # Prefer longer suffixes (more specific)
                if best_match is None or len(suffix) > len(best_match):
                    best_match = suffix
                    best_property = property_name
                    best_prefix = prefix

    if not best_property:
        logger.debug(f"Could not determine property for parameter '{param.name}'")
        return False

    # Try to find matching interface - check both stream and AXI-Lite interfaces
    interface = _find_stream_interface(best_prefix, kernel)
    if not interface:
        # Also check AXI-Lite interfaces
        interface = _find_axilite_interface(best_prefix, kernel)

    if interface:
        # Create DatatypeParameters if needed
        if not interface.dtype_params:
            interface.dtype_params = DatatypeParameters()

        # Set kernel_value to track the property type
        param.kernel_value = best_property

        # Use _assign_if_empty to set the property
        context = (
            f"'{param.name}' to interface '{interface.name}' as dtype property '{best_property}'"
        )
        return _assign_if_empty(interface.dtype_params, best_property, param, context)

    # No matching interface - parameter remains unlinked
    logger.debug(
        f"No interface found for dtype parameter '{param.name}' with prefix '{best_prefix}'"
    )
    return False


def link_axilite_parameter(
    param: Parameter, kernel: KernelMetadata, match: re.Match, metadata: dict | None
) -> bool:
    """Try to link parameter to AXI-Lite interface."""
    # Extract interface name from match groups
    interface_name = None
    for group in match.groups():
        if group:
            interface_name = group
            break

    if not interface_name:
        return False

    interface = _find_axilite_interface(interface_name, kernel)
    if not interface:
        logger.debug(f"No AXI-Lite interface found for '{interface_name}'")
        return False

    # Find which property this pattern matches
    for property_name, patterns in AXILITE_PATTERNS.items():
        for pattern_str in patterns:
            if re.match(pattern_str, param.name):
                # Map property name to interface attribute
                attr_map = {
                    "enable": "enable_param",
                    "data_width": "data_width_param",
                    "addr_width": "addr_width_param",
                }

                attr_name = attr_map.get(property_name)
                if attr_name:
                    context = (
                        f"'{param.name}' to AXI-Lite interface '{interface.name}' as {attr_name}"
                    )
                    return _assign_if_empty(interface, attr_name, param, context)

    return False


# Helper functions


def _assign_if_empty(obj, attr_name: str, value, logger_context: str = "") -> bool:
    """Assign value to attribute if it's currently empty/None/False.

    Args:
        obj: Object to set attribute on
        attr_name: Name of attribute to set
        value: Value to assign
        logger_context: Optional context for debug logging

    Returns:
        True if assignment was made, False if attribute was already set
    """
    current_value = getattr(obj, attr_name)
    if not current_value:
        setattr(obj, attr_name, value)
        if logger_context:
            logger.debug(f"Assigned {logger_context}")
        return True
    if logger_context:
        logger.debug(f"Skipped {logger_context} - already set")
    return False


def _find_stream_interface(name: str, kernel: KernelMetadata) -> AXIStreamMetadata | None:
    """Find an AXI-Stream interface by name."""
    # Check inputs
    for interface in kernel.inputs:
        if interface.name == name:
            return interface

    # Check outputs
    for interface in kernel.outputs:
        if interface.name == name:
            return interface

    return None


def _find_axilite_interface(name: str, kernel: KernelMetadata) -> AXILiteMetadata | None:
    """Find an AXI-Lite interface by name."""
    for interface in kernel.config:
        if interface.name == name or interface.name == f"s_axilite_{name}":
            return interface
    return None


def _add_indexed_dimension_param(
    interface: AXIStreamMetadata, attr_name: str, param: Parameter, index: int
) -> None:
    """Add an indexed parameter to a dimension list, handling gaps."""
    current_list = getattr(interface, attr_name)

    # If empty, create new list
    if not current_list:
        # Fill with "1" up to index
        new_list = ["1"] * (index + 1)
        new_list[index] = param
        setattr(interface, attr_name, new_list)
    else:
        # Extend existing list if needed
        while len(current_list) <= index:
            current_list.append("1")
        current_list[index] = param
