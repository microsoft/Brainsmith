# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared constants for Brainsmith core.

This module provides constants used across the core system to ensure
consistency and single source of truth.
"""

# Skip indicators in blueprint steps
# YAML can use ~, null, or empty string - all normalized to "~" during parsing
SKIP_VALUES = frozenset([None, "~", ""])  # For parsing only

# Canonical skip indicator (used internally after parsing)
SKIP_INDICATOR = "~"


def is_skip(value) -> bool:
    """Check if value is the skip indicator.

    Note: After parsing, all skip values are normalized to SKIP_INDICATOR.
    This function checks the canonical form only.
    """
    return value == SKIP_INDICATOR

