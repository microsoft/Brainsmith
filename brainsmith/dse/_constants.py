# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared constants for Brainsmith core.

This module provides constants used across the core system to ensure
consistency and single source of truth.
"""

# Skip indicators in blueprint steps
# These values in YAML all mean "skip this step"
SKIP_VALUES = frozenset([None, "~", ""])

# Normalized representation (internal)
# All skip values are normalized to this symbol
SKIP_INDICATOR = "~"
