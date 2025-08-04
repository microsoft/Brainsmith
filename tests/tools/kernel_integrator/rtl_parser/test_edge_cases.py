############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Tests for edge cases and unusual scenarios in the RTL parser.

This module tests handling of uncommon but valid SystemVerilog
constructs and edge cases, including:
- Unusual but valid syntax
- Extreme parameter values
- Complex nested structures
- Performance with large files
- Unicode and special characters
"""