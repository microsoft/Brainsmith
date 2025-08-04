############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Tests for error handling in the RTL parser.

This module tests error detection, reporting, and recovery across
all parser components, including:
- Syntax error handling
- Invalid pragma detection
- Missing parameter errors
- Circular dependency detection
- Error message quality and clarity
"""