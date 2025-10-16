############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
FINN integration components for Brainsmith.

This module provides the AutoRTLBackend base class for RTL-based custom operators.
"""

from .auto_rtl_backend import AutoRTLBackend

__all__ = ["AutoRTLBackend"]