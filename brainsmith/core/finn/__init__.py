############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
FINN integration components for Brainsmith.

This module provides base classes and utilities for integrating custom hardware
kernels with the FINN framework through the Brainsmith Kernel Modeling system.
"""

from .auto_hw_custom_op import AutoHWCustomOp
from .auto_rtl_backend import AutoRTLBackend

__all__ = ["AutoHWCustomOp", "AutoRTLBackend"]