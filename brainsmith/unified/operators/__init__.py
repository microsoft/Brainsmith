############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Unified Operator Implementations

This module contains concrete operator implementations using the unified
framework.
"""

from .thresholding import UnifiedThresholding

__all__ = ['UnifiedThresholding']