############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Unified Framework for Brainsmith

This module provides a clean implementation that integrates the Unified Kernel
Modeling Framework with FINN's HWCustomOp infrastructure, running in parallel
with the existing dataflow framework.
"""

from .core import (
    KernelDefinition,
    InterfaceDefinition,
    UnifiedHWCustomOp,
    UnifiedRTLBackend,
    UnifiedDSEMixin,
    KernelDefinitionFactory
)

__version__ = "0.1.0"

__all__ = [
    'KernelDefinition',
    'InterfaceDefinition', 
    'UnifiedHWCustomOp',
    'UnifiedRTLBackend',
    'UnifiedDSEMixin',
    'KernelDefinitionFactory'
]