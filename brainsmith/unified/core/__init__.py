############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Unified Framework Core Components

This module provides the core infrastructure for the unified implementation
that integrates the Unified Kernel Modeling Framework with FINN.
"""

from .kernel_definition import (
    KernelDefinition,
    InterfaceDefinition,
    ProtocolType,
    DatatypeConstraint,
    PerformanceModel,
    ResourceModel
)
from .unified_hw_custom_op import UnifiedHWCustomOp
from .unified_rtl_backend import UnifiedRTLBackend
from .dse_integration import UnifiedDSEMixin
from .kernel_factory import KernelDefinitionFactory

__all__ = [
    'KernelDefinition',
    'InterfaceDefinition',
    'ProtocolType',
    'DatatypeConstraint',
    'PerformanceModel',
    'ResourceModel',
    'UnifiedHWCustomOp',
    'UnifiedRTLBackend',
    'UnifiedDSEMixin',
    'KernelDefinitionFactory'
]