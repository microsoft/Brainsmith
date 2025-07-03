############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
RTL implementation domain for auto-generated thresholding kernels.

This module registers our AutoRTLBackend with QONNX/FINN using the proper
domain structure that FINN recognizes.
"""

from qonnx.custom_op.registry import register_op
from brainsmith.hw_kernels.thresholding.auto_thresholding.thresholding_axi_rtl import thresholding_axi_rtl

# Register with QONNX custom op registry
@register_op("brainsmith.hw_kernels.thresholding.auto_thresholding.rtl", "ThresholdingAxi")
class RegisteredThresholdingAxiRTL(thresholding_axi_rtl):
    """Registered version of ThresholdingAxi RTL backend for FINN integration."""
    pass