############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
HLS implementation domain for auto-generated thresholding kernels.

This module registers our AutoHWCustomOp with QONNX/FINN using the proper
domain structure that FINN recognizes.
"""

from qonnx.custom_op.registry import register_op
from brainsmith.hw_kernels.thresholding.auto_thresholding.thresholding_axi_hw_custom_op import ThresholdingAxi

# Register with QONNX custom op registry
@register_op("brainsmith.hw_kernels.thresholding.auto_thresholding.hls", "ThresholdingAxi")
class RegisteredThresholdingAxi(ThresholdingAxi):
    """Registered version of ThresholdingAxi for FINN integration."""
    pass