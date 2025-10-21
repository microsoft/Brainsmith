# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# flake8: noqa
# Disable linting from here, as all imports will be flagged E402 and maybe F401

"""
Brainsmith HLS Kernel Imports

This is a TEMPORARY measure to ensure HLS variants are properly registered
in the kernels.hls namepace until backend refactoring is complete.

Similar to how FINN imports its HLS variants in:
deps/finn/src/finn/custom_op/fpgadataflow/hls/__init__.py
"""

# Import all HLS custom ops - they will be discovered automatically via namespace
# Note: Using absolute imports to ensure proper registration

# Import Brainsmith HLS kernels
from brainsmith.kernels.crop.crop_hls import LegacyCrop_hls
from brainsmith.kernels.crop.auto_crop_hls import Crop_hls
from brainsmith.kernels.layernorm.layernorm_hls import LayerNorm_hls
from brainsmith.kernels.shuffle.legacy_shuffle_hls import LegacyShuffle_hls
from brainsmith.kernels.shuffle.shuffle_hls import Shuffle_hls
from brainsmith.kernels.softmax.softmax_hls import Softmax_hls
from brainsmith.kernels.thresholding.thresholding_hls import Thresholding_hls
from brainsmith.kernels.vvau.vvau_hls import VectorVectorActivation_hls
