# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Kernels

Plugin-based hardware kernel implementations.
"""

# Kernels
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.kernels.crop.crop import Crop
from brainsmith.kernels.softmax.hwsoftmax import HWSoftmax
from brainsmith.kernels.shuffle.shuffle import Shuffle

# Backends
from brainsmith.kernels.layernorm.layernorm_hls import LayerNorm_hls
from brainsmith.kernels.layernorm.layernorm_rtl import LayerNorm_rtl
from brainsmith.kernels.crop.crop_hls import Crop_hls
from brainsmith.kernels.softmax.hwsoftmax_hls import HWSoftmax_hls
from brainsmith.kernels.shuffle.shuffle_hls import Shuffle_hls

__all__ = [
    # Kernels
    'LayerNorm',
    'Crop',
    'HWSoftmax',
    'Shuffle',
    # Backends
    'LayerNorm_hls',
    'LayerNorm_rtl',
    'Crop_hls',
    'HWSoftmax_hls',
    'Shuffle_hls',
]
