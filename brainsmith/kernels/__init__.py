# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith hardware kernels.

Kernel implementations for FPGA deployment. All kernels are eagerly imported
for simplicity and debuggability. Manifest caching provides performance.
"""

# Kernels
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.kernels.crop.crop import Crop
from brainsmith.kernels.softmax.hwsoftmax import Softmax
from brainsmith.kernels.shuffle.shuffle import Shuffle

# Backends
from brainsmith.kernels.layernorm.layernorm_hls import LayerNorm_hls
from brainsmith.kernels.layernorm.layernorm_rtl import LayerNorm_rtl
from brainsmith.kernels.crop.crop_hls import Crop_hls
from brainsmith.kernels.softmax.hwsoftmax_hls import Softmax_hls
from brainsmith.kernels.shuffle.shuffle_hls import Shuffle_hls

__all__ = [
    # Kernels
    'LayerNorm',
    'Crop',
    'Softmax',
    'Shuffle',
    # Backends
    'LayerNorm_hls',
    'LayerNorm_rtl',
    'Crop_hls',
    'Softmax_hls',
    'Shuffle_hls',
]
