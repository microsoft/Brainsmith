# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
BrainSmith Kernels

Plugin-based hardware kernel implementations.
"""

# Import all custom kernels - they will be discovered automatically via namespace
from .crop.crop import Crop
from .layernorm.layernorm import LayerNorm
from .shuffle.shuffle import Shuffle
from .softmax.hwsoftmax import HWSoftmax

# Import HLS backends
from .crop.crop_hls import Crop_hls
from .layernorm.layernorm_hls import LayerNorm_hls
from .shuffle.shuffle_hls import Shuffle_hls
from .softmax.hwsoftmax_hls import HWSoftmax_hls

# Import kernel inference transforms to trigger registration
from .crop.infer_crop_from_gather import InferCropFromGather
from .layernorm.infer_layernorm import InferLayerNorm
from .shuffle.infer_shuffle import InferShuffle
from .softmax.infer_hwsoftmax import InferHWSoftmax

# Import HLS submodule to ensure proper registration in FINN namespace
# TEMPORARY: Until backend refactoring is complete
from . import hls