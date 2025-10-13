# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Legacy implementations
from .layernorm import LayerNorm
from .layernorm_hls import LayerNorm_hls as LayerNormHLS
from .infer_layernorm import InferLayerNorm

# Modern AutoHWCustomOp implementations
from .auto_layernorm import AutoLayerNorm
from .auto_layernorm_hls import AutoLayerNorm_hls as AutoLayerNormHLS
from .infer_auto_layernorm import InferAutoLayerNorm

__all__ = [
    # Legacy (deprecated but maintained for compatibility)
    "LayerNorm",
    "LayerNormHLS",
    "InferLayerNorm",

    # Modern AutoHWCustomOp implementation
    "AutoLayerNorm",
    "AutoLayerNormHLS",
    "InferAutoLayerNorm",
]