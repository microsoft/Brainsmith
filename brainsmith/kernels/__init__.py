# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Kernels - Lazy Loading for Performance

Plugin-based hardware kernel implementations using PEP 562 lazy loading.
Kernels are imported only when actually accessed, avoiding expensive upfront
imports of torch, numpy, scipy, etc.
"""

from brainsmith.plugin_helpers import create_lazy_module

# ============================================================================
# Kernel Registry (Metadata Only - NO imports!)
# ============================================================================

COMPONENTS = {
    'kernels': {
        'LayerNorm': '.layernorm.layernorm',
        'Crop': '.crop.crop',
        'Softmax': '.softmax',
        'Shuffle': '.shuffle.shuffle',
    },
    'backends': {
        'LayerNorm_hls': '.layernorm.layernorm_hls',
        'Crop_hls': '.crop.crop_hls',
        'Softmax_hls': '.softmax.hwsoftmax_hls',
        'Shuffle_hls': '.shuffle.shuffle_hls',
    }
}

# ============================================================================
# Lazy Loading (PEP 562) - Unified Pattern
# ============================================================================

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
