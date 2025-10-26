# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith Kernels - Lazy Loading for Performance

Plugin-based hardware kernel implementations using PEP 562 lazy loading.
Kernels are imported only when actually accessed, avoiding expensive upfront
imports of torch, numpy, scipy, etc.
"""

from brainsmith.registry import create_lazy_module

# ============================================================================
# Kernel Registry (Metadata Only - NO imports!)
# ============================================================================
# Enhanced format includes type-specific metadata for manifest caching (Issue #9)
# Supports both old format (string) and new format (dict) for backwards compat

COMPONENTS = {
    'kernels': {
        'LayerNorm': {
            'module': '.layernorm.layernorm',
            'infer_transform': 'brainsmith.kernels.layernorm.infer_layernorm:InferLayerNorm',
            'domain': 'finn.custom',
        },
        'Crop': {
            'module': '.crop.crop',
            'infer_transform': 'brainsmith.kernels.crop.infer_crop_from_gather:InferCropFromGather',
        },
        'Softmax': {
            'module': '.softmax',
            'infer_transform': 'brainsmith.kernels.softmax.infer_softmax:InferSoftmax',
        },
        'Shuffle': {
            'module': '.shuffle.shuffle',
            # No infer_transform for Shuffle
        },
    },
    'backends': {
        'LayerNorm_hls': {
            'module': '.layernorm.layernorm_hls',
            'target_kernel': 'brainsmith:LayerNorm',
            'language': 'hls',
        },
        'Crop_hls': {
            'module': '.crop.crop_hls',
            'target_kernel': 'brainsmith:Crop',
            'language': 'hls',
        },
        'Softmax_hls': {
            'module': '.softmax.hwsoftmax_hls',
            'target_kernel': 'brainsmith:Softmax',
            'language': 'hls',
        },
        'Shuffle_hls': {
            'module': '.shuffle.shuffle_hls',
            'target_kernel': 'brainsmith:Shuffle',
            'language': 'hls',
        },
    }
}

# ============================================================================
# Lazy Loading (PEP 562) - Unified Pattern
# ============================================================================

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
