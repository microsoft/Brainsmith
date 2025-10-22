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
        'Softmax': '.softmax',  # Package exports HWSoftmax (registered as Softmax)
        'Shuffle': '.shuffle.shuffle',
    }
}

# ============================================================================
# Lazy Loading (PEP 562) - Unified Pattern
# ============================================================================

__getattr__, __dir__ = create_lazy_module(COMPONENTS, __name__)
