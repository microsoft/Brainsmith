############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Softmax kernel package - Lazy Loading for Performance
############################################################################

def __getattr__(name):
    """Lazy import for performance (avoid loading scipy at discovery time)."""
    if name == "HWSoftmax":
        from .hwsoftmax import HWSoftmax
        return HWSoftmax
    elif name == "Softmax":
        # Alias - decorator registers as 'Softmax'
        from .hwsoftmax import HWSoftmax
        return HWSoftmax
    elif name == "HWSoftmax_hls":
        from .hwsoftmax_hls import HWSoftmax_hls
        return HWSoftmax_hls
    elif name == "InferHWSoftmax":
        from .infer_hwsoftmax import InferHWSoftmax
        return InferHWSoftmax
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return ["HWSoftmax", "HWSoftmax_hls", "InferHWSoftmax", "Softmax"]

__all__ = ["HWSoftmax", "HWSoftmax_hls", "InferHWSoftmax", "Softmax"]