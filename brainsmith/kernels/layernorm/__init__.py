# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# LayerNorm implementations using KernelOp and Dataflow Modeling
from .layernorm import LayerNorm
from .layernorm_hls import LayerNorm_hls

# Friendly alias for external use
LayerNormHLS = LayerNorm_hls

__all__ = [
    "LayerNorm",
    "LayerNorm_hls",  # Actual class name (for QONNX/FINN discovery)
    "LayerNormHLS",   # Friendly alias
]