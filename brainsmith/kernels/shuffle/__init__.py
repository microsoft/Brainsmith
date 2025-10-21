# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Modern KernelOp-based implementation (primary)
from .shuffle import Shuffle
from .shuffle_hls import Shuffle_hls
from .infer_shuffle import InferShuffle

# Legacy HWCustomOp implementation (for backward compatibility)
from .legacy_shuffle import LegacyShuffle
from .legacy_shuffle_hls import LegacyShuffle_hls

__all__ = [
    "Shuffle",
    "Shuffle_hls",
    "InferShuffle",
    "LegacyShuffle",
    "LegacyShuffle_hls"
]