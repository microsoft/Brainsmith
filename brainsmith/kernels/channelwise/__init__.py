# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Modern KernelOp-based ChannelwiseOp implementation
from .channelwise import ChannelwiseOp
from .channelwise_hls import ChannelwiseOp_hls

__all__ = [
    "ChannelwiseOp",
    "ChannelwiseOp_hls",
]
