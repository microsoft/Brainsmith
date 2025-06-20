############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# LayerNorm kernel package
############################################################################

# Import the main LayerNorm operator
from .layernorm import LayerNorm

# Import HLS backend if needed
from .layernorm_hls import LayerNorm_hls

__all__ = ["LayerNorm", "LayerNorm_hls"]