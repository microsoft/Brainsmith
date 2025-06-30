############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Softmax kernel package
############################################################################

# Import the main HWSoftmax operator
from .hwsoftmax import HWSoftmax

# Import HLS backend if needed
from .hwsoftmax_hls import HWSoftmax_hls

# Import inference transform
from .infer_hwsoftmax import InferHWSoftmax

__all__ = ["HWSoftmax", "HWSoftmax_hls", "InferHWSoftmax"]