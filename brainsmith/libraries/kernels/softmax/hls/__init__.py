############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Mapping module for FINN dynamic import compatibility
############################################################################

# Import the HLS implementation from parent directory
from ..hwsoftmax_hls import HWSoftmax_hls

# Export for FINN's dynamic import mechanism
custom_op = {"HWSoftmax_hls": HWSoftmax_hls}