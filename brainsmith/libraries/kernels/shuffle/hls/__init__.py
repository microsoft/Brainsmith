############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Mapping module for FINN dynamic import compatibility
############################################################################

# Import the HLS implementation from parent directory
from ..shuffle_hls import Shuffle_hls

# Export for FINN's dynamic import mechanism
# This allows FINN to find Shuffle_hls when it imports
# brainsmith.libraries.kernels.shuffle.hls
custom_op = {"Shuffle_hls": Shuffle_hls}