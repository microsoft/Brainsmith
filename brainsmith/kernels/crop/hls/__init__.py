############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Mapping module for FINN dynamic import compatibility
############################################################################

# Import the HLS implementation from parent directory
from ..crop_hls import Crop_hls

# Export for FINN's dynamic import mechanism
custom_op = {"Crop_hls": Crop_hls}