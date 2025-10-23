############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Softmax kernel package
#
# Components eagerly imported to ensure @kernel, @backend decorators fire.
# Lazy loading is at the top level (brainsmith.kernels), not within packages.
############################################################################

from .hwsoftmax import Softmax
from .hwsoftmax_hls import Softmax_hls
from .infer_hwsoftmax import InferSoftmax

__all__ = ["Softmax", "Softmax_hls", "InferSoftmax"]