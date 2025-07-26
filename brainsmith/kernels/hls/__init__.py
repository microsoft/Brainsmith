# Copyright (c) 2024 BrainSmith Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# flake8: noqa
# Disable linting from here, as all imports will be flagged E402 and maybe F401

"""
BrainSmith HLS Kernel Imports

This is a TEMPORARY measure to ensure HLS variants are properly registered
in the FINN namespace until backend refactoring is complete.

Similar to how FINN imports its HLS variants in:
deps/finn/src/finn/custom_op/fpgadataflow/hls/__init__.py
"""

# Import all HLS custom ops - they will be discovered automatically via namespace
# Note: Using absolute imports to ensure proper registration

# Import BrainSmith HLS kernels
from brainsmith.kernels.crop.crop_hls import Crop_hls
from brainsmith.kernels.layernorm.layernorm_hls import LayerNorm_hls
from brainsmith.kernels.shuffle.shuffle_hls import Shuffle_hls
from brainsmith.kernels.softmax.hwsoftmax_hls import HWSoftmax_hls

# Also ensure they're available in finn.custom_op.fpgadataflow namespace
# This is needed for FINN's transformations to find them
import finn.custom_op.fpgadataflow as finn_base
import finn.custom_op.fpgadataflow.hls as finn_hls

# Import base classes too
from brainsmith.kernels.crop.crop import Crop
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.kernels.shuffle.shuffle import Shuffle
from brainsmith.kernels.softmax.hwsoftmax import HWSoftmax

# Register BrainSmith kernels in FINN's base namespace
# TEMPORARY: This ensures FINN can find our custom implementations
finn_base.Crop = Crop
finn_base.LayerNorm = LayerNorm
finn_base.Shuffle = Shuffle
finn_base.HWSoftmax = HWSoftmax

# Register BrainSmith HLS kernels in FINN's HLS namespace
# TEMPORARY: This ensures FINN can find our custom HLS implementations
finn_hls.Crop_hls = Crop_hls
finn_hls.LayerNorm_hls = LayerNorm_hls
finn_hls.Shuffle_hls = Shuffle_hls
finn_hls.HWSoftmax_hls = HWSoftmax_hls

__all__ = [
    "Crop_hls",
    "LayerNorm_hls",
    "Shuffle_hls",
    "HWSoftmax_hls",
]