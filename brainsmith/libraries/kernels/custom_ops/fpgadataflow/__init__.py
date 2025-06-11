############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# Dictionary of HWCustomOp implementations
custom_op = dict()

# flake8: noqa
# Disable linting from here, as all import will be flagged E402 and maybe F401

# Import all HWCustomOps
from brainsmith.custom_op.fpgadataflow.layernorm import LayerNorm
from brainsmith.custom_op.fpgadataflow.hwsoftmax import HWSoftmax
from brainsmith.custom_op.fpgadataflow.shuffle import Shuffle
from brainsmith.custom_op.fpgadataflow.crop import Crop

# Register in custom_op dictionary for use in QONNX
custom_op["LayerNorm"] = LayerNorm
custom_op["HWSoftmax"] = HWSoftmax
custom_op["Shuffle"] = Shuffle
custom_op["Crop"] = Crop
