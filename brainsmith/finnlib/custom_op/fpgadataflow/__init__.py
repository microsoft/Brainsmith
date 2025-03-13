############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

from brainsmith.finnlib.custom_op.fpgadataflow.layernorm import LayerNorm
from brainsmith.finnlib.custom_op.fpgadataflow.hwsoftmax import HWSoftmax
from brainsmith.finnlib.custom_op.fpgadataflow.shuffle import Shuffle
from brainsmith.finnlib.custom_op.fpgadataflow.crop import Crop

custom_op = dict()

custom_op["LayerNorm"] = LayerNorm
custom_op["HWSoftmax"] = HWSoftmax
custom_op["Shuffle"] = Shuffle
custom_op["Crop"] = Crop
