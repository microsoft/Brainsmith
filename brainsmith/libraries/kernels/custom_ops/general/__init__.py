############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# Dictionary of CustomOp implementations
custom_op = dict()

# flake8: noqa
# Disable linting from here, as all import will be flagged E402 and maybe F401

# Import all CustomOps
from brainsmith.custom_op.general.norms import FuncLayerNorm

# Register in custom_op dictionary for use in QONNX
custom_op["FuncLayerNorm"] = FuncLayerNorm
