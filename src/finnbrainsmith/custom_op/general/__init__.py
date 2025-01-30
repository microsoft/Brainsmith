############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

# The base class of all generic custom operations before specializing to either
# HLS or RTL backend
from qonnx.custom_op.base import CustomOp

# Dictionary of HWCustomOp implementations
custom_op = dict()


# Registers a class into the custom_op dictionary
# Note: This must be defined first, before importing any custom op
# implementation to avoid "importing partially initialized module" issues.
def register_custom_op(cls):
    # The class must actually implement HWCustomOp
    assert issubclass(cls, CustomOp), f"{cls} must subclass {CustomOp}"
    # Insert the class into the custom_op dictionary by its name
    custom_op[cls.__name__] = cls  # noqa: Some weird type annotation issue?
    # Pass through the class unmodified
    return cls


# flake8: noqa
# Disable linting from here, as all import will be flagged E402 and maybe F401


# Import the submodule containing specializations of ElementwiseBinaryOperation
# Note: This will automatically register all decorated classes into this domain

from finnbrainsmith.custom_op.general.norms import FuncLayerNorm

# make sure new HLSCustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
custom_op["FuncLayerNorm"] = FuncLayerNorm
