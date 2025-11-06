# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# flake8: noqa
# Disable linting from here, as all imports will be flagged E402 and maybe F401

"""
Brainsmith RTL Kernel Imports

This is a TEMPORARY measure to ensure RTL variants are properly registered
in the kernels.rtl namepace until backend refactoring is complete.

Similar to how FINN imports its RTL variants in:
deps/finn/src/finn/custom_op/fpgadataflow/rtl/__init__.py
"""

# Import all RTL custom ops - they will be discovered automatically via namespace
# Note: Using absolute imports to ensure proper registration

# Import Brainsmith RTL kernels
from brainsmith.kernels.layernorm.layernorm_rtl import LayerNorm_rtl
