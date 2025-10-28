# Portions derived from FINN project
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Code generation utilities for HLS synthesis.

This module provides structured code generation tools for creating
high-quality HLS C++ code with automatic indentation management.
"""

from brainsmith.codegen.hls_builder import HLSCodeBuilder

__all__ = ["HLSCodeBuilder"]
