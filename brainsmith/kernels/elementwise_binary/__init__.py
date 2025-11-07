# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
ElementwiseBinary Kernel

Polymorphic hardware kernel for elementwise binary operations (arithmetic,
logical, comparison, bitwise) with PE parallelism.
"""

from .elementwise_binary import ElementwiseBinaryOp
from .elementwise_binary_hls import ElementwiseBinaryOp_hls
from .operations import BinaryOperations

__all__ = ["ElementwiseBinaryOp", "ElementwiseBinaryOp_hls", "BinaryOperations"]
