# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
ElementwiseBinary Kernel

Polymorphic hardware kernel for elementwise binary operations (arithmetic,
logical, comparison, bitwise) with PE parallelism.
"""

from .elementwise_binary import ElementwiseBinaryOp, ELEMENTWISE_BINARY_SCHEMA
from .elementwise_binary_hls import ElementwiseBinaryOp_hls

__all__ = ["ElementwiseBinaryOp", "ELEMENTWISE_BINARY_SCHEMA", "ElementwiseBinaryOp_hls"]
