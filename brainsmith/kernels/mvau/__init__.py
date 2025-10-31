# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MVAU (Matrix Vector Activation Unit) kernel.

This kernel implements matrix-vector multiplication with optional multi-threshold
activation for FPGA deployment.

Modes:
- noActivation=1: MV (matmul only)
- noActivation=0: MVTU (matmul + multi-threshold activation)

Memory Modes:
- internal_embedded: Weights in C++ header (small matrices)
- internal_decoupled: Weights in BRAM/URAM via memstream wrapper
- external: Weights streamed from external source

Migrated from FINN's MatrixVectorActivation kernel.
"""

from .mvau import MVAU

# HLS backend will be imported when implemented
try:
    from .mvau_hls import MVAU_hls
    __all__ = ["MVAU", "MVAU_hls"]
except ImportError:
    __all__ = ["MVAU"]
