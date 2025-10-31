# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper functions for MVAU weight and threshold tensor formatting.

These functions convert weight and threshold matrices from ONNX format
to hardware-compatible format for HLS synthesis.
"""

import numpy as np
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions


def format_weight_tensor_for_hw(
    weight_matrix: np.ndarray,
    pe: int,
    simd: int,
    wmem: int,
    is_bipolar: bool = False
) -> np.ndarray:
    """Convert weight matrix to HW-compatible format.

    HW-Specific Logic:
    1. Transpose (ONNX uses (MW, MH), HLS uses (MH, MW))
    2. Convert bipolar {-1,+1} to binary {0,1} if needed
    3. Interleave rows between PEs
    4. Reshape to (1, PE, WMEM, SIMD)
    5. Reverse SIMD dimension (HLS convention)

    Args:
        weight_matrix: Original weight matrix (MW, MH)
        pe: Processing elements (output parallelism)
        simd: Input parallelism
        wmem: Weight memory depth
        is_bipolar: Whether weights are bipolar {-1, +1}

    Returns:
        Formatted weight tensor (1, PE, WMEM, SIMD)
    """
    mw, mh = weight_matrix.shape

    # Transpose: ONNX (MW, MH) -> HLS (MH, MW)
    ret = weight_matrix.T

    # Convert bipolar {-1,+1} to binary {0,1} if needed
    if is_bipolar:
        ret = (ret + 1) / 2

    # Interleave rows between PEs
    ret = interleave_matrix_outer_dim_from_partitions(ret, pe)

    # Reshape to (1, PE, WMEM, SIMD)
    ret = ret.reshape(1, pe, wmem, simd)

    # Reverse SIMD dimension (HLS convention)
    ret = np.flip(ret, axis=-1)

    return ret


def format_threshold_tensor_for_hw(
    threshold_matrix: np.ndarray,
    pe: int,
    tmem: int,
    mh: int
) -> np.ndarray:
    """Convert threshold matrix to HW-compatible format.

    HW-Specific Logic:
    1. Ensure positive thresholds for bipolar×bipolar (checked by caller)
    2. Broadcast if single channel
    3. Interleave rows between PEs
    4. Reshape to (1, PE, TMEM, n_thres_steps)

    Args:
        threshold_matrix: Original threshold matrix (MH or 1, n_thres_steps)
        pe: Processing elements
        tmem: Threshold memory depth
        mh: Matrix height (output features)

    Returns:
        Formatted threshold tensor (1, PE, TMEM, n_thres_steps)
    """
    assert threshold_matrix.ndim == 2, \
        f"Threshold matrix must be 2D, got {threshold_matrix.ndim}D"

    n_thres_steps = threshold_matrix.shape[1]
    ret = threshold_matrix

    # Broadcast if single channel
    if ret.shape[0] == 1:
        ret = np.tile(ret, (mh, 1))

    assert ret.shape[0] == mh, \
        f"Threshold channels {ret.shape[0]} != MH {mh}"

    # Interleave rows between PEs
    ret = interleave_matrix_outer_dim_from_partitions(ret, pe)

    return ret.reshape(1, pe, tmem, n_thres_steps)
