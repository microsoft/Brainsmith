"""Shared pytest fixtures for kernel tests (v2.0).

Provides default fixtures for shape and dtype parameterization.
Tests can override by defining local fixtures with same name.

Usage:
    # Use defaults (single INT8 input, shape (1, 64))
    class TestMyKernel(KernelTest):
        # Uses default fixtures automatically
        pass

    # Override in test file for custom parameterization
    @pytest.fixture(params=[
        {"input": DataType["INT8"], "param": DataType["INT8"]},
        {"input": DataType["INT16"], "param": DataType["INT16"]},
    ])
    def input_datatypes(request):
        return request.param

    @pytest.fixture(params=[
        {"input": (1, 64), "param": (64,)},
        {"input": (4, 128), "param": (128,)},
    ])
    def input_shapes(request):
        return request.param

    class TestMyKernel(KernelTest):
        # Now runs with custom parameterization!
        pass
"""

import pytest
from qonnx.core.datatype import DataType


# ============================================================================
# Default fixtures - can be overridden per-test or per-directory
# ============================================================================


@pytest.fixture(
    params=[
        {"input": DataType["INT8"]},
    ]
)
def input_datatypes(request):
    """Default input datatypes - single INT8 input.

    Override in test file for custom parameterization:

        @pytest.fixture(params=[
            {"input": DataType["INT8"], "param": DataType["INT8"]},
            {"input": DataType["INT16"], "param": DataType["INT16"]},
            {"input": DataType["FLOAT<5,10,15>"], "param": DataType["FLOAT<5,10,15>"]},
        ])
        def input_datatypes(request):
            return request.param

    This will run tests with 3 different dtype configurations automatically!

    Multi-input example:

        @pytest.fixture(params=[
            {"x": DataType["INT8"], "y": DataType["INT8"]},
            {"x": DataType["INT16"], "y": DataType["INT16"]},
        ])
        def input_datatypes(request):
            return request.param

    Returns:
        Dict mapping input names to QONNX DataTypes
    """
    return request.param


@pytest.fixture(
    params=[
        {"input": (1, 64)},
    ]
)
def input_shapes(request):
    """Default input shapes - single (1, 64) input.

    Override in test file for custom parameterization:

        @pytest.fixture(params=[
            {"input": (1, 64), "param": (64,)},
            {"input": (4, 128), "param": (128,)},
            {"input": (1, 8, 8, 32), "param": (32,)},
        ])
        def input_shapes(request):
            return request.param

    This will run tests with 3 different shape configurations automatically!

    Multi-input example:

        @pytest.fixture(params=[
            {"x": (1, 64), "y": (1, 64)},
            {"x": (4, 128), "y": (4, 128)},
        ])
        def input_shapes(request):
            return request.param

    Returns:
        Dict mapping input names to shape tuples
    """
    return request.param


# ============================================================================
# Common dtype configurations (for reuse)
# ============================================================================


# Common integer types
INT_DTYPES_SINGLE = [
    {"input": DataType["INT8"]},
    {"input": DataType["INT16"]},
    {"input": DataType["INT32"]},
]

INT_DTYPES_BINARY = [
    {"input": DataType["INT8"], "param": DataType["INT8"]},
    {"input": DataType["INT16"], "param": DataType["INT16"]},
]

# Common unsigned integer types
UINT_DTYPES_SINGLE = [
    {"input": DataType["UINT8"]},
    {"input": DataType["UINT16"]},
    {"input": DataType["UINT32"]},
]

# FP8 types
FP8_DTYPES_SINGLE = [
    {"input": DataType["FLOAT<5,2,15>"]},  # E5M2 (common FP8)
    {"input": DataType["FLOAT<4,3,7>"]},  # E4M3 (alternative FP8)
]

# Mixed-type combinations
MIXED_DTYPES = [
    {"input": DataType["INT8"], "param": DataType["FLOAT32"]},
    {"input": DataType["FLOAT32"], "param": DataType["INT8"]},
]

# ============================================================================
# Common shape configurations (for reuse)
# ============================================================================

# 2D shapes (batch, features)
SHAPES_2D_SINGLE = [
    {"input": (1, 64)},
    {"input": (4, 128)},
    {"input": (8, 256)},
]

SHAPES_2D_BINARY = [
    {"input": (1, 64), "param": (64,)},
    {"input": (4, 128), "param": (128,)},
]

# 4D shapes (batch, height, width, channels) - NHWC layout
SHAPES_4D_NHWC = [
    {"input": (1, 8, 8, 32)},
    {"input": (1, 16, 16, 64)},
    {"input": (4, 32, 32, 128)},
]

# 4D shapes (batch, channels, height, width) - NCHW layout
SHAPES_4D_NCHW = [
    {"input": (1, 32, 8, 8)},
    {"input": (1, 64, 16, 16)},
    {"input": (4, 128, 32, 32)},
]
