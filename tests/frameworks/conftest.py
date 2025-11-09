"""Shared pytest fixtures for kernel tests.

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
