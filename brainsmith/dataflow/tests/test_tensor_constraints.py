"""Tests for TensorDimMatches and TensorSizeMatches constraints."""

import pytest
import numpy as np
from brainsmith.dataflow.constraints import TensorDimMatches, TensorSizeMatches
from brainsmith.dataflow.validation import DesignSpaceValidationContext, ShapeHierarchy
from brainsmith.dataflow.dse_models import InterfaceDesignSpace
from qonnx.core.datatype import DataType


class MockValidationContext:
    """Mock validation context for testing constraints."""

    def __init__(self, shapes, params=None):
        """Initialize with interface shapes and optional params.

        Args:
            shapes: Dict[str, tuple] - interface name to shape mapping
            params: Dict[str, int] - parameter name to value mapping
        """
        self.shapes = shapes
        self.params = params or {}

    def get_shape(self, interface: str, hierarchy: ShapeHierarchy):
        """Get shape for interface at hierarchy level."""
        if interface not in self.shapes:
            raise KeyError(f"Interface '{interface}' not found")
        # For testing, we only care about TENSOR hierarchy
        return self.shapes[interface]

    def get_param(self, name: str):
        """Get parameter value."""
        if name not in self.params:
            raise KeyError(f"Parameter '{name}' not found")
        return self.params[name]


class TestTensorDimMatches:
    """Test suite for TensorDimMatches constraint."""

    def test_int_literal_match(self):
        """Test matching against integer literal."""
        constraint = TensorDimMatches("input", -1, [64, 128])
        ctx = MockValidationContext({"input": (1, 224, 224, 64)})
        assert constraint.check(ctx) is None  # 64 matches

    def test_int_literal_no_match(self):
        """Test failing to match integer literal."""
        constraint = TensorDimMatches("input", -1, [32, 128])
        ctx = MockValidationContext({"input": (1, 224, 224, 64)})
        error = constraint.check(ctx)
        assert error is not None
        assert "64" in error
        assert "expected one of" in error

    def test_tuple_reference_match(self):
        """Test matching against other interface dimension."""
        constraint = TensorDimMatches("parameters", -1, [1, ("input", -1)])
        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (64,)
        })
        assert constraint.check(ctx) is None  # parameters[-1]=64 matches input[-1]=64

    def test_tuple_reference_scalar_match(self):
        """Test scalar broadcast case."""
        constraint = TensorDimMatches("parameters", -1, [1, ("input", -1)])
        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (1,)
        })
        assert constraint.check(ctx) is None  # parameters[-1]=1 matches allowed value 1

    def test_tuple_reference_no_match(self):
        """Test rejection of wrong dimension size."""
        constraint = TensorDimMatches("parameters", -1, [1, ("input", -1)])
        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (32,)  # Wrong size
        })
        error = constraint.check(ctx)
        assert error is not None
        assert "32" in error

    def test_negative_index(self):
        """Test negative indexing works correctly."""
        constraint = TensorDimMatches("input", -2, [224])
        ctx = MockValidationContext({"input": (1, 224, 224, 64)})
        assert constraint.check(ctx) is None

    def test_positive_index(self):
        """Test positive indexing works correctly."""
        constraint = TensorDimMatches("input", 0, [1, 2, 4])
        ctx = MockValidationContext({"input": (1, 224, 224, 64)})
        assert constraint.check(ctx) is None

    def test_param_reference(self):
        """Test parameter reference resolution."""
        constraint = TensorDimMatches("input", -1, ["CHANNELS"])
        ctx = MockValidationContext(
            {"input": (1, 224, 224, 64)},
            {"CHANNELS": 64}
        )
        assert constraint.check(ctx) is None

    def test_interface_not_found(self):
        """Test error when interface doesn't exist."""
        constraint = TensorDimMatches("missing", -1, [64])
        ctx = MockValidationContext({"input": (1, 224, 224, 64)})
        error = constraint.check(ctx)
        assert error is not None
        assert "not found" in error

    def test_dim_index_out_of_range(self):
        """Test error when dimension index is invalid."""
        constraint = TensorDimMatches("input", 10, [64])
        ctx = MockValidationContext({"input": (1, 224, 224, 64)})
        error = constraint.check(ctx)
        assert error is not None
        assert "out of range" in error

    def test_describe(self):
        """Test constraint description."""
        constraint = TensorDimMatches("input", -1, [64, 128])
        desc = constraint.describe()
        assert "input" in desc
        assert "[-1]" in desc or "tensor" in desc

    def test_evaluation_phase(self):
        """Test constraint evaluates in structural phase."""
        constraint = TensorDimMatches("input", -1, [64])
        assert constraint.evaluation_phase == 'structural'


class TestTensorSizeMatches:
    """Test suite for TensorSizeMatches constraint."""

    def test_int_literal_match(self):
        """Test matching against integer literal."""
        constraint = TensorSizeMatches("parameters", [1, 64])
        ctx = MockValidationContext({"parameters": (64,)})
        assert constraint.check(ctx) is None  # size=64 matches

    def test_int_literal_no_match(self):
        """Test failing to match integer literal."""
        constraint = TensorSizeMatches("parameters", [1, 32])
        ctx = MockValidationContext({"parameters": (64,)})
        error = constraint.check(ctx)
        assert error is not None
        assert "64" in error

    def test_scalar_match(self):
        """Test scalar size matching."""
        constraint = TensorSizeMatches("parameters", [1, 64])
        ctx = MockValidationContext({"parameters": (1,)})
        assert constraint.check(ctx) is None

    def test_multidim_total_size(self):
        """Test total element count for multi-dimensional tensors."""
        constraint = TensorSizeMatches("weights", [4096])
        ctx = MockValidationContext({"weights": (64, 64)})  # 64*64=4096
        assert constraint.check(ctx) is None

    def test_tuple_reference_match(self):
        """Test matching against other interface dimension."""
        constraint = TensorSizeMatches("parameters", [1, ("input", -1)])
        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (64,)
        })
        assert constraint.check(ctx) is None  # size=64 matches input[-1]=64

    def test_tuple_reference_scalar(self):
        """Test scalar broadcast case."""
        constraint = TensorSizeMatches("parameters", [1, ("input", -1)])
        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (1,)
        })
        assert constraint.check(ctx) is None

    def test_param_reference(self):
        """Test parameter reference resolution."""
        constraint = TensorSizeMatches("parameters", ["NUM_PARAMS"])
        ctx = MockValidationContext(
            {"parameters": (64,)},
            {"NUM_PARAMS": 64}
        )
        assert constraint.check(ctx) is None

    def test_interface_not_found(self):
        """Test error when interface doesn't exist."""
        constraint = TensorSizeMatches("missing", [64])
        ctx = MockValidationContext({"input": (64,)})
        error = constraint.check(ctx)
        assert error is not None
        assert "not found" in error

    def test_describe(self):
        """Test constraint description."""
        constraint = TensorSizeMatches("parameters", [1, 64])
        desc = constraint.describe()
        assert "parameters" in desc
        assert "np.prod" in desc or "tensor" in desc

    def test_evaluation_phase(self):
        """Test constraint evaluates in structural phase."""
        constraint = TensorSizeMatches("parameters", [64])
        assert constraint.evaluation_phase == 'structural'


class TestFINNCompatibility:
    """Test FINN-compatible broadcast validation pattern."""

    def test_channelwise_valid_scalar(self):
        """Test scalar parameter is valid."""
        size_constraint = TensorSizeMatches("parameters", [1, ("input", -1)])
        dim_constraint = TensorDimMatches("parameters", -1, [1, ("input", -1)])

        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (1,)  # Scalar broadcast
        })

        assert size_constraint.check(ctx) is None
        assert dim_constraint.check(ctx) is None

    def test_channelwise_valid_per_channel(self):
        """Test per-channel parameters are valid."""
        size_constraint = TensorSizeMatches("parameters", [1, ("input", -1)])
        dim_constraint = TensorDimMatches("parameters", -1, [1, ("input", -1)])

        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (64,)  # Per-channel
        })

        assert size_constraint.check(ctx) is None
        assert dim_constraint.check(ctx) is None

    def test_channelwise_invalid_multidim(self):
        """Test multi-dimensional parameters are rejected (FINN-compatible)."""
        size_constraint = TensorSizeMatches("parameters", [1, ("input", -1)])
        dim_constraint = TensorDimMatches("parameters", -1, [1, ("input", -1)])

        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (8, 8)  # Total size 64 but shape [8, 8]
        })

        # Size matches (8*8=64)
        assert size_constraint.check(ctx) is None

        # But dimension doesn't match (parameters[-1]=8 != 64)
        dim_error = dim_constraint.check(ctx)
        assert dim_error is not None
        assert "8" in dim_error

    def test_channelwise_invalid_wrong_size(self):
        """Test wrong total size is rejected."""
        size_constraint = TensorSizeMatches("parameters", [1, ("input", -1)])
        dim_constraint = TensorDimMatches("parameters", -1, [1, ("input", -1)])

        ctx = MockValidationContext({
            "input": (1, 224, 224, 64),
            "parameters": (32,)  # Wrong size
        })

        # Size doesn't match
        size_error = size_constraint.check(ctx)
        assert size_error is not None
        assert "32" in size_error

        # Dimension also doesn't match
        dim_error = dim_constraint.check(ctx)
        assert dim_error is not None
        assert "32" in dim_error
