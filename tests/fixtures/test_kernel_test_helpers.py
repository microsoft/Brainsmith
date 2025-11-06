# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for kernel_test_helpers module.

Validates that the OnnxModelBuilder and convenience functions produce
correct ONNX models for kernel testing.
"""

import pytest
import numpy as np
from onnx import TensorProto

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from tests.fixtures.model_builders import (
    OnnxModelBuilder,
    make_binary_op_model,
    make_parametric_op_model,
    make_unary_op_model,
)


class TestOnnxModelBuilder:
    """Test suite for OnnxModelBuilder class."""

    def test_default_values(self):
        """Verify builder has sensible defaults."""
        model, node = OnnxModelBuilder().build()

        assert node.op_type == "Add"
        assert list(node.input) == ["in0", "in1"]
        assert list(node.output) == ["output"]

        # Check shapes
        assert model.get_tensor_shape("in0") == [1, 8, 8, 64]
        assert model.get_tensor_shape("in1") == [1, 8, 8, 64]
        assert model.get_tensor_shape("output") == [1, 8, 8, 64]

        # Check datatypes
        assert model.get_tensor_datatype("in0") == DataType["INT8"]
        assert model.get_tensor_datatype("in1") == DataType["INT8"]
        assert model.get_tensor_datatype("output") == DataType["INT8"]

    def test_custom_op_type(self):
        """Verify op_type can be customized."""
        model, node = OnnxModelBuilder().op_type("Mul").build()
        assert node.op_type == "Mul"

    def test_custom_inputs(self):
        """Verify inputs can be customized."""
        model, node = (OnnxModelBuilder()
            .inputs(["stream0", "stream1"])
            .build())

        assert list(node.input) == ["stream0", "stream1"]
        assert model.get_tensor_shape("stream0") == [1, 8, 8, 64]

    def test_custom_outputs(self):
        """Verify outputs can be customized."""
        model, node = (OnnxModelBuilder()
            .outputs(["result"])
            .build())

        assert list(node.output) == ["result"]
        assert model.get_tensor_shape("result") == [1, 8, 8, 64]

    def test_custom_shape(self):
        """Verify shape can be customized."""
        custom_shape = [1, 224, 224, 128]
        model, node = OnnxModelBuilder().shape(custom_shape).build()

        assert model.get_tensor_shape("in0") == custom_shape
        assert model.get_tensor_shape("in1") == custom_shape
        assert model.get_tensor_shape("output") == custom_shape

    def test_custom_datatype(self):
        """Verify datatype can be customized."""
        model, node = (OnnxModelBuilder()
            .datatype(DataType["UINT8"])
            .build())

        assert model.get_tensor_datatype("in0") == DataType["UINT8"]
        assert model.get_tensor_datatype("in1") == DataType["UINT8"]

    def test_static_input_default(self):
        """Verify static_input with default shape (per-channel)."""
        model, node = (OnnxModelBuilder()
            .inputs(["data", "bias"])
            .shape([1, 8, 8, 64])
            .static_input("bias")  # Should default to [64]
            .build())

        # bias should be an initializer
        bias_init = model.get_initializer("bias")
        assert bias_init is not None
        assert bias_init.shape == (64,)
        np.testing.assert_array_equal(bias_init, np.zeros(64))

        # data should not be an initializer
        assert model.get_initializer("data") is None

    def test_static_input_scalar(self):
        """Verify static_input with scalar shape."""
        model, node = (OnnxModelBuilder()
            .inputs(["data", "scalar"])
            .static_input("scalar", shape=1)
            .build())

        scalar_init = model.get_initializer("scalar")
        assert scalar_init is not None
        assert scalar_init.shape == (1,)

    def test_static_input_custom_shape(self):
        """Verify static_input with custom shape."""
        model, node = (OnnxModelBuilder()
            .inputs(["data", "weight"])
            .static_input("weight", shape=[64, 32])
            .build())

        weight_init = model.get_initializer("weight")
        assert weight_init is not None
        assert weight_init.shape == (64, 32)

    def test_static_input_custom_values(self):
        """Verify static_input with custom values."""
        custom_values = np.array([1.0, 2.0, 3.0])
        model, node = (OnnxModelBuilder()
            .inputs(["data", "param"])
            .static_input("param", values=custom_values)
            .build())

        param_init = model.get_initializer("param")
        assert param_init is not None
        np.testing.assert_array_equal(param_init, custom_values)

    def test_input_shape_override(self):
        """Verify input_shape overrides default shape for specific input."""
        model, node = (OnnxModelBuilder()
            .shape([1, 8, 8, 64])
            .input_shape("in1", [1, 8, 8, 32])  # Different channels
            .build())

        assert model.get_tensor_shape("in0") == [1, 8, 8, 64]
        assert model.get_tensor_shape("in1") == [1, 8, 8, 32]

    def test_domain_custom(self):
        """Verify domain can be set."""
        model, node = (OnnxModelBuilder()
            .domain("brainsmith.kernels")
            .build())

        assert node.domain == "brainsmith.kernels"

    def test_name_custom(self):
        """Verify node name can be set."""
        model, node = (OnnxModelBuilder()
            .name("test_node")
            .build())

        assert node.name == "test_node"

    def test_fluent_api_chaining(self):
        """Verify fluent API allows method chaining."""
        model, node = (OnnxModelBuilder()
            .op_type("Add")
            .inputs(["data", "bias"])
            .outputs(["result"])
            .shape([1, 224, 224, 64])
            .static_input("bias", shape=[64])
            .datatype(DataType["INT8"])
            .domain("test.domain")
            .name("test_add")
            .build())

        assert node.op_type == "Add"
        assert list(node.input) == ["data", "bias"]
        assert list(node.output) == ["result"]
        assert node.domain == "test.domain"
        assert node.name == "test_add"
        assert model.get_tensor_shape("data") == [1, 224, 224, 64]
        assert model.get_initializer("bias") is not None


class TestMakeBinaryOpModel:
    """Test suite for make_binary_op_model convenience function."""

    def test_default_add(self):
        """Verify default Add model."""
        model, node = make_binary_op_model("Add")

        assert node.op_type == "Add"
        assert list(node.input) == ["in0", "in1"]
        assert list(node.output) == ["output"]
        assert model.get_tensor_shape("in0") == [1, 8, 8, 64]
        assert model.get_tensor_datatype("in0") == DataType["INT8"]

    def test_custom_shape(self):
        """Verify custom shape."""
        shape = [1, 224, 224, 128]
        model, node = make_binary_op_model("Mul", shape=shape)

        assert node.op_type == "Mul"
        assert model.get_tensor_shape("in0") == shape
        assert model.get_tensor_shape("in1") == shape

    def test_custom_names(self):
        """Verify custom input/output names."""
        model, node = make_binary_op_model(
            "Add",
            input0="stream0",
            input1="stream1",
            output="result"
        )

        assert list(node.input) == ["stream0", "stream1"]
        assert list(node.output) == ["result"]

    def test_custom_datatype(self):
        """Verify custom datatype."""
        model, node = make_binary_op_model("Add", datatype=DataType["UINT8"])

        assert model.get_tensor_datatype("in0") == DataType["UINT8"]
        assert model.get_tensor_datatype("in1") == DataType["UINT8"]

    def test_no_static_inputs(self):
        """Verify no static inputs are created."""
        model, node = make_binary_op_model("Add")

        # Neither input should be an initializer
        assert model.get_initializer("in0") is None
        assert model.get_initializer("in1") is None


class TestMakeParametricOpModel:
    """Test suite for make_parametric_op_model convenience function."""

    def test_default_parametric(self):
        """Verify default parametric model."""
        model, node = make_parametric_op_model("Add")

        assert node.op_type == "Add"
        assert list(node.input) == ["data", "param"]
        assert model.get_initializer("param") is not None
        assert model.get_initializer("data") is None

    def test_custom_names(self):
        """Verify custom input names."""
        model, node = make_parametric_op_model(
            "Add",
            dynamic_input="activations",
            param_input="bias"
        )

        assert list(node.input) == ["activations", "bias"]
        assert model.get_initializer("bias") is not None
        assert model.get_initializer("activations") is None

    def test_per_channel_parameter(self):
        """Verify per-channel parameter (default param_shape)."""
        model, node = make_parametric_op_model(
            "Add",
            shape=[1, 8, 8, 64],
            param_input="bias"
        )

        bias_init = model.get_initializer("bias")
        assert bias_init.shape == (64,)  # Matches last dim of shape

    def test_scalar_parameter(self):
        """Verify scalar parameter."""
        model, node = make_parametric_op_model(
            "Add",
            param_input="scalar",
            param_shape=1
        )

        scalar_init = model.get_initializer("scalar")
        assert scalar_init.shape == (1,)

    def test_custom_param_shape(self):
        """Verify custom parameter shape."""
        model, node = make_parametric_op_model(
            "Add",
            param_input="weight",
            param_shape=[64, 32]
        )

        weight_init = model.get_initializer("weight")
        assert weight_init.shape == (64, 32)

    def test_input_order_dynamic_first(self):
        """Verify canonical input order (dynamic, static)."""
        model, node = make_parametric_op_model(
            "Add",
            dynamic_input="data",
            param_input="bias",
            input_order="dynamic_first"
        )

        assert list(node.input) == ["data", "bias"]

    def test_input_order_static_first(self):
        """Verify swapped input order (static, dynamic)."""
        model, node = make_parametric_op_model(
            "Add",
            dynamic_input="data",
            param_input="bias",
            input_order="static_first"
        )

        assert list(node.input) == ["bias", "data"]

    def test_comparison_operation(self):
        """Verify comparison operations work correctly."""
        model, node = make_parametric_op_model(
            "LessOrEqual",
            param_input="threshold"
        )

        assert node.op_type == "LessOrEqual"
        assert model.get_initializer("threshold") is not None


class TestMakeUnaryOpModel:
    """Test suite for make_unary_op_model convenience function."""

    def test_default_unary(self):
        """Verify default unary model."""
        model, node = make_unary_op_model("Softmax")

        assert node.op_type == "Softmax"
        assert list(node.input) == ["input"]
        assert list(node.output) == ["output"]
        assert model.get_tensor_shape("input") == [1, 8, 8, 64]

    def test_custom_names(self):
        """Verify custom input/output names."""
        model, node = make_unary_op_model(
            "LayerNorm",
            input_name="activations",
            output="normalized"
        )

        assert list(node.input) == ["activations"]
        assert list(node.output) == ["normalized"]

    def test_custom_shape(self):
        """Verify custom shape."""
        shape = [1, 12, 768]
        model, node = make_unary_op_model("Softmax", shape=shape)

        assert model.get_tensor_shape("input") == shape
        assert model.get_tensor_shape("output") == shape

    def test_custom_datatype(self):
        """Verify custom datatype."""
        model, node = make_unary_op_model("Softmax", datatype=DataType["UINT8"])

        assert model.get_tensor_datatype("input") == DataType["UINT8"]

    def test_no_static_inputs(self):
        """Verify no static inputs are created."""
        model, node = make_unary_op_model("Softmax")

        assert model.get_initializer("input") is None


class TestIntegrationExamples:
    """Integration tests showing real-world usage patterns."""

    def test_channelwise_add_pattern(self):
        """Test pattern from channelwise Add kernel."""
        model, node = make_parametric_op_model("Add", param_input="bias")

        # Verify structure matches what channelwise expects
        assert node.op_type == "Add"
        assert len(node.input) == 2
        assert model.get_initializer(node.input[1]) is not None  # One static input

    def test_addstreams_pattern(self):
        """Test pattern from AddStreams kernel."""
        model, node = make_binary_op_model("Add", shape=[1, 224, 224, 64])

        # Verify structure matches what AddStreams expects
        assert node.op_type == "Add"
        assert len(node.input) == 2
        assert model.get_initializer(node.input[0]) is None  # Both dynamic
        assert model.get_initializer(node.input[1]) is None

    def test_comparison_pattern(self):
        """Test pattern from channelwise comparison operations."""
        model, node = make_parametric_op_model(
            "LessOrEqual",
            param_input="threshold"
        )

        assert node.op_type == "LessOrEqual"
        assert model.get_initializer("threshold") is not None

    def test_swapped_input_pattern(self):
        """Test pattern for testing canonical input ordering."""
        model, node = make_parametric_op_model(
            "Add",
            param_input="bias",
            input_order="static_first"
        )

        # Inputs are reversed (static first, dynamic second)
        assert node.input[0] == "bias"
        assert node.input[1] == "data"
        assert model.get_initializer("bias") is not None
