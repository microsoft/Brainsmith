"""Tests for inference helper functions."""

import pytest
import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model

from brainsmith.dataflow.inference_helpers import (
    find_static_dynamic_pair,
    find_dynamic_inputs,
    find_static_inputs,
    check_all_integer_types,
    check_shapes_equal,
    check_parameter_shape_matches_channels,
    expand_scalar_to_channels,
    lift_scalar_to_rank1,
)


class TestFindStaticDynamicPair:
    """Test find_static_dynamic_pair helper."""

    def make_model_with_inputs(self, dynamic_names, static_names):
        """Create model with specified dynamic and static inputs."""
        # Create graph inputs
        inputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 64])
            for name in dynamic_names + static_names
        ]

        # Create initializers for static inputs
        initializers = [
            helper.make_tensor(name, TensorProto.FLOAT, [64], np.zeros(64))
            for name in static_names
        ]

        # Create dummy output
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

        # Create identity node (doesn't matter for this test)
        # Use first available input (dynamic or static)
        first_input = dynamic_names[0] if dynamic_names else static_names[0]
        node = helper.make_node("Identity", [first_input], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=inputs,
            outputs=[output],
            initializer=initializers
        )

        model = qonnx_make_model(graph)
        return ModelWrapper(model)

    def test_correct_order(self):
        """Test with inputs already in correct order (dynamic, static)."""
        model = self.make_model_with_inputs(["data"], ["params"])

        result = find_static_dynamic_pair(["data", "params"], model)

        assert result == ("data", "params")

    def test_swapped_order(self):
        """Test with inputs in wrong order (static, dynamic) - should swap."""
        model = self.make_model_with_inputs(["data"], ["params"])

        result = find_static_dynamic_pair(["params", "data"], model)

        assert result == ("data", "params")  # Swapped to correct order

    def test_both_dynamic(self):
        """Test with both inputs dynamic - should return None."""
        model = self.make_model_with_inputs(["data1", "data2"], [])

        result = find_static_dynamic_pair(["data1", "data2"], model)

        assert result is None

    def test_both_static(self):
        """Test with both inputs static - should return None."""
        model = self.make_model_with_inputs([], ["params1", "params2"])

        result = find_static_dynamic_pair(["params1", "params2"], model)

        assert result is None

    def test_wrong_input_count(self):
        """Test with != 2 inputs - should return None."""
        model = self.make_model_with_inputs(["data"], ["params"])

        result = find_static_dynamic_pair(["data"], model)
        assert result is None

        result = find_static_dynamic_pair(["data", "params", "extra"], model)
        assert result is None


class TestFindDynamicInputs:
    """Test find_dynamic_inputs helper."""

    def make_model_with_inputs(self, dynamic_names, static_names):
        """Create model with specified dynamic and static inputs."""
        inputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [64])
            for name in dynamic_names + static_names
        ]

        initializers = [
            helper.make_tensor(name, TensorProto.FLOAT, [64], np.zeros(64))
            for name in static_names
        ]

        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [64])
        first_input = dynamic_names[0] if dynamic_names else static_names[0]
        node = helper.make_node("Identity", [first_input], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=inputs,
            outputs=[output],
            initializer=initializers
        )

        model = qonnx_make_model(graph)
        return ModelWrapper(model)

    def test_mixed_inputs(self):
        """Test with mix of dynamic and static inputs."""
        model = self.make_model_with_inputs(["data1", "data2"], ["params1", "params2"])

        result = find_dynamic_inputs(["data1", "params1", "data2", "params2"], model)

        assert result == ["data1", "data2"]

    def test_all_dynamic(self):
        """Test with all dynamic inputs."""
        model = self.make_model_with_inputs(["data1", "data2"], [])

        result = find_dynamic_inputs(["data1", "data2"], model)

        assert result == ["data1", "data2"]

    def test_all_static(self):
        """Test with all static inputs."""
        model = self.make_model_with_inputs([], ["params1", "params2"])

        result = find_dynamic_inputs(["params1", "params2"], model)

        assert result == []

    def test_preserves_order(self):
        """Test that original order is preserved."""
        model = self.make_model_with_inputs(["a", "b", "c"], ["p1", "p2"])

        result = find_dynamic_inputs(["c", "p1", "a", "b", "p2"], model)

        assert result == ["c", "a", "b"]  # Same order as input


class TestFindStaticInputs:
    """Test find_static_inputs helper."""

    def make_model_with_inputs(self, dynamic_names, static_names):
        """Create model with specified dynamic and static inputs."""
        inputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [64])
            for name in dynamic_names + static_names
        ]

        initializers = [
            helper.make_tensor(name, TensorProto.FLOAT, [64], np.zeros(64))
            for name in static_names
        ]

        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [64])
        node = helper.make_node("Identity", [dynamic_names[0]] if dynamic_names else [static_names[0]], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=inputs,
            outputs=[output],
            initializer=initializers
        )

        model = qonnx_make_model(graph)
        return ModelWrapper(model)

    def test_mixed_inputs(self):
        """Test with mix of dynamic and static inputs."""
        model = self.make_model_with_inputs(["data1", "data2"], ["params1", "params2"])

        result = find_static_inputs(["data1", "params1", "data2", "params2"], model)

        assert result == ["params1", "params2"]


class TestCheckAllIntegerTypes:
    """Test check_all_integer_types helper."""

    def make_model_with_datatypes(self, tensor_datatypes):
        """Create model with specified tensor datatypes.

        Args:
            tensor_datatypes: Dict mapping tensor_name → DataType
        """
        inputs = []
        for name, dt in tensor_datatypes.items():
            inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [64]))

        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [64])
        node = helper.make_node("Identity", [list(tensor_datatypes.keys())[0]], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=inputs,
            outputs=[output]
        )

        model = qonnx_make_model(graph)
        model_w = ModelWrapper(model)

        # Set datatypes
        for name, dt in tensor_datatypes.items():
            model_w.set_tensor_datatype(name, dt)

        return model_w

    def test_all_integer(self):
        """Test with all integer types."""
        model = self.make_model_with_datatypes({
            "a": DataType["INT8"],
            "b": DataType["INT16"],
            "c": DataType["UINT8"],
        })

        result = check_all_integer_types(["a", "b", "c"], model)

        assert result is True

    def test_one_float(self):
        """Test with one float type - should return False."""
        model = self.make_model_with_datatypes({
            "a": DataType["INT8"],
            "b": DataType["FLOAT32"],
        })

        result = check_all_integer_types(["a", "b"], model)

        assert result is False

    def test_all_float(self):
        """Test with all float types - should return False."""
        model = self.make_model_with_datatypes({
            "a": DataType["FLOAT32"],
            "b": DataType["FLOAT32"],
        })

        result = check_all_integer_types(["a", "b"], model)

        assert result is False


class TestCheckShapesEqual:
    """Test check_shapes_equal helper."""

    def make_model_with_shapes(self, tensor_shapes):
        """Create model with specified tensor shapes.

        Args:
            tensor_shapes: Dict mapping tensor_name → shape tuple
        """
        inputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))
            for name, shape in tensor_shapes.items()
        ]

        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", [list(tensor_shapes.keys())[0]], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=inputs,
            outputs=[output]
        )

        model = qonnx_make_model(graph)
        return ModelWrapper(model)

    def test_equal_shapes(self):
        """Test with equal shapes."""
        model = self.make_model_with_shapes({
            "a": (1, 224, 224, 64),
            "b": (1, 224, 224, 64),
            "c": (1, 224, 224, 64),
        })

        result = check_shapes_equal(["a", "b", "c"], model)

        assert result is True

    def test_unequal_shapes(self):
        """Test with unequal shapes."""
        model = self.make_model_with_shapes({
            "a": (1, 224, 224, 64),
            "b": (1, 112, 112, 64),
        })

        result = check_shapes_equal(["a", "b"], model)

        assert result is False


class TestCheckParameterShapeMatchesChannels:
    """Test check_parameter_shape_matches_channels helper."""

    def make_model_with_shapes(self, data_shape, param_shape):
        """Create model with data and param tensors."""
        inputs = [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(data_shape)),
            helper.make_tensor_value_info("params", TensorProto.FLOAT, list(param_shape)),
        ]

        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(data_shape))
        node = helper.make_node("Identity", ["data"], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=inputs,
            outputs=[output]
        )

        model = qonnx_make_model(graph)
        return ModelWrapper(model)

    def test_per_channel_match(self):
        """Test with per-channel parameters (match last dim)."""
        model = self.make_model_with_shapes(
            data_shape=(1, 224, 224, 64),
            param_shape=(64,)
        )

        result = check_parameter_shape_matches_channels("data", "params", model)

        assert result is True

    def test_scalar_broadcast(self):
        """Test with scalar parameter (broadcast)."""
        model = self.make_model_with_shapes(
            data_shape=(1, 224, 224, 64),
            param_shape=(1,)
        )

        result = check_parameter_shape_matches_channels("data", "params", model)

        assert result is True

    def test_mismatch(self):
        """Test with mismatched parameter count."""
        model = self.make_model_with_shapes(
            data_shape=(1, 224, 224, 64),
            param_shape=(32,)  # Wrong count
        )

        result = check_parameter_shape_matches_channels("data", "params", model)

        assert result is False


class TestExpandScalarToChannels:
    """Test expand_scalar_to_channels helper."""

    def make_model_with_scalar(self, scalar_value, dtype=DataType["INT8"]):
        """Create model with scalar initializer."""
        scalar_tensor = helper.make_tensor(
            "scalar",
            TensorProto.FLOAT,  # Use FLOAT for simplicity, we'll set datatype later
            [1],
            [scalar_value]
        )

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Identity", ["input"], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=[inp],
            outputs=[output],
            initializer=[scalar_tensor]
        )

        model = qonnx_make_model(graph)
        model_w = ModelWrapper(model)
        model_w.set_tensor_datatype("scalar", dtype)
        return model_w

    def test_expand_scalar(self):
        """Test expanding scalar to per-channel."""
        model = self.make_model_with_scalar(scalar_value=5)

        expanded_name = expand_scalar_to_channels("scalar", num_channels=64, model=model)

        # Check new tensor was created
        assert expanded_name == "scalar_expanded_64ch"

        # Check it was added to initializers
        init_names = [x.name for x in model.graph.initializer]
        assert expanded_name in init_names

        # Check values
        expanded_init = model.get_initializer(expanded_name)
        assert expanded_init.shape == (64,)
        assert np.all(expanded_init == 5)

    def test_expand_with_different_dtype(self):
        """Test expanding with INT16 datatype."""
        model = self.make_model_with_scalar(scalar_value=100, dtype=DataType["INT16"])

        expanded_name = expand_scalar_to_channels("scalar", num_channels=32, model=model)

        expanded_init = model.get_initializer(expanded_name)
        assert expanded_init.shape == (32,)
        assert np.all(expanded_init == 100)
        # Note: The tensor proto type is preserved, not the numpy dtype
        # (we create with TensorProto.FLOAT in test, so it's float32)

    def test_expand_non_initializer_fails(self):
        """Test that expanding non-initializer raises error."""
        # Create model without initializer for "not_init"
        inp = helper.make_tensor_value_info("not_init", TensorProto.FLOAT, [1])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", ["not_init"], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=[inp],
            outputs=[output]
        )

        model = qonnx_make_model(graph)
        model_w = ModelWrapper(model)

        with pytest.raises(ValueError, match="not an initializer"):
            expand_scalar_to_channels("not_init", num_channels=64, model=model_w)


class TestLiftScalarToRank1:
    """Test lift_scalar_to_rank1 helper."""

    def make_model_with_scalar(self, scalar_value=0.5):
        """Create model with a scalar initializer."""
        # Create scalar initializer (rank 0)
        scalar_init = helper.make_tensor(
            "scalar",
            TensorProto.FLOAT,
            [],  # Empty shape = scalar
            [scalar_value]
        )

        # Create dummy node to use the scalar
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Add", ["input", "scalar"], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=[inp],
            outputs=[output],
            initializer=[scalar_init]
        )

        model = qonnx_make_model(graph)
        model_w = ModelWrapper(model)
        return model_w

    def make_model_with_rank1(self, values):
        """Create model with a rank-1 tensor."""
        # Create rank-1 initializer
        tensor_init = helper.make_tensor(
            "tensor",
            TensorProto.FLOAT,
            [len(values)],
            values
        )

        # Create dummy node
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, len(values)])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, len(values)])
        node = helper.make_node("Add", ["input", "tensor"], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=[inp],
            outputs=[output],
            initializer=[tensor_init]
        )

        model = qonnx_make_model(graph)
        model_w = ModelWrapper(model)
        return model_w

    def test_lift_scalar_with_initializer(self):
        """Test lifting scalar tensor with initializer."""
        model = self.make_model_with_scalar(scalar_value=0.5)

        # Before lift
        assert model.get_tensor_shape("scalar") == []
        init_before = model.get_initializer("scalar")
        assert init_before.shape == ()
        assert float(init_before) == 0.5

        # Lift the scalar
        lifted = lift_scalar_to_rank1("scalar", model)

        # After lift
        assert lifted == True
        assert model.get_tensor_shape("scalar") == [1]
        init_after = model.get_initializer("scalar")
        assert init_after.shape == (1,)
        assert float(init_after[0]) == 0.5

    def test_lift_scalar_without_initializer(self):
        """Test lifting scalar tensor without initializer (shape only)."""
        # Create model with scalar shape but no initializer
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        # Note: "scalar_shape" is just a shape, not an initializer
        scalar_shape = helper.make_tensor_value_info("scalar_shape", TensorProto.FLOAT, [])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Add", ["input", "scalar_shape"], ["output"])

        graph = helper.make_graph(
            nodes=[node],
            name="test",
            inputs=[inp, scalar_shape],
            outputs=[output]
        )

        model = qonnx_make_model(graph)
        model_w = ModelWrapper(model)

        # Before lift
        assert model_w.get_tensor_shape("scalar_shape") == []
        assert model_w.get_initializer("scalar_shape") is None

        # Lift the scalar
        lifted = lift_scalar_to_rank1("scalar_shape", model_w)

        # After lift
        assert lifted == True
        assert model_w.get_tensor_shape("scalar_shape") == [1]
        assert model_w.get_initializer("scalar_shape") is None  # Still no initializer

    def test_lift_already_rank1(self):
        """Test that lifting rank-1 tensor is a no-op."""
        model = self.make_model_with_rank1([0.5])

        # Already rank 1
        assert model.get_tensor_shape("tensor") == [1]

        # Lift should return False (no-op)
        lifted = lift_scalar_to_rank1("tensor", model)

        assert lifted == False
        assert model.get_tensor_shape("tensor") == [1]  # Unchanged

    def test_lift_higher_rank(self):
        """Test that lifting higher rank tensor is a no-op."""
        model = self.make_model_with_rank1([1.0, 2.0, 3.0, 4.0])

        # Rank 1 with length > 1
        assert model.get_tensor_shape("tensor") == [4]

        # Lift should return False (no-op)
        lifted = lift_scalar_to_rank1("tensor", model)

        assert lifted == False
        assert model.get_tensor_shape("tensor") == [4]  # Unchanged

    def test_lift_preserves_onnx_semantics(self):
        """Verify that scalar [] and rank-1 [1] broadcast identically."""
        # Test using numpy broadcasting (matches ONNX semantics)
        lhs_shape = (1, 4, 32, 32)

        # Scalar broadcasts
        result_scalar = np.broadcast_shapes(lhs_shape, ())
        # Rank-1 broadcasts
        result_rank1 = np.broadcast_shapes(lhs_shape, (1,))

        # They should be identical
        assert result_scalar == result_rank1 == lhs_shape

    def test_lift_multiple_scalars(self):
        """Test lifting multiple scalar inputs in sequence."""
        # Create model with two scalar initializers
        scalar1 = helper.make_tensor("s1", TensorProto.FLOAT, [], [1.0])
        scalar2 = helper.make_tensor("s2", TensorProto.FLOAT, [], [2.0])

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node1 = helper.make_node("Add", ["input", "s1"], ["tmp"])
        node2 = helper.make_node("Add", ["tmp", "s2"], ["output"])

        graph = helper.make_graph(
            nodes=[node1, node2],
            name="test",
            inputs=[inp],
            outputs=[output],
            initializer=[scalar1, scalar2]
        )

        model = qonnx_make_model(graph)
        model_w = ModelWrapper(model)

        # Both are scalars initially
        assert model_w.get_tensor_shape("s1") == []
        assert model_w.get_tensor_shape("s2") == []

        # Lift both
        lifted1 = lift_scalar_to_rank1("s1", model_w)
        lifted2 = lift_scalar_to_rank1("s2", model_w)

        # Both should be lifted
        assert lifted1 == True
        assert lifted2 == True
        assert model_w.get_tensor_shape("s1") == [1]
        assert model_w.get_tensor_shape("s2") == [1]
