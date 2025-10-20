############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Tests for TransformationHelper utility class.

Tests transformation utilities used during ONNX → HW kernel conversion.
Validation methods have been removed - see constraints.py for validation.
"""

import pytest
import numpy as np
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import qonnx.core.data_layout as DataLayout

from brainsmith.dataflow.inference import TransformationHelper


# ===================================================================
# Helper Functions
# ===================================================================

def create_test_model():
    """Create simple test model with various tensor types.

    Creates a model with:
    - Integer tensors (INT8)
    - Float tensors (FLOAT32)
    - Static tensor (initializer)
    - Multiple nodes for graph topology testing
    """
    # Integer tensors
    int_in = helper.make_tensor_value_info("int_in", TensorProto.FLOAT, [1, 64])
    int_out = helper.make_tensor_value_info("int_out", TensorProto.FLOAT, [1, 64])

    # Float tensors
    float_in = helper.make_tensor_value_info("float_in", TensorProto.FLOAT, [1, 64])
    float_out = helper.make_tensor_value_info("float_out", TensorProto.FLOAT, [1, 64])

    # Static tensor (initializer)
    static_val = np.ones((64,), dtype=np.float32)
    static_init = helper.make_tensor("static", TensorProto.FLOAT, [64], static_val.flatten())

    # Intermediate tensor
    intermediate = helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, [1, 64])

    # Nodes: int_in + static -> intermediate -> int_out
    #        float_in -> float_out
    node1 = helper.make_node("Add", ["int_in", "static"], ["intermediate"], name="add_node")
    node2 = helper.make_node("Mul", ["intermediate"], ["int_out"], name="mul_node")
    node3 = helper.make_node("Identity", ["float_in"], ["float_out"], name="id_node")

    graph = helper.make_graph(
        [node1, node2, node3],
        "test",
        [int_in, float_in],
        [int_out, float_out],
        value_info=[intermediate],
        initializer=[static_init]
    )

    model = ModelWrapper(helper.make_model(graph))

    # Set datatypes
    model.set_tensor_datatype("int_in", DataType["INT8"])
    model.set_tensor_datatype("int_out", DataType["INT8"])
    model.set_tensor_datatype("intermediate", DataType["INT8"])
    model.set_tensor_datatype("float_in", DataType["FLOAT32"])
    model.set_tensor_datatype("float_out", DataType["FLOAT32"])
    model.set_tensor_datatype("static", DataType["INT8"])

    return model


# ===================================================================
# Domain Configurability Tests (Critical)
# ===================================================================

def test_default_brainsmith_domain():
    """Test TransformationHelper with default Brainsmith domain."""
    model = create_test_model()
    helper_obj = TransformationHelper(model)

    assert helper_obj.domain == "brainsmith.kernels"

    node = helper_obj.make_node(
        "TestOp",
        ["int_in"],
        ["int_out"],
        {"PE": 1}
    )

    assert node.domain == "brainsmith.kernels"
    assert node.op_type == "TestOp"


def test_custom_brainsmith_domain():
    """Test TransformationHelper with custom Brainsmith domain."""
    model = create_test_model()
    helper_obj = TransformationHelper(model, domain="brainsmith.kernels")

    assert helper_obj.domain == "brainsmith.kernels"

    node = helper_obj.make_node(
        "LayerNorm",
        ["int_in"],
        ["int_out"],
        {"epsilon": 1e-5}
    )

    assert node.domain == "brainsmith.kernels"
    assert node.op_type == "LayerNorm"


# ===================================================================
# Property Extraction Tests (Computed properties, not validation)
# ===================================================================

def test_get_spatial_dims():
    """Test spatial dimension extraction for 4D NHWC tensors."""
    inp_4d = helper.make_tensor_value_info("inp_4d", TensorProto.FLOAT, [1, 224, 224, 3])
    out_4d = helper.make_tensor_value_info("out_4d", TensorProto.FLOAT, [1, 112, 112, 64])
    node = helper.make_node("Conv", ["inp_4d"], ["out_4d"])
    graph = helper.make_graph([node], "test_4d", [inp_4d], [out_4d])
    model = ModelWrapper(helper.make_model(graph))

    helper_obj = TransformationHelper(model)

    h, w = helper_obj.get_spatial_dims("inp_4d")
    assert (h, w) == (224, 224)

    h, w = helper_obj.get_spatial_dims("out_4d")
    assert (h, w) == (112, 112)


def test_get_spatial_dims_error():
    """Test spatial dims raises error for non-4D tensors."""
    model = create_test_model()
    helper_obj = TransformationHelper(model)

    with pytest.raises(ValueError, match="not 4D"):
        helper_obj.get_spatial_dims("int_in")  # 2D tensor


# ===================================================================
# Graph Topology Tests (Pattern Matching)
# ===================================================================

def test_has_consumer_of_type():
    """Test consumer type matching."""
    model = create_test_model()
    helper_obj = TransformationHelper(model)

    # int_in is consumed by Add node
    assert helper_obj.has_consumer_of_type("int_in", "Add") == True
    assert helper_obj.has_consumer_of_type("int_in", "Mul") == False

    # intermediate is consumed by Mul node
    assert helper_obj.has_consumer_of_type("intermediate", "Mul") == True
    assert helper_obj.has_consumer_of_type("intermediate", "Add") == False

    # int_out is graph output (no consumer)
    assert helper_obj.has_consumer_of_type("int_out", "Add") == False


def test_has_multiple_consumers():
    """Test fanout detection."""
    # Create model with fanout
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
    out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 64])
    out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 64])

    node1 = helper.make_node("Identity", ["inp"], ["out1"])
    node2 = helper.make_node("Identity", ["inp"], ["out2"])  # Same input!

    graph = helper.make_graph([node1, node2], "test", [inp], [out1, out2])
    model = ModelWrapper(helper.make_model(graph))

    helper_obj = TransformationHelper(model)

    # inp has fanout of 2
    assert helper_obj.has_multiple_consumers("inp") == True

    # out1, out2 are graph outputs (no consumers)
    assert helper_obj.has_multiple_consumers("out1") == False
    assert helper_obj.has_multiple_consumers("out2") == False


def test_has_multiple_consumers_single():
    """Test single consumer case."""
    model = create_test_model()
    helper_obj = TransformationHelper(model)

    # int_in has single consumer (Add node)
    assert helper_obj.has_multiple_consumers("int_in") == False

    # intermediate has single consumer (Mul node)
    assert helper_obj.has_multiple_consumers("intermediate") == False


# ===================================================================
# Integration Tests
# ===================================================================

def test_existing_methods_unchanged():
    """Ensure existing TransformationHelper methods still work."""
    model = create_test_model()
    helper_obj = TransformationHelper(model)

    # Test get_num_channels (existing)
    assert helper_obj.get_num_channels("int_in") == 64

    # Test get_num_input_vectors (existing)
    assert helper_obj.get_num_input_vectors("int_in") == [1]


def test_helper_with_layernorm_domain():
    """Test TransformationHelper with LayerNorm use case."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 128, 768])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 128, 768])
    node = helper.make_node("FuncLayerNorm", ["inp"], ["out"])
    graph = helper.make_graph([node], "test", [inp], [out])
    model = ModelWrapper(helper.make_model(graph))

    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])

    # Use Brainsmith domain for LayerNorm
    helper_obj = TransformationHelper(model, domain="brainsmith.kernels")

    # Create LayerNorm node
    ln_node = helper_obj.make_node(
        "LayerNorm",
        ["inp"],
        ["out"],
        {
            "SIMD": 1,
            "epsilon": 1e-5,
            "input0Datatype": "INT8",
            "output0Datatype": "INT8"
        },
        name_prefix="LayerNorm_test"
    )

    assert ln_node.domain == "brainsmith.kernels"
    assert ln_node.op_type == "LayerNorm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
