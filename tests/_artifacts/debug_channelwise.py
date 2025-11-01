#!/usr/bin/env python3
"""Debug script to investigate ChannelwiseOp inference issues."""

import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
import qonnx.core.data_layout as DataLayout

from finn.transformation.fpgadataflow.convert_to_hw_layers import InferChannelwiseLinearLayer
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList


def test_add_inference():
    """Test Add operation inference."""
    print("\n" + "="*80)
    print("Testing Add Operation Inference")
    print("="*80)

    # Create Add node with static param
    batch, h, w, ch = 1, 8, 8, 64
    shape = [batch, h, w, ch]

    inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

    np.random.seed(42)
    param_data = gen_finn_dt_tensor(DataType["INT8"], [ch])
    param_tensor = helper.make_tensor(
        "param", TensorProto.FLOAT, [ch], param_data.flatten().tolist()
    )

    node = helper.make_node("Add", ["data", "param"], ["output"], name="Add_test")

    graph = helper.make_graph(
        nodes=[node],
        name="test",
        inputs=[inp],
        outputs=[outp],
        initializer=[param_tensor],
    )
    model = ModelWrapper(qonnx_make_model(graph))

    model.set_tensor_datatype("data", DataType["INT8"])
    model.set_tensor_datatype("param", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["INT9"])
    model.set_tensor_layout("data", DataLayout.NHWC)
    model.set_tensor_layout("output", DataLayout.NHWC)

    print(f"\nOriginal node: {model.graph.node[0].op_type}")

    # Test manual transform (FINN)
    print("\n--- FINN Manual Transform (InferChannelwiseLinearLayer) ---")
    model_manual = model.transform(InferChannelwiseLinearLayer())
    manual_node = model_manual.graph.node[0]
    print(f"Resulting node: op_type={manual_node.op_type}, domain={manual_node.domain}")

    # Test auto transform (Brainsmith)
    print("\n--- Brainsmith Auto Transform (InferKernelList) ---")
    model_auto = model.transform(InferKernelList())
    auto_node = model_auto.graph.node[0]
    print(f"Resulting node: op_type={auto_node.op_type}, domain={auto_node.domain}")


def test_lessorequal_inference():
    """Test LessOrEqual operation inference."""
    print("\n" + "="*80)
    print("Testing LessOrEqual Operation Inference")
    print("="*80)

    # Create LessOrEqual node with static param
    batch, h, w, ch = 1, 8, 8, 64
    shape = [batch, h, w, ch]

    inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

    np.random.seed(42)
    param_data = gen_finn_dt_tensor(DataType["INT8"], [ch])
    param_tensor = helper.make_tensor(
        "param", TensorProto.FLOAT, [ch], param_data.flatten().tolist()
    )

    node = helper.make_node("LessOrEqual", ["data", "param"], ["output"], name="LessOrEqual_test")

    graph = helper.make_graph(
        nodes=[node],
        name="test",
        inputs=[inp],
        outputs=[outp],
        initializer=[param_tensor],
    )
    model = ModelWrapper(qonnx_make_model(graph))

    model.set_tensor_datatype("data", DataType["INT8"])
    model.set_tensor_datatype("param", DataType["INT8"])
    model.set_tensor_datatype("output", DataType["BINARY"])
    model.set_tensor_layout("data", DataLayout.NHWC)
    model.set_tensor_layout("output", DataLayout.NHWC)

    print(f"\nOriginal node: {model.graph.node[0].op_type}")

    # Test manual transform (FINN)
    print("\n--- FINN Manual Transform (InferChannelwiseLinearLayer) ---")
    try:
        model_manual = model.transform(InferChannelwiseLinearLayer())
        manual_node = model_manual.graph.node[0]
        print(f"Resulting node: op_type={manual_node.op_type}, domain={manual_node.domain}")
        for attr in manual_node.attribute:
            print(f"  {attr.name} = {attr}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

    # Test auto transform (Brainsmith)
    print("\n--- Brainsmith Auto Transform (InferKernelList) ---")
    try:
        model_auto = model.transform(InferKernelList())
        auto_node = model_auto.graph.node[0]
        print(f"Resulting node: op_type={auto_node.op_type}, domain={auto_node.domain}")
        for attr in auto_node.attribute:
            print(f"  {attr.name} = {attr}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_add_inference()
    test_lessorequal_inference()
