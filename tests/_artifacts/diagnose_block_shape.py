#!/usr/bin/env python3
"""Diagnostic script to compare design_point block_shape between manual and auto pipelines."""

import sys
import numpy as np
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
import qonnx.core.data_layout as DataLayout
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from finn.transformation.fpgadataflow.convert_to_hw_layers import InferChannelwiseLinearLayer
from finn.util.basic import getHWCustomOp

from brainsmith.primitives.transforms.infer_kernel import InferKernel
from brainsmith.kernels.channelwise import ChannelwiseOp


def make_channelwise_model():
    """Create test model matching test_channelwise_backend.py."""
    batch = 1
    h = w = 8
    ch = 64
    shape = [batch, h, w, ch]  # NHWC format
    node_name = "Add_test"

    # Add datatypes
    idt = DataType["INT8"]
    pdt = DataType["INT8"]
    odt = DataType["INT9"]

    # Create input tensor info
    inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)

    # Generate parameter tensor (per-channel values)
    np.random.seed(42)
    param_data = gen_finn_dt_tensor(pdt, [ch])
    param_tensor = helper.make_tensor(
        "param", TensorProto.FLOAT, [ch], param_data.flatten().tolist()
    )

    # Create ONNX node
    node = helper.make_node(
        "Add", ["data", "param"], ["output"], name=node_name
    )

    # Build graph and model
    graph = helper.make_graph(
        nodes=[node],
        name="test_channelwise",
        inputs=[inp],
        outputs=[outp],
        initializer=[param_tensor],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="channelwise-test"))

    # Set datatypes
    model.set_tensor_datatype("data", idt)
    model.set_tensor_datatype("param", pdt)
    model.set_tensor_datatype("output", odt)

    # Set data layout
    model.set_tensor_layout("data", DataLayout.NHWC)
    model.set_tensor_layout("output", DataLayout.NHWC)

    return model, node_name


def run_manual_pipeline():
    """Run FINN manual pipeline."""
    print("=" * 80)
    print("MANUAL PIPELINE (FINN InferChannelwiseLinearLayer)")
    print("=" * 80)

    model, node_name = make_channelwise_model()

    # Standard preprocessing
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Apply manual transform
    model = model.transform(InferChannelwiseLinearLayer())

    # Find node
    hw_node = model.get_node_from_name(node_name)
    if hw_node is None:
        hw_node = model.graph.node[0]

    # Get op
    op = getHWCustomOp(hw_node, model)

    print(f"\nNode type: {hw_node.op_type}")
    print(f"Op class: {op.__class__.__name__}")

    # Check if it has design_point
    if hasattr(op, 'design_point'):
        dp = op.design_point
        print(f"\nHas design_point: YES")
        print(f"Input tensor_shape: {dp.inputs['input'].tensor_shape}")
        print(f"Input block_shape: {dp.inputs['input'].block_shape}")
        print(f"Input stream_shape: {dp.inputs['input'].stream_shape}")

        # Compute spatial_dim as the code does
        block_shape = dp.inputs["input"].block_shape
        if len(block_shape) == 4:
            spatial_dim = block_shape[1] * block_shape[2]
            print(f"\nSpatial dim (H*W): {spatial_dim}")
        elif len(block_shape) == 2:
            spatial_dim = 1
            print(f"\nSpatial dim (FC): {spatial_dim}")
        else:
            print(f"\nUnexpected block_shape length: {len(block_shape)}")
    else:
        print(f"\nHas design_point: NO (legacy FINN kernel)")

    return op, model


def run_auto_pipeline():
    """Run Brainsmith auto pipeline."""
    print("\n" + "=" * 80)
    print("AUTO PIPELINE (Brainsmith InferKernel(ChannelwiseOp))")
    print("=" * 80)

    model, node_name = make_channelwise_model()

    # Standard preprocessing
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Apply auto transform
    model = model.transform(InferKernel(ChannelwiseOp))

    # Find node
    hw_node = model.get_node_from_name(node_name)
    if hw_node is None:
        hw_node = model.graph.node[0]

    # Get op
    op = getHWCustomOp(hw_node, model)

    print(f"\nNode type: {hw_node.op_type}")
    print(f"Op class: {op.__class__.__name__}")

    # Check if it has design_point
    if hasattr(op, 'design_point'):
        dp = op.design_point
        print(f"\nHas design_point: YES")
        print(f"Input tensor_shape: {dp.inputs['input'].tensor_shape}")
        print(f"Input block_shape: {dp.inputs['input'].block_shape}")
        print(f"Input stream_shape: {dp.inputs['input'].stream_shape}")

        # Compute spatial_dim as the code does
        block_shape = dp.inputs["input"].block_shape
        if len(block_shape) == 4:
            spatial_dim = block_shape[1] * block_shape[2]
            print(f"\nSpatial dim (H*W): {spatial_dim}")
        elif len(block_shape) == 2:
            spatial_dim = 1
            print(f"\nSpatial dim (FC): {spatial_dim}")
        else:
            print(f"\nUnexpected block_shape length: {len(block_shape)}")
    else:
        print(f"\nHas design_point: NO")

    return op, model


if __name__ == "__main__":
    try:
        manual_op, manual_model = run_manual_pipeline()
        auto_op, auto_model = run_auto_pipeline()

        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)

        # Compare designs
        if hasattr(manual_op, 'design_point') and hasattr(auto_op, 'design_point'):
            manual_block = manual_op.design_point.inputs['input'].block_shape
            auto_block = auto_op.design_point.inputs['input'].block_shape

            if manual_block == auto_block:
                print(f"✅ Block shapes MATCH: {manual_block}")
            else:
                print(f"❌ Block shapes DIFFER:")
                print(f"  Manual: {manual_block}")
                print(f"  Auto:   {auto_block}")
        elif not hasattr(manual_op, 'design_point'):
            print("❌ Manual op has no design_point (legacy FINN)")
        else:
            print("❌ Auto op has no design_point")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
