"""ONNX model fixtures for DSE testing.

Ported from OLD_FOR_REFERENCE_ONLY/fixtures/model_utils.py
Real ONNX models - no mocks.
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
from pathlib import Path
from typing import List
import pytest


def create_simple_model(
    input_shape: List[int] = [1, 3, 32, 32],
    output_shape: List[int] = [1, 10],
    op_type: str = "MatMul",
    model_name: str = "simple_test_model"
) -> onnx.ModelProto:
    """Create a simple single-operation ONNX model.

    Args:
        input_shape: Shape of input tensor
        output_shape: Shape of output tensor
        op_type: Type of operation to use (MatMul, Add, or generic)
        model_name: Name of the model

    Returns:
        ONNX model proto
    """
    # Create input/output tensors
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )

    # Create weight initializer for MatMul
    if op_type == "MatMul":
        weight_shape = [np.prod(input_shape[1:]), np.prod(output_shape[1:])]
        weight_tensor = helper.make_tensor(
            "weight",
            TensorProto.FLOAT,
            weight_shape,
            np.random.randn(*weight_shape).astype(np.float32).flatten()
        )

        # Flatten input first
        flatten_node = helper.make_node(
            "Flatten",
            inputs=["input"],
            outputs=["input_flat"],
            axis=1
        )

        # MatMul operation
        matmul_node = helper.make_node(
            "MatMul",
            inputs=["input_flat", "weight"],
            outputs=["output"]
        )

        nodes = [flatten_node, matmul_node]
        initializers = [weight_tensor]

    elif op_type == "Add":
        # Simple Add with constant
        const_tensor = helper.make_tensor(
            "constant",
            TensorProto.FLOAT,
            input_shape,
            np.ones(input_shape).astype(np.float32).flatten()
        )

        add_node = helper.make_node(
            "Add",
            inputs=["input", "constant"],
            outputs=["output"]
        )

        nodes = [add_node]
        initializers = [const_tensor]

    else:
        # Generic single operation
        node = helper.make_node(
            op_type,
            inputs=["input"],
            outputs=["output"]
        )
        nodes = [node]
        initializers = []

    # Create graph
    graph_proto = helper.make_graph(
        nodes,
        model_name,
        [input_tensor],
        [output_tensor],
        initializer=initializers
    )

    # Create model
    model_proto = helper.make_model(
        graph_proto,
        producer_name="brainsmith_test"
    )

    # Set opset version
    model_proto.opset_import[0].version = 11

    return model_proto


@pytest.fixture
def simple_onnx_model(tmp_path) -> Path:
    """Create a simple ONNX model for testing.

    Returns:
        Path to saved ONNX model file
    """
    model = create_simple_model()
    model_path = tmp_path / "simple_model.onnx"
    onnx.save(model, str(model_path))

    return model_path


@pytest.fixture
def quantized_onnx_model(tmp_path) -> Path:
    """FINN-compatible quantized ONNX model.

    Creates a simple quantized neural network compatible with FINN/QONNX:
    - Architecture: Input (1,16) → Quant → MatMul → Quant → Output (1,10)
    - 4-bit integer quantization (INT4)
    - Uses QONNX custom quantization ops
    - Compatible with FINN streamlining transformations

    This is the minimal quantized model that can go through FINN pipelines.
    Fast to generate (<1 second) for integration tests.

    Returns:
        Path to saved quantized ONNX model file
    """
    import numpy as np
    import onnx.parser as oprs
    from qonnx.core.modelwrapper import ModelWrapper

    # Model dimensions
    input_shape = [1, 16]
    output_shape = [1, 10]

    # ONNX text format model with QONNX Quant ops
    model_text = f"""
    <
        ir_version: 7,
        opset_import: ["" : 9, "qonnx.custom_op.general" : 1]
    >
    quant_matmul (float{input_shape} inp) => (float{output_shape} out)
    <
        float[1] inp_scale,
        float[1] inp_zeropt,
        float inp_bitwidth,
        float[16, 10] weights,
        float[1] out_scale,
        float[1] out_zeropt,
        float out_bitwidth
    >
    {{
        # Input quantization
        inp_q = qonnx.custom_op.general.Quant<
            signed=1,
            narrow=1,
            rounding_mode="ROUND"
        >(inp, inp_scale, inp_zeropt, inp_bitwidth)

        # Matrix multiplication
        matmul_out = MatMul(inp_q, weights)

        # Output quantization
        out = qonnx.custom_op.general.Quant<
            signed=1,
            narrow=1,
            rounding_mode="ROUND"
        >(matmul_out, out_scale, out_zeropt, out_bitwidth)
    }}
    """

    # Parse model
    model = oprs.parse_model(model_text)
    model = ModelWrapper(model)

    # Set quantization parameters (4-bit)
    np.random.seed(42)  # Deterministic

    # Input quantization
    model.set_initializer("inp_scale", np.array([0.1], dtype=np.float32))
    model.set_initializer("inp_zeropt", np.array([0.0], dtype=np.float32))
    model.set_initializer("inp_bitwidth", np.array(4.0, dtype=np.float32))

    # Weights (small random values for fast test)
    model.set_initializer("weights", np.random.randn(16, 10).astype(np.float32) * 0.1)

    # Output quantization
    model.set_initializer("out_scale", np.array([0.1], dtype=np.float32))
    model.set_initializer("out_zeropt", np.array([0.0], dtype=np.float32))
    model.set_initializer("out_bitwidth", np.array(4.0, dtype=np.float32))

    # Save model
    model_path = tmp_path / "quantized_model.onnx"
    model.save(str(model_path))

    return model_path


@pytest.fixture(scope="session")
def brevitas_fc_model(tmp_path_factory):
    """Brevitas-based FC network compatible with FINN hardware conversion.

    Architecture:
    - Input: (1, 28*28) flattened grayscale image
    - FC1: 784 -> 64, 2-bit weights, 2-bit activations
    - FC2: 64 -> 10, 2-bit weights
    - Output: (1, 10) logits

    Compatible with FULL hardware conversion pipeline:
    - finn:streamline ✅
    - finn:convert_to_hw ✅
    - finn:generate_estimate_reports ✅

    Expected hardware layers after convert_to_hw:
    - Thresholding_rtl (input quantizer)
    - MVAU_hls x2 (matrix-vector activation units)
    - Thresholding_rtl (activation quantizers)

    Session-scoped: Built once per test session (~15 seconds).

    Returns:
        Path: Path to generated FINN-compatible ONNX model
    """
    import torch
    import torch.nn as nn
    from brevitas.nn import QuantLinear, QuantReLU
    from brevitas.core.quant import QuantType
    from brevitas.export import export_qonnx
    from qonnx.util.cleanup import cleanup as qonnx_cleanup
    from qonnx.core.modelwrapper import ModelWrapper
    from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
    from qonnx.transformation.infer_shapes import InferShapes
    from qonnx.transformation.fold_constants import FoldConstants
    from qonnx.transformation.general import (
        GiveUniqueNodeNames,
        GiveReadableTensorNames,
        RemoveStaticGraphInputs
    )
    from qonnx.transformation.infer_datatypes import InferDataTypes

    # Use session-scoped cache directory
    cache_dir = tmp_path_factory.mktemp("brevitas_models")
    model_path = cache_dir / "brevitas_fc.onnx"

    # Return cached model if exists
    if model_path.exists():
        return model_path

    # 1. Build Brevitas model
    class BrevitasFC(nn.Module):
        """Simple 2-layer FC network with Brevitas quantization."""

        def __init__(self):
            super().__init__()
            self.fc1 = QuantLinear(
                in_features=784,
                out_features=64,
                bias=True,
                weight_bit_width=2,
                weight_quant_type=QuantType.INT,
            )
            self.act1 = QuantReLU(
                bit_width=2,
                quant_type=QuantType.INT,
            )
            self.fc2 = QuantLinear(
                in_features=64,
                out_features=10,
                bias=True,
                weight_bit_width=2,
                weight_quant_type=QuantType.INT,
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            return x

    model = BrevitasFC()
    model.eval()

    # 2. Export to QONNX
    ishape = (1, 1, 28, 28)
    export_qonnx(
        model,
        torch.randn(ishape),
        str(model_path),
        opset_version=13
    )

    # 3. Cleanup ONNX
    qonnx_cleanup(str(model_path), out_file=str(model_path))

    # 4. Convert to FINN ONNX
    model_wrapper = ModelWrapper(str(model_path))
    model_wrapper = model_wrapper.transform(ConvertQONNXtoFINN())

    # 5. Prepare for hardware conversion
    model_wrapper = model_wrapper.transform(InferShapes())
    model_wrapper = model_wrapper.transform(FoldConstants())
    model_wrapper = model_wrapper.transform(GiveUniqueNodeNames())
    model_wrapper = model_wrapper.transform(GiveReadableTensorNames())
    model_wrapper = model_wrapper.transform(InferDataTypes())
    model_wrapper = model_wrapper.transform(RemoveStaticGraphInputs())

    model_wrapper.save(str(model_path))

    return model_path


# TODO (Phase 5 or later): Add BERT-like model for transformer tests
# @pytest.fixture
# def bert_tiny_model(tmp_path) -> Path:
#     """Tiny BERT-like model for realistic pipeline tests.
#
#     Complexity: 2 layers, 128 hidden dim, 2 attention heads (~50K params)
#     Generation time: ~10-30 seconds
#     Use case: End-to-end transformer acceleration tests
#
#     Deferred: Not needed for Phase 4. Implement when testing
#     transformer-specific FINN features (LayerNorm, MultiHeadAttention).
#     """
#     pass
