# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Focused test for FINN elementwise operation registration.

Tests the complete workflow after registering all 16 elementwise operations:
1. Registry discovery: All kernels and backends are registered
2. Inference: ONNX ops → InferElementwiseBinaryOperation → ElementwiseAdd/etc
3. Specialization: ElementwiseAdd → SpecializeKernels → ElementwiseAdd_hls
4. Domain mutation: Correct domain and op_type transformations
"""

import pytest
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferElementwiseBinaryOperation
from finn.util.basic import getHWCustomOp
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.primitives.transforms.specialize_kernels import SpecializeKernels
from brainsmith.registry import (
    discover_components,
    get_backend,
    get_kernel,
    list_backends_for_kernel,
)

# Ensure registry is initialized
discover_components()


# All 16 elementwise operations
ELEMENTWISE_OPERATIONS = [
    "Add",
    "Sub",
    "Mul",
    "Div",
    "And",
    "Or",
    "Xor",
    "Equal",
    "Less",
    "LessOrEqual",
    "Greater",
    "GreaterOrEqual",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BitShift",
]


# ============================================================================
# Test 1: Registry Lookup Validation
# ============================================================================


@pytest.mark.parametrize("operation", ELEMENTWISE_OPERATIONS)
def test_elementwise_kernel_registered(operation):
    """Verify all 16 elementwise kernels are discoverable in registry."""
    kernel_name = f"finn:Elementwise{operation}"

    # Should not raise KeyError
    kernel_class = get_kernel(kernel_name)

    # Verify it's the correct class
    assert kernel_class.__name__ == f"Elementwise{operation}"
    assert hasattr(kernel_class, "_operation")  # Base class has _operation tuple


@pytest.mark.parametrize("operation", ELEMENTWISE_OPERATIONS)
def test_elementwise_backend_registered(operation):
    """Verify all 16 elementwise HLS backends are discoverable in registry."""
    backend_name = f"finn:Elementwise{operation}_hls"

    # Should not raise KeyError
    backend_class = get_backend(backend_name)

    # Verify it's the correct class
    assert backend_class.__name__ == f"Elementwise{operation}_hls"


@pytest.mark.parametrize("operation", ELEMENTWISE_OPERATIONS)
def test_backend_targets_correct_kernel(operation):
    """Verify backend metadata correctly links to kernel."""
    kernel_name = f"finn:Elementwise{operation}"

    # Get backends for this kernel (returns list of backend name strings)
    backend_names = list_backends_for_kernel(kernel_name)

    # Should have at least one backend (HLS)
    assert len(backend_names) >= 1

    # Verify HLS backend is in the list (names are fully qualified: "finn:ElementwiseAdd_hls")
    assert f"finn:Elementwise{operation}_hls" in backend_names


# ============================================================================
# Test 2: Inference Transform (ONNX → Base Kernel)
# ============================================================================


def make_elementwise_onnx_model(onnx_op_type: str, lhs_dtype="INT8", rhs_dtype="INT8"):
    """Create minimal ONNX model for testing elementwise inference.

    Args:
        onnx_op_type: ONNX operation type (e.g., "Add", "Mul")
        lhs_dtype: Left operand datatype
        rhs_dtype: Right operand datatype

    Returns:
        ModelWrapper with quantized elementwise operation
    """
    shape = [1, 8, 8, 64]

    # Create tensors (use FLOAT as container)
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, shape)
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)

    # Create node
    node = helper.make_node(onnx_op_type, ["in0", "in1"], ["out"], name=f"{onnx_op_type}_0")

    # Build graph and model
    graph = helper.make_graph([node], "elementwise_test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Set FINN datatypes (this is where quantization info lives)
    model.set_tensor_datatype("in0", DataType[lhs_dtype])
    model.set_tensor_datatype("in1", DataType[rhs_dtype])

    return model


@pytest.mark.parametrize(
    "onnx_op,finn_op",
    [
        ("Add", "ElementwiseAdd"),
        ("Mul", "ElementwiseMul"),
        ("Sub", "ElementwiseSub"),
    ],
)
def test_inference_transform_onnx_to_finn_kernel(onnx_op, finn_op):
    """Test InferElementwiseBinaryOperation transforms ONNX → FINN base kernel."""
    # Stage 1: Create ONNX model
    model = make_elementwise_onnx_model(onnx_op)

    # Verify initial state
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == onnx_op

    # Stage 2: Apply FINN inference transform
    model = model.transform(InferElementwiseBinaryOperation())

    # Validate transformation
    assert len(model.graph.node) == 1
    node = model.graph.node[0]

    # Check op_type mutation
    assert node.op_type == finn_op, f"Expected {finn_op}, got {node.op_type}"

    # Check domain
    assert (
        node.domain == "finn.custom_op.fpgadataflow"
    ), f"Expected finn.custom_op.fpgadataflow domain, got {node.domain}"

    # Check backend attribute
    op = getHWCustomOp(node, model)
    assert op.get_nodeattr("backend") == "fpgadataflow"


# ============================================================================
# Test 3: Specialization Transform (Base Kernel → HLS Backend)
# ============================================================================


def make_finn_elementwise_model(finn_op_type: str):
    """Create FINN elementwise model (post-inference) for specialization testing.

    Args:
        finn_op_type: FINN operation type (e.g., "ElementwiseAdd")

    Returns:
        ModelWrapper with FINN elementwise base kernel
    """
    shape = [1, 8, 8, 64]

    # Create tensors
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, shape)
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)

    # Create FINN elementwise node
    node = helper.make_node(
        finn_op_type,
        ["in0", "in1"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        lhs_shape=shape,
        rhs_shape=shape,
        out_shape=shape,
        lhs_dtype="INT8",
        rhs_dtype="INT8",
        out_dtype="INT16",
        name=f"{finn_op_type}_0",
    )

    # Build model
    graph = helper.make_graph([node], "elementwise_test", [in0, in1], [out])
    model = ModelWrapper(helper.make_model(graph))

    # Set datatypes
    model.set_tensor_datatype("in0", DataType["INT8"])
    model.set_tensor_datatype("in1", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT16"])

    return model


@pytest.fixture
def test_config():
    """Create test configuration for SpecializeKernels."""
    cfg = DataflowBuildConfig(
        output_dir="/tmp/test_output",
        synth_clk_period_ns=10.0,
        board="Pynq-Z1",
        shell_flow_type="vivado_zynq",
        generate_outputs=["estimate"],
    )

    # Set kernel_selections for all elementwise operations
    cfg.kernel_selections = [
        (f"finn:Elementwise{op}", [f"finn:Elementwise{op}_hls"]) for op in ["Add", "Mul", "Sub"]
    ]

    return cfg


@pytest.mark.parametrize(
    "finn_op",
    [
        "ElementwiseAdd",
        "ElementwiseMul",
        "ElementwiseSub",
    ],
)
def test_specialization_transform_finn_to_backend(finn_op, test_config):
    """Test SpecializeKernels transforms FINN base kernel → HLS backend."""
    # Stage 1: Create FINN elementwise model (post-inference)
    model = make_finn_elementwise_model(finn_op)

    # Verify initial state
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == finn_op
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow"

    # Stage 2: Apply specialization transform
    model = model.transform(SpecializeKernels(test_config))

    # Validate transformation
    assert len(model.graph.node) == 1
    node = model.graph.node[0]

    # Check op_type mutation (base → backend)
    expected_backend_op = f"{finn_op}_hls"
    assert (
        node.op_type == expected_backend_op
    ), f"Expected {expected_backend_op}, got {node.op_type}"

    # Check domain mutation (FINN adds language suffix)
    assert (
        node.domain == "finn.custom_op.fpgadataflow.hls"
    ), f"Expected finn.custom_op.fpgadataflow.hls domain, got {node.domain}"

    # Check backend attribute
    op = getHWCustomOp(node, model)
    assert op.get_nodeattr("backend") == "hls"


# ============================================================================
# Test 4: End-to-End Pipeline
# ============================================================================


@pytest.mark.parametrize(
    "onnx_op,finn_op",
    [
        ("Add", "ElementwiseAdd"),
        ("Mul", "ElementwiseMul"),
    ],
)
def test_end_to_end_onnx_to_hls_backend(onnx_op, finn_op, test_config):
    """Test complete pipeline: ONNX → FINN kernel → HLS backend."""
    # Stage 1: Create ONNX model
    model = make_elementwise_onnx_model(onnx_op)

    assert model.graph.node[0].op_type == onnx_op

    # Stage 2: Inference (ONNX → FINN)
    model = model.transform(InferElementwiseBinaryOperation())

    assert model.graph.node[0].op_type == finn_op
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow"

    # Stage 3: Specialization (FINN → Backend)
    model = model.transform(SpecializeKernels(test_config))

    assert model.graph.node[0].op_type == f"{finn_op}_hls"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow.hls"

    # Validate backend capabilities
    node = model.graph.node[0]
    op = getHWCustomOp(node, model)
    assert hasattr(op, "code_gen_dict"), "Backend should have HLS code generation capability"
    assert op.get_nodeattr("backend") == "hls"
