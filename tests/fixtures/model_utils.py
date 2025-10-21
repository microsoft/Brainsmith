"""Utilities for creating test ONNX models."""

import onnx
from onnx import helper, TensorProto, ValueInfoProto
import numpy as np
from typing import List, Optional, Dict, Any
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
        op_type: Type of operation to use
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


def _save_model(model: onnx.ModelProto, path: str) -> None:
    """Save ONNX model to file (internal use only).
    
    Args:
        model: ONNX model proto
        path: Path to save the model
    """
    onnx.save(model, path)


def create_softmax_model(
    batch_size: int = 1,
    seq_len: int = 128,
    channels: int = 768,
    input_dtype: str = "INT8",
    output_dtype: str = "FLOAT32",
    model_name: str = "softmax_test_model"
) -> onnx.ModelProto:
    """Create an ONNX model with a Softmax operation.

    Args:
        batch_size: Batch dimension
        seq_len: Sequence length dimension
        channels: Channel dimension (softmax normalizes over this)
        input_dtype: QONNX DataType name for input (e.g., "INT8", "FLOAT32")
        output_dtype: QONNX DataType name for output (usually "FLOAT32")
        model_name: Name of the model

    Returns:
        ONNX model proto with Softmax node
    """
    from qonnx.core.datatype import DataType

    # Map QONNX DataType names to ONNX TensorProto types
    dtype_map = {
        "INT8": TensorProto.INT8,
        "UINT8": TensorProto.UINT8,
        "INT16": TensorProto.INT16,
        "UINT16": TensorProto.UINT16,
        "INT32": TensorProto.INT32,
        "FLOAT32": TensorProto.FLOAT,
        "FLOAT16": TensorProto.FLOAT16,
    }

    input_shape = [batch_size, seq_len, channels]
    output_shape = [batch_size, seq_len, channels]

    # Create input/output tensors with specified datatypes
    input_tensor = helper.make_tensor_value_info(
        "input", dtype_map.get(input_dtype, TensorProto.FLOAT), input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output", dtype_map.get(output_dtype, TensorProto.FLOAT), output_shape
    )

    # Create Softmax node (normalizes over axis=-1, the channel dimension)
    softmax_node = helper.make_node(
        "Softmax",
        inputs=["input"],
        outputs=["output"],
        axis=-1  # Normalize over last dimension (channels)
    )

    # Create graph
    graph_proto = helper.make_graph(
        [softmax_node],
        model_name,
        [input_tensor],
        [output_tensor]
    )

    # Create model
    model_proto = helper.make_model(
        graph_proto,
        producer_name="brainsmith_test"
    )

    # Set opset version
    model_proto.opset_import[0].version = 11

    return model_proto


def create_gather_crop_model(
    input_shape: List[int] = [1, 224, 224, 64],
    crop_start: int = 12,
    crop_end: int = 212,
    axis: int = 1,
    input_dtype: str = "INT8",
    model_name: str = "gather_crop_test_model"
) -> onnx.ModelProto:
    """Create ONNX model with Gather node that represents a crop operation.

    Creates a pattern that can be transformed to Crop hardware operation:
    - Gather with consecutive indices on spatial axis (height or width)
    - Represents cropping pixels from image edges

    Args:
        input_shape: Input tensor shape in NHWC format [N, H, W, C]
        crop_start: Starting index for crop (crop_north or crop_west)
        crop_end: Ending index for crop (exclusive)
        axis: Axis to crop (1=height, 2=width in NHWC)
        input_dtype: Input datatype as string (e.g., "INT8")
        model_name: Name of the model

    Returns:
        ONNX ModelProto with Gather node representing crop

    Example:
        # Crop height dimension: [1, 224, 224, 64] -> [1, 200, 224, 64]
        # crop_north=12, crop_south=12 -> indices [12:212]
        model = create_gather_crop_model(
            input_shape=[1, 224, 224, 64],
            crop_start=12,
            crop_end=212,
            axis=1
        )
    """
    # Map QONNX DataType names to ONNX TensorProto types
    dtype_map = {
        "INT8": TensorProto.INT8,
        "UINT8": TensorProto.UINT8,
        "INT16": TensorProto.INT16,
        "UINT16": TensorProto.UINT16,
        "INT32": TensorProto.INT32,
        "FLOAT32": TensorProto.FLOAT,
        "FLOAT16": TensorProto.FLOAT16,
    }

    # Compute output shape after crop
    output_shape = list(input_shape)
    output_shape[axis] = crop_end - crop_start

    # Create input tensor
    input_tensor = helper.make_tensor_value_info(
        "input",
        dtype_map.get(input_dtype, TensorProto.INT8),
        input_shape
    )

    # Create output tensor
    output_tensor = helper.make_tensor_value_info(
        "output",
        dtype_map.get(input_dtype, TensorProto.INT8),
        output_shape
    )

    # Create indices for Gather (consecutive sequence for crop)
    indices = np.arange(crop_start, crop_end, dtype=np.int64)
    indices_tensor = helper.make_tensor(
        "indices",
        TensorProto.INT64,
        [len(indices)],
        indices.tolist()
    )

    # Create Gather node
    gather_node = helper.make_node(
        "Gather",
        inputs=["input", "indices"],
        outputs=["output"],
        axis=axis,
        name="Gather_0"
    )

    # Create graph
    graph_proto = helper.make_graph(
        [gather_node],
        model_name,
        [input_tensor],
        [output_tensor],
        initializer=[indices_tensor]
    )

    # Create model
    model_proto = helper.make_model(
        graph_proto,
        producer_name="brainsmith_test"
    )

    # Set opset version
    model_proto.opset_import[0].version = 11

    return model_proto


def create_transpose_shuffle_model(
    input_shape: List[int] = [1, 56, 56, 128],
    perm: List[int] = [0, 2, 1, 3],
    input_dtype: str = "INT8",
    model_name: str = "transpose_shuffle_test_model"
) -> onnx.ModelProto:
    """Create ONNX model with Transpose node for Shuffle testing.

    Creates a Transpose node that can be transformed to Shuffle hardware operation.
    This represents tensor rearrangement operations common in neural networks.

    Args:
        input_shape: Input tensor shape (e.g., [1, 56, 56, 128] for NHWC)
        perm: Permutation array (e.g., [0, 2, 1, 3] swaps H and W)
        input_dtype: Input datatype as string (e.g., "INT8")
        model_name: Name of the model

    Returns:
        ONNX ModelProto with Transpose node

    Example:
        # Swap height and width: [1, 56, 56, 128] -> [1, 56, 56, 128]
        model = create_transpose_shuffle_model(
            input_shape=[1, 56, 56, 128],
            perm=[0, 2, 1, 3]
        )
    """
    # Map QONNX DataType names to ONNX TensorProto types
    dtype_map = {
        "INT8": TensorProto.INT8,
        "UINT8": TensorProto.UINT8,
        "INT16": TensorProto.INT16,
        "UINT16": TensorProto.UINT16,
        "INT32": TensorProto.INT32,
        "FLOAT32": TensorProto.FLOAT,
        "FLOAT16": TensorProto.FLOAT16,
    }

    # Compute output shape after permutation
    output_shape = [input_shape[i] for i in perm]

    # Create input tensor
    input_tensor = helper.make_tensor_value_info(
        "input",
        dtype_map.get(input_dtype, TensorProto.INT8),
        input_shape
    )

    # Create output tensor
    output_tensor = helper.make_tensor_value_info(
        "output",
        dtype_map.get(input_dtype, TensorProto.INT8),
        output_shape
    )

    # Create Transpose node
    transpose_node = helper.make_node(
        "Transpose",
        inputs=["input"],
        outputs=["output"],
        perm=perm,
        name="Transpose_0"
    )

    # Create graph
    graph_proto = helper.make_graph(
        [transpose_node],
        model_name,
        [input_tensor],
        [output_tensor],
        initializer=[]
    )

    # Create model
    model_proto = helper.make_model(
        graph_proto,
        producer_name="brainsmith_test"
    )

    # Set opset version
    model_proto.opset_import[0].version = 11

    return model_proto


def create_multithreshold_model(
    input_shape: List[int] = [1, 56, 56, 128],
    num_thresholds: int = 7,
    input_dtype: str = "INT8",
    threshold_dtype: str = "INT8",
    output_dtype: str = "UINT4",
    out_bias: int = 0,
    out_scale: float = 1.0,
    model_name: str = "multithreshold_test_model"
) -> onnx.ModelProto:
    """Create an ONNX model with a MultiThreshold operation.

    MultiThreshold is a QONNX operation that applies multi-threshold quantization:
    - Compares input against multiple thresholds per channel
    - Outputs quantized value based on which threshold range input falls into
    - Used for quantized activation functions and multi-bit quantization

    Args:
        input_shape: Shape of input tensor (e.g., [batch, height, width, channels])
        num_thresholds: Number of threshold steps per channel (e.g., 7 for 3-bit output)
        input_dtype: QONNX DataType name for input (e.g., "INT8")
        threshold_dtype: QONNX DataType name for thresholds (e.g., "INT8")
        output_dtype: QONNX DataType name for output (e.g., "UINT4" for 3-bit)
        out_bias: Bias value added to output (default 0)
        out_scale: Scale factor for output (must be 1.0 for HLS conversion)
        model_name: Name of the model

    Returns:
        ONNX model proto with MultiThreshold node

    Example:
        # Create a 3-bit quantization model (7 thresholds → 8 levels → 3 bits)
        model = create_multithreshold_model(
            input_shape=[1, 56, 56, 128],
            num_thresholds=7,
            input_dtype="INT8",
            output_dtype="UINT4"
        )
    """
    from qonnx.core.datatype import DataType

    # Map QONNX DataType names to ONNX TensorProto types
    dtype_map = {
        "INT8": TensorProto.INT8,
        "UINT8": TensorProto.UINT8,
        "INT4": TensorProto.INT8,  # Use INT8 as container for INT4
        "UINT4": TensorProto.UINT8,  # Use UINT8 as container for UINT4
        "INT16": TensorProto.INT16,
        "UINT16": TensorProto.UINT16,
        "INT32": TensorProto.INT32,
        "UINT32": TensorProto.UINT32,
        "FLOAT32": TensorProto.FLOAT,
    }

    num_channels = input_shape[-1]
    threshold_shape = [num_channels, num_thresholds]
    output_shape = input_shape  # Same shape as input

    # Create threshold values (one row per channel)
    # Use evenly spaced thresholds for testing
    thresholds = np.zeros(threshold_shape, dtype=np.float32)
    for ch in range(num_channels):
        # Create ascending thresholds for this channel
        # Example for INT8 input: [-100, -50, 0, 50, 100, 120, 127]
        idt = DataType[input_dtype]
        tdt = DataType[threshold_dtype]
        min_val = max(idt.min(), tdt.min())
        max_val = min(idt.max(), tdt.max())
        ch_thresholds = np.linspace(min_val, max_val, num_thresholds + 2)[1:-1]
        thresholds[ch, :] = ch_thresholds

    # Convert to appropriate integer type if needed
    if DataType[threshold_dtype].is_integer():
        thresholds = np.round(thresholds).astype(np.int32)

    # Create initializer for thresholds
    threshold_tensor = helper.make_tensor(
        "thresholds",
        dtype_map.get(threshold_dtype, TensorProto.FLOAT),
        threshold_shape,
        thresholds.flatten().tolist()
    )

    # Create input/output tensors
    input_tensor = helper.make_tensor_value_info(
        "input",
        dtype_map.get(input_dtype, TensorProto.FLOAT),
        input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        "output",
        dtype_map.get(output_dtype, TensorProto.UINT8),
        output_shape
    )

    # Create MultiThreshold node
    multithreshold_node = helper.make_node(
        "MultiThreshold",
        inputs=["input", "thresholds"],
        outputs=["output"],
        domain="qonnx.custom_op.general",
        out_dtype=output_dtype,
        out_scale=out_scale,
        out_bias=out_bias,
        name="MultiThreshold_0"
    )

    # Create graph
    graph_proto = helper.make_graph(
        [multithreshold_node],
        model_name,
        [input_tensor],
        [output_tensor],
        initializer=[threshold_tensor]
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
def simple_onnx_model(tmp_path):
    """Create a simple ONNX model for testing."""
    model = create_simple_model()
    model_path = tmp_path / "simple_model.onnx"
    _save_model(model, str(model_path))

    return model_path