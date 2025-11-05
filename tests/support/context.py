"""Test execution context generation.

This module provides utilities for generating test data (execution contexts)
for kernel testing. Provides a single source of truth for test data generation
used by executors and test frameworks.

Key Features:
- Deterministic random data generation with seed support
- ONNX-native and QONNX-native data generation (stage separation)
- Automatic shape and datatype inference from operators
- Handles both streaming inputs and initializers (weights)
- Pre-allocates output tensors

Architecture:
    make_execution_context_onnx() - Generate from TensorProto types (Stage 1)
    make_execution_context_qonnx() - Generate from QONNX DataType (Stage 2+)

Usage:
    from tests.support.context import make_execution_context_onnx, make_execution_context_qonnx

    # Stage 1: Pure ONNX (for golden reference)
    context = make_execution_context_onnx(model, input_names, seed=42)

    # Stage 2+: QONNX/FINN (for hardware execution)
    context = make_execution_context_qonnx(model, op, seed=42)
"""

import numpy as np
from typing import Dict, List, Optional
from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from tests.support.constants import (
    UNSIGNED_TEST_DATA_CAP,
    SIGNED_TEST_DATA_MIN,
    SIGNED_TEST_DATA_MAX,
)
from tests.support.onnx_utils import generate_onnx_test_data, get_onnx_tensor_type


def make_execution_context_onnx(
    model: ModelWrapper,
    input_names: List[str],
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Create execution context from pure ONNX model (TensorProto types).

    Generates test data based on ONNX TensorProto types, independent of
    QONNX DataType annotations. Used for golden reference validation.

    This is Stage 1 data generation:
    - Reads TensorProto types from model (TensorProto.FLOAT, INT8, etc.)
    - Generates appropriate test data for each type
    - Independent of FINN/Brainsmith QONNX annotations

    Args:
        model: ONNX model wrapper (pure ONNX, no QONNX annotations required)
        input_names: List of input tensor names to generate data for
        seed: Random seed for reproducibility (optional)

    Returns:
        Dict mapping tensor names → numpy arrays with test data

    Raises:
        ValueError: If input tensor not found or has no type info

    Example:
        >>> # Stage 1: Generate data for pure ONNX model
        >>> onnx_model, _ = make_onnx_model()  # Pure ONNX
        >>> input_names = [inp.name for inp in onnx_model.graph.input]
        >>> context = make_execution_context_onnx(onnx_model, input_names, seed=42)
        >>>
        >>> # Execute with ONNX Runtime for golden reference
        >>> import onnxruntime as ort
        >>> sess = ort.InferenceSession(onnx_model.model.SerializeToString())
        >>> outputs = sess.run(None, context)
    """
    if seed is not None:
        np.random.seed(seed)

    context = {}

    for inp_name in input_names:
        # Skip if it's an initializer (weight/parameter)
        if model.get_initializer(inp_name) is not None:
            context[inp_name] = model.get_initializer(inp_name)
            continue

        # Get TensorProto type and shape
        try:
            tensor_type = get_onnx_tensor_type(model, inp_name)
            shape = model.get_tensor_shape(inp_name)

            # Generate ONNX-native test data
            data = generate_onnx_test_data(tensor_type, tuple(shape), seed=None)
            context[inp_name] = data

        except Exception as e:
            raise ValueError(
                f"Cannot generate ONNX test data for input '{inp_name}': {e}\n"
                f"This typically means the tensor is missing type information.\n"
                f"Ensure the ONNX model has proper type annotations."
            )

    return context


def make_execution_context_qonnx(
    model: ModelWrapper,
    op: HWCustomOp,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Create execution context from QONNX model (DataType annotations).

    Generates test data based on QONNX DataType annotations. Used for
    FINN/Brainsmith hardware execution (Python, cppsim, rtlsim).

    This is Stage 2+ data generation:
    - Reads QONNX DataType from operator (DataType["INT8"], etc.)
    - Generates data in DataType's valid range
    - Uses FINN/QONNX conventions (FLOAT containers for integers)

    Args:
        model: QONNX model wrapper (with DataType annotations)
        op: Hardware operator instance (HWCustomOp with DataType info)
        seed: Random seed for reproducibility (optional)

    Returns:
        Dict mapping tensor names to numpy arrays containing:
        - Random inputs (for streaming data)
        - Initializers (weights/parameters from model)
        - Pre-allocated outputs (zeros)

    Raises:
        ValueError: If shape or datatype cannot be determined for any input/output

    Example:
        >>> # Stage 2+: Generate data for QONNX/FINN execution
        >>> qonnx_model, op = run_inference_pipeline()  # QONNX annotations added
        >>> context = make_execution_context_qonnx(qonnx_model, op, seed=42)
        >>>
        >>> # Execute with FINN
        >>> op.execute_node(context, qonnx_model.graph)
        >>> output = context[op.onnx_node.output[0]]
    """
    if seed is not None:
        np.random.seed(seed)

    context = {}
    node = op.onnx_node

    # Create inputs (streaming data or initializers)
    for i, inp_name in enumerate(node.input):
        if not inp_name:  # Optional input (empty string)
            continue

        # Check if it's an initializer (weight/parameter)
        init = model.get_initializer(inp_name)
        if init is not None:
            context[inp_name] = init
            continue

        # Generate random input for streaming data
        try:
            shape = op.get_normal_input_shape(i)
            dtype = op.get_input_datatype(i)

            # Generate data in datatype's valid range
            if dtype.min() >= 0:
                # Unsigned type (e.g., UINT8)
                # Cap at UNSIGNED_TEST_DATA_CAP to prevent extreme values from wide
                # datatypes (e.g., UINT32 max is 4,294,967,295). This ensures:
                # - Numerical stability: Prevents overflow in fixed-point arithmetic
                # - Test practicality: Human-readable values (0-255) aid debugging
                # - Representative sampling: Real-world data rarely uses full range
                data = np.random.randint(
                    max(dtype.min(), 0),  # Ensure non-negative
                    min(dtype.max() + 1, UNSIGNED_TEST_DATA_CAP),
                    size=shape
                ).astype(np.float32)
            else:
                # Signed type (e.g., INT8)
                # Cap at INT8 range (-128 to 127) for consistency and stability.
                # Same benefits as unsigned: prevents overflow, aids debugging,
                # and represents typical real-world value distributions.
                data = np.random.randint(
                    max(dtype.min(), SIGNED_TEST_DATA_MIN),
                    min(dtype.max() + 1, SIGNED_TEST_DATA_MAX),
                    size=shape
                ).astype(np.float32)

            context[inp_name] = data

        except Exception as e:
            raise ValueError(
                f"Cannot generate input {i} ({inp_name}): {e}\n"
                f"Operator: {op.__class__.__name__}\n"
                f"This typically means the operator doesn't implement "
                f"get_normal_input_shape() or get_input_datatype() correctly."
            )

    # Pre-allocate output tensors
    for i, out_name in enumerate(node.output):
        try:
            shape = op.get_normal_output_shape(i)
            context[out_name] = np.zeros(shape, dtype=np.float32)

        except Exception as e:
            raise ValueError(
                f"Cannot pre-allocate output {i} ({out_name}): {e}\n"
                f"Operator: {op.__class__.__name__}\n"
                f"This typically means the operator doesn't implement "
                f"get_normal_output_shape() correctly."
            )

    return context


def make_execution_context(
    model: ModelWrapper,
    op: HWCustomOp,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """[DEPRECATED] Use make_execution_context_qonnx() instead.

    This function is kept for backward compatibility but will be removed
    in a future version. It assumes QONNX annotations are present.

    For new code:
    - Use make_execution_context_onnx() for golden reference (Stage 1)
    - Use make_execution_context_qonnx() for FINN execution (Stage 2+)
    """
    import warnings
    warnings.warn(
        "make_execution_context() is deprecated. "
        "Use make_execution_context_onnx() for golden reference or "
        "make_execution_context_qonnx() for FINN/Brainsmith execution.",
        DeprecationWarning,
        stacklevel=2
    )
    return make_execution_context_qonnx(model, op, seed)
