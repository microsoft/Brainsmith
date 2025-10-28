"""Test fixture generation for parity testing.

This module provides utilities for generating test data (execution contexts)
for parity tests. It eliminates duplication between base_parity_test.py and
executors.py by providing a single source of truth for test data generation.

Key Features:
- Deterministic random data generation with seed support
- Automatic shape and datatype inference from operators
- Handles both streaming inputs and initializers (weights)
- Pre-allocates output tensors

Usage:
    from tests.parity.test_fixtures import make_execution_context

    # Generate test context with random seed
    context = make_execution_context(model, op, seed=42)

    # Execute operator
    op.execute_node(context, model.graph)
"""

import numpy as np
from typing import Dict, Optional
from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from tests.common.constants import (
    UNSIGNED_TEST_DATA_CAP,
    SIGNED_TEST_DATA_MIN,
    SIGNED_TEST_DATA_MAX,
)


def make_execution_context(
    model: ModelWrapper,
    op: HWCustomOp,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Create execution context with random inputs for testing.

    Generates random test data based on operator's shape and datatype
    specifications. This is the single source of truth for test data
    generation, used by both ParityTestBase and BackendExecutor classes.

    Args:
        model: ONNX model wrapper containing the operator
        op: Hardware custom operator to generate context for
        seed: Random seed for reproducibility (optional)
              If provided, ensures deterministic data generation

    Returns:
        Dict mapping tensor names to numpy arrays containing:
        - Random inputs (for streaming data)
        - Initializers (weights/parameters from model)
        - Pre-allocated outputs (zeros)

    Raises:
        ValueError: If shape or datatype cannot be determined for any input/output

    Example:
        >>> # Generate deterministic test data
        >>> context = make_execution_context(model, op, seed=42)
        >>> op.execute_node(context, model.graph)
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
