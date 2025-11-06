"""Golden reference execution utilities for test framework.

This module provides utilities for executing golden reference models using
ONNX Runtime. Golden references are executed on Stage 1 (pure ONNX) models
without QONNX annotations to ensure correct semantics independent of
FINN/Brainsmith conventions.

Key Concepts:
- Stage 1 (Pure ONNX): Uses actual TensorProto types (INT8, INT16, etc.)
- ONNX Runtime: Executes Stage 1 models with correct type semantics
- NumPy Fallback: Optional fallback for ops ONNX Runtime doesn't support

Example:
    >>> from tests.support.golden_reference import execute_onnx_runtime_golden
    >>>
    >>> # Execute with ONNX Runtime
    >>> model, _ = make_onnx_model()  # Stage 1 model
    >>> outputs = execute_onnx_runtime_golden(model, inputs)
    >>>
    >>> # Execute with NumPy fallback
    >>> def numpy_impl(inputs):
    ...     return {"output": np.add(inputs["input0"], inputs["input1"])}
    >>>
    >>> outputs = execute_onnx_runtime_golden(model, inputs, numpy_fallback=numpy_impl)
"""

import numpy as np
from typing import Dict, Optional, Callable
from qonnx.core.modelwrapper import ModelWrapper


def execute_onnx_runtime_golden(
    model: ModelWrapper,
    inputs: Dict[str, np.ndarray],
    numpy_fallback: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]] = None
) -> Dict[str, np.ndarray]:
    """Execute Stage 1 ONNX model with ONNX Runtime for golden reference.

    This function executes a pure ONNX model (Stage 1, before QONNX annotations)
    using ONNX Runtime to obtain the canonical "correct" behavior according to
    the ONNX specification. This is independent of FINN/Brainsmith conventions.

    If ONNX Runtime execution fails (e.g., unsupported op or type validation),
    falls back to NumPy implementation if provided.

    Args:
        model: Pure ONNX model (Stage 1, no QONNX annotations)
        inputs: Dict mapping input names to NumPy arrays
        numpy_fallback: Optional function to compute outputs using NumPy
            if ONNX Runtime fails. Signature: (inputs) -> outputs

    Returns:
        Dict mapping output names to NumPy arrays

    Raises:
        RuntimeError: If ONNX Runtime fails and no fallback provided

    Example:
        >>> # Simple ONNX Runtime execution
        >>> model = make_add_model()  # Stage 1 model
        >>> inputs = {"a": np.array([1, 2]), "b": np.array([3, 4])}
        >>> outputs = execute_onnx_runtime_golden(model, inputs)
        >>> outputs["output"]
        array([4, 6])

        >>> # With NumPy fallback for unsupported ops
        >>> def add_fallback(inputs):
        ...     return {"output": inputs["a"] + inputs["b"]}
        >>> outputs = execute_onnx_runtime_golden(model, inputs, numpy_fallback=add_fallback)

    Note:
        - Input dtypes are NOT automatically cast - caller must ensure correct types
        - Initializers are automatically filtered out (not passed to ONNX Runtime)
        - ONNX Runtime exceptions are caught and trigger fallback if available
    """
    import onnxruntime as ort

    try:
        # Create ONNX Runtime session
        sess = ort.InferenceSession(
            model.model.SerializeToString(),
            providers=['CPUExecutionProvider']
        )

        # Filter inputs to only include runtime inputs (exclude initializers)
        # ONNX Runtime rejects initializers passed as runtime inputs
        runtime_input_names = [inp.name for inp in sess.get_inputs()]
        runtime_inputs = {
            name: inputs[name]
            for name in runtime_input_names
            if name in inputs
        }

        # Execute with ONNX Runtime
        ort_outputs = sess.run(None, runtime_inputs)

        # Convert list of outputs to dict
        output_names = [out.name for out in sess.get_outputs()]
        return {name: output for name, output in zip(output_names, ort_outputs)}

    except (ort.capi.onnxruntime_pybind11_state.InvalidGraph,
            ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.RuntimeException) as e:
        # ONNX Runtime failed - try fallback
        if numpy_fallback is not None:
            return numpy_fallback(inputs)
        else:
            raise RuntimeError(
                f"ONNX Runtime execution failed and no NumPy fallback provided. "
                f"Error: {e}"
            ) from e
