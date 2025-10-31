"""Tensor name mapping utilities for test data generation.

Bridges the gap between ONNX tensor names (arbitrary) and golden reference
standard naming conventions (input0, input1, output).

This module was extracted from DualPipelineParityTest to enable reuse across
all test frameworks. It solves the common problem where:
- ONNX models use arbitrary tensor names: "inp1", "inp2", "outp", "data", "bias"
- Golden reference implementations expect standard names: "input0", "input1", "output"

Key Features:
- Bidirectional mapping (ONNX ↔ Golden)
- Handles single and multiple inputs/outputs
- Preserves tensor data (zero-copy reference semantics)
- Type-safe with clear error messages

Usage:
    from tests.support.tensor_mapping import map_onnx_to_golden_names

    # ONNX model has arbitrary names
    onnx_context = {"inp1": arr1, "inp2": arr2, "outp": arr3}

    # Golden reference expects standard names
    golden_inputs = map_onnx_to_golden_names(onnx_context, num_inputs=2)
    # → {"input0": arr1, "input1": arr2}

    golden_outputs = kernel.compute_golden_reference(golden_inputs)
    # golden_outputs = {"output": arr3}
"""

from typing import Dict, List, Tuple
import numpy as np


def map_onnx_to_golden_names(
    onnx_tensors: Dict[str, np.ndarray],
    num_inputs: int
) -> Dict[str, np.ndarray]:
    """Map ONNX tensor names to golden reference standard names.

    Converts arbitrary ONNX tensor names to the standard naming convention
    used by golden reference implementations:
    - Inputs: "input0", "input1", ..., "input{N-1}"
    - Outputs: "output" (single) or "output0", "output1", ... (multiple)

    This enables golden reference implementations to use predictable,
    standardized names while ONNX models retain their original names.

    Args:
        onnx_tensors: Dict with ONNX tensor names as keys, numpy arrays as values
        num_inputs: Number of input tensors to map (remaining tensors are outputs)

    Returns:
        Dict with standard golden reference names as keys, same numpy arrays as values

    Example:
        >>> import numpy as np
        >>> onnx = {
        ...     "inp1": np.array([1, 2, 3]),
        ...     "inp2": np.array([4, 5, 6]),
        ...     "outp": np.array([5, 7, 9])
        ... }
        >>> golden = map_onnx_to_golden_names(onnx, num_inputs=2)
        >>> golden.keys()
        dict_keys(['input0', 'input1'])
        >>> np.array_equal(golden["input0"], onnx["inp1"])
        True

    Note:
        This function only maps input tensors. Output names are typically
        handled by the golden reference implementation itself.
    """
    if num_inputs < 0:
        raise ValueError(f"num_inputs must be non-negative, got {num_inputs}")

    if num_inputs > len(onnx_tensors):
        raise ValueError(
            f"num_inputs={num_inputs} exceeds available tensors ({len(onnx_tensors)}). "
            f"Available: {list(onnx_tensors.keys())}"
        )

    # Take first N tensors as inputs (preserve insertion order)
    input_items = list(onnx_tensors.items())[:num_inputs]

    # Map to standard names
    golden_inputs = {}
    for i, (_, tensor_value) in enumerate(input_items):
        golden_inputs[f"input{i}"] = tensor_value

    return golden_inputs


def map_golden_to_onnx_names(
    golden_tensors: Dict[str, np.ndarray],
    onnx_names: List[str]
) -> Dict[str, np.ndarray]:
    """Reverse mapping: golden reference names → ONNX tensor names.

    Converts standard golden reference names back to ONNX tensor names.
    Useful for mapping golden outputs back to ONNX context for execution.

    Args:
        golden_tensors: Dict with standard golden names (input0, input1, output, etc.)
        onnx_names: List of ONNX tensor names in order

    Returns:
        Dict with ONNX tensor names as keys, same numpy arrays as values

    Example:
        >>> golden = {
        ...     "input0": np.array([1, 2, 3]),
        ...     "input1": np.array([4, 5, 6]),
        ...     "output": np.array([5, 7, 9])
        ... }
        >>> onnx_names = ["inp1", "inp2", "outp"]
        >>> onnx = map_golden_to_onnx_names(golden, onnx_names)
        >>> onnx.keys()
        dict_keys(['inp1', 'inp2', 'outp'])
        >>> np.array_equal(onnx["outp"], golden["output"])
        True

    Raises:
        ValueError: If counts don't match or mapping is ambiguous
    """
    if len(golden_tensors) != len(onnx_names):
        raise ValueError(
            f"Mismatch: {len(golden_tensors)} golden tensors but {len(onnx_names)} ONNX names. "
            f"Golden: {list(golden_tensors.keys())}, ONNX: {onnx_names}"
        )

    # Sort golden keys to ensure predictable mapping
    # input0, input1, ..., output or output0, output1, ...
    golden_keys = sorted(golden_tensors.keys())

    # Map in order
    onnx_tensors = {}
    for onnx_name, golden_key in zip(onnx_names, golden_keys):
        onnx_tensors[onnx_name] = golden_tensors[golden_key]

    return onnx_tensors


def infer_num_inputs_from_golden(golden_tensors: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """Infer number of inputs and outputs from golden reference tensor names.

    Analyzes golden reference tensor names to determine how many are inputs
    vs outputs based on naming convention:
    - Inputs: "input0", "input1", ..., "input{N-1}"
    - Outputs: "output" or "output0", "output1", ..., "output{M-1}"

    Args:
        golden_tensors: Dict with golden reference tensor names

    Returns:
        Tuple of (num_inputs, num_outputs)

    Example:
        >>> golden = {"input0": arr1, "input1": arr2, "output": arr3}
        >>> infer_num_inputs_from_golden(golden)
        (2, 1)

        >>> golden = {"input0": arr1, "output0": arr2, "output1": arr3}
        >>> infer_num_inputs_from_golden(golden)
        (1, 2)
    """
    num_inputs = sum(1 for k in golden_tensors.keys() if k.startswith("input"))
    num_outputs = sum(1 for k in golden_tensors.keys() if k.startswith("output"))

    return num_inputs, num_outputs


def extract_inputs_only(
    tensors: Dict[str, np.ndarray],
    num_inputs: int
) -> Dict[str, np.ndarray]:
    """Extract only input tensors from a mixed dict of inputs and outputs.

    Helper for separating inputs from outputs when both are in the same dict.
    Preserves insertion order (Python 3.7+).

    Args:
        tensors: Dict with both inputs and outputs
        num_inputs: Number of tensors that are inputs (from the start)

    Returns:
        Dict with only the first num_inputs tensors

    Example:
        >>> context = {"inp1": arr1, "inp2": arr2, "outp": arr3}
        >>> inputs = extract_inputs_only(context, num_inputs=2)
        >>> inputs.keys()
        dict_keys(['inp1', 'inp2'])
    """
    if num_inputs < 0:
        raise ValueError(f"num_inputs must be non-negative, got {num_inputs}")

    if num_inputs > len(tensors):
        raise ValueError(
            f"num_inputs={num_inputs} exceeds available tensors ({len(tensors)})"
        )

    # Take first N items (preserves insertion order)
    input_items = list(tensors.items())[:num_inputs]
    return dict(input_items)
