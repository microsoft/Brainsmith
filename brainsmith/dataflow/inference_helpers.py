############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Helper utilities for ONNX → KernelOp inference.

These helpers handle common patterns during inference (e.g., commutative ops,
scalar broadcasting). They are ONLY used during can_infer_from() and infer_from()
to handle ONNX's flexibility.

Once a KernelOp node is created, inputs are in schema-defined order and the
kernel expects them to stay that way. If a user manually modifies inputs after
inference, constraint validation may fail - this is the user's responsibility.

Key Principle: Rigid Kernels + Smart Inference
    - Schemas define fixed input positions with rigid constraints
    - Inference methods use these helpers to detect correct input ordering
    - After node creation, input order is locked (no runtime reordering)

Example Usage:
    >>> @classmethod
    >>> def can_infer_from(cls, node, model):
    ...     # Use helper to detect pattern
    ...     pair = find_static_dynamic_pair(node.input, model)
    ...     return pair is not None
    ...
    >>> @classmethod
    >>> def infer_from(cls, node, model, insert_index):
    ...     # Use helper to get correct order
    ...     dynamic, static = find_static_dynamic_pair(node.input, model)
    ...     # Create node with fixed order
    ...     hw_node = helper.make_node("ChannelwiseOp",
    ...                                inputs=[dynamic, static], ...)
"""

import logging

import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

logger = logging.getLogger(__name__)

# Forward declaration for BroadcastInfo (avoid circular import)
try:
    from brainsmith.dataflow.broadcast_helpers import BroadcastInfo
except ImportError:
    BroadcastInfo = None  # Will be available after package initialization


# ============================================================================
# Commutative Input Detection
# ============================================================================

def find_static_dynamic_pair(
    inputs: list[str],
    model: ModelWrapper
) -> tuple[str, str] | None:
    """Find (dynamic, static) pair from commutative binary op inputs.

    For operations like Add, Mul where input order doesn't affect semantics,
    identify which input is dynamic (data) and which is static (parameters).

    Args:
        inputs: List of input tensor names (typically 2 for binary ops)
        model: ModelWrapper for checking initializers

    Returns:
        (dynamic_tensor, static_tensor) tuple, or None if pattern doesn't match

    Example:
        >>> # Correct order
        >>> find_static_dynamic_pair(["data", "params"], model)
        ("data", "params")

        >>> # Swapped - returns corrected order
        >>> find_static_dynamic_pair(["params", "data"], model)
        ("data", "params")

        >>> # Both dynamic - not the pattern
        >>> find_static_dynamic_pair(["data1", "data2"], model)
        None

        >>> # Both static - not the pattern
        >>> find_static_dynamic_pair(["params1", "params2"], model)
        None
    """
    if len(inputs) != 2:
        return None

    # Check which are static (initializers)
    initializer_names = {x.name for x in model.graph.initializer}
    is_static = [inp in initializer_names for inp in inputs]

    # We need exactly one static, one dynamic
    if sum(is_static) != 1:
        return None

    # Return in (dynamic, static) order
    if is_static[0]:
        # inputs[0] is static, inputs[1] is dynamic → swap
        return (inputs[1], inputs[0])
    else:
        # inputs[0] is dynamic, inputs[1] is static → keep order
        return (inputs[0], inputs[1])


def find_dynamic_inputs(
    inputs: list[str],
    model: ModelWrapper
) -> list[str]:
    """Find all dynamic (non-initializer) inputs.

    Used for operations that expect all-dynamic inputs (e.g., AddStreams).

    Args:
        inputs: List of input tensor names
        model: ModelWrapper for checking initializers

    Returns:
        List of dynamic input names (preserves original order)

    Example:
        >>> find_dynamic_inputs(["data1", "params", "data2"], model)
        ["data1", "data2"]
    """
    initializer_names = {x.name for x in model.graph.initializer}
    return [inp for inp in inputs if inp not in initializer_names]


def find_static_inputs(
    inputs: list[str],
    model: ModelWrapper
) -> list[str]:
    """Find all static (initializer) inputs.

    Args:
        inputs: List of input tensor names
        model: ModelWrapper for checking initializers

    Returns:
        List of static input names (preserves original order)

    Example:
        >>> find_static_inputs(["data", "params1", "params2"], model)
        ["params1", "params2"]
    """
    initializer_names = {x.name for x in model.graph.initializer}
    return [inp for inp in inputs if inp in initializer_names]


def find_dynamic_dynamic_pair(
    inputs: list[str],
    model: ModelWrapper
) -> tuple[str, str] | None:
    """Find (lhs, rhs) pair from both-dynamic binary op inputs.

    For operations like Add, Mul where both inputs are streaming/dynamic,
    identify the pair and return in original order.

    Args:
        inputs: List of input tensor names (typically 2 for binary ops)
        model: ModelWrapper for checking initializers

    Returns:
        (lhs, rhs) tuple, or None if pattern doesn't match

    Example:
        >>> # Both dynamic - returns in order
        >>> find_dynamic_dynamic_pair(["stream1", "stream2"], model)
        ("stream1", "stream2")

        >>> # One static - not the pattern
        >>> find_dynamic_dynamic_pair(["stream", "params"], model)
        None

        >>> # Both static - not the pattern
        >>> find_dynamic_dynamic_pair(["params1", "params2"], model)
        None
    """
    if len(inputs) != 2:
        return None

    # Check which are static (initializers)
    initializer_names = {x.name for x in model.graph.initializer}
    is_static = [inp in initializer_names for inp in inputs]

    # We need both dynamic (neither static)
    if any(is_static):
        return None

    # Return in original order (lhs, rhs)
    return (inputs[0], inputs[1])


# ============================================================================
# Tensor Property Checks (For Inference-Time Validation)
# ============================================================================

def check_all_integer_types(
    tensors: list[str],
    model: ModelWrapper
) -> bool:
    """Check if all tensors have integer datatypes.

    Args:
        tensors: List of tensor names
        model: ModelWrapper for accessing datatypes

    Returns:
        True if all tensors are integer types, False otherwise
    """
    try:
        for tensor in tensors:
            dt = model.get_tensor_datatype(tensor)
            if not dt.is_integer():
                return False
        return True
    except:
        return False


def check_shapes_equal(
    tensors: list[str],
    model: ModelWrapper
) -> bool:
    """Check if all tensors have identical shapes.

    Args:
        tensors: List of tensor names
        model: ModelWrapper for accessing shapes

    Returns:
        True if all shapes are identical, False otherwise
    """
    try:
        shapes = [tuple(model.get_tensor_shape(t)) for t in tensors]
        return all(s == shapes[0] for s in shapes)
    except:
        return False


def check_parameter_shape_matches_channels(
    data_tensor: str,
    param_tensor: str,
    model: ModelWrapper
) -> bool:
    """Check if parameter shape matches channel count or is scalar.

    Common pattern for channelwise operations where parameters are either:
    - Scalar (broadcast to all channels)
    - Per-channel (one value per channel in last dimension)

    Args:
        data_tensor: Dynamic input tensor name
        param_tensor: Static parameter tensor name
        model: ModelWrapper for accessing shapes

    Returns:
        True if param is scalar or matches last dim of data, False otherwise

    Example:
        >>> # Per-channel match
        >>> check_parameter_shape_matches_channels(
        ...     "data",      # shape: [1, 224, 224, 64]
        ...     "params",    # shape: [64]
        ...     model
        ... )
        True

        >>> # Scalar broadcast
        >>> check_parameter_shape_matches_channels(
        ...     "data",      # shape: [1, 224, 224, 64]
        ...     "scalar",    # shape: [1]
        ...     model
        ... )
        True

        >>> # Mismatch
        >>> check_parameter_shape_matches_channels(
        ...     "data",      # shape: [1, 224, 224, 64]
        ...     "params",    # shape: [32]
        ...     model
        ... )
        False
    """
    try:
        data_shape = model.get_tensor_shape(data_tensor)
        param_shape = model.get_tensor_shape(param_tensor)

        num_channels = data_shape[-1]
        num_params = int(np.prod(param_shape))

        # Valid: scalar (broadcast) or per-channel match
        return num_params == 1 or num_params == num_channels
    except:
        return False


# ============================================================================
# Scalar Broadcasting Helper
# ============================================================================

def expand_scalar_to_channels(
    scalar_tensor_name: str,
    num_channels: int,
    model: ModelWrapper
) -> str:
    """Expand scalar parameter to per-channel vector.

    Creates a new initializer with the scalar value replicated num_channels times.
    The new initializer is added to the model graph.

    Args:
        scalar_tensor_name: Name of scalar initializer
        num_channels: Number of channels to expand to
        model: ModelWrapper to add new initializer

    Returns:
        Name of new expanded initializer

    Example:
        >>> # Before: scalar with value 5
        >>> # After: [5, 5, 5, ..., 5] (64 times)
        >>> expanded_name = expand_scalar_to_channels("bias_scalar", 64, model)
        >>> # expanded_name: "bias_scalar_expanded_64ch"
        >>> # New initializer added to model.graph.initializer
    """
    # Get scalar value (as numpy array)
    scalar_init_array = model.get_initializer(scalar_tensor_name)
    if scalar_init_array is None:
        raise ValueError(f"Tensor '{scalar_tensor_name}' is not an initializer")

    scalar_value = scalar_init_array.flatten()[0]

    # Create expanded tensor
    expanded_values = np.full((num_channels,), scalar_value, dtype=scalar_init_array.dtype)

    # Create new initializer name
    expanded_name = f"{scalar_tensor_name}_expanded_{num_channels}ch"

    # Find original TensorProto in graph.initializer to get data_type
    orig_tensor_proto = None
    for init in model.graph.initializer:
        if init.name == scalar_tensor_name:
            orig_tensor_proto = init
            break

    if orig_tensor_proto is None:
        raise ValueError(f"Could not find TensorProto for '{scalar_tensor_name}'")

    # Create tensor (reuse the data_type from original TensorProto)
    expanded_tensor = helper.make_tensor(
        expanded_name,
        orig_tensor_proto.data_type,  # Reuse ONNX tensor proto type
        [num_channels],
        expanded_values.flatten().tolist()
    )

    # Add to graph
    model.graph.initializer.append(expanded_tensor)

    logger.info(
        f"Expanded scalar parameter '{scalar_tensor_name}' "
        f"→ '{expanded_name}' ({num_channels} channels)"
    )

    return expanded_name


# ============================================================================
# Broadcasting Analysis (Phase 2)
# ============================================================================

def check_shapes_broadcastable(
    tensor1: str,
    tensor2: str,
    model: ModelWrapper
) -> bool:
    """Check if two tensors have broadcastable shapes.

    Uses numpy broadcasting semantics (matches ONNX).

    Args:
        tensor1: First tensor name
        tensor2: Second tensor name
        model: ModelWrapper for accessing shapes

    Returns:
        True if shapes are broadcastable, False otherwise

    Example:
        >>> check_shapes_broadcastable("data", "bias", model)
        True  # [1,64,64,128] and [128] are broadcastable
    """
    try:
        shape1 = tuple(model.get_tensor_shape(tensor1))
        shape2 = tuple(model.get_tensor_shape(tensor2))

        # Use numpy to check broadcastability
        _ = np.broadcast_shapes(shape1, shape2)
        return True
    except (ValueError, AttributeError):
        return False


def get_broadcast_info(
    lhs_tensor: str,
    rhs_tensor: str,
    model: ModelWrapper
):
    """Get broadcasting metadata for two tensors.

    Returns BroadcastInfo object with broadcasting analysis, or None if
    shapes are not broadcastable.

    Args:
        lhs_tensor: Left-hand side tensor name
        rhs_tensor: Right-hand side tensor name
        model: ModelWrapper for accessing shapes

    Returns:
        BroadcastInfo object, or None if not broadcastable

    Example:
        >>> info = get_broadcast_info("activation", "bias", model)
        >>> if info and info.has_broadcast:
        ...     print(f"Broadcasting from {info.rhs_shape} to {info.output_shape}")
    """
    # Import here to avoid circular dependency
    from brainsmith.dataflow.broadcast_helpers import BroadcastInfo

    try:
        lhs_shape = tuple(model.get_tensor_shape(lhs_tensor))
        rhs_shape = tuple(model.get_tensor_shape(rhs_tensor))

        return BroadcastInfo.compute(lhs_shape, rhs_shape)
    except (ValueError, AttributeError):
        return None


# ============================================================================
# Scalar Tensor Normalization
# ============================================================================

def lift_scalar_to_rank1(
    tensor_name: str,
    model: ModelWrapper
) -> bool:
    """Lift scalar tensor (rank 0) to rank-1 tensor with single element.

    ONNX Broadcasting Semantics: Scalar [] and rank-1 [1] are semantically
    equivalent for all broadcasting operations. This function normalizes
    scalar inputs to rank-1 to enable template-based tiling in schema system.

    Why This Is Needed:
    - Brainsmith's schema system uses templates (e.g., block_tiling=[FULL_DIM])
    - Templates expect rank ≥ 1 tensors for dimension specifications
    - ONNX models often use scalar parameters (shape=[]) for operations
    - Lifting [] → [1] preserves semantics while enabling template resolution

    IMPORTANT: This function is safe to call during inference transforms because:
    - QONNX transforms receive deep copies of models by default (make_deepcopy=True)
    - Mutations happen on the copy, not the original model
    - If validation fails, the copy is discarded

    Mutates model in place:
    - Updates tensor shape: [] → [1]
    - Reshapes initializer if present: scalar → array[1]

    Args:
        tensor_name: Name of tensor to potentially lift
        model: ModelWrapper to mutate (operates on deep copy in practice)

    Returns:
        True if tensor was lifted, False if already rank ≥ 1

    Example:
        >>> # Before lift
        >>> model.get_tensor_shape("bias")
        []
        >>> model.get_initializer("bias")
        array(0.5)

        >>> # Lift the scalar
        >>> lifted = lift_scalar_to_rank1("bias", model)
        >>> lifted
        True

        >>> # After lift
        >>> model.get_tensor_shape("bias")
        [1]
        >>> model.get_initializer("bias")
        array([0.5])

        >>> # No-op on rank ≥ 1 tensors
        >>> lift_scalar_to_rank1("weights", model)  # shape=[64]
        False

    Usage in Kernel Inference:
        >>> @classmethod
        >>> def infer_from(cls, node, model, insert_index):
        ...     # Normalize scalar inputs for template-based tiling
        ...     for inp in node.input:
        ...         lift_scalar_to_rank1(inp, model)
        ...
        ...     # Now schema validation will work with rank-1 tensors
        ...     schema = cls.get_schema()
        ...     design_space = schema.build_design_space(...)

    References:
        - FINN's lift_to_rank1() in convert_to_hw_layers.py (lines 1740-1748)
        - ONNX broadcasting semantics: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    """
    # Get current tensor shape
    shape = model.get_tensor_shape(tensor_name)

    # Only lift if scalar (rank 0)
    if len(shape) != 0:
        return False

    # Update shape metadata: [] → [1]
    model.set_tensor_shape(tensor_name, [1])

    # Update initializer if present
    initializer = model.get_initializer(tensor_name)
    if initializer is not None:
        # Reshape scalar to rank-1 array
        model.set_initializer(tensor_name, initializer.reshape(1))

    logger.debug(
        f"Lifted scalar tensor '{tensor_name}' from shape [] to [1] "
        f"(ONNX semantic equivalence for template-based tiling)"
    )

    return True
