"""Backend detection and specialization helpers for parity testing.

This module provides utilities for detecting backend types (HLS vs RTL) and
converting base kernel operators to specialized backend implementations via
the SpecializeLayers transform.

Key Features:
- Backend type detection (HLS, RTL)
- SpecializeLayers transform workflow
- Reusable across test utilities and integration tests
- No dependency on test class state

Usage:
    from tests.parity.backend_helpers import is_hls_backend, setup_hls_backend_via_specialize

    # Check backend type
    if is_hls_backend(op):
        print("Operator supports HLS cppsim execution")

    # Convert base kernel to HLS backend
    hls_op, hls_model = setup_hls_backend_via_specialize(base_op, base_model)
"""

from typing import Tuple
from qonnx.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


def is_hls_backend(op: HWCustomOp) -> bool:
    """Check if op is an HLS backend (has cppsim capability).

    HLS backends inherit from HLSBackend and have code generation methods.
    Only HLS backends can use cppsim execution mode.

    Args:
        op: HWCustomOp instance to check

    Returns:
        True if HLS backend, False otherwise

    Example:
        >>> from finn.custom_op.fpgadataflow.thresholding_hls import Thresholding_hls
        >>> op = Thresholding_hls(node)
        >>> is_hls_backend(op)
        True
    """
    try:
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
        return isinstance(op, HLSBackend)
    except ImportError:
        return False


def is_rtl_backend(op: HWCustomOp) -> bool:
    """Check if op is an RTL backend (has rtlsim capability).

    RTL backends inherit from RTLBackend and have HDL generation methods.
    Only RTL backends can use rtlsim execution mode.

    Args:
        op: HWCustomOp instance to check

    Returns:
        True if RTL backend, False otherwise

    Example:
        >>> from finn.custom_op.fpgadataflow.thresholding_rtl import Thresholding_rtl
        >>> op = Thresholding_rtl(node)
        >>> is_rtl_backend(op)
        True
    """
    try:
        from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
        return isinstance(op, RTLBackend)
    except ImportError:
        return False


def setup_hls_backend_via_specialize(
    base_op: HWCustomOp,
    base_model: ModelWrapper,
    fpgapart: str = "xcvu9p-flgb2104-2-i"
) -> Tuple[HWCustomOp, ModelWrapper]:
    """Setup HLS backend by applying SpecializeLayers transform.

    This matches the production FINN workflow:
    1. Base kernel node with preferred_impl_style="hls"
    2. SpecializeLayers transform → Kernel_hls node
    3. getCustomOp() returns HLS backend class

    Args:
        base_op: Base kernel op instance (e.g., Shuffle, AutoShuffle)
        base_model: Model containing the base kernel node
        fpgapart: FPGA part name for specialization (default: Virtex UltraScale+ VU9P)

    Returns:
        Tuple of (HLS backend op instance, transformed model)

    Raises:
        RuntimeError: If SpecializeLayers does not create specialized backend node

    Example:
        >>> # Create base Shuffle node
        >>> from brainsmith.kernels.shuffle import AutoShuffle
        >>> shuffle_op, model = create_shuffle_model()
        >>>
        >>> # Convert to Shuffle_hls via SpecializeLayers
        >>> shuffle_hls_op, hls_model = setup_hls_backend_via_specialize(
        ...     shuffle_op, model
        ... )
        >>> assert is_hls_backend(shuffle_hls_op)
        >>> assert shuffle_hls_op.onnx_node.op_type == "Shuffle_hls"
    """
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from qonnx.custom_op.registry import getCustomOp

    # Set preferred implementation style to HLS
    base_op.set_nodeattr("preferred_impl_style", "hls")

    # Apply SpecializeLayers transform (matches FINN build flow)
    # This converts domain from "brainsmith.kernels" to "brainsmith.kernels.hls"
    # and op_type from "Shuffle" to "Shuffle_hls"
    model = base_model.transform(SpecializeLayers(fpgapart))

    # Find the specialized node (should be the same node, transformed)
    # The node's domain and op_type have been modified by SpecializeLayers
    specialized_node = None
    for node in model.graph.node:
        if node.domain.endswith(".hls") or node.domain.endswith(".rtl"):
            specialized_node = node
            break

    if specialized_node is None:
        raise RuntimeError(
            f"SpecializeLayers did not create HLS/RTL backend node. "
            f"Node domain: {base_op.onnx_node.domain}, "
            f"preferred_impl_style: {base_op.get_nodeattr('preferred_impl_style')}"
        )

    # Use getCustomOp to retrieve the HLS backend class
    # This will look up the class based on the specialized domain and op_type
    hls_op = getCustomOp(specialized_node)

    return hls_op, model


def setup_rtl_backend_via_specialize(
    base_op: HWCustomOp,
    base_model: ModelWrapper,
    fpgapart: str = "xcvc1902-vsvd1760-2MP-e-S",
    clk_ns: float = 3.0
) -> Tuple[HWCustomOp, ModelWrapper]:
    """Setup RTL backend by applying SpecializeLayers transform.

    This matches the production FINN workflow:
    1. Base kernel node with preferred_impl_style="rtl"
    2. Set clk_ns and fpgapart attributes
    3. SpecializeLayers transform → Kernel_rtl node
    4. getCustomOp() returns RTL backend class

    Args:
        base_op: Base kernel op instance (e.g., Shuffle, AutoShuffle)
        base_model: Model containing the base kernel node
        fpgapart: FPGA part name (default: Versal xcvc1902, DSP58-based)
        clk_ns: Clock period in nanoseconds (default: 3.0ns = ~333MHz)

    Returns:
        Tuple of (RTL backend op instance, transformed model)

    Raises:
        RuntimeError: If SpecializeLayers does not create RTL backend node

    Example:
        >>> # Create base Shuffle node
        >>> from brainsmith.kernels.shuffle import AutoShuffle
        >>> shuffle_op, model = create_shuffle_model()
        >>>
        >>> # Convert to Shuffle_rtl via SpecializeLayers
        >>> shuffle_rtl_op, rtl_model = setup_rtl_backend_via_specialize(
        ...     shuffle_op, model,
        ...     fpgapart="xcvc1902-vsvd1760-2MP-e-S",
        ...     clk_ns=3.0
        ... )
        >>> assert is_rtl_backend(shuffle_rtl_op)
        >>> assert shuffle_rtl_op.onnx_node.op_type == "Shuffle_rtl"
    """
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from qonnx.custom_op.registry import getCustomOp

    # Set preferred implementation style to RTL
    base_op.set_nodeattr("preferred_impl_style", "rtl")

    # Set RTL-specific attributes
    base_op.set_nodeattr("fpgapart", fpgapart)
    base_op.set_nodeattr("clk_ns", clk_ns)

    # Apply SpecializeLayers transform (matches FINN build flow)
    # This converts domain from "brainsmith.kernels" to "brainsmith.kernels.rtl"
    # and op_type from "Shuffle" to "Shuffle_rtl"
    model = base_model.transform(SpecializeLayers(fpgapart))

    # Find the specialized node (should be the same node, transformed)
    # The node's domain and op_type have been modified by SpecializeLayers
    specialized_node = None
    for node in model.graph.node:
        if node.domain.endswith(".rtl"):
            specialized_node = node
            break

    if specialized_node is None:
        raise RuntimeError(
            f"SpecializeLayers did not create RTL backend node. "
            f"Node domain: {base_op.onnx_node.domain}, "
            f"preferred_impl_style: {base_op.get_nodeattr('preferred_impl_style')}"
        )

    # Use getCustomOp to retrieve the RTL backend class
    # This will look up the class based on the specialized domain and op_type
    rtl_op = getCustomOp(specialized_node)

    return rtl_op, model
