"""Backend specialization utilities for test infrastructure.

This module provides utilities for transforming base HWCustomOp kernels
into backend-specialized variants (HLS or RTL) that support code generation
and hardware simulation.

Pattern validated by: tests/spike_backend_specialization.py
"""

from typing import Tuple, List, Type

from finn.transformation.fpgadataflow.specialize_kernel import SpecializeKernel
from finn.util.basic import getHWCustomOp
from qonnx.core.modelwrapper import ModelWrapper


def specialize_to_backend(
    op,  # HWCustomOp (avoid import to prevent circular deps)
    model: ModelWrapper,
    fpgapart: str,
    backend_variants: List[Type]
) -> Tuple:
    """Specialize base kernel to backend variant with code generation capability.

    This function transforms a base HWCustomOp kernel (e.g., ElementwiseBinaryOp) into
    a backend-specialized variant (e.g., ElementwiseBinaryOp_hls) that inherits from
    HLSBackend or RTLBackend, enabling cppsim/rtlsim execution.

    Transformation flow:
        Stage 2: ElementwiseBinaryOp (base kernel)
                    ↓ SpecializeKernel (with explicit backend classes)
        Stage 3: ElementwiseBinaryOp_hls (backend with HLSBackend inheritance)

    This uses FINN's SpecializeKernel transform which:
    - Takes explicit backend classes (no string matching)
    - Extracts correct domain from backend.__module__
    - Handles both FINN and Brainsmith backends correctly
    - Supports priority ordering (try RTL first, fallback to HLS)

    Args:
        op: Base HWCustomOp kernel instance (e.g., ElementwiseBinaryOp).
            Must have a valid op_type that has a registered backend variant.
        model: ModelWrapper containing the ONNX model with the base kernel.
        fpgapart: FPGA part string for specialization (e.g., "xc7z020clg400-1").
            This determines which backend implementations are available.
        backend_variants: List of backend classes in priority order.
            Example: [ElementwiseBinaryOp_hls]
            Example: [MVAU_rtl, MVAU_hls]  # Try RTL first, fallback to HLS

    Returns:
        Tuple[HWCustomOp, ModelWrapper]: Specialized operator and transformed model.
            - The operator will have HLSBackend or RTLBackend inheritance
            - The model will contain the specialized node
            - Configuration (PE, folding, etc.) is preserved

    Raises:
        RuntimeError: If specialized node cannot be found after transformation.
            This usually means:
            - No backend could satisfy constraints for the given FPGA part
            - Backend classes are not properly registered in QONNX

    Example:
        >>> # Import backend class
        >>> from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp_hls
        >>>
        >>> # Specialize to backend
        >>> backend_op, backend_model = specialize_to_backend(
        ...     op, model,
        ...     fpgapart="xc7z020clg400-1",
        ...     backend_variants=[ElementwiseBinaryOp_hls]
        ... )
        >>> assert backend_op.onnx_node.op_type == "ElementwiseBinaryOp_hls"
        >>> assert isinstance(backend_op, HLSBackend)
        >>> assert backend_op.get_nodeattr("PE") == 4  # Config preserved

    Example (priority order):
        >>> # Try RTL first, fallback to HLS
        >>> from brainsmith.kernels.mvau import MVAU_rtl, MVAU_hls
        >>> backend_op, backend_model = specialize_to_backend(
        ...     op, model,
        ...     fpgapart="xc7z020clg400-1",
        ...     backend_variants=[MVAU_rtl, MVAU_hls]
        ... )

    See Also:
        - deps/finn/src/finn/transformation/fpgadataflow/specialize_kernel.py
        - tests/spike_backend_specialization.py: Spike test validating this pattern
    """
    kernel_class = type(op)
    base_op_type = op.onnx_node.op_type

    # Use FINN's SpecializeKernel with explicit backend variants
    # It will try each backend in priority order and select the first one that:
    # 1. Exists in QONNX registry (via hasCustomOp check)
    # 2. Meets constraints for the given FPGA part
    model = model.transform(SpecializeKernel(kernel_class, backend_variants, fpgapart))

    # Find specialized node (SpecializeKernel mutates op_type to add backend suffix)
    # Try each backend variant to find which one was selected
    specialized_node = None
    for backend_cls in backend_variants:
        # Get expected op_type from backend class name
        expected_op_type = backend_cls.__name__

        for node in model.graph.node:
            if node.op_type == expected_op_type:
                specialized_node = node
                break

        if specialized_node:
            break

    # Raise helpful error if node not found
    if specialized_node is None:
        available_nodes = [(n.name, n.op_type, n.domain) for n in model.graph.node]
        variant_names = [v.__name__ for v in backend_variants]
        raise RuntimeError(
            f"Failed to find specialized node after SpecializeKernel transform.\n"
            f"Base kernel: {kernel_class.__name__} (op_type={base_op_type})\n"
            f"FPGA part: {fpgapart}\n"
            f"Tried backend variants: {variant_names}\n"
            f"Available nodes: {available_nodes}\n\n"
            f"Possible causes:\n"
            f"  - No backend met constraints for FPGA part '{fpgapart}'\n"
            f"  - Backend classes not registered in QONNX (check __init__.py imports)\n"
            f"  - Backend domain/op_type mismatch in QONNX registry"
        )

    # Get specialized operator instance
    specialized_op = getHWCustomOp(specialized_node, model)

    return specialized_op, model


def verify_backend_inheritance(op, backend: str = "hls") -> bool:
    """Verify that operator has correct backend inheritance.

    Args:
        op: HWCustomOp instance to check.
        backend: Expected backend type ("hls" or "rtl").

    Returns:
        bool: True if operator has correct backend inheritance.

    Example:
        >>> op, model = specialize_to_backend(base_op, model, "xc7z020clg400-1")
        >>> assert verify_backend_inheritance(op, "hls")
    """
    if backend == "hls":
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
        return isinstance(op, HLSBackend)
    elif backend == "rtl":
        from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
        return isinstance(op, RTLBackend)
    else:
        raise ValueError(f"Unknown backend type: {backend}")


def get_backend_suffix(backend: str) -> str:
    """Get op_type suffix for backend variant.

    Args:
        backend: Backend type ("hls" or "rtl").

    Returns:
        str: Suffix for backend variant ("_hls" or "_rtl").

    Example:
        >>> get_backend_suffix("hls")
        '_hls'
    """
    if backend not in ("hls", "rtl"):
        raise ValueError(f"Unknown backend type: {backend}. Must be 'hls' or 'rtl'.")
    return f"_{backend}"
