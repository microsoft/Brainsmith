"""Backend specialization utilities for test infrastructure.

This module provides utilities for transforming base HWCustomOp kernels
into backend-specialized variants (HLS or RTL) that support code generation
and hardware simulation.

Pattern validated by: tests/spike_backend_specialization.py
"""

from typing import Tuple

from brainsmith.primitives.transforms import SpecializeKernels
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import getHWCustomOp
from qonnx.core.modelwrapper import ModelWrapper


def specialize_to_backend(
    op,  # HWCustomOp (avoid import to prevent circular deps)
    model: ModelWrapper,
    fpgapart: str,
    backend: str = "hls"
):
    """Specialize base kernel to backend variant with code generation capability.

    This function transforms a base HWCustomOp kernel (e.g., AddStreams) into
    a backend-specialized variant (e.g., AddStreams_hls) that inherits from
    HLSBackend or RTLBackend, enabling cppsim/rtlsim execution.

    Transformation flow:
        Stage 2: AddStreams (base kernel)
                    ↓ SpecializeLayers
        Stage 3: AddStreams_hls (backend with HLSBackend inheritance)

    This mirrors the production FINN transformation pipeline and enables
    complete end-to-end testing from ONNX → Base Kernel → Backend → Hardware.

    Args:
        op: Base HWCustomOp kernel instance (e.g., AddStreams).
            Must have a valid op_type that has a registered backend variant.
        model: ModelWrapper containing the ONNX model with the base kernel.
        fpgapart: FPGA part string for specialization (e.g., "xc7z020clg400-1").
            This determines which backend implementations are available.
        backend: Backend type ("hls" or "rtl"). Default: "hls".
            Determines the suffix used to find the specialized node.

    Returns:
        Tuple[HWCustomOp, ModelWrapper]: Specialized operator and transformed model.
            - The operator will have HLSBackend or RTLBackend inheritance
            - The model will contain the specialized node
            - Configuration (PE, folding, etc.) is preserved

    Raises:
        RuntimeError: If specialized node cannot be found after transformation.
            This usually means:
            - No backend registered for the base op_type
            - FPGA part not supported
            - Backend type mismatch

    Example:
        >>> # Create base kernel
        >>> runner = PipelineRunner()
        >>> op, model = runner.run(
        ...     model_factory=make_test_model,
        ...     transform=InferAddStreamsLayer,
        ...     configure_fn=lambda op, model: op.set_nodeattr("PE", 8)
        ... )
        >>> assert op.onnx_node.op_type == "AddStreams"
        >>> assert not isinstance(op, HLSBackend)
        >>>
        >>> # Specialize to backend
        >>> backend_op, backend_model = specialize_to_backend(
        ...     op, model, fpgapart="xc7z020clg400-1", backend="hls"
        ... )
        >>> assert backend_op.onnx_node.op_type == "AddStreams_hls"
        >>> assert isinstance(backend_op, HLSBackend)
        >>> assert backend_op.get_nodeattr("PE") == 8  # Config preserved
        >>>
        >>> # Execute with CppSimExecutor (will NOT skip)
        >>> executor = CppSimExecutor()
        >>> outputs = executor.execute(backend_op, backend_model, inputs)

    Implementation Note:
        This function searches for the specialized node by op_type (e.g.,
        "AddStreams_hls"), NOT by node name. This is critical because
        SpecializeLayers does NOT preserve node names in all cases.

        Pattern discovered by spike test (tests/spike_backend_specialization.py):
        - Before: node.name = "AddStreams_Add_0", op_type = "AddStreams"
        - After:  node.name = "", op_type = "AddStreams_hls"

        Therefore, we MUST search by op_type pattern, not name.

    See Also:
        - tests/spike_backend_specialization.py: Spike test validating this pattern
        - tests/PIPELINE_IMPLEMENTATION_PLAN.md: Staged implementation plan
        - tests/WHOLISTIC_PIPELINE_DESIGN.md: Architecture design document
    """
    # Store original op_type for finding specialized node
    base_op_type = op.onnx_node.op_type

    # Apply SpecializeLayers transform
    # This creates backend variant (e.g., AddStreams → AddStreams_hls)
    model = model.transform(SpecializeLayers(fpgapart))

    # Find specialized node by op_type (NOT by name!)
    # SpecializeLayers does NOT preserve node names reliably
    backend_suffix = f"_{backend}"  # "_hls" or "_rtl"
    expected_op_type = f"{base_op_type}{backend_suffix}"

    specialized_node = None
    for node in model.graph.node:
        if node.op_type == expected_op_type:
            specialized_node = node
            break

    # Raise helpful error if node not found
    if specialized_node is None:
        available_nodes = [(n.name, n.op_type) for n in model.graph.node]
        raise RuntimeError(
            f"Failed to find specialized node with op_type='{expected_op_type}' "
            f"after SpecializeLayers transform.\n"
            f"Base op_type: {base_op_type}\n"
            f"FPGA part: {fpgapart}\n"
            f"Backend: {backend}\n"
            f"Available nodes: {available_nodes}\n"
            f"Possible causes:\n"
            f"  - No {backend.upper()} backend registered for {base_op_type}\n"
            f"  - FPGA part '{fpgapart}' not supported\n"
            f"  - Backend type mismatch (expected '{backend}')"
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
