"""Unified pipeline execution for test frameworks.

This module provides PipelineRunner, which consolidates the duplicated pipeline
execution logic from CoreParityTest, HWEstimationParityTest, and IntegratedPipelineTest.

Design Philosophy:
- Single source of truth for ONNX → Hardware transformation
- Composable: works with any model factory and transform
- Flexible: optional node finding and configuration hooks
- Testable: pure functions, no hidden state

Usage:
    runner = PipelineRunner()

    # Manual (FINN) pipeline
    op, model = runner.run(
        model_factory=lambda: make_test_model(),
        transform=InferAddStreamsLayer(),
        configure_fn=lambda op, model: op.set_nodeattr("PE", 8)
    )

    # Auto (Brainsmith) pipeline
    op, model = runner.run(
        model_factory=lambda: make_test_model(),
        transform=InferKernels()
    )
"""

from typing import Callable, Optional, Tuple, Dict

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
import qonnx.core.data_layout as DataLayout
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.util.basic import getHWCustomOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth


def annotate_model_with_qonnx(
    model: ModelWrapper,
    annotations: Dict[str, DataType],
    layouts: Optional[Dict[str, DataLayout]] = None
) -> ModelWrapper:
    """Add QONNX DataType and layout annotations to pure ONNX model.

    This is the Stage 1 → Stage 2 transition:
    - Takes pure ONNX model (TensorProto types only)
    - Adds QONNX DataType annotations
    - Optionally adds DataLayout annotations
    - Enables InferDataTypes to propagate semantics

    Args:
        model: Pure ONNX model (no QONNX annotations)
        annotations: Dict mapping tensor names → QONNX DataTypes
        layouts: Optional dict mapping tensor names → DataLayouts

    Returns:
        Model with QONNX annotations added

    Example:
        >>> # Stage 1: Create pure ONNX
        >>> onnx_model = make_onnx_model()
        >>>
        >>> # Stage 1 → Stage 2: Add QONNX annotations
        >>> annotations = {
        ...     "input": DataType["INT8"],
        ...     "param": DataType["INT8"],
        ...     "output": DataType["INT16"]
        ... }
        >>> qonnx_model = annotate_model_with_qonnx(onnx_model, annotations)
        >>>
        >>> # Now InferDataTypes can propagate
        >>> qonnx_model = qonnx_model.transform(InferDataTypes())
    """
    # Add DataType annotations
    for tensor_name, dtype in annotations.items():
        model.set_tensor_datatype(tensor_name, dtype)

    # Add layout annotations (optional)
    if layouts:
        for tensor_name, layout in layouts.items():
            model.set_tensor_layout(tensor_name, layout)

    return model


class PipelineRunner:
    """Unified pipeline execution for manual/auto/custom kernel inference.

    Replaces duplicated run_manual_pipeline/run_auto_pipeline/run_inference_pipeline
    methods across CoreParityTest, HWEstimationParityTest, and IntegratedPipelineTest.

    The pipeline follows this standard sequence:
    1. Create model via model_factory
    2. Infer shapes (QONNX)
    3. Infer datatypes (QONNX)
    4. Apply kernel transform (FINN/Brainsmith)
    5. Minimize weight bit width (FINN optimization, no-op if not supported)
    6. Minimize accumulator width (FINN optimization, no-op if not supported)
    7. Find hardware node (by name or default to first node)
    8. Get HWCustomOp wrapper
    9. Configure node (optional)
    10. Initialize node (optional, if init_fn provided)

    This sequence is consistent across all test frameworks, with only the
    model creation, transform type, and optional configuration varying.

    Note: Steps 5-6 match FINN's standard workflow. If a kernel doesn't implement
    minimize_weight_bit_width() or minimize_accumulator_width(), these transforms
    have no effect and are safely skipped.
    """

    def run(
        self,
        model_factory: Callable[[], Tuple[ModelWrapper, Optional[str]]],
        transform: Transformation,
        configure_fn: Optional[Callable[[HWCustomOp, ModelWrapper], None]] = None,
        init_fn: Optional[Callable[[HWCustomOp, ModelWrapper], None]] = None,
        qonnx_annotations: Optional[Dict[str, DataType]] = None,
        qonnx_layouts: Optional[Dict[str, DataLayout]] = None
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run ONNX → Hardware transformation pipeline.

        Standard pipeline sequence:
        1. Create model (from factory)
        2. Add QONNX annotations (if provided) ← NEW
        3. InferShapes
        4. InferDataTypes (requires QONNX annotations)
        5. Apply kernel transform
        6. MinimizeWeightBitWidth
        7. MinimizeAccumulatorWidth
        8. Find HW node
        9. Configure node (optional)
        10. Initialize node (optional)

        Args:
            model_factory: Function that creates (model, node_name).
                The node_name may be None if transform renames the node.
            transform: Kernel inference transform (manual FINN or auto Brainsmith).
            configure_fn: Optional function to configure the node (set PE, SIMD, etc.).
                Called with (op, model) after node is found but before initialization.
            init_fn: Optional function to initialize the node (prepare_cppsim, etc.).
                Called with (op, model) after configuration.
            qonnx_annotations: QONNX DataType annotations (Stage 1→2 transition)
            qonnx_layouts: QONNX DataLayout annotations (optional)

        Returns:
            (op, model): Hardware operator and transformed model

        Example:
            # Manual FINN pipeline with configuration
            op, model = runner.run(
                model_factory=lambda: make_addstreams_model(),
                transform=InferAddStreamsLayer(),
                configure_fn=lambda op, model: op.set_nodeattr("PE", 8)
            )

            # Auto Brainsmith pipeline with cppsim initialization
            op, model = runner.run(
                model_factory=lambda: make_addstreams_model(),
                transform=InferKernels(),
                init_fn=lambda op, model: op.prepare_cppsim(model)
            )
        """
        # Create model (Stage 1: Pure ONNX)
        model, node_name = model_factory()

        # Stage 1 → Stage 2: Add QONNX annotations (if provided)
        if qonnx_annotations:
            model = annotate_model_with_qonnx(model, qonnx_annotations, qonnx_layouts)

        # Standard preprocessing (requires QONNX annotations for InferDataTypes)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Apply kernel transform
        model = model.transform(transform())

        # Apply standard FINN optimizations
        # These are no-ops if the kernel doesn't support them
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())

        # Find hardware node
        hw_node = None
        if node_name is not None:
            hw_node = model.get_node_from_name(node_name)

        if hw_node is None:
            # Transform may have renamed the node, or node_name was None
            # Default to first node (common in single-node test models)
            if len(model.graph.node) == 0:
                raise ValueError(
                    "Model has no nodes after transformation. "
                    "Transform may have failed or removed the target node."
                )
            hw_node = model.graph.node[0]

        # Get HWCustomOp wrapper
        op = getHWCustomOp(hw_node, model)

        # Initialize KernelOp design space if applicable (before configuration)
        from brainsmith.dataflow.kernel_op import KernelOp
        if isinstance(op, KernelOp):
            op._ensure_ready(model)

        # Configure if needed (PE, SIMD, etc.)
        if configure_fn is not None:
            configure_fn(op, model)

            # Re-initialize KernelOp if configuration changed dimension parameters
            if isinstance(op, KernelOp):
                op._ensure_ready(model)

        # Initialize if needed (prepare_cppsim, prepare_rtlsim, etc.)
        if init_fn is not None:
            init_fn(op, model)

        return op, model


# Convenience factory for common test patterns

def make_manual_pipeline_runner(
    configure_fn: Optional[Callable[[HWCustomOp, ModelWrapper], None]] = None
) -> Callable[[Callable[[], Tuple[ModelWrapper, Optional[str]]], Transformation], Tuple[HWCustomOp, ModelWrapper]]:
    """Create a pipeline runner bound with manual (FINN) configuration.

    This is a convenience factory for the common pattern of running manual
    FINN pipelines with optional configuration.

    Args:
        configure_fn: Optional configuration function for manual ops

    Returns:
        Function that takes (model_factory, transform) and returns (op, model)

    Example:
        run_manual = make_manual_pipeline_runner(
            configure_fn=lambda op, model: op.set_nodeattr("PE", 8)
        )
        op, model = run_manual(make_test_model, InferAddStreamsLayer())
    """
    runner = PipelineRunner()
    return lambda model_factory, transform: runner.run(
        model_factory=model_factory,
        transform=transform,
        configure_fn=configure_fn
    )


def make_auto_pipeline_runner(
    configure_fn: Optional[Callable[[HWCustomOp, ModelWrapper], None]] = None
) -> Callable[[Callable[[], Tuple[ModelWrapper, Optional[str]]], Transformation], Tuple[HWCustomOp, ModelWrapper]]:
    """Create a pipeline runner bound with auto (Brainsmith) configuration.

    This is a convenience factory for the common pattern of running auto
    Brainsmith pipelines with optional configuration.

    Args:
        configure_fn: Optional configuration function for auto ops

    Returns:
        Function that takes (model_factory, transform) and returns (op, model)

    Example:
        run_auto = make_auto_pipeline_runner(
            configure_fn=lambda op, model: (
                op.set_nodeattr("PE", 8),
                op._ensure_ready(model) if isinstance(op, KernelOp) else None
            )
        )
        op, model = run_auto(make_test_model, InferKernels())
    """
    runner = PipelineRunner()
    return lambda model_factory, transform: runner.run(
        model_factory=model_factory,
        transform=transform,
        configure_fn=configure_fn
    )
