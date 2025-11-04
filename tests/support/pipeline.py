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

from typing import Callable, Optional, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.util.basic import getHWCustomOp
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth


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
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run ONNX → Hardware transformation pipeline.

        Args:
            model_factory: Function that creates (model, node_name).
                The node_name may be None if transform renames the node.
            transform: Kernel inference transform (manual FINN or auto Brainsmith).
            configure_fn: Optional function to configure the node (set PE, SIMD, etc.).
                Called with (op, model) after node is found but before initialization.
            init_fn: Optional function to initialize the node (prepare_cppsim, etc.).
                Called with (op, model) after configuration.

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
        # Create model
        model, node_name = model_factory()

        # Standard preprocessing
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
