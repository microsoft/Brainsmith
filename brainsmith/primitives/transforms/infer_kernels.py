############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Meta-transform for inferring multiple hardware kernels via pattern matching.

This module provides InferKernels, a smart dispatcher that handles both
new KernelOp kernels and legacy HWCustomOp kernels with their inference
transforms.

IMPORTANT: Requires explicit kernel list to avoid ambiguity in tests.
Auto-discovery has been disabled.

Example usage:
    from brainsmith.primitives.transforms.infer_kernels import InferKernels
    from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp
    from brainsmith.kernels.softmax import Softmax
    from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU

    # Explicit list of kernels to infer (REQUIRED)
    model = model.transform(InferKernels([
        ElementwiseBinaryOp,  # New KernelOp → uses InferKernel
        Softmax,              # New KernelOp → uses InferKernel
        MVAU,                 # Legacy HWCustomOp → uses InferQuantizedMatrixVectorActivation
    ]))

    # Error: No auto-discovery
    model = model.transform(InferKernels())  # ValueError!
"""

import logging

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from .infer_kernel import InferKernel

logger = logging.getLogger(__name__)


class InferKernels(Transformation):
    """Meta-transform for inferring multiple hardware kernels.

    Accepts a list of kernel classes (KernelOp or HWCustomOp) and dispatches
    to the appropriate inference mechanism:
    - KernelOp subclasses → InferKernel(kernel_cls)
    - HWCustomOp (legacy FINN) → lookup registered Infer* transform via metadata

    This replaces the old kernel_filter approach with an explicit list,
    making the inference process more transparent and controllable.

    Args:
        kernel_classes: List of kernel classes to infer. REQUIRED - must be
                       an explicit list to avoid ambiguity about which kernels
                       are being tested.

    Example:
        # Explicit list of kernels to infer
        from brainsmith.kernels.addstreams import AddStreams
        from brainsmith.kernels.softmax import Softmax
        from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU

        model = model.transform(InferKernels([
            AddStreams,  # New KernelOp
            Softmax,     # New KernelOp
            MVAU,        # Legacy FINN kernel
        ]))

    Implementation Notes:
        - Type-based dispatch: checks issubclass(cls, KernelOp)
        - Metadata-driven lookup for legacy transforms
        - Graceful handling of missing transforms (logs warning, continues)
        - Each kernel is processed independently (one failure doesn't stop others)
        - Auto-discovery disabled: explicit list required for clarity
    """

    def __init__(self, kernel_classes: list[type]):
        """Initialize transform with kernel classes.

        Args:
            kernel_classes: List of kernel classes to infer. REQUIRED.

        Raises:
            ValueError: If kernel_classes is None or not provided
        """
        super().__init__()
        if kernel_classes is None:
            raise ValueError(
                "InferKernels requires an explicit list of kernel classes. "
                "Auto-discovery has been disabled to avoid ambiguity in tests. "
                "Example: InferKernels([ElementwiseBinaryOp])"
            )
        self.kernel_classes = kernel_classes

    def apply(self, model: ModelWrapper):
        """Apply kernel inference to the model.

        Dispatches to appropriate inference mechanism for each kernel class.

        Args:
            model: QONNX ModelWrapper to transform

        Returns:
            Tuple of (transformed_model, graph_modified_flag)
        """
        from brainsmith.dataflow import KernelOp
        from brainsmith.registry import get_kernel_infer

        graph_modified = False

        # Use explicit list (None check already done in __init__)
        kernels_to_process = self.kernel_classes

        if not kernels_to_process:
            logger.warning("Empty kernel list provided to InferKernels")
            return (model, False)

        # Process each kernel
        for kernel_cls in kernels_to_process:
            kernel_name = kernel_cls.__name__

            try:
                if issubclass(kernel_cls, KernelOp):
                    # New style: use InferKernel transform
                    logger.debug(f"Inferring {kernel_name} (KernelOp) via InferKernel")
                    transform = InferKernel(kernel_cls)
                    model, modified = transform.apply(model)
                    graph_modified = graph_modified or modified

                else:
                    # Legacy style: lookup registered transform via metadata
                    logger.debug(f"Inferring {kernel_name} (legacy) via metadata lookup")

                    # Use attached registry name (fallback to __name__ for non-registered classes)
                    registry_name = getattr(kernel_cls, "__registry_name__", kernel_cls.__name__)

                    try:
                        transform_cls = get_kernel_infer(registry_name)
                    except KeyError:
                        logger.warning(
                            f"No inference transform found for {kernel_name}. "
                            f"Ensure the kernel is registered with an infer_transform attribute."
                        )
                        continue

                    # Apply the legacy transform
                    logger.debug(f"  Using transform: {transform_cls.__name__}")
                    transform = transform_cls()
                    model, modified = transform.apply(model)
                    graph_modified = graph_modified or modified

            except Exception as e:
                logger.warning(f"Failed to infer {kernel_name}: {e}", exc_info=True)
                # Continue with other kernels (don't fail entire transform)

        return (model, graph_modified)
