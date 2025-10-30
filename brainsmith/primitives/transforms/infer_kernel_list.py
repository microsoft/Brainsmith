############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Meta-transform for inferring multiple hardware kernels.

This module provides InferKernelList, a smart dispatcher that handles both
new KernelOp kernels and legacy HWCustomOp kernels with their inference
transforms.

Example usage:
    from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
    from brainsmith.kernels.addstreams import AddStreams
    from brainsmith.kernels.softmax import Softmax
    from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU

    # Infer specific kernels (mix of new and legacy)
    model = model.transform(InferKernelList([
        AddStreams,  # New KernelOp → uses InferKernel
        Softmax,     # New KernelOp → uses InferKernel
        MVAU,        # Legacy HWCustomOp → uses InferQuantizedMatrixVectorActivation
    ]))

    # Infer all registered kernels (backward compatible)
    model = model.transform(InferKernelList())
"""

import logging
from typing import List, Optional, Type

from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper

from .infer_kernel import InferKernel

logger = logging.getLogger(__name__)


class InferKernelList(Transformation):
    """Meta-transform for inferring multiple hardware kernels.

    Accepts a list of kernel classes (KernelOp or HWCustomOp) and dispatches
    to the appropriate inference mechanism:
    - KernelOp subclasses → InferKernel(kernel_cls)
    - HWCustomOp (legacy FINN) → lookup registered Infer* transform via metadata

    This replaces the old kernel_filter approach with an explicit list,
    making the inference process more transparent and controllable.

    Args:
        kernel_classes: List of kernel classes to infer. If None, infers all
                       registered kernels (backward compatible).

    Example:
        # Explicit list of kernels to infer
        from brainsmith.kernels.addstreams import AddStreams
        from brainsmith.kernels.softmax import Softmax
        from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU

        model = model.transform(InferKernelList([
            AddStreams,  # New KernelOp
            Softmax,     # New KernelOp
            MVAU,        # Legacy FINN kernel
        ]))

        # Backward compatible: infer all registered kernels
        model = model.transform(InferKernelList())

    Implementation Notes:
        - Type-based dispatch: checks issubclass(cls, KernelOp)
        - Metadata-driven lookup for legacy transforms
        - Graceful handling of missing transforms (logs warning, continues)
        - Each kernel is processed independently (one failure doesn't stop others)
    """

    def __init__(self, kernel_classes: Optional[List[Type]] = None):
        """Initialize transform with kernel classes.

        Args:
            kernel_classes: List of kernel classes to infer.
                          If None, infers all registered KernelOp kernels.
        """
        super().__init__()
        self.kernel_classes = kernel_classes

    def apply(self, model: ModelWrapper):
        """Apply kernel inference to the model.

        Dispatches to appropriate inference mechanism for each kernel class.

        Args:
            model: QONNX ModelWrapper to transform

        Returns:
            Tuple of (transformed_model, graph_modified_flag)
        """
        from brainsmith.registry import list_kernels, get_kernel, get_kernel_infer
        from brainsmith.dataflow import KernelOp

        graph_modified = False

        # Determine which kernels to process
        if self.kernel_classes is None:
            # Backward compatible: infer all registered KernelOp kernels
            logger.info("No kernel classes specified, inferring all registered KernelOp kernels")
            all_kernel_names = list_kernels()
            kernels_to_process = []
            for name in all_kernel_names:
                cls = get_kernel(name)
                if issubclass(cls, KernelOp):
                    # Include all KernelOp subclasses
                    # can_infer_from() will determine if transformation applies
                    kernels_to_process.append(cls)
        else:
            # Use explicit list
            kernels_to_process = self.kernel_classes

        if not kernels_to_process:
            logger.warning("No kernels to infer")
            return (model, False)

        # Process each kernel
        for kernel_cls in kernels_to_process:
            kernel_name = kernel_cls.__name__

            try:
                if issubclass(kernel_cls, KernelOp):
                    # New style: use InferKernel transform
                    logger.info(f"Inferring {kernel_name} (KernelOp) via InferKernel")
                    transform = InferKernel(kernel_cls)
                    model, modified = transform.apply(model)
                    graph_modified = graph_modified or modified

                else:
                    # Legacy style: lookup registered transform via metadata
                    logger.info(f"Inferring {kernel_name} (legacy) via metadata lookup")

                    # Use attached registry name (fallback to __name__ for non-registered classes)
                    registry_name = getattr(kernel_cls, '__registry_name__', kernel_cls.__name__)

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
                logger.warning(
                    f"Failed to infer {kernel_name}: {e}",
                    exc_info=True
                )
                # Continue with other kernels (don't fail entire transform)

        return (model, graph_modified)
