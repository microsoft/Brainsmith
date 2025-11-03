# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kernel inference step for hardware mapping.

This is a thin wrapper around InferKernelList that adapts the build system's
kernel_selections format to the transform's expected input.

The step layer focuses on orchestration (extracting kernel classes from config),
while InferKernelList handles the actual inference dispatch logic.
"""
import logging
from typing import Any

from brainsmith.registry import get_kernel, step
from brainsmith.primitives.transforms import InferKernelList
from qonnx.transformation.general import GiveUniqueNodeNames

logger = logging.getLogger(__name__)


@step(name='infer_kernels')
def infer_kernels_step(model: Any, cfg: Any) -> Any:
    """Infer kernels from configuration (delegates to InferKernelList).

    Extracts kernel classes from cfg.kernel_selections and delegates to
    InferKernelList for actual inference. The backend class is used only
    for logging metadata at the step level.

    Args:
        model: ONNX model to transform
        cfg: Build configuration with kernel_selections attribute

    Returns:
        Transformed model with inferred kernel nodes
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping inference")
        return model

    logger.info(f"Inferring {len(kernel_selections)} kernels...")

    # Extract kernel classes from selections (log backend info)
    kernel_classes = []
    for kernel_name, backend_class in kernel_selections:
        try:
            kernel_class = get_kernel(kernel_name)
            kernel_classes.append(kernel_class)

            # Log backend metadata for user visibility
            backend_name = backend_class.__name__
            backend_language = getattr(backend_class, 'language', 'unknown')
            logger.info(
                f"  {kernel_name} backend={backend_name} language={backend_language}"
            )
        except KeyError:
            logger.warning(f"  Kernel not found in registry: {kernel_name}")

    # Delegate to InferKernelList (single source of truth for inference logic)
    model = model.transform(InferKernelList(kernel_classes))

    # Ensure all nodes have unique names after inference
    # Some legacy FINN transforms (e.g., InferElementwiseBinaryOperation) create
    # nodes without names, which causes issues in downstream steps like partitioning
    model = model.transform(GiveUniqueNodeNames())
    logger.debug("Assigned unique names to all nodes after kernel inference")

    return model
