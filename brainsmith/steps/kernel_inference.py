# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kernel inference step for hardware mapping."""
import logging
import os
from typing import Any

from brainsmith.loader import get_kernel_infer

logger = logging.getLogger(__name__)


def infer_kernels_step(model: Any, cfg: Any) -> Any:
    """Infer kernels using transforms stored in kernel metadata.

    In the new system, InferTransforms are stored as metadata on kernels,
    retrieved via get_kernel_infer(kernel_name).
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping inference")
        return model

    logger.info(f"Inferring {len(kernel_selections)} kernels...")

    # Apply inference for each selected kernel
    for kernel_name, backend in kernel_selections:
        try:
            # Get InferTransform from kernel metadata
            InferTransform = get_kernel_infer(kernel_name)
            transform_name = InferTransform.__name__
            logger.info(f"  {kernel_name} ({backend}) using {transform_name}")
            model = model.transform(InferTransform())
        except KeyError:
            logger.warning(f"  No inference transform found for kernel: {kernel_name}")

    return model
