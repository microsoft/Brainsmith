# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kernel inference step for hardware mapping."""
import logging
import os
from typing import Any

from brainsmith.registry import get_kernel_infer
from brainsmith.registry import step

logger = logging.getLogger(__name__)


@step(name='infer_kernels')
def infer_kernels_step(model: Any, cfg: Any) -> Any:
    """Infer kernels using transforms stored in kernel metadata.

    Retrieves InferTransforms from kernel metadata via get_kernel_infer().
    The backend class is used only for logging/metadata display.
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping inference")
        return model

    logger.info(f"Inferring {len(kernel_selections)} kernels...")

    # Apply inference for each selected kernel
    for kernel_name, backend_class in kernel_selections:
        try:
            # Get InferTransform from kernel metadata (by kernel name only)
            InferTransform = get_kernel_infer(kernel_name)

            # Log with full backend metadata
            backend_name = backend_class.__name__
            backend_language = getattr(backend_class, 'language', 'unknown')
            logger.info(
                f"  {kernel_name} "
                f"backend={backend_name} "
                f"language={backend_language} "
                f"using {InferTransform.__name__}"
            )

            model = model.transform(InferTransform())
        except KeyError:
            logger.warning(f"  No inference transform found for kernel: {kernel_name}")

    return model
