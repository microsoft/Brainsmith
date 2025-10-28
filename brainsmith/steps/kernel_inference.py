# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kernel inference step for hardware mapping."""
import logging
from typing import Any

from brainsmith.registry import get_kernel_infer, get_kernel
from brainsmith.registry import step

logger = logging.getLogger(__name__)


@step(name='infer_kernels')
def infer_kernels_step(model: Any, cfg: Any) -> Any:
    """Infer kernels using appropriate inference method.

    For modern KernelOp kernels with built-in inference (can_infer_from),
    uses InferKernel wrapper. For legacy kernels, retrieves explicit
    InferTransform from kernel metadata via get_kernel_infer().

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
            # Get kernel class to check its capabilities
            kernel_class = get_kernel(kernel_name)
        except KeyError:
            logger.warning(f"  Kernel not found in registry: {kernel_name}")
            continue

        backend_name = backend_class.__name__
        backend_language = getattr(backend_class, 'language', 'unknown')

        # Check if kernel has built-in inference (duck typing)
        if hasattr(kernel_class, 'can_infer_from'):
            # Modern KernelOp with built-in inference methods
            from brainsmith.transforms.infer_kernel import InferKernel

            logger.info(
                f"  {kernel_name} "
                f"backend={backend_name} "
                f"language={backend_language} "
                f"using InferKernel wrapper"
            )

            model = model.transform(InferKernel(kernel_class))
        else:
            # Legacy kernel - needs explicit InferTransform from registry
            try:
                InferTransform = get_kernel_infer(kernel_name)

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
