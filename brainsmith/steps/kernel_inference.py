# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kernel inference step for hardware mapping."""
import logging
from brainsmith.core.plugins import step, get_kernel
from brainsmith.transforms.infer_kernel_list import InferKernelList

logger = logging.getLogger(__name__)

@step(
    name="infer_kernels",
    category="hardware",
    description="Infer hardware kernels based on blueprint selections"
)
def infer_kernels_step(model, cfg):
    """Infer kernels using InferKernelList meta-transform.

    Converts the kernel_selections from the blueprint into kernel classes
    and passes them to InferKernelList for unified inference.
    """
    if not hasattr(cfg, 'kernel_selections'):
        logger.warning("No kernel_selections in config, skipping kernel inference")
        logger.warning(f"Config attributes: {[attr for attr in dir(cfg) if not attr.startswith('_')]}")
        return model

    if cfg.kernel_selections is None:
        logger.warning("kernel_selections is None, skipping kernel inference")
        return model

    logger.info(f"Inferring {len(cfg.kernel_selections)} kernels...")

    # Get kernel classes from kernel_selections
    kernel_classes = []
    for kernel_name, backend in cfg.kernel_selections:
        try:
            kernel_cls = get_kernel(kernel_name)
            kernel_classes.append(kernel_cls)
            logger.info(f"  {kernel_name} ({backend})")
        except KeyError:
            logger.warning(f"  Kernel not found in registry: {kernel_name}")

    if not kernel_classes:
        logger.warning("No valid kernel classes found, skipping inference")
        return model

    # Apply InferKernelList with all selected kernels
    # InferKernelList handles dispatch to InferKernel or legacy transforms
    logger.info(f"Applying InferKernelList with {len(kernel_classes)} kernel classes")
    model = model.transform(InferKernelList(kernel_classes))

    # Save model for debugging
    import os
    debug_path = os.path.join(cfg.output_dir, "debug_infer_kernels_output.onnx")
    model.save(debug_path)
    logger.info(f"Saved infer_kernels output to {debug_path}")

    return model