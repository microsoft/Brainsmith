# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kernel inference step for hardware mapping."""
import logging
import os
from typing import Any

from brainsmith.registry import step, get_transforms_by_metadata, get_transform

logger = logging.getLogger(__name__)

@step(
    name="infer_kernels",
    category="hardware",
    description="Infer hardware kernels based on blueprint selections"
)
def infer_kernels_step(model: Any, cfg: Any) -> Any:
    """Infer kernels using transforms with matching kernel metadata.

    Finds inference transforms by their 'kernel' metadata attribute,
    avoiding any name-based guessing.
    """
    kernel_selections = getattr(cfg, 'kernel_selections', None)
    if not kernel_selections:
        logger.debug("No kernel selections configured, skipping inference")
        return model

    logger.info(f"Inferring {len(kernel_selections)} kernels...")

    # Apply inference for each selected kernel
    for kernel_name, backend in kernel_selections:
        # Find transforms that infer this kernel
        inference_transforms = get_transforms_by_metadata(kernel=kernel_name)
        
        if inference_transforms:
            # Use the first matching transform name
            transform_name = inference_transforms[0]
            Transform = get_transform(transform_name)
            logger.info(f"  {kernel_name} ({backend}) using {transform_name}")
            model = model.transform(Transform())
        else:
            logger.warning(f"  No inference transform found for kernel: {kernel_name}")

    return model
