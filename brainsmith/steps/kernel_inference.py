# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kernel inference step for hardware mapping."""
import logging
from brainsmith.registry import step, get_transforms_by_metadata, get_transform

logger = logging.getLogger(__name__)

@step(
    name="infer_kernels",
    category="hardware",
    description="Infer hardware kernels based on blueprint selections"
)
def infer_kernels_step(model, cfg):
    """Infer kernels using transforms with matching kernel metadata.
    
    Finds inference transforms by their 'kernel' metadata attribute,
    avoiding any name-based guessing.
    """
    if not hasattr(cfg, 'kernel_selections'):
        logger.warning("No kernel_selections in config, skipping kernel inference")
        logger.warning(f"Config attributes: {[attr for attr in dir(cfg) if not attr.startswith('_')]}")
        return model
    
    if cfg.kernel_selections is None:
        logger.warning("kernel_selections is None, skipping kernel inference")
        return model
    
    logger.info(f"Inferring {len(cfg.kernel_selections)} kernels...")
    
    # Apply inference for each selected kernel
    for kernel_name, backend in cfg.kernel_selections:
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
    
    # Save model for debugging
    import os
    debug_path = os.path.join(cfg.output_dir, "debug_infer_kernels_output.onnx")
    model.save(debug_path)
    logger.info(f"Saved infer_kernels output to {debug_path}")

    return model