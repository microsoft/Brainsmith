"""Kernel inference step for hardware mapping."""
import logging
from brainsmith.core.plugins import step, transforms

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
        logger.debug("No kernel_selections in config, skipping kernel inference")
        return model
    
    logger.info(f"Inferring {len(cfg.kernel_selections)} kernels...")
    
    # Apply inference for each selected kernel
    for kernel_name, backend in cfg.kernel_selections:
        # Find transforms that infer this kernel
        inference_transforms = transforms.find(kernel=kernel_name)
        
        if inference_transforms:
            # Use the first matching transform
            transform = inference_transforms[0]
            logger.info(f"  {kernel_name} ({backend}) using {transform.__name__}")
            model = model.transform(transform())
        else:
            logger.warning(f"  No inference transform found for kernel: {kernel_name}")
    
    # Ensure custom opsets are imported
    ensure_imports = transforms.find(name="EnsureCustomOpsetImports")
    if ensure_imports:
        model = model.transform(ensure_imports[0]())
    
    return model