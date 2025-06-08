"""
Transforms Library - Week 4 Implementation

Organizes and provides access to existing steps/ functionality through
a structured library interface that integrates with the blueprint system.
"""

from .library import TransformsLibrary
from .registry import TransformRegistry, discover_transforms
from .pipeline import TransformPipeline

# Convenience functions
def get_available_transforms():
    """Get list of available transforms."""
    registry = TransformRegistry()
    return registry.get_available_transforms()

def create_pipeline(transforms_config):
    """Create transform pipeline from configuration."""
    pipeline = TransformPipeline()
    return pipeline.from_config(transforms_config)

def apply_transforms(model, transforms_config):
    """Apply transforms to a model."""
    library = TransformsLibrary()
    library.initialize()
    return library.execute("apply_transforms", {
        'model': model,
        'transforms_config': transforms_config
    })

__all__ = [
    'TransformsLibrary',
    'TransformRegistry',
    'TransformPipeline',
    'discover_transforms',
    'get_available_transforms',
    'create_pipeline',
    'apply_transforms'
]

# Version info
__version__ = "1.0.0"  # Week 4 implementation