"""
Simplified Blueprint System - North Star Aligned Implementation

Provides simple functions for working with blueprint YAML files,
following the North Star axioms:
- Functions Over Frameworks
- Simplicity Over Sophistication 
- Focus Over Feature Creep
"""

from .functions import (
    load_blueprint_yaml,
    validate_blueprint_yaml,
    get_build_steps,
    get_objectives,
    get_constraints,
    get_kernels,
    get_transforms,
    create_simple_blueprint,
    save_blueprint_yaml
)

# Backward compatibility aliases
def load_blueprint(path: str):
    """Load a blueprint from file (backward compatibility)."""
    return load_blueprint_yaml(path)

def validate_blueprint(blueprint_data):
    """Validate a blueprint (backward compatibility)."""
    return validate_blueprint_yaml(blueprint_data)

__all__ = [
    # Primary simplified functions
    'load_blueprint_yaml',
    'validate_blueprint_yaml', 
    'get_build_steps',
    'get_objectives',
    'get_constraints',
    'get_kernels',
    'get_transforms',
    'create_simple_blueprint',
    'save_blueprint_yaml',
    
    # Backward compatibility
    'load_blueprint',
    'validate_blueprint'
]

# Version info
__version__ = "2.0.0"  # Simplified implementation aligned with North Star axioms
