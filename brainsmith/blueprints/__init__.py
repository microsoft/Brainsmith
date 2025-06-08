"""
Brainsmith Blueprint System - Week 3 Implementation

Comprehensive blueprint system that leverages the Week 2 library structure
to provide blueprint-driven design space exploration.
"""

from .core import Blueprint, BlueprintLoader, BlueprintValidator

# Convenience functions
def load_blueprint(path: str):
    """Load a blueprint from file."""
    loader = BlueprintLoader()
    return loader.load(path)

def validate_blueprint(blueprint):
    """Validate a blueprint."""
    validator = BlueprintValidator()
    return validator.validate(blueprint)

__all__ = [
    'Blueprint',
    'BlueprintLoader',
    'BlueprintValidator', 
    'load_blueprint',
    'validate_blueprint'
]

# Version info
__version__ = "1.0.0"  # Week 3 implementation
